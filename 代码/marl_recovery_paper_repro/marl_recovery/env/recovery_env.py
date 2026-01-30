# -*- coding: utf-8 -*-
"""事件驱动的灾后恢复环境（MDP）。

对应论文 2.5：
- 状态：基础设施组件损伤/功能/可达性等（图结构）
- 动作：每个修复单元（RU）选择一个组件修复
- 状态转移：执行联合动作后，推进到“下一次修复完成事件”
- 奖励：目标1 Eq.(20)，目标2 Eq.(21)

本环境不依赖 Gym；采用一个轻量级 API：
- reset(seed) -> obs
- step(actions) -> (obs, reward, done, info)

obs 是一个字典，包含：
- node_features: (N,5) float32，特征顺序为 [ds, FN, NRU, AC, Er]
- action_masks: List[np.ndarray]，每个 agent 的动作 mask（1=可选，0=非法）
- time: 当前时间（天）

注意：为了尽量对齐论文，环境时间是连续的，但决策在离散事件点发生。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from marl_recovery.data.loader import DataBundle, load_data_bundle
from marl_recovery.env.interdependency import SystemIndicators, compute_functionality_and_building_indicators
from marl_recovery.env.repair_time import sample_truncated_lognormal
from marl_recovery.env.resilience import compute_Q


@dataclass
class EnvConfig:
    """环境配置。"""

    objective: str = "resilience"  # 'resilience' or 'time80'
    control_time: float = 100.0
    threshold_q: float = 0.8

    # 维修时间分布参数
    lognormal_sigma: float = 0.4
    trunc_low_ratio: float = 0.1
    trunc_high_ratio: float = 10.0


@dataclass
class AgentSpec:
    """一个修复单元（agent）的规格。"""

    name: str
    comp_type: str  # pipeline/substation/bridge
    local_index: int  # 同类型中的编号


class RecoveryEnv:
    """灾后恢复环境。"""

    def __init__(self, bundle: DataBundle, ru: Dict[str, int], cfg: EnvConfig):
        self.bundle = bundle
        self.cfg = cfg

        self.ru = dict(ru)  # 保存 RU 配置（用于 clone / rollout）

        # 生成 agent 列表
        self.agents: List[AgentSpec] = []
        for comp_type, n in ru.items():
            for i in range(int(n)):
                self.agents.append(AgentSpec(name=f"{comp_type}_{i}", comp_type=comp_type, local_index=i))

        self.n_agents = len(self.agents)

        # 每个 agent 的动作空间：len(type_indices) + 1（0=空/继续）
        self.type_to_action_nodes: Dict[str, List[int]] = {}
        for comp_type in {a.comp_type for a in self.agents}:
            self.type_to_action_nodes[comp_type] = self.bundle.type_to_indices.get(comp_type, [])

        self.rng = np.random.default_rng(0)

        # 状态变量
        self.time: float = 0.0
        self.phys_ok: np.ndarray = np.ones(self.bundle.num_nodes, dtype=np.float32)
        self.damage_state: np.ndarray = np.zeros(self.bundle.num_nodes, dtype=np.int32)
        self.expected_repair_time: np.ndarray = np.zeros(self.bundle.num_nodes, dtype=np.float32)

        self.nru: np.ndarray = np.zeros(self.bundle.num_nodes, dtype=np.int32)

        # 维修任务：component_index -> remaining_work（单位：天 * 1 RU）
        self.remaining_work: Dict[int, float] = {}
        self.comp_to_agents: Dict[int, List[int]] = {}

        # agent 当前任务
        self.agent_target: List[Optional[int]] = [None for _ in range(self.n_agents)]
        self.agent_busy: np.ndarray = np.zeros(self.n_agents, dtype=np.int32)

        # 指标
        self.indicators: Optional[SystemIndicators] = None

    @staticmethod
    def from_data_dir(data_dir: str, ru: Dict[str, int], cfg: EnvConfig) -> "RecoveryEnv":
        bundle = load_data_bundle(data_dir)
        return RecoveryEnv(bundle=bundle, ru=ru, cfg=cfg)

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """重置环境到初始损伤状态。"""
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        self.time = 0.0

        n = self.bundle.num_nodes
        self.damage_state = np.array([int(node.damage_state) for node in self.bundle.nodes], dtype=np.int32)
        self.expected_repair_time = np.array([float(node.expected_repair_time) for node in self.bundle.nodes], dtype=np.float32)

        # 物理功能：damage_state==0 视为完好，否则需要修复
        self.phys_ok = (self.damage_state == 0).astype(np.float32)

        self.nru[:] = 0
        self.remaining_work.clear()
        self.comp_to_agents.clear()
        self.agent_target = [None for _ in range(self.n_agents)]
        self.agent_busy[:] = 0

        # 计算 FN/AC/建筑 I 指标
        self.indicators = compute_functionality_and_building_indicators(self.bundle, phys_ok=self.phys_ok)

        return self._build_obs()

    def _build_obs(self) -> Dict[str, np.ndarray]:
        """构造观测。"""
        assert self.indicators is not None
        n = self.bundle.num_nodes

        ds = self.damage_state.astype(np.float32)
        fn = self.indicators.fn.astype(np.float32)
        nru = self.nru.astype(np.float32)
        ac = self.indicators.ac.astype(np.float32)
        er = self.expected_repair_time.astype(np.float32)

        node_features = np.stack([ds, fn, nru, ac, er], axis=1).astype(np.float32)  # (N,5)

        masks = self.get_action_masks()
        return {
            "node_features": node_features,
            "time": np.array([self.time], dtype=np.float32),
            "action_masks": masks,
        }

    def get_action_masks(self) -> List[np.ndarray]:
        """获取每个 agent 的动作 mask（1=可选，0=非法）。"""
        assert self.indicators is not None

        masks: List[np.ndarray] = []
        for ai, agent in enumerate(self.agents):
            action_nodes = self.type_to_action_nodes.get(agent.comp_type, [])
            dim = len(action_nodes) + 1
            mask = np.zeros(dim, dtype=np.float32)

            # action 0：空/继续，总是允许（busy 时只能选这个）
            mask[0] = 1.0

            if self.agent_busy[ai] == 1:
                # 忙碌时只能继续当前任务
                masks.append(mask)
                continue

            # 空闲时：对每个组件动作判断是否合法
            for j, comp in enumerate(action_nodes):
                # comp 必须未修复（phys_ok==0）且可达（AC==1）
                if self.phys_ok[comp] <= 0.0 and self.indicators.ac[comp] > 0.0:
                    mask[j + 1] = 1.0

            masks.append(mask)

        return masks

    def _action_to_component(self, agent: AgentSpec, action: int) -> Optional[int]:
        """将离散动作映射为组件索引。action=0 表示不选择/继续。"""
        if action <= 0:
            return None
        action_nodes = self.type_to_action_nodes.get(agent.comp_type, [])
        idx = action - 1
        if idx < 0 or idx >= len(action_nodes):
            return None
        return int(action_nodes[idx])

    def step(self, actions: List[int]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, float]]:
        """执行一步。

        actions: 长度 = n_agents，每个 agent 一个离散动作。

        返回
        ----
        obs, reward, done, info
        """
        if len(actions) != self.n_agents:
            raise ValueError(f"actions 长度必须等于 {self.n_agents}")
        assert self.indicators is not None

        # 当前 Q(t)：用于计算本 step 的 reward（Q 在 dt 内视为常数）
        q_now = compute_Q(self.bundle.buildings, self.indicators.i_w, self.indicators.i_p, self.indicators.i_t)

        # 1) 处理动作：为空闲 agent 分配新的维修目标
        masks = self.get_action_masks()
        for ai, a in enumerate(actions):
            # 先做一个 mask 校验：如果策略输出非法动作，这里强制改为 0
            if a < 0 or a >= masks[ai].shape[0] or masks[ai][a] <= 0.0:
                a = 0

            # 忙碌 agent 忽略动作（只能 continue）
            if self.agent_busy[ai] == 1:
                continue

            comp = self._action_to_component(self.agents[ai], int(a))
            if comp is None:
                continue

            # 再做一次合法性判断（保险）
            if self.phys_ok[comp] > 0.0:
                continue
            if self.indicators.ac[comp] <= 0.0:
                continue

            self._assign_agent_to_component(ai, comp)

        # 2) 如果没有任何正在维修的组件，则 episode 结束（无法推进）
        if len(self.remaining_work) == 0:
            obs = self._build_obs()
            info = {
                "Q": float(q_now),
                "time": float(self.time),
            }
            return obs, 0.0, True, info

        # 3) 推进到下一次维修完成
        dt = self._advance_to_next_completion()

        # 4) 根据 objective 计算 reward
        if self.cfg.objective == "resilience":
            reward = - (1.0 - float(q_now)) * float(dt)
        elif self.cfg.objective == "time80":
            reward = - float(dt) if float(q_now) < float(self.cfg.threshold_q) else 0.0
        else:
            raise ValueError("objective 必须是 resilience 或 time80")

        # 5) 重新计算指标
        self.indicators = compute_functionality_and_building_indicators(self.bundle, phys_ok=self.phys_ok)

        # done 条件：所有损伤组件都修复完成，或者超过控制时间
        all_repaired = bool(self.phys_ok.min() > 0.5)
        time_up = bool(self.time >= float(self.cfg.control_time))
        done = bool(all_repaired or time_up)

        obs = self._build_obs()
        q_next = compute_Q(self.bundle.buildings, self.indicators.i_w, self.indicators.i_p, self.indicators.i_t)

        info = {
            "Q": float(q_next),
            "Q_prev": float(q_now),
            "time": float(self.time),
            "dt": float(dt),
            "all_repaired": float(1.0 if all_repaired else 0.0),
        }
        return obs, float(reward), done, info

    def _assign_agent_to_component(self, agent_index: int, comp_index: int) -> None:
        """将 agent 分配到某个组件维修任务上。"""
        # 标记 agent 忙碌
        self.agent_busy[agent_index] = 1
        self.agent_target[agent_index] = int(comp_index)

        # 组件 -> agents
        self.comp_to_agents.setdefault(int(comp_index), []).append(agent_index)
        self.nru[comp_index] = len(self.comp_to_agents[int(comp_index)])

        # 若该组件此前没有任务，则采样一个“总工作量”（=单 RU 修复时间）
        if int(comp_index) not in self.remaining_work:
            mean = float(self.expected_repair_time[comp_index])
            sample = sample_truncated_lognormal(
                rng=self.rng,
                mean=mean,
                sigma=float(self.cfg.lognormal_sigma),
                low_ratio=float(self.cfg.trunc_low_ratio),
                high_ratio=float(self.cfg.trunc_high_ratio),
            )
            self.remaining_work[int(comp_index)] = float(sample)

    def _advance_to_next_completion(self) -> float:
        """推进到下一次维修完成事件，返回 dt。"""
        # 计算每个任务的完成时间 remaining_work / nru
        min_dt = None
        for comp, work in self.remaining_work.items():
            k = len(self.comp_to_agents.get(comp, []))
            if k <= 0:
                continue
            ttf = float(work) / float(k)
            if min_dt is None or ttf < min_dt:
                min_dt = ttf

        if min_dt is None:
            return 0.0

        dt = float(max(0.0, min_dt))
        if dt <= 0.0:
            dt = 0.0

        # 所有任务按 k*dt 消耗 work
        completed: List[int] = []
        for comp, work in list(self.remaining_work.items()):
            k = len(self.comp_to_agents.get(comp, []))
            if k <= 0:
                continue
            new_work = float(work) - float(k) * dt
            self.remaining_work[comp] = new_work
            if new_work <= 1e-8:
                completed.append(comp)

        # 更新时间
        self.time += dt

        # 处理完成的组件：置为 phys_ok=1，damage_state=0，释放 agents
        for comp in completed:
            self.phys_ok[comp] = 1.0
            self.damage_state[comp] = 0
            self.remaining_work.pop(comp, None)

            agents = self.comp_to_agents.pop(comp, [])
            self.nru[comp] = 0
            for ai in agents:
                self.agent_busy[ai] = 0
                self.agent_target[ai] = None

        return dt

    # ---------------------------------------------------------------------
    # 下面这些接口主要用于 Rollout/SA 这类需要“复制环境并做蒙特卡洛模拟”的基线方法
    # ---------------------------------------------------------------------

    def get_state_dict(self) -> Dict[str, object]:
        """获取当前环境可序列化状态（用于 clone / Monte Carlo）。"""
        state = {
            "time": float(self.time),
            "phys_ok": self.phys_ok.copy(),
            "damage_state": self.damage_state.copy(),
            "nru": self.nru.copy(),
            "remaining_work": dict(self.remaining_work),
            "comp_to_agents": {int(k): list(v) for k, v in self.comp_to_agents.items()},
            "agent_target": list(self.agent_target),
            "agent_busy": self.agent_busy.copy(),
            "rng_state": self.rng.bit_generator.state,
        }
        return state

    def set_state_dict(self, state: Dict[str, object]) -> None:
        """从 state_dict 恢复环境状态。"""
        self.time = float(state["time"])
        self.phys_ok = np.array(state["phys_ok"], dtype=np.float32)
        self.damage_state = np.array(state["damage_state"], dtype=np.int32)
        self.nru = np.array(state["nru"], dtype=np.int32)

        self.remaining_work = {int(k): float(v) for k, v in dict(state["remaining_work"]).items()}
        self.comp_to_agents = {int(k): list(v) for k, v in dict(state["comp_to_agents"]).items()}
        self.agent_target = list(state["agent_target"])
        self.agent_busy = np.array(state["agent_busy"], dtype=np.int32)

        # 恢复 RNG
        rng_state = state.get("rng_state", None)
        if isinstance(rng_state, dict):
            self.rng.bit_generator.state = rng_state

        # 重新计算指标
        self.indicators = compute_functionality_and_building_indicators(self.bundle, phys_ok=self.phys_ok)

    def clone(self) -> "RecoveryEnv":
        """深拷贝一个新的环境对象（共享 bundle，不共享状态）。"""
        new_env = RecoveryEnv(bundle=self.bundle, ru=self.ru, cfg=self.cfg)
        new_env.reset(seed=0)
        new_env.set_state_dict(self.get_state_dict())
        return new_env
