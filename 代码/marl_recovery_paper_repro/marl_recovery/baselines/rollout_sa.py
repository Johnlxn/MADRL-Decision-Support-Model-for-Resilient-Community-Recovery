# -*- coding: utf-8 -*-
"""Rollout + Simulated Annealing（Rollout-SA）基线。

论文基线描述：
- Rollout：通过蒙特卡洛模拟估计 Q 函数 (Eq.(22))
- SA：由于动作空间巨大，用模拟退火在每个决策点近似搜索最优动作（非穷举）

注意：论文中 Rollout-SA 的计算量很大（案例报告耗时可达数小时级）。
本实现给出一个“可运行”的版本，并提供可调参数（sa_iters, n_mc, horizon 等）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from marl_recovery.env.recovery_env import RecoveryEnv
from marl_recovery.baselines.importance_policy import ImportancePolicy


@dataclass
class RolloutSAConfig:
    gamma: float = 0.999
    n_mc: int = 8
    rollout_horizon: int = 15
    sa_iters: int = 30
    temp_start: float = 1.0
    temp_end: float = 0.05
    seed: int = 0


class RolloutSAPolicy:
    def __init__(self, bundle, objective: str, cfg: RolloutSAConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.seed))
        self.base_policy = ImportancePolicy(bundle=bundle, objective=objective)

    def _estimate_return(self, env: RecoveryEnv, first_action: List[int]) -> float:
        """估计 Q^(pi)(s, a)（论文 Eq.(22) 的有限时域近似）。"""
        rets = []
        for _ in range(int(self.cfg.n_mc)):
            sim = env.clone()
            # 给每次模拟一个不同的 RNG（否则 clone 会复制同一 RNG state）
            sim.rng = np.random.default_rng(int(self.rng.integers(0, 1_000_000_000)))

            total = 0.0
            discount = 1.0

            _obs, r, done, _info = sim.step(list(first_action))
            total += discount * float(r)
            discount *= float(self.cfg.gamma)

            for _t in range(int(self.cfg.rollout_horizon) - 1):
                if done:
                    break
                a = self.base_policy.act(sim)
                _obs, r, done, _info = sim.step(a)
                total += discount * float(r)
                discount *= float(self.cfg.gamma)

            rets.append(total)

        return float(np.mean(rets)) if len(rets) > 0 else 0.0

    def act(self, env: RecoveryEnv) -> List[int]:
        """在当前状态下用 SA 搜索一个较优的联合动作。"""
        # 初始化动作：用重要度策略作为初值
        cur_action = self.base_policy.act(env)
        cur_value = self._estimate_return(env, cur_action)
        best_action = list(cur_action)
        best_value = float(cur_value)

        # 预取 mask 与可行动作集合
        masks = env.get_action_masks()
        valid_lists: List[np.ndarray] = []
        for m in masks:
            valid = np.where(m > 0.0)[0]
            # busy agent 的 mask 只有 0，这里也没问题
            valid_lists.append(valid)

        # SA
        for it in range(int(self.cfg.sa_iters)):
            # 温度线性退火
            frac = it / max(1, int(self.cfg.sa_iters) - 1)
            temp = float(self.cfg.temp_start) * (1.0 - frac) + float(self.cfg.temp_end) * frac
            temp = max(1e-8, temp)

            # 生成邻域解：随机挑一个 agent 改动作
            cand_action = list(cur_action)
            ai = int(self.rng.integers(0, env.n_agents))
            candidates = valid_lists[ai]
            if len(candidates) == 0:
                continue

            # 从合法动作中随机采样一个（尽量不同于当前）
            if len(candidates) == 1:
                new_a = int(candidates[0])
            else:
                # 重采样几次避免相同
                new_a = int(candidates[int(self.rng.integers(0, len(candidates)))])
                for _ in range(5):
                    if new_a != cand_action[ai]:
                        break
                    new_a = int(candidates[int(self.rng.integers(0, len(candidates)))])

            cand_action[ai] = new_a

            cand_value = self._estimate_return(env, cand_action)

            delta = float(cand_value - cur_value)
            accept = False
            if delta >= 0:
                accept = True
            else:
                p = float(np.exp(delta / temp))
                if float(self.rng.random()) < p:
                    accept = True

            if accept:
                cur_action = cand_action
                cur_value = cand_value

                if cand_value > best_value:
                    best_value = cand_value
                    best_action = list(cand_action)

        return best_action
