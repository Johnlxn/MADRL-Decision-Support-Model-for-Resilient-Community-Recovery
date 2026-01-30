# -*- coding: utf-8 -*-
"""重要度基线（Importance-based）。

论文中给出的基线方法之一（Eq.(23)(24)）：
- 目标1（韧性）：
  - 对管线与变电站：IF1_i = (\sum_{j\in B(i)} \alpha_j S_j) / E_i
  - 对桥梁：优先修复损伤程度较低的桥
- 目标2（T80）：
  - 对所有组件：IF2_i = 1 / E_i

其中：
- B(i) 为由组件 i 提供服务的建筑集合
- E_i 为组件的维修时间（论文中通常用“期望维修时间”）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np

from marl_recovery.data.loader import DataBundle
from marl_recovery.env.recovery_env import RecoveryEnv


class ImportancePolicy:
    def __init__(self, bundle: DataBundle, objective: str = "resilience", allow_pipeline_duplicate: bool = False):
        self.bundle = bundle
        self.objective = str(objective)
        self.allow_pipeline_duplicate = bool(allow_pipeline_duplicate)

        # 预计算：每个 pipeline/substation 节点对应的 \sum(alpha*S)
        n = bundle.num_nodes
        self.load_sum = np.zeros(n, dtype=np.float32)

        for b in bundle.buildings:
            self.load_sum[b.pipeline_node] += float(b.alpha) * float(b.area)
            self.load_sum[b.substation_node] += float(b.alpha) * float(b.area)

    def _score(self, env: RecoveryEnv, comp_index: int, comp_type: str) -> float:
        E = float(env.expected_repair_time[comp_index])
        ds = float(env.damage_state[comp_index])

        if self.objective == "resilience":
            if comp_type in {"pipeline", "substation"}:
                if E <= 0:
                    return -1e9
                return float(self.load_sum[comp_index] / E)
            if comp_type == "bridge":
                # 论文：优先修复损伤较低的桥（ds 越小越好）
                return float(-ds)
            # 其他类型：退化为 1/E
            return float(1.0 / E) if E > 0 else -1e9

        # time80
        if E <= 0:
            return -1e9
        return float(1.0 / E)

    def act(self, env: RecoveryEnv) -> List[int]:
        masks = env.get_action_masks()
        actions: List[int] = [0 for _ in range(env.n_agents)]

        # 对 pipeline agent 做一个“协同选择”（可选唯一）
        chosen_components: Set[int] = set()

        for ai, agent in enumerate(env.agents):
            if env.agent_busy[ai] == 1:
                actions[ai] = 0
                continue

            m = masks[ai]
            valid_actions = np.where(m > 0.0)[0]
            # 去掉 0 动作（空闲）以便选择组件；如果只有 0 则只能空闲
            valid_actions = valid_actions[valid_actions != 0]
            if len(valid_actions) == 0:
                actions[ai] = 0
                continue

            best_a = 0
            best_s = -1e18
            action_nodes = env.type_to_action_nodes.get(agent.comp_type, [])

            for a in valid_actions:
                comp = action_nodes[int(a) - 1]

                # 可选：pipeline 不允许重复选择同一组件
                if (not self.allow_pipeline_duplicate) and agent.comp_type == "pipeline":
                    if comp in chosen_components:
                        continue

                s = self._score(env, comp_index=int(comp), comp_type=agent.comp_type)
                if s > best_s:
                    best_s = s
                    best_a = int(a)

            actions[ai] = int(best_a)

            # 记录已选组件
            if agent.comp_type == "pipeline" and actions[ai] != 0:
                comp = action_nodes[int(actions[ai]) - 1]
                chosen_components.add(int(comp))

        return actions
