# -*- coding: utf-8 -*-
"""随机基线（Random）。

在每个决策点，对每个空闲 RU 从合法动作中均匀随机选择一个。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from marl_recovery.env.recovery_env import RecoveryEnv


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(int(seed))

    def act(self, env: RecoveryEnv) -> List[int]:
        masks = env.get_action_masks()
        actions: List[int] = []
        for m in masks:
            valid = np.where(m > 0.0)[0]
            if len(valid) == 0:
                actions.append(0)
            else:
                actions.append(int(self.rng.choice(valid)))
        return actions
