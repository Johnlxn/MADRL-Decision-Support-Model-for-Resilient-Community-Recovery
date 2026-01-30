# -*- coding: utf-8 -*-
"""强化学习常用工具：GAE、mini-batch 采样等。"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import torch


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算广义优势估计（GAE）。

    参数
    ----
    rewards: (T, N)  每步共享奖励
    dones: (T, N)  0/1，表示该步结束后是否终止
    values: (T, N)  V(s_t)
    last_values: (N,)  V(s_{T})

    返回
    ----
    advantages: (T, N)
    returns: (T, N)
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N, dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(T)):
        next_value = last_values if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def iter_minibatches(batch_size: int, minibatch_size: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    """生成一系列 minibatch 索引。"""
    idx = np.arange(batch_size)
    rng.shuffle(idx)
    for start in range(0, batch_size, minibatch_size):
        yield idx[start : start + minibatch_size]
