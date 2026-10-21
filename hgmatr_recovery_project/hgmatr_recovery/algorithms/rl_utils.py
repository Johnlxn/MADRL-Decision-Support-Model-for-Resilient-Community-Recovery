# -*- coding: utf-8 -*-
"""RL 工具函数。"""

from __future__ import annotations

from typing import Tuple
import torch


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算 GAE 优势与回报。

    参数形状:
    - rewards, dones, values: (T, B)
    - last_values: (B,)
    """
    if rewards.dim() != 2 or dones.dim() != 2 or values.dim() != 2:
        raise ValueError('rewards/dones/values 必须是 (T,B)')
    T, B = rewards.shape
    if dones.shape != (T, B) or values.shape != (T, B):
        raise ValueError('rewards/dones/values 形状必须一致')
    if last_values.shape != (B,):
        raise ValueError('last_values 必须是 (B,)')

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)
    next_values = last_values
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae
        next_values = values[t]
    returns = advantages + values
    return advantages, returns
