# -*- coding: utf-8 -*-
"""韧性指标与社区功能 Q(t)。

对应论文：
- RL（resilience loss）: Eq.(1)
- Q(t): Eq.(2)
- C: Eq.(3)
- Qphy: Eq.(4)

本模块只负责“公式计算”，不负责判定 Iw/Ip/It 的取值（...由 interdependency.py 给出）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from marl_recovery.data.loader import BuildingRecord


def normalization_constant(buildings: Sequence[BuildingRecord]) -> float:
    """计算论文 Eq.(3) 的归一化常数 C = sum(alpha_i * S_i)。"""
    c = 0.0
    for b in buildings:
        c += float(b.alpha) * float(b.area)
    # 防止除零
    return float(c if c > 0 else 1.0)


def compute_Q(
    buildings: Sequence[BuildingRecord],
    i_w: np.ndarray,
    i_p: np.ndarray,
    i_t: np.ndarray,
) -> float:
    """计算论文 Eq.(2)-(4) 的社区功能 Q(t)。

    参数
    ----
    buildings: 建筑列表
    i_w/i_p/i_t: 0/1 指示变量数组，长度等于 buildings 数量

    返回
    ----
    Q: [0,1] 之间的功能度（在 beta 权重合理时）
    """
    if len(buildings) == 0:
        return 1.0

    assert i_w.shape[0] == len(buildings)
    assert i_p.shape[0] == len(buildings)
    assert i_t.shape[0] == len(buildings)

    c = normalization_constant(buildings)
    total = 0.0
    for idx, b in enumerate(buildings):
        s = float(b.area)
        q_phy = s * (float(i_w[idx]) * float(b.beta_w) + float(i_p[idx]) * float(b.beta_p) + float(i_t[idx]) * float(b.beta_t))
        total += float(b.alpha) * q_phy

    q = total / c
    # 由于 beta 可能不是严格归一，做一个裁剪更稳健
    return float(max(0.0, min(1.0, q)))


def compute_subsystem_Q(
    buildings: Sequence[BuildingRecord],
    i_w: np.ndarray,
    i_p: np.ndarray,
    i_t: np.ndarray,
    which: str,
) -> float:
    """计算单个子系统的功能（论文 Fig.12/16 说明）。

    论文描述：通过 Eq.(2)-(4) 计算，只是将其他系统的 beta 设为 0。

    which:
      - 'w'：WDN
      - 'p'：EPN
      - 't'：TN
    """
    if which not in {"w", "p", "t"}:
        raise ValueError("which 必须是 'w'/'p'/'t'")

    if len(buildings) == 0:
        return 1.0

    c = normalization_constant(buildings)
    total = 0.0
    for idx, b in enumerate(buildings):
        s = float(b.area)
        if which == "w":
            q_phy = s * (float(i_w[idx]) * float(b.beta_w))
        elif which == "p":
            q_phy = s * (float(i_p[idx]) * float(b.beta_p))
        else:
            q_phy = s * (float(i_t[idx]) * float(b.beta_t))
        total += float(b.alpha) * q_phy

    q = total / c
    return float(max(0.0, min(1.0, q)))


def resilience_loss(times: Sequence[float], qs: Sequence[float], control_time: float) -> float:
    """根据分段常数 Q(t) 序列计算 RL。

    参数
    ----
    times: 时间点序列（长度 = len(qs)），表示每段开始的时间
    qs: 每段的 Q 值（假设在 [times[k], times[k+1]) 内保持常数）
    control_time: TLC

    返回
    ----
    RL = ∫(1-Q)dt / TLC
    """
    if len(times) == 0:
        return 0.0
    if len(times) != len(qs):
        raise ValueError("times 与 qs 长度必须相同")

    area = 0.0
    for k in range(len(times) - 1):
        t0 = float(times[k])
        t1 = float(times[k + 1])
        dt = max(0.0, min(t1, control_time) - t0)
        if dt <= 0:
            continue
        q = float(qs[k])
        area += (1.0 - q) * dt

    # 最后一段延拓到 control_time
    last_t = float(times[-1])
    if last_t < control_time:
        dt = control_time - last_t
        area += (1.0 - float(qs[-1])) * dt

    return float(area / float(control_time))
