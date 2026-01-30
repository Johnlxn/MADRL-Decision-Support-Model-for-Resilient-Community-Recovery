# -*- coding: utf-8 -*-
"""维修时间分布。

论文 2.4.1：采用对数正态分布（lognormal），离散程度（标准差）取 0.4，
并对维修时间进行截断，避免出现极端值。

本模块提供：
- sample_truncated_lognormal(mean, sigma, low_ratio, high_ratio, rng)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def sample_truncated_lognormal(
    rng: np.random.Generator,
    mean: float,
    sigma: float = 0.4,
    low_ratio: float = 0.1,
    high_ratio: float = 10.0,
) -> float:
    """采样截断对数正态维修时间。

    参数
    ----
    rng: numpy 随机数生成器
    mean: 期望值（论文中的 Er 或 Erepair）
    sigma: 对数正态分布对数域标准差（论文建议 0.4）
    low_ratio/high_ratio: 截断区间比例，最终区间是 [low_ratio*mean, high_ratio*mean]

    返回
    ----
    sample: 非负的维修时间（单位：天）

    说明
    ----
    对数正态：如果 X ~ LogNormal(mu, sigma)，则 E[X] = exp(mu + 0.5*sigma^2)
    所以给定 mean 后，mu = ln(mean) - 0.5*sigma^2
    """
    if mean <= 0:
        return 0.0

    mu = float(np.log(mean) - 0.5 * sigma * sigma)
    x = float(rng.lognormal(mean=mu, sigma=sigma))

    low = float(low_ratio * mean)
    high = float(high_ratio * mean)
    if x < low:
        x = low
    if x > high:
        x = high
    return float(x)
