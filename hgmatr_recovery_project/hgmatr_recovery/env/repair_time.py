# -*- coding: utf-8 -*-
"""维修时间分布。"""

from __future__ import annotations

import math
import numpy as np


def _std_norm_cdf(x: float) -> float:
    """标准正态分布 CDF。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _truncated_lognormal_mean(mu: float, sigma: float, low: float, high: float) -> float:
    """计算截断对数正态在 [low, high] 上的条件均值。"""
    if low <= 0.0:
        raise ValueError(f"low must be > 0 for lognormal truncation, got {low}")
    if not (low < high):
        raise ValueError(f"Require low < high, got low={low}, high={high}")

    alpha = (math.log(low) - mu) / sigma
    beta = (math.log(high) - mu) / sigma

    # 数值稳定保护：
    # mu 远小于 low 时，截断后几乎全堆在 low 附近
    if alpha > 8.0:
        return float(low)

    # mu 远大于 high 时，截断后几乎全堆在 high 附近
    if beta < -8.0:
        return float(high)

    z = _std_norm_cdf(beta) - _std_norm_cdf(alpha)

    # 极端情况下 z 非常小，不再报错，直接用边界极限近似
    if z < 1e-14:
        if mu <= math.log(low):
            return float(low)
        if mu >= math.log(high):
            return float(high)
        return 0.5 * (low + high)

    num = math.exp(mu + 0.5 * sigma * sigma) * (
        _std_norm_cdf(beta - sigma) - _std_norm_cdf(alpha - sigma)
    )
    return float(num / z)


def _solve_mu_for_truncated_mean(
    target_mean: float,
    sigma: float,
    low: float,
    high: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """求解使截断后均值等于 target_mean 的 mu。"""
    if target_mean <= 0.0:
        raise ValueError(f"target_mean must be positive, got {target_mean}")
    if low <= 0.0:
        raise ValueError(f"low must be > 0, got {low}")
    if not (low < high):
        raise ValueError(f"Require low < high, got low={low}, high={high}")

    # 截断分布的均值一定落在 [low, high] 内
    if target_mean < low or target_mean > high:
        raise ValueError(
            f"target_mean must lie in [low, high], got target_mean={target_mean}, "
            f"low={low}, high={high}"
        )

    # target_mean 恰好在边界时，直接返回接近边界的 mu
    if abs(target_mean - low) < 1e-12:
        return math.log(low) - 10.0 * sigma
    if abs(target_mean - high) < 1e-12:
        return math.log(high) + 10.0 * sigma

    lo = math.log(low) - 6.0 * sigma
    hi = math.log(high) + 6.0 * sigma

    f_lo = _truncated_lognormal_mean(lo, sigma, low, high) - target_mean
    f_hi = _truncated_lognormal_mean(hi, sigma, low, high) - target_mean

    expand_iter = 0
    while f_lo > 0.0 and expand_iter < 50:
        lo -= 2.0 * sigma
        f_lo = _truncated_lognormal_mean(lo, sigma, low, high) - target_mean
        expand_iter += 1

    expand_iter = 0
    while f_hi < 0.0 and expand_iter < 50:
        hi += 2.0 * sigma
        f_hi = _truncated_lognormal_mean(hi, sigma, low, high) - target_mean
        expand_iter += 1

    if not (f_lo <= 0.0 <= f_hi):
        raise RuntimeError(
            f"Failed to bracket mu for target_mean={target_mean}, "
            f"sigma={sigma}, low={low}, high={high}"
        )

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = _truncated_lognormal_mean(mid, sigma, low, high) - target_mean

        if abs(f_mid) < tol:
            return mid

        if f_mid < 0.0:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def sample_truncated_lognormal(
    rng: np.random.Generator,
    mean: float,
    sigma: float = 0.4,
    low_ratio: float = 0.1,
    high_ratio: float = 10.0,
    max_tries: int = 10000,
) -> float:
    """采样截断对数正态维修时间。

    这里的 mean 被视为“截断后”的目标均值。
    截断区间为 [low_ratio * mean, high_ratio * mean]。
    """
    if mean <= 0.0:
        raise ValueError(f"mean must be positive, got {mean}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if low_ratio <= 0.0:
        raise ValueError(f"low_ratio must be > 0, got {low_ratio}")
    if high_ratio <= low_ratio:
        raise ValueError(
            f"Require high_ratio > low_ratio, got low_ratio={low_ratio}, high_ratio={high_ratio}"
        )

    low = float(low_ratio * mean)
    high = float(high_ratio * mean)

    mu = _solve_mu_for_truncated_mean(
        target_mean=float(mean),
        sigma=float(sigma),
        low=low,
        high=high,
    )

    for _ in range(max_tries):
        x = float(rng.lognormal(mean=mu, sigma=sigma))
        if low <= x <= high:
            return x

    raise RuntimeError(
        "Failed to sample truncated lognormal within bounds "
        f"[{low}, {high}] after {max_tries} tries. "
        f"target_mean={mean}, sigma={sigma}"
    )