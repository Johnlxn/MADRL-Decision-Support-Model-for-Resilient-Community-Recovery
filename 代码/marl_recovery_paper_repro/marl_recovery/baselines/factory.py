# -*- coding: utf-8 -*-
"""Baseline policy factory."""

from __future__ import annotations

from typing import Any, Dict

from marl_recovery.data.loader import DataBundle
from marl_recovery.baselines.random_policy import RandomPolicy
from marl_recovery.baselines.importance_policy import ImportancePolicy
from marl_recovery.baselines.rollout_sa import RolloutSAConfig, RolloutSAPolicy


def build_baseline_policy(
    name: str,
    bundle: DataBundle,
    objective: str,
    cfg: Dict[str, Any] | None = None,
):
    """Build a baseline policy from config.

    Supported names:
    - random
    - importance
    - rollout_sa

    cfg may contain shared keys like seed/gamma, plus a nested rollout_sa
    dict for rollout-specific hyper-parameters.
    """
    cfg = cfg or {}
    name = str(name).lower()

    if name == "random":
        return RandomPolicy(seed=int(cfg.get("seed", 0)))

    if name == "importance":
        return ImportancePolicy(
            bundle=bundle,
            objective=objective,
            allow_pipeline_duplicate=bool(cfg.get("allow_pipeline_duplicate", False)),
        )

    if name == "rollout_sa":
        rollout_cfg = cfg.get("rollout_sa", {})
        sa_cfg = RolloutSAConfig(
            gamma=float(rollout_cfg.get("gamma", cfg.get("gamma", 0.999))),
            n_mc=int(rollout_cfg.get("n_mc", 8)),
            rollout_horizon=int(rollout_cfg.get("rollout_horizon", 15)),
            sa_iters=int(rollout_cfg.get("sa_iters", 30)),
            temp_start=float(rollout_cfg.get("temp_start", 1.0)),
            temp_end=float(rollout_cfg.get("temp_end", 0.05)),
            seed=int(rollout_cfg.get("seed", cfg.get("seed", 0))),
        )
        return RolloutSAPolicy(bundle=bundle, objective=objective, cfg=sa_cfg)

    raise ValueError(f"不支持的 baseline policy: {name}")
