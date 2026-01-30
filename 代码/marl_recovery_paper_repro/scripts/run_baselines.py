# -*- coding: utf-8 -*-
"""运行基线策略（Random / Importance-based / Rollout-SA）。

用法：
python scripts/run_baselines.py --config configs/train_resilience.yaml --out outputs/baselines

说明：
- Random：随机选择合法动作
- Importance-based：论文 Eq.(23)(24)
- Rollout-SA：论文 Eq.(22) + 模拟退火搜索
"""

from __future__ import annotations
import os
import sys

# 让脚本在命令行与 PyCharm 中都能直接 import 本工程包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

from marl_recovery.data.loader import load_data_bundle
from marl_recovery.env.recovery_env import EnvConfig, RecoveryEnv
from marl_recovery.env.resilience import compute_Q, resilience_loss
from marl_recovery.baselines.random_policy import RandomPolicy
from marl_recovery.baselines.importance_policy import ImportancePolicy
from marl_recovery.baselines.rollout_sa import RolloutSAConfig, RolloutSAPolicy
from marl_recovery.utils.config import ensure_dir, load_yaml
from marl_recovery.utils.io import save_json


def run_policy(env: RecoveryEnv, policy, episodes: int, control_time: float, threshold_q: float) -> Dict[str, float]:
    rls = []
    t80s = []
    qends = []

    for ep in range(int(episodes)):
        obs = env.reset(seed=100 + ep)
        times = [float(obs["time"][0])]
        q0 = compute_Q(env.bundle.buildings, env.indicators.i_w, env.indicators.i_p, env.indicators.i_t)  # type: ignore
        qs = [float(q0)]

        reached = None
        done = False
        while not done:
            a = policy.act(env)
            obs, _r, done, info = env.step(a)
            times.append(float(info.get("time", 0.0)))
            qs.append(float(info.get("Q", 0.0)))
            if reached is None and qs[-1] >= threshold_q:
                reached = float(info.get("time", 0.0))
            if float(info.get("time", 0.0)) >= control_time:
                break

        rls.append(resilience_loss(times, qs, control_time=control_time))
        t80s.append(reached if reached is not None else control_time)
        qends.append(qs[-1])

    return {
        "RL_mean": float(np.mean(rls)),
        "T80_mean": float(np.mean(t80s)),
        "Q_end_mean": float(np.mean(qends)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_dir = str(cfg["data_dir"])
    objective = str(cfg.get("objective", "resilience"))

    control_time = float(cfg.get("control_time", 100.0))
    threshold_q = float(cfg.get("threshold_q", 0.8))

    ru = cfg.get("ru", {"pipeline": 2, "substation": 1, "bridge": 1})
    ru = {k: int(v) for k, v in ru.items()}

    env_cfg = EnvConfig(
        objective=objective,
        control_time=control_time,
        threshold_q=threshold_q,
        lognormal_sigma=float(cfg.get("repair_time", {}).get("lognormal_sigma", 0.4)),
        trunc_low_ratio=float(cfg.get("repair_time", {}).get("trunc_low_ratio", 0.1)),
        trunc_high_ratio=float(cfg.get("repair_time", {}).get("trunc_high_ratio", 10.0)),
    )

    bundle = load_data_bundle(data_dir)

    ensure_dir(args.out)

    episodes = int(cfg.get("eval", {}).get("episodes", 20))

    # Random
    env = RecoveryEnv(bundle=bundle, ru=ru, cfg=env_cfg)
    rand = RandomPolicy(seed=0)
    m_rand = run_policy(env, rand, episodes=episodes, control_time=control_time, threshold_q=threshold_q)
    save_json(os.path.join(args.out, "random.json"), m_rand)
    print("[Random]", m_rand)

    # Importance-based
    env = RecoveryEnv(bundle=bundle, ru=ru, cfg=env_cfg)
    imp = ImportancePolicy(bundle=bundle, objective=objective)
    m_imp = run_policy(env, imp, episodes=episodes, control_time=control_time, threshold_q=threshold_q)
    save_json(os.path.join(args.out, "importance.json"), m_imp)
    print("[Importance]", m_imp)

    # Rollout-SA
    env = RecoveryEnv(bundle=bundle, ru=ru, cfg=env_cfg)
    sa_cfg = RolloutSAConfig(
        gamma=float(cfg.get("train", {}).get("gamma", 0.999)),
        n_mc=8,
        rollout_horizon=15,
        sa_iters=30,
        temp_start=1.0,
        temp_end=0.05,
        seed=0,
    )
    rsa = RolloutSAPolicy(bundle=bundle, objective=objective, cfg=sa_cfg)
    m_rsa = run_policy(env, rsa, episodes=max(1, episodes // 4), control_time=control_time, threshold_q=threshold_q)
    save_json(os.path.join(args.out, "rollout_sa.json"), m_rsa)
    print("[Rollout-SA]", m_rsa)

    print(f"已保存到: {args.out}")


if __name__ == "__main__":
    main()
