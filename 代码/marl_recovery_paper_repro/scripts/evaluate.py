# -*- coding: utf-8 -*-
"""评估脚本：生成 Q(t) 曲线并计算 RL / T80。
（已修复路径问题，可自动识别项目根目录）

用法：
python scripts/evaluate.py --config configs/train_resilience.yaml --ckpt outputs/run/checkpoints/best.pth --out outputs/run/eval
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# ====== 路径修复核心逻辑 ======
# 自动定位项目根目录：当前脚本 -> scripts -> 项目根目录
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parent.parent

# 将项目根目录加入环境变量，确保能 import marl_recovery
sys.path.insert(0, str(ROOT_DIR))
# ===========================

import argparse
from typing import Dict, List

import torch
import matplotlib.pyplot as plt

# 确保包能被导入
try:
    from marl_recovery.data.loader import load_data_bundle
    from marl_recovery.env.recovery_env import EnvConfig, RecoveryEnv
    from marl_recovery.env.resilience import compute_Q, compute_subsystem_Q, resilience_loss
    from marl_recovery.models.actor_critic import ActorNet, CriticNet
    from marl_recovery.utils.config import ensure_dir, load_yaml
    from marl_recovery.utils.io import save_json
except ImportError as e:
    print(f"[ERROR] 无法导入工程包，请检查路径设置。")
    print(f"检测到的 ROOT_DIR: {ROOT_DIR}")
    raise e


def _edges_to_torch(bundle, device: str) -> Dict[str, torch.Tensor]:
    edges = {}
    for etype, arr in bundle.edges.items():
        edges[etype] = torch.as_tensor(arr, dtype=torch.long, device=device)
    return edges


def run_one_episode(env: RecoveryEnv, actors: List[ActorNet], edges: Dict[str, torch.Tensor], device: str,
                    control_time: float, threshold_q: float) -> Dict[str, object]:
    obs = env.reset(seed=123)
    times = [float(obs["time"][0])]

    q = compute_Q(env.bundle.buildings, env.indicators.i_w, env.indicators.i_p, env.indicators.i_t)
    qw = compute_subsystem_Q(env.bundle.buildings, env.indicators.i_w, env.indicators.i_p, env.indicators.i_t,
                             which="w")
    qp = compute_subsystem_Q(env.bundle.buildings, env.indicators.i_w, env.indicators.i_p, env.indicators.i_t,
                             which="p")
    qt = compute_subsystem_Q(env.bundle.buildings, env.indicators.i_w, env.indicators.i_p, env.indicators.i_t,
                             which="t")

    Qs = [float(q)]
    Qw = [float(qw)]
    Qp = [float(qp)]
    Qt = [float(qt)]

    t80 = None
    done = False

    while not done:
        node_feat = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
        masks_np = obs["action_masks"]

        acts = []
        for ai, actor in enumerate(actors):
            mask = torch.as_tensor(masks_np[ai], dtype=torch.float32, device=device)
            a, _ = actor.sample_action(node_feat, edges, mask)
            acts.append(a)

        obs, _r, done, info = env.step(acts)

        times.append(float(info.get("time", 0.0)))
        Qs.append(float(info.get("Q", 0.0)))

        # 子系统 Q
        qw = compute_subsystem_Q(env.bundle.buildings, env.indicators.i_w, env.indicators.i_p, env.indicators.i_t,
                                 which="w")
        qp = compute_subsystem_Q(env.bundle.buildings, env.indicators.i_w, env.indicators.i_p, env.indicators.i_t,
                                 which="p")
        qt = compute_subsystem_Q(env.bundle.buildings, env.indicators.i_w, env.indicators.i_p, env.indicators.i_t,
                                 which="t")
        Qw.append(float(qw))
        Qp.append(float(qp))
        Qt.append(float(qt))

        if t80 is None and Qs[-1] >= threshold_q:
            t80 = float(info.get("time", 0.0))

        if float(info.get("time", 0.0)) >= control_time:
            break

    rl = resilience_loss(times, Qs, control_time=control_time)
    if t80 is None:
        t80 = control_time

    return {
        "times": times,
        "Q": Qs,
        "Q_w": Qw,
        "Q_p": Qp,
        "Q_t": Qt,
        "RL": float(rl),
        "T80": float(t80),
        "Q_end": float(Qs[-1]),
    }


def plot_curve(times: List[float], ys: List[float], title: str, out_path: str) -> None:
    plt.figure()
    plt.plot(times, ys)
    plt.xlabel("Time (days)")
    plt.ylabel(title)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # ====== [新增] 自动处理相对路径 ======
    def resolve_path(p: str) -> Path:
        """如果路径是相对的，将其解析为相对于 ROOT_DIR 的路径"""
        path_obj = Path(p)
        if not path_obj.is_absolute():
            return ROOT_DIR / path_obj
        return path_obj

    print(f"[DEBUG] Project Root: {ROOT_DIR}")

    cfg_path = resolve_path(args.config)
    ckpt_path = resolve_path(args.ckpt)
    out_dir = resolve_path(args.out)

    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {ckpt_path}")

    print(f"[INFO] Config: {cfg_path}")
    print(f"[INFO] Checkpoint: {ckpt_path}")

    cfg = load_yaml(str(cfg_path))

    # ====== [关键修复] 处理 data_dir ======
    raw_data_dir = cfg.get("data_dir")
    data_dir_path = Path(raw_data_dir)

    # 如果 YAML 里写的是相对路径，强制拼接到 ROOT_DIR
    if not data_dir_path.is_absolute():
        data_dir_path = ROOT_DIR / data_dir_path

    data_dir = str(data_dir_path.resolve())
    if not data_dir_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    print(f"[INFO] Data Directory: {data_dir}")
    # ====================================

    control_time = float(cfg.get("control_time", 100.0))
    threshold_q = float(cfg.get("threshold_q", 0.8))

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] cuda 不可用，将使用 cpu")
        device = "cpu"

    bundle = load_data_bundle(data_dir)
    edges = _edges_to_torch(bundle, device=device)
    relations = list(edges.keys())

    env_cfg = EnvConfig(
        objective=str(cfg.get("objective", "resilience")),
        control_time=control_time,
        threshold_q=threshold_q,
        lognormal_sigma=float(cfg.get("repair_time", {}).get("lognormal_sigma", 0.4)),
        trunc_low_ratio=float(cfg.get("repair_time", {}).get("trunc_low_ratio", 0.1)),
        trunc_high_ratio=float(cfg.get("repair_time", {}).get("trunc_high_ratio", 10.0)),
    )

    ru = cfg.get("ru", {"pipeline": 2, "substation": 1, "bridge": 1})
    ru = {k: int(v) for k, v in ru.items()}

    env = RecoveryEnv(bundle=bundle, ru=ru, cfg=env_cfg)

    # 创建网络并加载 ckpt
    print(f"[INFO] Loading checkpoint...")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    action_dims = ckpt.get("action_dims")
    if action_dims is None:
        # fallback：根据 env 计算
        action_dims = []
        for agent in env.agents:
            action_dims.append(len(env.type_to_action_nodes.get(agent.comp_type, [])) + 1)

    actors: List[ActorNet] = []
    for ad in action_dims:
        actor = ActorNet(bundle=bundle, relations=relations, action_dim=int(ad)).to(device)
        actors.append(actor)

    critic = CriticNet(bundle=bundle, relations=relations).to(device)

    for a, sd in zip(actors, ckpt["actors"]):
        a.load_state_dict(sd)
    critic.load_state_dict(ckpt["critic"])

    ensure_dir(str(out_dir))

    print("[INFO] Running evaluation...")
    result = run_one_episode(env, actors, edges, device=device, control_time=control_time, threshold_q=threshold_q)

    save_json(os.path.join(out_dir, "metrics.json"), {k: result[k] for k in ["RL", "T80", "Q_end"]})
    save_json(os.path.join(out_dir, "curve.json"), result)

    plot_curve(result["times"], result["Q"], "Community Function Q(t)", os.path.join(out_dir, "Q_curve.png"))
    plot_curve(result["times"], result["Q_w"], "WDN Function", os.path.join(out_dir, "Q_w.png"))
    plot_curve(result["times"], result["Q_p"], "EPN Function", os.path.join(out_dir, "Q_p.png"))
    plot_curve(result["times"], result["Q_t"], "TN Function", os.path.join(out_dir, "Q_t.png"))

    print(f"RL={result['RL']:.4f}, T80={result['T80']:.2f}, Q_end={result['Q_end']:.3f}")
    print(f"结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()