# -*- coding: utf-8 -*-
"""训练脚本：Seq-MAPPO + GNN-FiLM。

支持三种算法模式：
- seq_mappo：纯顺序更新 PPO
- heuristic_guided：带启发式 imitation guidance 的 Seq-MAPPO
- heuristic_only：仅运行启发式 baseline 评估

用法：
python scripts/train.py --config configs/train_resilience.yaml
"""

from __future__ import annotations
import os
import sys

# 让脚本在命令行与 PyCharm 中都能直接 import 本工程包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import replace
from pathlib import Path
import argparse
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from marl_recovery.data.loader import load_data_bundle
from marl_recovery.env.recovery_env import EnvConfig, RecoveryEnv
from marl_recovery.env.resilience import compute_Q, resilience_loss
from marl_recovery.algorithms.rl_utils import compute_gae
from marl_recovery.algorithms.seq_mappo import RolloutBatch, SeqMAPPOTrainer
from marl_recovery.baselines.factory import build_baseline_policy
from marl_recovery.models.actor_critic import ActorNet, CriticNet
from marl_recovery.utils.config import ensure_dir, load_yaml, set_global_seeds
from marl_recovery.utils.io import save_json


ActionSelector = Callable[[RecoveryEnv, Dict[str, np.ndarray]], List[int]]


def _edges_to_torch(bundle, device: str) -> Dict[str, torch.Tensor]:
    edges = {}
    for etype, arr in bundle.edges.items():
        edges[etype] = torch.as_tensor(arr, dtype=torch.long, device=device)
    return edges


def _linear_schedule(start: float, end: float, progress: float, until: float = 1.0) -> float:
    if until <= 0.0:
        return float(end)
    frac = min(max(float(progress) / float(until), 0.0), 1.0)
    return float(start + (end - start) * frac)


def _make_actor_selector(
    actors: List[ActorNet],
    edges: Dict[str, torch.Tensor],
    device: str,
) -> ActionSelector:
    def _select(_env: RecoveryEnv, obs: Dict[str, np.ndarray]) -> List[int]:
        node_feat = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
        masks_np = obs["action_masks"]
        acts: List[int] = []
        for ai, actor in enumerate(actors):
            mask = torch.as_tensor(masks_np[ai], dtype=torch.float32, device=device)
            a, _ = actor.sample_action(node_feat, edges, mask)
            acts.append(a)
        return acts

    return _select


def _make_baseline_selector(policy) -> ActionSelector:
    def _select(env: RecoveryEnv, _obs: Dict[str, np.ndarray]) -> List[int]:
        return policy.act(env)

    return _select


def _eval_policy(
    env: RecoveryEnv,
    action_selector: ActionSelector,
    episodes: int,
    control_time: float,
    threshold_q: float,
) -> Dict[str, float]:
    rls = []
    t80s = []
    qends = []

    for ep in range(int(episodes)):
        obs = env.reset(seed=1000 + ep)
        times = [float(obs["time"][0])]
        q0 = compute_Q(env.bundle.buildings, env.indicators.i_w, env.indicators.i_p, env.indicators.i_t)  # type: ignore
        qs = [float(q0)]

        reached_t80 = None
        done = False
        while not done:
            acts = action_selector(env, obs)
            obs, _, done, info = env.step(acts)
            times.append(float(info.get("time", 0.0)))
            qs.append(float(info.get("Q", 0.0)))

            if reached_t80 is None and qs[-1] >= threshold_q:
                reached_t80 = float(info.get("time", 0.0))

            if float(info.get("time", 0.0)) >= control_time:
                break

        rls.append(resilience_loss(times, qs, control_time=control_time))
        t80s.append(reached_t80 if reached_t80 is not None else control_time)
        qends.append(qs[-1])

    return {
        "RL_mean": float(np.mean(rls)),
        "T80_mean": float(np.mean(t80s)),
        "Q_end_mean": float(np.mean(qends)),
    }


def _save_checkpoint(
    path: str,
    actors: List[ActorNet],
    critic: CriticNet,
    action_dims: List[int],
    relations: List[str],
    objective: str,
    cfg: Dict,
    algorithm_mode: str,
) -> None:
    torch.save(
        {
            "actors": [a.state_dict() for a in actors],
            "critic": critic.state_dict(),
            "action_dims": action_dims,
            "relations": relations,
            "objective": objective,
            "algorithm_mode": algorithm_mode,
            "config": cfg,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_resilience.yaml", help="训练配置 YAML")
    parser.add_argument("--device", type=str, default="cuda", help="cpu 或 cuda")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {cfg_path}")

    print(f"[INFO] 使用配置文件: {cfg_path}")
    cfg = load_yaml(str(cfg_path))

    seed = int(cfg.get("seed", 42))
    set_global_seeds(seed)

    data_dir = str(cfg["data_dir"])
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    data_dir = str(data_path)
    print(f"[INFO] 使用数据目录: {data_dir}")

    objective = str(cfg.get("objective", "resilience"))
    control_time = float(cfg.get("control_time", 100.0))
    threshold_q = float(cfg.get("threshold_q", 0.8))

    out_dir = str(cfg.get("output_dir", "outputs/run"))
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "checkpoints"))
    save_json(os.path.join(out_dir, "config_snapshot.json"), cfg)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] 你选择了 cuda，但当前环境检测不到 GPU，将回退到 cpu")
        device = "cpu"
    print(f"[INFO] device={device} | torch={torch.__version__} | cuda_available={torch.cuda.is_available()}")

    bundle = load_data_bundle(data_dir)

    beta_cfg = cfg.get("beta", None)
    if isinstance(beta_cfg, dict):
        bw = float(beta_cfg.get("w", 0.55))
        bp = float(beta_cfg.get("p", 0.40))
        bt = float(beta_cfg.get("t", 0.05))
        s = bw + bp + bt
        if s > 0:
            bw, bp, bt = bw / s, bp / s, bt / s
        bundle.buildings = [replace(b, beta_w=bw, beta_p=bp, beta_t=bt) for b in bundle.buildings]
        print(f"[INFO] Override building betas from YAML: beta_w={bw:.3f}, beta_p={bp:.3f}, beta_t={bt:.3f}")

    edges = _edges_to_torch(bundle, device=device)

    env_cfg = EnvConfig(
        objective=objective,
        control_time=control_time,
        threshold_q=threshold_q,
        lognormal_sigma=float(cfg.get("repair_time", {}).get("lognormal_sigma", 0.4)),
        trunc_low_ratio=float(cfg.get("repair_time", {}).get("trunc_low_ratio", 0.1)),
        trunc_high_ratio=float(cfg.get("repair_time", {}).get("trunc_high_ratio", 10.0)),
    )

    ru = cfg.get("ru", {"pipeline": 2, "substation": 1, "bridge": 1})
    ru = {k: int(v) for k, v in ru.items()}

    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    algo_cfg = cfg.get("algorithm", {})

    mode = str(algo_cfg.get("mode", "seq_mappo")).lower()
    baseline_name = str(algo_cfg.get("baseline_policy", "importance"))
    baseline_cfg = dict(algo_cfg.get("baseline", {}))
    baseline_cfg.setdefault("gamma", float(train_cfg.get("gamma", 0.999)))

    eval_episodes = int(eval_cfg.get("episodes", 5))
    print(f"[INFO] algorithm.mode={mode}")

    if mode == "heuristic_only":
        policy = build_baseline_policy(name=baseline_name, bundle=bundle, objective=objective, cfg=baseline_cfg)
        eval_env = RecoveryEnv(bundle=bundle, ru=ru, cfg=env_cfg)
        metrics = _eval_policy(
            env=eval_env,
            action_selector=_make_baseline_selector(policy),
            episodes=eval_episodes,
            control_time=control_time,
            threshold_q=threshold_q,
        )
        metrics["algorithm_mode"] = mode
        metrics["baseline_policy"] = baseline_name
        save_json(os.path.join(out_dir, "metrics.json"), metrics)
        print(
            f"[HEURISTIC] RL_mean={metrics['RL_mean']:.4f} "
            f"T80_mean={metrics['T80_mean']:.2f} Q_end_mean={metrics['Q_end_mean']:.3f}"
        )
        return

    num_envs = int(train_cfg.get("num_envs", 8))
    horizon = int(train_cfg.get("rollout_horizon", 128))
    envs = [RecoveryEnv(bundle=bundle, ru=ru, cfg=env_cfg) for _ in range(num_envs)]
    obs_list = [env.reset(seed=seed + i) for i, env in enumerate(envs)]

    relations = list(edges.keys())
    tmp_env = envs[0]
    action_dims = [len(tmp_env.type_to_action_nodes.get(agent.comp_type, [])) + 1 for agent in tmp_env.agents]

    gnn_hidden = int(cfg.get("model", {}).get("gnn_hidden", 32))
    gnn_out = int(cfg.get("model", {}).get("gnn_out", 1))
    mlp_hidden = int(cfg.get("model", {}).get("mlp_hidden", 128))
    mlp_layers = int(cfg.get("model", {}).get("mlp_layers", 5))

    actors: List[ActorNet] = []
    for ad in action_dims:
        actor = ActorNet(
            bundle=bundle,
            relations=relations,
            action_dim=ad,
            gnn_hidden=gnn_hidden,
            gnn_out=gnn_out,
            mlp_hidden=mlp_hidden,
            mlp_layers=mlp_layers,
        ).to(device)
        actors.append(actor)

    critic = CriticNet(
        bundle=bundle,
        relations=relations,
        gnn_hidden=gnn_hidden,
        gnn_out=gnn_out,
        mlp_hidden=mlp_hidden,
        mlp_layers=mlp_layers,
    ).to(device)

    actor_lr = float(train_cfg.get("actor_lr", 3e-4))
    critic_lr = float(train_cfg.get("critic_lr", 3e-4))
    clip_eps = float(train_cfg.get("ppo_clip", 0.2))
    update_epochs = int(train_cfg.get("update_epochs", 4))
    minibatch_size = int(train_cfg.get("minibatch_size", 256))
    grad_clip = float(train_cfg.get("grad_norm_clip", 0.5))

    trainer = SeqMAPPOTrainer(
        actors=actors,
        critic=critic,
        edges=edges,
        actor_lrs=[actor_lr for _ in range(len(actors))],
        critic_lr=critic_lr,
        clip_eps=clip_eps,
        entropy_coef=float(algo_cfg.get("entropy_coef_start", 0.0)),
        grad_clip=grad_clip,
        device=device,
        seed=seed,
    )

    total_env_steps = int(train_cfg.get("total_env_steps", 500000))
    gamma = float(train_cfg.get("gamma", 0.999))
    lam = float(train_cfg.get("gae_lambda", 0.95))
    eval_every = int(eval_cfg.get("every_iters", 20))

    normalize_advantage = bool(algo_cfg.get("normalize_advantage", True))
    entropy_coef_start = float(algo_cfg.get("entropy_coef_start", 0.0))
    entropy_coef_end = float(algo_cfg.get("entropy_coef_end", entropy_coef_start))
    actor_lr_end = float(algo_cfg.get("actor_lr_end", actor_lr))
    critic_lr_end = float(algo_cfg.get("critic_lr_end", critic_lr))

    guide_cfg = algo_cfg.get("guide", {})
    guide_coef_start = float(guide_cfg.get("imitation_coef_start", 0.0))
    guide_coef_end = float(guide_cfg.get("imitation_coef_end", 0.0))
    guide_handoff_fraction = float(guide_cfg.get("handoff_fraction", 0.5))
    guide_policy = None
    if mode == "heuristic_guided":
        guide_policy = build_baseline_policy(name=baseline_name, bundle=bundle, objective=objective, cfg=baseline_cfg)

    global_step = 0
    it = 0
    best_metric = None
    n_agents = len(actors)
    N = bundle.num_nodes

    print(f"开始训练：objective={objective}, total_env_steps={total_env_steps}, num_envs={num_envs}, horizon={horizon}")

    while global_step < total_env_steps:
        it += 1
        progress = min(global_step / max(1, total_env_steps), 1.0)
        cur_entropy_coef = _linear_schedule(entropy_coef_start, entropy_coef_end, progress)
        cur_actor_lr = _linear_schedule(actor_lr, actor_lr_end, progress)
        cur_critic_lr = _linear_schedule(critic_lr, critic_lr_end, progress)
        cur_guide_coef = 0.0
        if guide_policy is not None:
            cur_guide_coef = _linear_schedule(
                guide_coef_start,
                guide_coef_end,
                progress,
                until=max(guide_handoff_fraction, 1e-8),
            )

        trainer.set_entropy_coef(cur_entropy_coef)
        trainer.set_learning_rates(cur_actor_lr, cur_critic_lr)

        PRINT_STEP = True
        PRINT_EVERY_IT = 10
        PRINT_FIRST_STEPS = 10
        PRINT_ENV_ID = 0
        PRINT_ROLLOUT_EVERY_IT = 1

        obs_buf = torch.zeros((horizon, num_envs, N, 5), dtype=torch.float32, device=device)
        actions_buf = torch.zeros((horizon, num_envs, n_agents), dtype=torch.long, device=device)
        logps_buf = torch.zeros((horizon, num_envs, n_agents), dtype=torch.float32, device=device)
        values_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        rewards_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        dones_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        guide_actions_buf = (
            torch.zeros((horizon, num_envs, n_agents), dtype=torch.long, device=device) if guide_policy is not None else None
        )

        masks_buf: List[torch.Tensor] = [
            torch.zeros((horizon, num_envs, action_dims[ai]), dtype=torch.float32, device=device) for ai in range(n_agents)
        ]

        for t in range(horizon):
            node_feat_batch = torch.as_tensor(
                np.stack([o["node_features"] for o in obs_list], axis=0),
                dtype=torch.float32,
                device=device,
            )

            with torch.no_grad():
                values_buf[t] = critic(node_feat_batch, edges)

            if guide_actions_buf is not None:
                for ei, env in enumerate(envs):
                    guide_actions_buf[t, ei] = torch.as_tensor(guide_policy.act(env), dtype=torch.long, device=device)

            actions_np = np.zeros((num_envs, n_agents), dtype=np.int64)
            with torch.no_grad():
                for ai, actor in enumerate(actors):
                    mask_batch_np = np.stack([o["action_masks"][ai] for o in obs_list], axis=0)
                    mask_batch = torch.as_tensor(mask_batch_np, dtype=torch.float32, device=device)

                    logits = actor(node_feat_batch, edges, action_mask=mask_batch)
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    a = dist.sample()
                    logp = dist.log_prob(a)

                    actions_np[:, ai] = a.cpu().numpy()
                    actions_buf[t, :, ai] = a
                    logps_buf[t, :, ai] = logp
                    masks_buf[ai][t] = mask_batch

            obs_buf[t] = node_feat_batch

            new_obs_list = []
            for ei, env in enumerate(envs):
                obs, r, done, info = env.step(actions_np[ei].tolist())
                rewards_buf[t, ei] = float(r)
                dones_buf[t, ei] = 1.0 if done else 0.0
                global_step += 1

                if PRINT_STEP and (it % PRINT_EVERY_IT == 0) and (ei == PRINT_ENV_ID) and (t < PRINT_FIRST_STEPS):
                    print(
                        f"[STEP] it={it} t={t} env={ei} reward={float(r):.4f} done={done} "
                        f"time={info.get('time', None)} Q={info.get('Q', None)}"
                    )

                if done:
                    obs = env.reset(seed=seed + 10_000 + ei + it * 100)
                new_obs_list.append(obs)

            obs_list = new_obs_list

        if it % PRINT_ROLLOUT_EVERY_IT == 0:
            print(
                f"[ROLLOUT] it={it} step={global_step} mean_r={rewards_buf.mean().item():.4f} "
                f"sum_r={rewards_buf.sum().item():.4f} min_r={rewards_buf.min().item():.4f} "
                f"max_r={rewards_buf.max().item():.4f} entropy_coef={cur_entropy_coef:.4f} "
                f"guide_coef={cur_guide_coef:.4f} actor_lr={cur_actor_lr:.2e} critic_lr={cur_critic_lr:.2e}"
            )

        node_feat_batch = torch.as_tensor(
            np.stack([o["node_features"] for o in obs_list], axis=0),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            last_values = critic(node_feat_batch, edges)

        advantages, returns = compute_gae(
            rewards=rewards_buf,
            dones=dones_buf,
            values=values_buf,
            last_values=last_values,
            gamma=gamma,
            lam=lam,
        )
        print(
            f"[GAE] it={it} adv_mean={advantages.mean().item():.4f} adv_std={advantages.std().item():.4f} "
            f"return_mean={returns.mean().item():.4f}"
        )

        B = horizon * num_envs
        batch = RolloutBatch(
            node_features=obs_buf.reshape(B, N, 5),
            actions=actions_buf.reshape(B, n_agents),
            old_logps=logps_buf.reshape(B, n_agents),
            action_masks=[m.reshape(B, m.shape[-1]) for m in masks_buf],
            advantages=advantages.reshape(B),
            returns=returns.reshape(B),
            guide_actions=guide_actions_buf.reshape(B, n_agents) if guide_actions_buf is not None else None,
        )

        stats = trainer.update(
            batch=batch,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            normalize_adv=normalize_advantage,
            guide_coef=cur_guide_coef,
        )

        if it % 5 == 0:
            print(
                f"it={it:04d} step={global_step} actor_loss={stats['actor_loss']:.4f} "
                f"critic_loss={stats['critic_loss']:.4f} guide_loss={stats['guide_loss']:.4f}"
            )

        if eval_every > 0 and it % eval_every == 0:
            eval_env = RecoveryEnv(bundle=bundle, ru=ru, cfg=env_cfg)
            metrics = _eval_policy(
                env=eval_env,
                action_selector=_make_actor_selector(actors, edges, device),
                episodes=eval_episodes,
                control_time=control_time,
                threshold_q=threshold_q,
            )
            metrics["algorithm_mode"] = mode
            metrics["baseline_policy"] = baseline_name if guide_policy is not None else None
            save_json(os.path.join(out_dir, f"eval_it{it:04d}.json"), metrics)
            print(
                f"[EVAL] it={it} RL_mean={metrics['RL_mean']:.4f} "
                f"T80_mean={metrics['T80_mean']:.2f} Q_end_mean={metrics['Q_end_mean']:.3f}"
            )

            key = "RL_mean" if objective == "resilience" else "T80_mean"
            cur = float(metrics[key])
            is_best = best_metric is None or cur < best_metric
            if is_best:
                best_metric = cur
                ckpt_path = os.path.join(out_dir, "checkpoints", "best.pth")
                _save_checkpoint(ckpt_path, actors, critic, action_dims, relations, objective, cfg, mode)
                print(f"[SAVE] best checkpoint -> {ckpt_path}")

        if it % 50 == 0:
            ckpt_path = os.path.join(out_dir, "checkpoints", f"iter_{it:04d}.pth")
            _save_checkpoint(ckpt_path, actors, critic, action_dims, relations, objective, cfg, mode)
            print(f"[SAVE] latest checkpoint -> {ckpt_path}")

    ckpt_path = os.path.join(out_dir, "checkpoints", "final.pth")
    _save_checkpoint(ckpt_path, actors, critic, action_dims, relations, objective, cfg, mode)
    print(f"训练结束，已保存 final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
