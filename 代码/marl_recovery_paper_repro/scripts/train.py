# -*- coding: utf-8 -*-
"""训练脚本：Seq-MAPPO + GNN-FiLM。

对齐论文：
- 目标1：最大化韧性（最小化 RL），奖励 Eq.(20)
- 目标2：最小化恢复到 80% 的时间，奖励 Eq.(21)
- 算法：顺序更新多智能体 PPO（Eq.(16)(17) + Algorithm 1）
- 网络：GNN-FiLM（Eq.(18)(19)） + MLP（Table 2）

用法：
python scripts/train.py --config configs/train_resilience.yaml
"""

from __future__ import annotations
import os
import sys

# 让脚本在命令行与 PyCharm 中都能直接 import 本工程包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import argparse
from typing import Dict, List

import numpy as np
import torch

from marl_recovery.data.loader import load_data_bundle
from marl_recovery.env.recovery_env import EnvConfig, RecoveryEnv
from marl_recovery.env.resilience import compute_Q, resilience_loss
from marl_recovery.algorithms.rl_utils import compute_gae
from marl_recovery.algorithms.seq_mappo import RolloutBatch, SeqMAPPOTrainer
from marl_recovery.models.actor_critic import ActorNet, CriticNet  #切换htg
from marl_recovery.utils.config import ensure_dir, load_yaml, set_global_seeds
from marl_recovery.utils.io import save_json

from dataclasses import replace

def _edges_to_torch(bundle, device: str) -> Dict[str, torch.Tensor]:
    edges = {}
    for etype, arr in bundle.edges.items():
        edges[etype] = torch.as_tensor(arr, dtype=torch.long, device=device)
    return edges


def _eval_policy(
    env: RecoveryEnv,
    actors: List[ActorNet],
    edges: Dict[str, torch.Tensor],
    episodes: int,
    device: str,
    control_time: float,
    threshold_q: float,
) -> Dict[str, float]:
    """简单评估：返回平均 RL、平均 T80、平均 Q_end。"""
    rls = []
    t80s = []
    qends = []

    for ep in range(int(episodes)):
        obs = env.reset(seed=1000 + ep)
        times = [float(obs["time"][0])]

        # 初始 Q
        q0 = compute_Q(env.bundle.buildings, env.indicators.i_w, env.indicators.i_p, env.indicators.i_t)  # type: ignore
        qs = [float(q0)]

        reached_t80 = None
        done = False

        while not done:
            node_feat = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
            masks_np = obs["action_masks"]

            acts = []
            for ai, actor in enumerate(actors):
                mask = torch.as_tensor(masks_np[ai], dtype=torch.float32, device=device)
                a, _ = actor.sample_action(node_feat, edges, mask)
                acts.append(a)

            obs, _, done, info = env.step(acts)
            times.append(float(info.get("time", 0.0)))
            qs.append(float(info.get("Q", 0.0)))

            if reached_t80 is None and qs[-1] >= threshold_q:
                reached_t80 = float(info.get("time", 0.0))

            if float(info.get("time", 0.0)) >= control_time:
                break

        rl = resilience_loss(times, qs, control_time=control_time)
        rls.append(rl)
        t80s.append(reached_t80 if reached_t80 is not None else control_time)
        qends.append(qs[-1])

    return {
        "RL_mean": float(np.mean(rls)),
        "T80_mean": float(np.mean(t80s)),
        "Q_end_mean": float(np.mean(qends)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_resilience.yaml", help="训练配置 YAML")
    parser.add_argument("--device", type=str, default="cuda", help="cpu 或 cuda")
    args = parser.parse_args()

    # 以脚本位置推断项目根目录：.../marl_recovery_paper_repro
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

    # data_dir 解析为绝对路径（兼容相对路径）
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

    # 保存配置快照
    save_json(os.path.join(out_dir, "config_snapshot.json"), cfg)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] 你选择了 cuda，但当前环境检测不到 GPU，将回退到 cpu")
        device = "cpu"

    print(f"[INFO] device={device} | torch={torch.__version__} | cuda_available={torch.cuda.is_available()}")

    # # 读取数据
    # bundle = load_data_bundle(data_dir)
    # edges = _edges_to_torch(bundle, device=device)

    # 读取数据
    bundle = load_data_bundle(data_dir)

    # ====== 用 YAML beta 覆盖 buildings.csv 的 beta（让 compute_Q 立刻生效）======
    beta_cfg = cfg.get("beta", None)
    if isinstance(beta_cfg, dict):
        bw = float(beta_cfg.get("w", 0.55))
        bp = float(beta_cfg.get("p", 0.40))
        bt = float(beta_cfg.get("t", 0.05))

        # 可选：做个归一化（防止你写的 beta 不加和为 1）
        s = bw + bp + bt
        if s > 0:
            bw, bp, bt = bw / s, bp / s, bt / s

        bundle.buildings = [replace(b, beta_w=bw, beta_p=bp, beta_t=bt) for b in bundle.buildings]
        print(f"[INFO] Override building betas from YAML: beta_w={bw:.3f}, beta_p={bp:.3f}, beta_t={bt:.3f}")
    # =======================================================================
    edges = _edges_to_torch(bundle, device=device)


    # 环境
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

    num_envs = int(cfg.get("train", {}).get("num_envs", 8))
    horizon = int(cfg.get("train", {}).get("rollout_horizon", 128))

    envs = [RecoveryEnv(bundle=bundle, ru=ru, cfg=env_cfg) for _ in range(num_envs)]
    obs_list = [env.reset(seed=seed + i) for i, env in enumerate(envs)]

    # 创建网络
    relations = list(edges.keys())

    # 每个 agent 的动作维度：按 env0 的 agent specs
    tmp_env = envs[0]
    action_dims = []
    for agent in tmp_env.agents:
        action_dims.append(len(tmp_env.type_to_action_nodes.get(agent.comp_type, [])) + 1)

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

    # Trainer
    actor_lr = float(cfg.get("train", {}).get("actor_lr", 3e-4))
    critic_lr = float(cfg.get("train", {}).get("critic_lr", 3e-4))
    clip_eps = float(cfg.get("train", {}).get("ppo_clip", 0.2))
    update_epochs = int(cfg.get("train", {}).get("update_epochs", 4))
    minibatch_size = int(cfg.get("train", {}).get("minibatch_size", 256))
    grad_clip = float(cfg.get("train", {}).get("grad_norm_clip", 0.5))

    trainer = SeqMAPPOTrainer(
        actors=actors,
        critic=critic,
        edges=edges,
        actor_lrs=[actor_lr for _ in range(len(actors))],
        critic_lr=critic_lr,
        clip_eps=clip_eps,
        entropy_coef=0.0,
        grad_clip=grad_clip,
        device=device,
        seed=seed,
    )

    # 训练循环
    total_env_steps = int(cfg.get("train", {}).get("total_env_steps", 500000))
    gamma = float(cfg.get("train", {}).get("gamma", 0.999))
    lam = float(cfg.get("train", {}).get("gae_lambda", 0.95))

    eval_every = int(cfg.get("eval", {}).get("every_iters", 20))
    eval_episodes = int(cfg.get("eval", {}).get("episodes", 5))

    global_step = 0
    it = 0
    best_metric = None

    print(f"开始训练：objective={objective}, total_env_steps={total_env_steps}, num_envs={num_envs}, horizon={horizon}")

    while global_step < total_env_steps:
        it += 1

        # ====== 打印控制（你可以改这里）======
        PRINT_STEP = True              # 是否打印 step 级别 reward
        PRINT_EVERY_IT = 10            # 每隔多少个 it 打印一次 step 细节
        PRINT_FIRST_STEPS = 10         # 每次打印前多少个 step
        PRINT_ENV_ID = 0               # 只看第几个并行环境（0 表示 envs[0]）
        PRINT_ROLLOUT_EVERY_IT = 1     # rollout 汇总每隔多少 it 打印一次（默认每次）

        # rollout buffer
        N = bundle.num_nodes
        n_agents = len(actors)

        obs_buf = torch.zeros((horizon, num_envs, N, 5), dtype=torch.float32, device=device)
        actions_buf = torch.zeros((horizon, num_envs, n_agents), dtype=torch.long, device=device)
        logps_buf = torch.zeros((horizon, num_envs, n_agents), dtype=torch.float32, device=device)
        values_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        rewards_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        dones_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)

        masks_buf: List[torch.Tensor] = []
        for ai in range(n_agents):
            masks_buf.append(torch.zeros((horizon, num_envs, action_dims[ai]), dtype=torch.float32, device=device))

        for t in range(horizon):
            # 拼 batch
            node_feat_batch = torch.as_tensor(
                np.stack([o["node_features"] for o in obs_list], axis=0),
                dtype=torch.float32,
                device=device,
            )

            # critic value
            with torch.no_grad():
                v = critic(node_feat_batch, edges)  # (num_envs,)
            values_buf[t] = v

            # actor actions
            actions_np = np.zeros((num_envs, n_agents), dtype=np.int64)

            with torch.no_grad():
                for ai, actor in enumerate(actors):
                    mask_batch_np = np.stack([o["action_masks"][ai] for o in obs_list], axis=0)
                    mask_batch = torch.as_tensor(mask_batch_np, dtype=torch.float32, device=device)

                    logits = actor(node_feat_batch, edges, action_mask=mask_batch)
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    a = dist.sample()  # (num_envs,)
                    logp = dist.log_prob(a)

                    actions_np[:, ai] = a.cpu().numpy()
                    actions_buf[t, :, ai] = a
                    logps_buf[t, :, ai] = logp
                    masks_buf[ai][t] = mask_batch

            # 记录 obs
            obs_buf[t] = node_feat_batch

            # step envs
            new_obs_list = []
            for ei, env in enumerate(envs):
                obs, r, done, _info = env.step(actions_np[ei].tolist())
                rewards_buf[t, ei] = float(r)
                dones_buf[t, ei] = 1.0 if done else 0.0
                global_step += 1

                # ====== 打印每一步奖励（只打印部分步数，防止刷屏）======
                if PRINT_STEP and (it % PRINT_EVERY_IT == 0) and (ei == PRINT_ENV_ID) and (t < PRINT_FIRST_STEPS):
                    print(
                        f"[STEP] it={it} t={t} env={ei} "
                        f"reward={float(r):.4f} done={done} "
                        f"time={_info.get('time', None)} Q={_info.get('Q', None)}"
                    )

                if done:
                    obs = env.reset(seed=seed + 10_000 + ei + it * 100)
                new_obs_list.append(obs)

            obs_list = new_obs_list

        # ====== rollout 完成后打印本轮 reward 汇总 ======
        if it % PRINT_ROLLOUT_EVERY_IT == 0:
            rollout_mean_reward = rewards_buf.mean().item()
            rollout_sum_reward = rewards_buf.sum().item()
            rollout_max_reward = rewards_buf.max().item()
            rollout_min_reward = rewards_buf.min().item()
            print(
                f"[ROLLOUT] it={it} step={global_step} "
                f"mean_r={rollout_mean_reward:.4f} sum_r={rollout_sum_reward:.4f} "
                f"min_r={rollout_min_reward:.4f} max_r={rollout_max_reward:.4f}"
            )

        # bootstrap values
        node_feat_batch = torch.as_tensor(
            np.stack([o["node_features"] for o in obs_list], axis=0),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            last_values = critic(node_feat_batch, edges)  # (num_envs,)

        advantages, returns = compute_gae(
            rewards=rewards_buf,
            dones=dones_buf,
            values=values_buf,
            last_values=last_values,
            gamma=gamma,
            lam=lam,
        )

        # ====== 打印 GAE 统计（了解 advantage 是否稳定）======
        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item()
        ret_mean = returns.mean().item()
        print(f"[GAE] it={it} adv_mean={adv_mean:.4f} adv_std={adv_std:.4f} return_mean={ret_mean:.4f}")

        # flatten
        B = horizon * num_envs
        batch = RolloutBatch(
            node_features=obs_buf.reshape(B, N, 5),
            actions=actions_buf.reshape(B, n_agents),
            old_logps=logps_buf.reshape(B, n_agents),
            action_masks=[m.reshape(B, m.shape[-1]) for m in masks_buf],
            advantages=advantages.reshape(B),
            returns=returns.reshape(B),
        )

        stats = trainer.update(batch=batch, update_epochs=update_epochs, minibatch_size=minibatch_size)

        if it % 5 == 0:
            print(f"it={it:04d} step={global_step} actor_loss={stats['actor_loss']:.4f} critic_loss={stats['critic_loss']:.4f}")

        # evaluate & save
        if eval_every > 0 and it % eval_every == 0:
            eval_env = RecoveryEnv(bundle=bundle, ru=ru, cfg=env_cfg)
            metrics = _eval_policy(
                env=eval_env,
                actors=actors,
                edges=edges,
                episodes=eval_episodes,
                device=device,
                control_time=control_time,
                threshold_q=threshold_q,
            )

            save_json(os.path.join(out_dir, f"eval_it{it:04d}.json"), metrics)
            print(
                f"[EVAL] it={it} RL_mean={metrics['RL_mean']:.4f} "
                f"T80_mean={metrics['T80_mean']:.2f} Q_end_mean={metrics['Q_end_mean']:.3f}"
            )

            # 选择 best：目标1 用 RL（越小越好），目标2 用 T80（越小越好）
            key = "RL_mean" if objective == "resilience" else "T80_mean"
            cur = float(metrics[key])
            is_best = False
            if best_metric is None or cur < best_metric:
                best_metric = cur
                is_best = True

            if is_best:
                ckpt_path = os.path.join(out_dir, "checkpoints", "best.pth")
                torch.save(
                    {
                        "actors": [a.state_dict() for a in actors],
                        "critic": critic.state_dict(),
                        "action_dims": action_dims,
                        "relations": relations,
                        "objective": objective,
                        "config": cfg,
                    },
                    ckpt_path,
                )
                print(f"[SAVE] best checkpoint -> {ckpt_path}")

        # 定期保存 latest
        if it % 50 == 0:
            ckpt_path = os.path.join(out_dir, "checkpoints", f"iter_{it:04d}.pth")
            torch.save(
                {
                    "actors": [a.state_dict() for a in actors],
                    "critic": critic.state_dict(),
                    "action_dims": action_dims,
                    "relations": relations,
                    "objective": objective,
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"[SAVE] latest checkpoint -> {ckpt_path}")

    # 保存 final
    ckpt_path = os.path.join(out_dir, "checkpoints", "final.pth")
    torch.save(
        {
            "actors": [a.state_dict() for a in actors],
            "critic": critic.state_dict(),
            "action_dims": action_dims,
            "relations": relations,
            "objective": objective,
            "config": cfg,
        },
        ckpt_path,
    )
    print(f"训练结束，已保存 final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
