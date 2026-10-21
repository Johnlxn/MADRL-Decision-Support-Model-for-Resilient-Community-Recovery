
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, math, os, sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from hgmatr_recovery.algorithms.hgmatr_trainer import HGMATRRolloutBatch, HGMATRTrainer
from hgmatr_recovery.algorithms.rl_utils import compute_gae
from hgmatr_recovery.data.loader import load_data_bundle
from hgmatr_recovery.env.recovery_env import EnvConfig, RecoveryEnv
from hgmatr_recovery.env.resilience import compute_Q, resilience_loss
from hgmatr_recovery.models.hgmatr import FeatureScale, HGMATRConfig, HGMATRNet
from hgmatr_recovery.utils.config import ensure_dir, load_yaml, set_global_seeds
from hgmatr_recovery.utils.io import save_json

def _edges_to_torch(bundle, device: str) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.long, device=device) for k, v in bundle.edges.items()}

def _reward_components(q_prev: float, dt: float, threshold_q: float) -> Tuple[float, float]:
    reward_res = -(1.0 - float(q_prev)) * float(dt)
    reward_t80 = -float(dt) if float(q_prev) < float(threshold_q) else 0.0
    return float(reward_res), float(reward_t80)

def _mixed_reward(objective: str, reward_res: float, reward_t80: float, alpha_mix: float) -> float:
    objective = str(objective).lower()
    if objective == 'resilience':
        return float(reward_res)
    if objective == 'time80':
        return float(reward_t80)
    if objective == 'mixed':
        return float(alpha_mix * reward_res + (1.0 - alpha_mix) * reward_t80)
    raise ValueError('objective 必须是 resilience / time80 / mixed')

def _parse_model_config(model_cfg: Dict[str, object], bundle) -> Tuple[HGMATRConfig, FeatureScale]:
    max_er = max(1.0, np.max([n.expected_repair_time for n in bundle.nodes]))
    cfg = HGMATRConfig(
        hgt_hidden=int(model_cfg.get('hgt_hidden', 64)), hgt_out=int(model_cfg.get('hgt_out', 64)), hgt_heads=int(model_cfg.get('hgt_heads', 4)), hgt_layers=int(model_cfg.get('hgt_layers', 2)), hgt_dropout=float(model_cfg.get('hgt_dropout', 0.1)), hgt_attn_dropout=float(model_cfg.get('hgt_attn_dropout', 0.0)), hgt_use_norm=bool(model_cfg.get('hgt_use_norm', True)), hgt_use_residual=bool(model_cfg.get('hgt_use_residual', True)), hgt_ffn_multiplier=float(model_cfg.get('hgt_ffn_multiplier', 2.0)), hgt_activation=str(model_cfg.get('hgt_activation', 'gelu')), hgt_qkv_bias=bool(model_cfg.get('hgt_qkv_bias', True)), hgt_input_bias=bool(model_cfg.get('hgt_input_bias', True)), hgt_output_bias=bool(model_cfg.get('hgt_output_bias', True)), hgt_use_relation_pri=bool(model_cfg.get('hgt_use_relation_pri', True)), hgt_readout=str(model_cfg.get('hgt_readout', 'attention')), hgt_gate_hidden=int(model_cfg.get('hgt_gate_hidden', 64)), memory_dim=int(model_cfg.get('memory_dim', 128)), action_embed_dim=int(model_cfg.get('action_embed_dim', 32)), agent_type_embed_dim=int(model_cfg.get('agent_type_embed_dim', 24)), model_dim=int(model_cfg.get('model_dim', 128)), decoder_heads=int(model_cfg.get('decoder_heads', 4)), decoder_layers=int(model_cfg.get('decoder_layers', 2)), decoder_ffn_dim=int(model_cfg.get('decoder_ffn_dim', 256)), decoder_dropout=float(model_cfg.get('decoder_dropout', 0.1)), decoder_activation=str(model_cfg.get('decoder_activation', 'gelu')), decoder_norm_first=bool(model_cfg.get('decoder_norm_first', False)), policy_hidden=int(model_cfg.get('policy_hidden', 128)), value_hidden=int(model_cfg.get('value_hidden', 128)), aux_hidden=int(model_cfg.get('aux_hidden', 128)), time_scale=float(model_cfg.get('time_scale', 100.0)),
    )
    feature_scale = FeatureScale(ds=float(model_cfg.get('scale_ds', 4.0)), fn=float(model_cfg.get('scale_fn', 1.0)), nru=float(model_cfg.get('scale_nru', 2.0)), ac=float(model_cfg.get('scale_ac', 1.0)), er=float(model_cfg.get('scale_er', max_er)))
    return cfg, feature_scale

def _eval_policy(env: RecoveryEnv, model: HGMATRNet, edges: Dict[str, torch.Tensor], episodes: int, device: str, control_time: float, threshold_q: float, objective: str, alpha_mix: float) -> Dict[str, float]:
    rls, t80s, qends = [], [], []
    for ep in range(int(episodes)):
        obs = env.reset(seed=1000 + ep)
        hidden = model.init_memory(1, device=device)
        prev_actions = torch.zeros(1, env.n_agents, dtype=torch.long, device=device)
        prev_reward = torch.zeros(1, dtype=torch.float32, device=device)
        episode_start = torch.ones(1, dtype=torch.float32, device=device)
        times = [float(obs['time'][0])]
        q0 = compute_Q(env.bundle.buildings, env.indicators.i_w, env.indicators.i_p, env.indicators.i_t)
        qs = [float(q0)]
        reached_t80 = None
        done = False
        while not done:
            node_feat = torch.as_tensor(obs['node_features'], dtype=torch.float32, device=device).unsqueeze(0)
            mask_batch = [torch.as_tensor(obs['action_masks'][ai], dtype=torch.float32, device=device).unsqueeze(0) for ai in range(env.n_agents)]
            time_batch = torch.as_tensor(obs['time'], dtype=torch.float32, device=device).view(1)
            out = model.act(node_features=node_feat, edges=edges, action_masks=mask_batch, prev_memory=hidden, prev_actions=prev_actions, prev_reward=prev_reward, times=time_batch, episode_start=episode_start, deterministic=True)
            actions = out['actions'].squeeze(0).tolist()
            hidden = out['memory'].unsqueeze(0) if out['memory'].dim() == 1 else out['memory']
            obs_next, _, done, info = env.step(actions)
            times.append(float(info.get('time', 0.0)))
            qs.append(float(info.get('Q', 0.0)))
            if reached_t80 is None and qs[-1] >= threshold_q:
                reached_t80 = float(info.get('time', 0.0))
            reward_res, reward_t80 = _reward_components(info.get('Q_prev', info.get('Q', 0.0)), info.get('dt', 0.0), threshold_q)
            reward_mix = _mixed_reward(objective, reward_res, reward_t80, alpha_mix)
            prev_actions = torch.as_tensor(actions, dtype=torch.long, device=device).view(1, -1)
            prev_reward = torch.tensor([reward_mix], dtype=torch.float32, device=device)
            episode_start = torch.zeros(1, dtype=torch.float32, device=device)
            obs = obs_next
            if float(info.get('time', 0.0)) >= control_time:
                break
        rls.append(float(resilience_loss(times, qs, control_time=control_time)))
        t80s.append(float(reached_t80 if reached_t80 is not None else control_time))
        qends.append(float(qs[-1]))
    return {'RL_mean': float(np.mean(rls)), 'T80_mean': float(np.mean(t80s)), 'Q_end_mean': float(np.mean(qends))}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_hgmatr_resilience.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    cfg_path = ROOT / args.config if not os.path.isabs(args.config) else Path(args.config)
    cfg = load_yaml(str(cfg_path))
    seed = int(cfg.get('seed', 42))
    set_global_seeds(seed)
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('[WARN] CUDA 不可用，自动回退到 CPU')
        device = 'cpu'
    data_dir = str(ROOT / cfg.get('data', {}).get('data_dir', 'data/synth_campus'))
    out_dir = str(ROOT / cfg.get('output', {}).get('out_dir', 'outputs/hgmatr'))
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, 'checkpoints'))
    save_json(os.path.join(out_dir, 'config_snapshot.json'), cfg)
    objective = str(cfg.get('env', {}).get('objective', 'resilience')).lower()
    control_time = float(cfg.get('env', {}).get('control_time', 100.0))
    threshold_q = float(cfg.get('env', {}).get('threshold_q', 0.8))
    bundle = load_data_bundle(data_dir)
    edges = _edges_to_torch(bundle, device=device)
    env_cfg = EnvConfig(objective='resilience' if objective == 'mixed' else objective, control_time=control_time, threshold_q=threshold_q, lognormal_sigma=float(cfg.get('repair_time', {}).get('lognormal_sigma', 0.4)), trunc_low_ratio=float(cfg.get('repair_time', {}).get('trunc_low_ratio', 0.1)), trunc_high_ratio=float(cfg.get('repair_time', {}).get('trunc_high_ratio', 10.0)))
    ru = {k: int(v) for k, v in cfg.get('ru', {'pipeline': 2, 'substation': 1, 'bridge': 1}).items()}
    train_cfg = cfg.get('train', {})
    num_envs = int(train_cfg.get('num_envs', 8)); horizon = int(train_cfg.get('rollout_horizon', 128)); total_env_steps = int(train_cfg.get('total_env_steps', 200000)); gamma = float(train_cfg.get('gamma', 0.999)); lam = float(train_cfg.get('gae_lambda', 0.95)); update_epochs = int(train_cfg.get('update_epochs', 4)); env_minibatch_size = int(train_cfg.get('env_minibatch_size', max(1, num_envs // 2))); print_every = int(train_cfg.get('print_every', 1)); save_every = int(train_cfg.get('save_every', 50)); total_updates = max(1, math.ceil(total_env_steps / max(1, num_envs * horizon)))
    envs = [RecoveryEnv(bundle=bundle, ru=ru, cfg=env_cfg) for _ in range(num_envs)]
    obs_list = [env.reset(seed=seed + i) for i, env in enumerate(envs)]
    tmp_env = envs[0]
    action_dims = [len(tmp_env.type_to_action_nodes.get(agent.comp_type, [])) + 1 for agent in tmp_env.agents]
    agent_types = [agent.comp_type for agent in tmp_env.agents]
    relations = list(edges.keys())
    model_cfg, feature_scale = _parse_model_config(cfg.get('model', {}), bundle)
    model = HGMATRNet(bundle=bundle, relations=relations, agent_types=agent_types, action_dims=action_dims, cfg=model_cfg, feature_scale=feature_scale).to(device)
    print(f'[INFO] HGMAT-R | hgt_out={model_cfg.hgt_out} | memory_dim={model_cfg.memory_dim} | model_dim={model_cfg.model_dim} | decoder_layers={model_cfg.decoder_layers}')
    trainer = HGMATRTrainer(model=model, edges=edges, lr=float(train_cfg.get('lr', 3e-4)), clip_eps=float(train_cfg.get('ppo_clip', 0.2)), entropy_coef=float(train_cfg.get('entropy_coef', 0.01)), value_coef=float(train_cfg.get('value_coef', 0.5)), aux_coef=float(train_cfg.get('aux_coef', 0.1)), grad_clip=float(train_cfg.get('grad_norm_clip', 0.5)), alpha_start=float(train_cfg.get('alpha_start', 0.6)), alpha_end=float(train_cfg.get('alpha_end', 0.3)), device=device, seed=seed)
    eval_cfg = cfg.get('eval', {})
    eval_every = int(eval_cfg.get('every_iters', 20)); eval_episodes = int(eval_cfg.get('episodes', 5))
    global_step = 0; it = 0; best_metric = None
    hidden_state = model.init_memory(num_envs, device=device)
    prev_actions_state = torch.zeros(num_envs, len(action_dims), dtype=torch.long, device=device)
    prev_reward_state = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_start_state = torch.ones(num_envs, dtype=torch.float32, device=device)
    print(f'开始训练 HGMAT-R：objective={objective}, total_env_steps={total_env_steps}, num_envs={num_envs}, horizon={horizon}, device={device}')
    while global_step < total_env_steps:
        it += 1
        alpha_curr = trainer._mix_alpha(it - 1, total_updates)
        n_nodes = bundle.num_nodes; n_agents = len(action_dims)
        obs_buf = torch.zeros((horizon, num_envs, n_nodes, 5), dtype=torch.float32, device=device)
        times_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        prev_actions_buf = torch.zeros((horizon, num_envs, n_agents), dtype=torch.long, device=device)
        prev_reward_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        episode_starts_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        actions_buf = torch.zeros((horizon, num_envs, n_agents), dtype=torch.long, device=device)
        logps_buf = torch.zeros((horizon, num_envs, n_agents), dtype=torch.float32, device=device)
        values_res_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        values_t80_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        rewards_res_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        rewards_t80_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        dones_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        target_next_fn_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        target_next_ac_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        target_completion_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        aux_valid_buf = torch.zeros((horizon, num_envs), dtype=torch.float32, device=device)
        masks_buf = [torch.zeros((horizon, num_envs, action_dims[ai]), dtype=torch.float32, device=device) for ai in range(n_agents)]
        for t in range(horizon):
            node_feat_batch_np = np.stack([o['node_features'] for o in obs_list], axis=0)
            times_batch_np = np.stack([o['time'] for o in obs_list], axis=0).reshape(num_envs)
            node_feat_batch = torch.as_tensor(node_feat_batch_np, dtype=torch.float32, device=device)
            times_batch = torch.as_tensor(times_batch_np, dtype=torch.float32, device=device)
            masks_batch = [torch.as_tensor(np.stack([o['action_masks'][ai] for o in obs_list], axis=0), dtype=torch.float32, device=device) for ai in range(n_agents)]
            obs_buf[t] = node_feat_batch; times_buf[t] = times_batch; prev_actions_buf[t] = prev_actions_state; prev_reward_buf[t] = prev_reward_state; episode_starts_buf[t] = episode_start_state
            for ai in range(n_agents): masks_buf[ai][t] = masks_batch[ai]
            with torch.no_grad():
                out = model.act(node_features=node_feat_batch, edges=edges, action_masks=masks_batch, prev_memory=hidden_state, prev_actions=prev_actions_state, prev_reward=prev_reward_state, times=times_batch, episode_start=episode_start_state, deterministic=False)
            actions = out['actions']; logps = out['logps']; new_hidden = out['memory']
            values_res_buf[t] = out['value_res']; values_t80_buf[t] = out['value_t80']; actions_buf[t] = actions; logps_buf[t] = logps
            actions_np = actions.detach().cpu().numpy()
            new_obs_list = []; next_hidden_state = hidden_state.clone(); next_prev_actions_state = prev_actions_state.clone(); next_prev_reward_state = prev_reward_state.clone(); next_episode_start_state = torch.zeros_like(episode_start_state)
            for ei, env in enumerate(envs):
                cur_obs = obs_list[ei]
                obs_next, _, done, info = env.step(actions_np[ei].tolist())
                q_prev = float(info.get('Q_prev', info.get('Q', 0.0))); dt = float(info.get('dt', 0.0))
                reward_res, reward_t80 = _reward_components(q_prev, dt, threshold_q); reward_mix = _mixed_reward(objective, reward_res, reward_t80, alpha_curr)
                rewards_res_buf[t, ei] = reward_res; rewards_t80_buf[t, ei] = reward_t80; dones_buf[t, ei] = 1.0 if done else 0.0; global_step += 1
                if not done:
                    target_next_fn_buf[t, ei] = float(np.mean(obs_next['node_features'][:, 1])); target_next_ac_buf[t, ei] = float(np.mean(obs_next['node_features'][:, 3])); target_completion_buf[t, ei] = float(np.any(obs_next['node_features'][:, 0] < cur_obs['node_features'][:, 0])); aux_valid_buf[t, ei] = 1.0
                if done:
                    obs_reset = env.reset(seed=seed + 10_000 + ei + it * 100); new_obs_list.append(obs_reset); next_hidden_state[ei].zero_(); next_prev_actions_state[ei].zero_(); next_prev_reward_state[ei] = 0.0; next_episode_start_state[ei] = 1.0
                else:
                    new_obs_list.append(obs_next); next_hidden_state[ei] = new_hidden[ei]; next_prev_actions_state[ei] = actions[ei]; next_prev_reward_state[ei] = reward_mix; next_episode_start_state[ei] = 0.0
            obs_list = new_obs_list; hidden_state = next_hidden_state; prev_actions_state = next_prev_actions_state; prev_reward_state = next_prev_reward_state; episode_start_state = next_episode_start_state
        node_feat_batch_np = np.stack([o['node_features'] for o in obs_list], axis=0)
        times_batch_np = np.stack([o['time'] for o in obs_list], axis=0).reshape(num_envs)
        node_feat_batch = torch.as_tensor(node_feat_batch_np, dtype=torch.float32, device=device)
        times_batch = torch.as_tensor(times_batch_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            last_heads = model.bootstrap_values(node_features=node_feat_batch, edges=edges, times=times_batch, prev_actions=prev_actions_state, prev_reward=prev_reward_state, prev_memory=hidden_state, episode_start=episode_start_state)
        advantages_res, returns_res = compute_gae(rewards=rewards_res_buf, dones=dones_buf, values=values_res_buf, last_values=last_heads['value_res'], gamma=gamma, lam=lam)
        advantages_t80, returns_t80 = compute_gae(rewards=rewards_t80_buf, dones=dones_buf, values=values_t80_buf, last_values=last_heads['value_t80'], gamma=gamma, lam=lam)
        batch = HGMATRRolloutBatch(node_features=obs_buf, times=times_buf, prev_actions=prev_actions_buf, prev_reward=prev_reward_buf, episode_starts=episode_starts_buf, actions=actions_buf, old_logps=logps_buf, action_masks=masks_buf, rewards_res=rewards_res_buf, rewards_t80=rewards_t80_buf, dones=dones_buf, values_res=values_res_buf, values_t80=values_t80_buf, advantages_res=advantages_res, advantages_t80=advantages_t80, returns_res=returns_res, returns_t80=returns_t80, target_next_fn=target_next_fn_buf, target_next_ac=target_next_ac_buf, target_completion=target_completion_buf, aux_valid=aux_valid_buf)
        stats = trainer.update(batch=batch, update_idx=it - 1, total_updates=total_updates, update_epochs=update_epochs, env_minibatch_size=env_minibatch_size)
        if it % print_every == 0:
            progress = (global_step / total_env_steps) * 100.0
            print(f"[Train] Iter={it:04d} | Step={global_step}/{total_env_steps} ({progress:.1f}%) | R_res={rewards_res_buf.mean().item():.4f} | R_t80={rewards_t80_buf.mean().item():.4f} | Loss={stats['loss']:.4f} | Actor={stats['actor_loss']:.4f} | Critic={stats['critic_loss']:.4f} | Aux={stats['aux_loss']:.4f} | Ent={stats['entropy']:.4f} | alpha={stats['alpha_mix']:.3f}")
        if eval_every > 0 and it % eval_every == 0:
            eval_env = RecoveryEnv(bundle=bundle, ru=ru, cfg=env_cfg)
            metrics = _eval_policy(eval_env, model, edges, eval_episodes, device, control_time, threshold_q, objective, float(stats['alpha_mix']))
            save_json(os.path.join(out_dir, f'eval_it{it:04d}.json'), metrics)
            print(f"[EVAL] Iter={it} | RL_mean={metrics['RL_mean']:.4f} | T80_mean={metrics['T80_mean']:.2f} | Q_end={metrics['Q_end_mean']:.3f}")
            key = 'RL_mean' if objective != 'time80' else 'T80_mean'; cur = float(metrics[key])
            if best_metric is None or cur < best_metric:
                best_metric = cur
                ckpt_path = os.path.join(out_dir, 'checkpoints', 'best.pth')
                torch.save({'model': model.state_dict(), 'action_dims': action_dims, 'agent_types': agent_types, 'relations': relations, 'objective': objective, 'config': cfg}, ckpt_path)
                print(f'[SAVE] new best -> {ckpt_path}')
        if save_every > 0 and it % save_every == 0:
            ckpt_path = os.path.join(out_dir, 'checkpoints', f'iter_{it:04d}.pth')
            torch.save({'model': model.state_dict(), 'action_dims': action_dims, 'agent_types': agent_types, 'relations': relations, 'objective': objective, 'config': cfg}, ckpt_path)
    ckpt_path = os.path.join(out_dir, 'checkpoints', 'final.pth')
    torch.save({'model': model.state_dict(), 'action_dims': action_dims, 'agent_types': agent_types, 'relations': relations, 'objective': objective, 'config': cfg}, ckpt_path)
    print(f'训练结束，已保存 final checkpoint: {ckpt_path}')
if __name__ == '__main__':
    main()
