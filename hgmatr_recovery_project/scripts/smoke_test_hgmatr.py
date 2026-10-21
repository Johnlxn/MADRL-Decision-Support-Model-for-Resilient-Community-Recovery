
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from hgmatr_recovery.data.loader import load_data_bundle
from hgmatr_recovery.env.recovery_env import EnvConfig, RecoveryEnv
from hgmatr_recovery.models.hgmatr import FeatureScale, HGMATRConfig, HGMATRNet

def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bundle = load_data_bundle(str(ROOT / 'data' / 'synth_campus'))
    env = RecoveryEnv(bundle=bundle, ru={'pipeline': 2, 'substation': 1, 'bridge': 1}, cfg=EnvConfig(control_time=30.0))
    obs = env.reset(seed=7)
    action_dims = [len(env.type_to_action_nodes.get(agent.comp_type, [])) + 1 for agent in env.agents]
    agent_types = [agent.comp_type for agent in env.agents]
    relations = list(bundle.edges.keys())
    edges = {k: torch.as_tensor(v, dtype=torch.long, device=device) for k, v in bundle.edges.items()}
    cfg = HGMATRConfig(hgt_hidden=8, hgt_out=8, hgt_heads=2, hgt_layers=1, hgt_dropout=0.0, hgt_attn_dropout=0.0, memory_dim=8, action_embed_dim=4, agent_type_embed_dim=4, model_dim=8, decoder_heads=2, decoder_layers=1, decoder_ffn_dim=16, decoder_dropout=0.0, decoder_norm_first=False, policy_hidden=8, value_hidden=8, aux_hidden=8, time_scale=30.0)
    model = HGMATRNet(bundle=bundle, relations=relations, agent_types=agent_types, action_dims=action_dims, cfg=cfg, feature_scale=FeatureScale()).to(device).eval()
    hidden = model.init_memory(1, device=device)
    prev_actions = torch.zeros(1, len(action_dims), dtype=torch.long, device=device)
    prev_reward = torch.zeros(1, dtype=torch.float32, device=device)
    episode_start = torch.ones(1, dtype=torch.float32, device=device)
    node_feat = torch.as_tensor(obs['node_features'], dtype=torch.float32, device=device).unsqueeze(0)
    masks = [torch.as_tensor(obs['action_masks'][i], dtype=torch.float32, device=device).unsqueeze(0) for i in range(len(action_dims))]
    time_batch = torch.as_tensor(obs['time'], dtype=torch.float32, device=device).view(1)
    with torch.no_grad():
        out = model.act(node_features=node_feat, edges=edges, action_masks=masks, prev_memory=hidden, prev_actions=prev_actions, prev_reward=prev_reward, times=time_batch, episode_start=episode_start, deterministic=False)
    _, reward, _, info = env.step(out['actions'].squeeze(0).tolist())
    seq_out = model.evaluate_sequence(node_features=node_feat.unsqueeze(0), edges=edges, times=time_batch.unsqueeze(0), prev_actions=prev_actions.unsqueeze(0), prev_reward=prev_reward.unsqueeze(0), episode_starts=episode_start.unsqueeze(0), actions=out['actions'].view(1, 1, -1), action_masks=[m.unsqueeze(0) for m in masks])
    print(f"[HGMAT-R] OK | device={device} | actions={out['actions'].tolist()} | value_res={float(out['value_res']):.4f} | value_t80={float(out['value_t80']):.4f} | reward={float(reward):.4f} | next_time={float(info.get('time', 0.0)):.4f} | joint_logp_shape={tuple(seq_out['joint_logp'].shape)}")
if __name__ == '__main__':
    main()
