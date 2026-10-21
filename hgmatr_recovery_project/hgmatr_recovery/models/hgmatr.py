
# -*- coding: utf-8 -*-
"""HGMAT-R 主网络：Heterogeneous Graph Memory-augmented Autoregressive Transformer for Recovery."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from hgmatr_recovery.data.loader import DataBundle
from hgmatr_recovery.models.hgt import GlobalAttentionPooling, HGTEncoder

@dataclass
class FeatureScale:
    ds: float = 4.0
    fn: float = 1.0
    nru: float = 2.0
    ac: float = 1.0
    er: float = 230.0

@dataclass
class HGMATRConfig:
    hgt_hidden: int = 64
    hgt_out: int = 64
    hgt_heads: int = 4
    hgt_layers: int = 2
    hgt_dropout: float = 0.1
    hgt_attn_dropout: float = 0.0
    hgt_use_norm: bool = True
    hgt_use_residual: bool = True
    hgt_ffn_multiplier: float = 2.0
    hgt_activation: str = 'gelu'
    hgt_qkv_bias: bool = True
    hgt_input_bias: bool = True
    hgt_output_bias: bool = True
    hgt_use_relation_pri: bool = True
    hgt_readout: str = 'attention'
    hgt_gate_hidden: int = 64
    memory_dim: int = 128
    action_embed_dim: int = 32
    agent_type_embed_dim: int = 24
    model_dim: int = 128
    decoder_heads: int = 4
    decoder_layers: int = 2
    decoder_ffn_dim: int = 256
    decoder_dropout: float = 0.1
    decoder_activation: str = 'gelu'
    decoder_norm_first: bool = False
    policy_hidden: int = 128
    value_hidden: int = 128
    aux_hidden: int = 128
    time_scale: float = 100.0

def normalize_node_features(x: torch.Tensor, scale: FeatureScale) -> torch.Tensor:
    s = torch.tensor([scale.ds, scale.fn, scale.nru, scale.ac, scale.er], dtype=x.dtype, device=x.device)
    return x / s

def _ensure_action_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    mask = mask.clone()
    invalid_rows = mask.sum(dim=-1) <= 0.0
    if torch.any(invalid_rows):
        mask[invalid_rows, 0] = 1.0
    return mask.squeeze(0) if squeeze else mask

class HGMATRNet(nn.Module):
    def __init__(self, bundle: DataBundle, relations: Sequence[str], agent_types: Sequence[str], action_dims: Sequence[int], cfg: HGMATRConfig, feature_scale: Optional[FeatureScale] = None):
        super().__init__()
        self.bundle = bundle
        self.relations = list(relations)
        self.agent_types = list(agent_types)
        self.action_dims = [int(x) for x in action_dims]
        self.cfg = cfg
        self.num_agents = len(self.action_dims)
        er_scale = max(1.0, max(float(n.expected_repair_time) for n in bundle.nodes))
        self.feature_scale = feature_scale or FeatureScale(er=er_scale)
        node_types: List[str] = []
        for n in bundle.nodes:
            if n.node_type not in node_types:
                node_types.append(n.node_type)
        self.node_types = node_types
        self.type_to_id = {t: i for i, t in enumerate(self.node_types)}
        node_type_ids = torch.tensor([self.type_to_id[n.node_type] for n in bundle.nodes], dtype=torch.long)
        self.register_buffer('node_type_ids', node_type_ids, persistent=True)
        self.graph_encoder = HGTEncoder(
            in_dim=5, hidden_dim=int(cfg.hgt_hidden), out_dim=int(cfg.hgt_out),
            node_types=self.node_types, relations=self.relations,
            num_heads=int(cfg.hgt_heads), num_layers=int(cfg.hgt_layers),
            dropout=float(cfg.hgt_dropout), attn_dropout=float(cfg.hgt_attn_dropout),
            use_norm=bool(cfg.hgt_use_norm), use_residual=bool(cfg.hgt_use_residual),
            ffn_multiplier=float(cfg.hgt_ffn_multiplier), activation=str(cfg.hgt_activation),
            input_bias=bool(cfg.hgt_input_bias), output_bias=bool(cfg.hgt_output_bias),
            qkv_bias=bool(cfg.hgt_qkv_bias), use_relation_pri=bool(cfg.hgt_use_relation_pri),
        )
        self.graph_readout = GlobalAttentionPooling(in_dim=int(cfg.hgt_out), mode=str(cfg.hgt_readout), gate_hidden=int(cfg.hgt_gate_hidden), dropout=float(cfg.hgt_dropout))
        self.graph_dim = int(cfg.hgt_out)
        uniq_agent_types: List[str] = []
        for t in self.agent_types:
            if t not in uniq_agent_types:
                uniq_agent_types.append(t)
        self.uniq_agent_types = uniq_agent_types
        self.agent_type_to_id = {t: i for i, t in enumerate(self.uniq_agent_types)}
        agent_type_ids = torch.tensor([self.agent_type_to_id[t] for t in self.agent_types], dtype=torch.long)
        self.register_buffer('agent_type_ids', agent_type_ids, persistent=True)
        self.agent_type_embed = nn.Embedding(len(self.uniq_agent_types), int(cfg.agent_type_embed_dim))
        self.agent_pos_embed = nn.Embedding(self.num_agents, int(cfg.model_dim))
        self.register_buffer('agent_pos_ids', torch.arange(self.num_agents, dtype=torch.long), persistent=True)
        self.action_embeddings = nn.ModuleList([nn.Embedding(dim, int(cfg.action_embed_dim)) for dim in self.action_dims])
        self.prev_action_proj = nn.Sequential(nn.Linear(int(cfg.action_embed_dim), int(cfg.action_embed_dim)), nn.GELU())
        self.token_ctx_proj = nn.Linear(int(cfg.action_embed_dim), int(cfg.model_dim))
        token_in_dim = self.graph_dim + int(cfg.memory_dim) + int(cfg.agent_type_embed_dim) + 1
        self.token_proj = nn.Sequential(nn.Linear(token_in_dim, int(cfg.model_dim)), nn.GELU(), nn.Linear(int(cfg.model_dim), int(cfg.model_dim)))
        mem_input_dim = self.graph_dim + int(cfg.action_embed_dim) + 2
        self.memory_cell = nn.GRUCell(input_size=mem_input_dim, hidden_size=int(cfg.memory_dim))
        self.memory_ln = nn.LayerNorm(int(cfg.memory_dim))
        act_name = str(cfg.decoder_activation).lower()
        if act_name not in {'relu', 'gelu'}:
            act_name = 'gelu'
        dec_layer = nn.TransformerEncoderLayer(
            d_model=int(cfg.model_dim), nhead=int(cfg.decoder_heads), dim_feedforward=int(cfg.decoder_ffn_dim),
            dropout=float(cfg.decoder_dropout), activation=act_name, batch_first=True, norm_first=bool(cfg.decoder_norm_first),
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=int(cfg.decoder_layers))
        self.decoder_ln = nn.LayerNorm(int(cfg.model_dim))
        self._causal_mask_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self.policy_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(int(cfg.model_dim)), nn.Linear(int(cfg.model_dim), int(cfg.policy_hidden)), nn.GELU(), nn.Linear(int(cfg.policy_hidden), dim))
            for dim in self.action_dims
        ])
        value_in_dim = self.graph_dim + int(cfg.memory_dim)
        self.value_trunk = nn.Sequential(nn.Linear(value_in_dim, int(cfg.value_hidden)), nn.GELU(), nn.Linear(int(cfg.value_hidden), int(cfg.value_hidden)), nn.GELU())
        self.value_head_res = nn.Linear(int(cfg.value_hidden), 1)
        self.value_head_t80 = nn.Linear(int(cfg.value_hidden), 1)
        self.aux_trunk = nn.Sequential(nn.Linear(value_in_dim, int(cfg.aux_hidden)), nn.GELU(), nn.Linear(int(cfg.aux_hidden), int(cfg.aux_hidden)), nn.GELU())
        self.aux_next_fn = nn.Linear(int(cfg.aux_hidden), 1)
        self.aux_next_ac = nn.Linear(int(cfg.aux_hidden), 1)
        self.aux_completion = nn.Linear(int(cfg.aux_hidden), 1)
    def init_memory(self, batch_size: int, device: Optional[torch.device | str] = None) -> torch.Tensor:
        dev = device if device is not None else next(self.parameters()).device
        return torch.zeros(int(batch_size), int(self.cfg.memory_dim), dtype=torch.float32, device=dev)
    def _ensure_batch_graph(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        if x.dim() != 3:
            raise ValueError('node_features 必须是 (N,F) 或 (B,N,F)')
        return x, squeeze
    def _ensure_batch_scalar(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if x.dim() == 2 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.dim() != 1:
            raise ValueError('标量张量必须是 (B,) / (B,1) / ()')
        return x
    def _encode_graph(self, node_features: torch.Tensor, edges: Dict[str, torch.Tensor]) -> torch.Tensor:
        x, _ = self._ensure_batch_graph(node_features)
        x = normalize_node_features(x, self.feature_scale)
        h = self.graph_encoder(x, edges, self.node_type_ids, self.node_types)
        g = self.graph_readout(h)
        if g.dim() == 1:
            g = g.unsqueeze(0)
        return g
    def _encode_graph_sequence(self, node_features: torch.Tensor, edges: Dict[str, torch.Tensor]) -> torch.Tensor:
        if node_features.dim() != 4:
            raise ValueError('node_features must be (T,B,N,F)')
        T, B, N, F = node_features.shape
        flat = node_features.reshape(T * B, N, F)
        flat_g = self._encode_graph(flat, edges)
        return flat_g.reshape(T, B, -1)
    def _embed_joint_prev_actions(self, prev_actions: torch.Tensor) -> torch.Tensor:
        if prev_actions.dim() == 1:
            prev_actions = prev_actions.unsqueeze(0)
        embs = [emb(prev_actions[:, ai].long()) for ai, emb in enumerate(self.action_embeddings)]
        joint = torch.stack(embs, dim=1).mean(dim=1)
        return self.prev_action_proj(joint)
    def _prefix_action_context(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        B, A = actions.shape
        if A != self.num_agents:
            raise ValueError('actions second dim must equal num_agents')
        ctxs: List[torch.Tensor] = []
        prefix = torch.zeros(B, int(self.cfg.action_embed_dim), dtype=torch.float32, device=actions.device)
        for ai, emb in enumerate(self.action_embeddings):
            ctxs.append(prefix)
            prefix = prefix + emb(actions[:, ai].long())
        return torch.stack(ctxs, dim=1)
    def _build_memory(self, graph_emb: torch.Tensor, prev_memory: torch.Tensor, prev_actions: torch.Tensor, prev_reward: torch.Tensor, times: torch.Tensor, episode_start: Optional[torch.Tensor] = None) -> torch.Tensor:
        if episode_start is not None:
            if episode_start.dim() == 2 and episode_start.shape[-1] == 1:
                episode_start = episode_start.squeeze(-1)
            reset = (1.0 - episode_start.float()).unsqueeze(-1)
            prev_memory = prev_memory * reset
        prev_joint = self._embed_joint_prev_actions(prev_actions)
        prev_reward = self._ensure_batch_scalar(prev_reward).to(graph_emb.device).unsqueeze(-1)
        times = self._ensure_batch_scalar(times).to(graph_emb.device).unsqueeze(-1) / max(1.0, float(self.cfg.time_scale))
        mem_input = torch.cat([graph_emb, prev_joint, prev_reward, times], dim=-1)
        new_mem = self.memory_cell(mem_input, prev_memory)
        return self.memory_ln(new_mem)
    def _build_base_tokens(self, graph_emb: torch.Tensor, memory: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        B = graph_emb.shape[0]
        times = self._ensure_batch_scalar(times).to(graph_emb.device).unsqueeze(-1) / max(1.0, float(self.cfg.time_scale))
        type_emb = self.agent_type_embed(self.agent_type_ids.to(graph_emb.device)).unsqueeze(0).expand(B, -1, -1)
        g_rep = graph_emb.unsqueeze(1).expand(B, self.num_agents, -1)
        m_rep = memory.unsqueeze(1).expand(B, self.num_agents, -1)
        t_rep = times.unsqueeze(1).expand(B, self.num_agents, 1)
        token_in = torch.cat([g_rep, m_rep, type_emb, t_rep], dim=-1)
        tokens = self.token_proj(token_in)
        return tokens + self.agent_pos_embed(self.agent_pos_ids.to(graph_emb.device)).unsqueeze(0)
    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (str(device), int(seq_len))
        cached = self._causal_mask_cache.get(key)
        if cached is None:
            cached = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
            self._causal_mask_cache[key] = cached
        return cached
    def _value_aux_from_state(self, graph_emb: torch.Tensor, memory: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.cat([graph_emb, memory], dim=-1)
        v = self.value_trunk(x)
        aux = self.aux_trunk(x)
        return {'value_res': self.value_head_res(v).squeeze(-1), 'value_t80': self.value_head_t80(v).squeeze(-1), 'pred_next_fn': self.aux_next_fn(aux).squeeze(-1), 'pred_next_ac': self.aux_next_ac(aux).squeeze(-1), 'pred_completion_logit': self.aux_completion(aux).squeeze(-1)}
    def _decode_with_teacher(self, base_tokens: torch.Tensor, actions: torch.Tensor, action_masks: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        prefix_ctx = self._prefix_action_context(actions).to(base_tokens.device)
        tokens = base_tokens + self.token_ctx_proj(prefix_ctx)
        out = self.decoder(tokens, mask=self._causal_mask(self.num_agents, device=base_tokens.device))
        out = self.decoder_ln(out)
        logps, entropies, logits_list = [], [], []
        for ai, head in enumerate(self.policy_heads):
            logits = head(out[:, ai, :])
            mask_ai = _ensure_action_mask(action_masks[ai].to(logits.device))
            logits = logits.masked_fill(mask_ai <= 0.0, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            a = actions[:, ai].long()
            logps.append(dist.log_prob(a))
            entropies.append(dist.entropy())
            logits_list.append(logits)
        stacked_logps = torch.stack(logps, dim=-1)
        return {'logits': logits_list, 'logps': stacked_logps, 'joint_logp': stacked_logps.sum(dim=-1), 'entropy': torch.stack(entropies, dim=-1).mean(dim=-1)}
    @torch.no_grad()
    def act(self, node_features: torch.Tensor, edges: Dict[str, torch.Tensor], action_masks: Sequence[torch.Tensor], prev_memory: torch.Tensor, prev_actions: torch.Tensor, prev_reward: torch.Tensor, times: torch.Tensor, episode_start: Optional[torch.Tensor] = None, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        was_single = node_features.dim() == 2
        x, _ = self._ensure_batch_graph(node_features)
        B = x.shape[0]
        if prev_memory.dim() == 1:
            prev_memory = prev_memory.unsqueeze(0)
        if prev_actions.dim() == 1:
            prev_actions = prev_actions.unsqueeze(0)
        prev_reward = self._ensure_batch_scalar(prev_reward)
        times = self._ensure_batch_scalar(times)
        if episode_start is not None:
            episode_start = self._ensure_batch_scalar(episode_start)
        graph_emb = self._encode_graph(x, edges)
        memory = self._build_memory(graph_emb, prev_memory, prev_actions, prev_reward, times, episode_start)
        base_tokens = self._build_base_tokens(graph_emb, memory, times)
        sampled_actions, sampled_logps, sampled_entropy, prefix_action_embs = [], [], [], []
        for ai in range(self.num_agents):
            ctxs: List[torch.Tensor] = []
            prefix = torch.zeros(B, int(self.cfg.action_embed_dim), dtype=torch.float32, device=graph_emb.device)
            for j in range(ai + 1):
                ctxs.append(prefix)
                if j < ai:
                    prefix = prefix + prefix_action_embs[j]
            prefix_ctx = torch.stack(ctxs, dim=1)
            tokens = base_tokens[:, : ai + 1, :] + self.token_ctx_proj(prefix_ctx)
            out = self.decoder(tokens, mask=self._causal_mask(ai + 1, device=graph_emb.device))
            out = self.decoder_ln(out)
            logits = self.policy_heads[ai](out[:, -1, :])
            mask_ai = action_masks[ai]
            if mask_ai.dim() == 1:
                mask_ai = mask_ai.unsqueeze(0)
            mask_ai = _ensure_action_mask(mask_ai.to(logits.device))
            logits = logits.masked_fill(mask_ai <= 0.0, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            sampled_actions.append(action)
            sampled_logps.append(dist.log_prob(action))
            sampled_entropy.append(dist.entropy())
            prefix_action_embs.append(self.action_embeddings[ai](action))
        heads = self._value_aux_from_state(graph_emb, memory)
        actions = torch.stack(sampled_actions, dim=-1)
        logps = torch.stack(sampled_logps, dim=-1)
        entropy = torch.stack(sampled_entropy, dim=-1).mean(dim=-1)
        out: Dict[str, torch.Tensor] = {'actions': actions.squeeze(0) if was_single else actions, 'logps': logps.squeeze(0) if was_single else logps, 'joint_logp': logps.sum(dim=-1).squeeze(0) if was_single else logps.sum(dim=-1), 'entropy': entropy.squeeze(0) if was_single else entropy, 'memory': memory.squeeze(0) if was_single else memory}
        for k, v in heads.items():
            out[k] = v.squeeze(0) if was_single else v
        return out
    def evaluate_sequence(self, node_features: torch.Tensor, edges: Dict[str, torch.Tensor], times: torch.Tensor, prev_actions: torch.Tensor, prev_reward: torch.Tensor, episode_starts: torch.Tensor, actions: torch.Tensor, action_masks: Sequence[torch.Tensor], initial_memory: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if node_features.dim() != 4:
            raise ValueError('node_features must be (T,B,N,F)')
        T, B, _, _ = node_features.shape
        graph_seq = self._encode_graph_sequence(node_features, edges)
        hidden = initial_memory if initial_memory is not None else self.init_memory(B, device=node_features.device)
        joint_logps, logps_all, entropies = [], [], []
        values_res, values_t80 = [], []
        pred_next_fn, pred_next_ac, pred_completion = [], [], []
        memories = []
        masks_by_agent = list(action_masks)
        for t in range(T):
            hidden = self._build_memory(graph_seq[t], hidden, prev_actions[t], prev_reward[t], times[t], episode_start=episode_starts[t])
            memories.append(hidden)
            base_tokens = self._build_base_tokens(graph_seq[t], hidden, times[t])
            step_masks = [masks_by_agent[ai][t] for ai in range(self.num_agents)]
            dec = self._decode_with_teacher(base_tokens, actions[t], step_masks)
            aux = self._value_aux_from_state(graph_seq[t], hidden)
            joint_logps.append(dec['joint_logp'])
            logps_all.append(dec['logps'])
            entropies.append(dec['entropy'])
            values_res.append(aux['value_res'])
            values_t80.append(aux['value_t80'])
            pred_next_fn.append(aux['pred_next_fn'])
            pred_next_ac.append(aux['pred_next_ac'])
            pred_completion.append(aux['pred_completion_logit'])
        return {'joint_logp': torch.stack(joint_logps, dim=0), 'logps': torch.stack(logps_all, dim=0), 'entropy': torch.stack(entropies, dim=0), 'value_res': torch.stack(values_res, dim=0), 'value_t80': torch.stack(values_t80, dim=0), 'pred_next_fn': torch.stack(pred_next_fn, dim=0), 'pred_next_ac': torch.stack(pred_next_ac, dim=0), 'pred_completion_logit': torch.stack(pred_completion, dim=0), 'memory': torch.stack(memories, dim=0)}
    @torch.no_grad()
    def bootstrap_values(self, node_features: torch.Tensor, edges: Dict[str, torch.Tensor], times: torch.Tensor, prev_actions: torch.Tensor, prev_reward: torch.Tensor, prev_memory: torch.Tensor, episode_start: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x, _ = self._ensure_batch_graph(node_features)
        if prev_memory.dim() == 1:
            prev_memory = prev_memory.unsqueeze(0)
        if prev_actions.dim() == 1:
            prev_actions = prev_actions.unsqueeze(0)
        graph_emb = self._encode_graph(x, edges)
        memory = self._build_memory(graph_emb, prev_memory, prev_actions, prev_reward, times, episode_start)
        heads = self._value_aux_from_state(graph_emb, memory)
        heads['memory'] = memory
        return heads
