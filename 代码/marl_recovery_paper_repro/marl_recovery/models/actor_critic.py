# -*- coding: utf-8 -*-
"""Actor-Critic 网络。

论文结构（Fig.6）：
- 输入：图状态（节点特征 + 异构边）
- 图编码：两层 GNN-FiLM（论文 Eq.(18)(19)）
- 展平为向量
- MLP：输出 actor 的 action logits 或 critic 的 V(s)

实现要点：
- 支持动作掩码（论文 2.4.2 的 invalid action mask）
- 网络默认参数对齐论文案例（Table 2 / 文中描述）：GNN 5->32->1，MLP 5 层 128。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from marl_recovery.data.loader import DataBundle
from marl_recovery.models.gnn_film import GNNFiLMEncoder


class MLP(nn.Module):
    """简单的 MLP。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_hidden_layers: int = 5):
        super().__init__()
        layers: List[nn.Module] = []
        d = int(in_dim)
        for _ in range(int(num_hidden_layers)):
            layers.append(nn.Linear(d, int(hidden_dim)))
            layers.append(nn.ReLU())
            d = int(hidden_dim)
        layers.append(nn.Linear(d, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class FeatureScale:
    """输入特征归一化系数。"""

    ds: float = 4.0
    fn: float = 1.0
    nru: float = 2.0
    ac: float = 1.0
    er: float = 230.0  # 论文表 1 中桥梁 complete 的均值可达 230


def normalize_node_features(x: torch.Tensor, scale: FeatureScale) -> torch.Tensor:
    """对节点特征做简单尺度归一化，以便训练更稳定。

    x: (N,5) 或 (B,N,5)
    """
    s = torch.tensor([scale.ds, scale.fn, scale.nru, scale.ac, scale.er], dtype=x.dtype, device=x.device)
    return x / s


class ActorNet(nn.Module):
    """单个 agent 的 actor 网络：π(a|s)。"""

    def __init__(
        self,
        bundle: DataBundle,
        relations: List[str],
        action_dim: int,
        gnn_hidden: int = 32,
        gnn_out: int = 1,
        mlp_hidden: int = 128,
        mlp_layers: int = 5,
        feature_scale: Optional[FeatureScale] = None,
    ):
        super().__init__()
        self.bundle = bundle
        self.relations = relations
        self.action_dim = int(action_dim)
        self.feature_scale = feature_scale or FeatureScale(er=float(max(1.0, np.max([n.expected_repair_time for n in bundle.nodes]))))

        self.encoder = GNNFiLMEncoder(in_dim=5, hidden_dim=gnn_hidden, out_dim=gnn_out, relations=self.relations)
        vec_dim = bundle.num_nodes * gnn_out
        self.mlp = MLP(in_dim=vec_dim, hidden_dim=mlp_hidden, out_dim=self.action_dim, num_hidden_layers=mlp_layers)

    def forward(self, node_features: torch.Tensor, edges: Dict[str, torch.Tensor], action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """输出 logits。

        node_features: (N,5) 或 (B,N,5)
        action_mask: (A) 或 (B,A) 的 0/1 mask
        """
        x = normalize_node_features(node_features, self.feature_scale)
        h = self.encoder(x, edges)  # (N,1) 或 (B,N,1)

        if h.dim() == 2:
            v = h.reshape(1, -1)
        else:
            v = h.reshape(h.shape[0], -1)

        logits = self.mlp(v)  # (B,A)

        # mask: 将非法动作的 logits 设为极小
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(action_mask <= 0.0, -1e9)

        return logits

    @torch.no_grad()
    def sample_action(self, node_features: torch.Tensor, edges: Dict[str, torch.Tensor], action_mask: torch.Tensor) -> Tuple[int, float]:
        """采样动作并返回 log_prob。"""
        logits = self.forward(node_features, edges, action_mask=action_mask)  # (1,A)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), float(logp.item())

    def log_prob(self, node_features: torch.Tensor, edges: Dict[str, torch.Tensor], action_mask: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """计算给定 actions 的 log_prob。

        actions: (B,)
        返回: (B,)
        """
        logits = self.forward(node_features, edges, action_mask=action_mask)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        return dist.log_prob(actions)

    def entropy(self, node_features: torch.Tensor, edges: Dict[str, torch.Tensor], action_mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward(node_features, edges, action_mask=action_mask)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        return dist.entropy()


class CriticNet(nn.Module):
    """Critic 网络：V(s)。"""

    def __init__(
        self,
        bundle: DataBundle,
        relations: List[str],
        gnn_hidden: int = 32,
        gnn_out: int = 1,
        mlp_hidden: int = 128,
        mlp_layers: int = 5,
        feature_scale: Optional[FeatureScale] = None,
    ):
        super().__init__()
        self.bundle = bundle
        self.relations = relations
        self.feature_scale = feature_scale or FeatureScale(er=float(max(1.0, np.max([n.expected_repair_time for n in bundle.nodes]))))

        self.encoder = GNNFiLMEncoder(in_dim=5, hidden_dim=gnn_hidden, out_dim=gnn_out, relations=self.relations)
        vec_dim = bundle.num_nodes * gnn_out
        self.mlp = MLP(in_dim=vec_dim, hidden_dim=mlp_hidden, out_dim=1, num_hidden_layers=mlp_layers)

    def forward(self, node_features: torch.Tensor, edges: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = normalize_node_features(node_features, self.feature_scale)
        h = self.encoder(x, edges)
        if h.dim() == 2:
            v = h.reshape(1, -1)
        else:
            v = h.reshape(h.shape[0], -1)
        value = self.mlp(v)
        return value.squeeze(-1)  # (B,)
