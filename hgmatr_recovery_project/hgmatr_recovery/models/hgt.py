# -*- coding: utf-8 -*-
"""Heterogeneous Graph Transformer (HGT) 编码器。

特点：
- 不依赖 torch_geometric，直接使用 edge_index=(2,E) 做消息传递。
- 支持单图 (N,F) 与批量图 (B,N,F) 输入。
- 支持多节点类型、多关系类型。
- 适合作为当前异构基础设施恢复图状态的替代编码器。

建议搭配：
- Actor/Critic 前端编码器
- 全局 attention pooling 读出图级向量
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TypedLinear(nn.Module):
    """按节点类型应用不同线性层。"""

    def __init__(self, in_dim: int, out_dim: int, node_types: Iterable[str], bias: bool = True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.node_types: List[str] = list(node_types)
        self.layers = nn.ModuleDict({t: nn.Linear(self.in_dim, self.out_dim, bias=bias) for t in self.node_types})

    def forward(self, x: torch.Tensor, node_type_ids: torch.Tensor, id_to_type: List[str]) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        if x.dim() != 3:
            raise ValueError("x 必须是 (N,F) 或 (B,N,F)")

        if node_type_ids.dim() != 1:
            raise ValueError("node_type_ids 必须是 (N,)")
        if x.shape[1] != node_type_ids.shape[0]:
            raise ValueError(f"节点数不匹配: x.shape[1]={x.shape[1]} vs node_type_ids={node_type_ids.shape[0]}")

        out = torch.zeros(x.shape[0], x.shape[1], self.out_dim, dtype=x.dtype, device=x.device)
        for type_id, type_name in enumerate(id_to_type):
            mask = (node_type_ids == type_id)
            if not torch.any(mask):
                continue
            out[:, mask, :] = self.layers[type_name](x[:, mask, :])

        return out.squeeze(0) if squeeze else out


class RelationMessageTransform(nn.Module):
    """关系特异的 head-wise 线性变换。"""

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.weight = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.head_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,E,H,D)
        return torch.einsum("behd,hdf->behf", x, self.weight)


class HGTLayer(nn.Module):
    """单层 HGT。

    结构：
    1. type-specific Q/K/V 投影
    2. relation-specific attention/message transform
    3. 按 dst 节点聚合消息
    4. residual + norm + FFN
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        node_types: Iterable[str],
        relations: Iterable[str],
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        use_norm: bool = True,
        use_residual: bool = True,
        ffn_multiplier: float = 2.0,
        activation: str = "gelu",
        qkv_bias: bool = True,
        out_bias: bool = True,
        use_relation_pri: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} 必须能被 num_heads={num_heads} 整除")
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.node_types: List[str] = list(node_types)
        self.relations: List[str] = list(relations)
        self.use_norm = bool(use_norm)
        self.use_residual = bool(use_residual)
        self.use_relation_pri = bool(use_relation_pri)

        self.q_proj = TypedLinear(self.hidden_dim, self.hidden_dim, self.node_types, bias=qkv_bias)
        self.k_proj = TypedLinear(self.hidden_dim, self.hidden_dim, self.node_types, bias=qkv_bias)
        self.v_proj = TypedLinear(self.hidden_dim, self.hidden_dim, self.node_types, bias=qkv_bias)
        self.out_proj = TypedLinear(self.hidden_dim, self.hidden_dim, self.node_types, bias=out_bias)

        self.rel_att = nn.ModuleDict({r: RelationMessageTransform(self.num_heads, self.head_dim) for r in self.relations})
        self.rel_msg = nn.ModuleDict({r: RelationMessageTransform(self.num_heads, self.head_dim) for r in self.relations})
        if self.use_relation_pri:
            self.rel_pri = nn.ParameterDict({r: nn.Parameter(torch.ones(self.num_heads)) for r in self.relations})
        else:
            self.rel_pri = None

        self.dropout = nn.Dropout(float(dropout))
        self.attn_dropout = nn.Dropout(float(attn_dropout))

        ffn_hidden = max(self.hidden_dim, int(round(self.hidden_dim * float(ffn_multiplier))))
        act_layer: nn.Module
        if activation.lower() == "relu":
            act_layer = nn.ReLU()
        elif activation.lower() == "elu":
            act_layer = nn.ELU()
        else:
            act_layer = nn.GELU()

        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, ffn_hidden),
            act_layer,
            nn.Dropout(float(dropout)),
            nn.Linear(ffn_hidden, self.hidden_dim),
        )

        self.norm1 = nn.LayerNorm(self.hidden_dim) if self.use_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(self.hidden_dim) if self.use_norm else nn.Identity()

    def _edge_softmax(self, scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """对每个目标节点的入边做 softmax（向量化实现）。

        scores: (B,E,H)
        dst: (E,)
        return: (B,E,H)
        """
        B, E, H = scores.shape
        if E == 0:
            return scores

        # 变形为 (B*H, E)，每个 (batch, head) 独立做一次按 dst 分组的 softmax
        scores_bhe = scores.permute(0, 2, 1).reshape(B * H, E)
        dst_index = dst.view(1, E).expand(B * H, E)

        max_per_dst = torch.full(
            (B * H, num_nodes),
            fill_value=torch.finfo(scores.dtype).min,
            dtype=scores.dtype,
            device=scores.device,
        )
        max_per_dst.scatter_reduce_(1, dst_index, scores_bhe, reduce="amax", include_self=True)
        centered = scores_bhe - max_per_dst.gather(1, dst_index)

        exp_scores = centered.exp()
        denom = torch.zeros((B * H, num_nodes), dtype=scores.dtype, device=scores.device)
        denom.scatter_add_(1, dst_index, exp_scores)

        alpha = exp_scores / (denom.gather(1, dst_index) + 1e-12)
        return alpha.reshape(B, H, E).permute(0, 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        edges_by_type: Dict[str, torch.Tensor],
        node_type_ids: torch.Tensor,
        id_to_type: List[str],
    ) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        if x.dim() != 3:
            raise ValueError("x 必须是 (N,D) 或 (B,N,D)")

        B, N, D = x.shape
        if D != self.hidden_dim:
            raise ValueError(f"x 最后一维应为 hidden_dim={self.hidden_dim}，当前为 {D}")

        q = self.q_proj(x, node_type_ids=node_type_ids, id_to_type=id_to_type).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x, node_type_ids=node_type_ids, id_to_type=id_to_type).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x, node_type_ids=node_type_ids, id_to_type=id_to_type).view(B, N, self.num_heads, self.head_dim)

        agg = torch.zeros((B, N, self.num_heads, self.head_dim), dtype=x.dtype, device=x.device)

        for r in self.relations:
            edge_index = edges_by_type.get(r)
            if edge_index is None or edge_index.numel() == 0:
                continue
            src = edge_index[0].long()
            dst = edge_index[1].long()
            if src.numel() == 0:
                continue

            q_dst = q[:, dst, :, :]  # (B,E,H,Dh)
            k_src = k[:, src, :, :]
            v_src = v[:, src, :, :]

            k_rel = self.rel_att[r](k_src)
            v_rel = self.rel_msg[r](v_src)

            scores = (q_dst * k_rel).sum(dim=-1) * self.scale  # (B,E,H)
            if self.use_relation_pri and self.rel_pri is not None:
                scores = scores * self.rel_pri[r].view(1, 1, self.num_heads)

            alpha = self._edge_softmax(scores, dst=dst, num_nodes=N)
            alpha = self.attn_dropout(alpha)
            msg = alpha.unsqueeze(-1) * v_rel  # (B,E,H,Dh)

            index = dst.view(1, -1, 1, 1).expand(B, -1, self.num_heads, self.head_dim)
            agg.scatter_add_(dim=1, index=index, src=msg)

        out = agg.reshape(B, N, self.hidden_dim)
        out = self.out_proj(out, node_type_ids=node_type_ids, id_to_type=id_to_type)
        out = self.dropout(out)

        if self.use_residual:
            x = x + out
        else:
            x = out
        x = self.norm1(x)

        ff = self.ffn(x)
        ff = self.dropout(ff)
        if self.use_residual:
            x = x + ff
        else:
            x = ff
        x = self.norm2(x)

        return x.squeeze(0) if squeeze else x


class HGTEncoder(nn.Module):
    """多层 HGT 编码器。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        node_types: Iterable[str],
        relations: Iterable[str],
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        use_norm: bool = True,
        use_residual: bool = True,
        ffn_multiplier: float = 2.0,
        activation: str = "gelu",
        input_bias: bool = True,
        output_bias: bool = True,
        qkv_bias: bool = True,
        use_relation_pri: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.node_types: List[str] = list(node_types)
        self.relations: List[str] = list(relations)

        self.input_proj = TypedLinear(self.in_dim, self.hidden_dim, self.node_types, bias=input_bias)
        self.layers = nn.ModuleList([
            HGTLayer(
                hidden_dim=self.hidden_dim,
                num_heads=num_heads,
                node_types=self.node_types,
                relations=self.relations,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_norm=use_norm,
                use_residual=use_residual,
                ffn_multiplier=ffn_multiplier,
                activation=activation,
                qkv_bias=qkv_bias,
                out_bias=output_bias,
                use_relation_pri=use_relation_pri,
            )
            for _ in range(int(num_layers))
        ])
        self.output_proj = TypedLinear(self.hidden_dim, self.out_dim, self.node_types, bias=output_bias)
        self.output_norm = nn.LayerNorm(self.out_dim) if use_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edges_by_type: Dict[str, torch.Tensor],
        node_type_ids: torch.Tensor,
        id_to_type: List[str],
    ) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        if x.dim() != 3:
            raise ValueError("x 必须是 (N,F) 或 (B,N,F)")

        h = self.input_proj(x, node_type_ids=node_type_ids, id_to_type=id_to_type)
        for layer in self.layers:
            h = layer(h, edges_by_type=edges_by_type, node_type_ids=node_type_ids, id_to_type=id_to_type)
        h = self.output_proj(h, node_type_ids=node_type_ids, id_to_type=id_to_type)
        h = self.output_norm(h)
        return h.squeeze(0) if squeeze else h


class GlobalAttentionPooling(nn.Module):
    """图级读出层。

    mode:
    - attention: 可学习门控注意力池化
    - mean: 平均池化
    - sum: 求和池化
    - max: 最大池化
    """

    def __init__(self, in_dim: int, mode: str = "attention", gate_hidden: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.mode = str(mode).lower()
        self.dropout = nn.Dropout(float(dropout))

        gate_hidden = int(gate_hidden) if gate_hidden is not None else max(32, self.in_dim)
        if self.mode == "attention":
            self.gate = nn.Sequential(
                nn.Linear(self.in_dim, gate_hidden),
                nn.Tanh(),
                nn.Linear(gate_hidden, 1),
            )
        elif self.mode in {"mean", "sum", "max"}:
            self.gate = None
        else:
            raise ValueError(f"不支持的 readout mode: {mode}")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if h.dim() == 2:
            h = h.unsqueeze(0)
            squeeze = True
        if h.dim() != 3:
            raise ValueError("h 必须是 (N,D) 或 (B,N,D)")

        h = self.dropout(h)
        if self.mode == "attention":
            score = self.gate(h).squeeze(-1)  # (B,N)
            alpha = F.softmax(score, dim=1).unsqueeze(-1)
            g = (alpha * h).sum(dim=1)
        elif self.mode == "mean":
            g = h.mean(dim=1)
        elif self.mode == "sum":
            g = h.sum(dim=1)
        else:  # max
            g = h.max(dim=1).values

        return g.squeeze(0) if squeeze else g
