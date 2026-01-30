# -*- coding: utf-8 -*-
"""GNN-FiLM（论文 Eq.(18)(19)）。

论文中使用异构图（多种 edge type），在每一层卷积中：

h_{v}^{(t+1)} = \sum_{k \in R} \sum_{j \in N(v)} ReLU( \omega_{k,v} \odot W_k h_{j}^{(t)} + \mu_{k,v})

其中 (\omega_{k,v}, \mu_{k,v}) = g(h_v^{(t)})。

本实现不依赖 torch_geometric，而是直接用 edge_index（2,E）做消息传递。
同时支持：
- 单图：x 形状 (N,in_dim)
- 批量：x 形状 (B,N,in_dim)（同一拓扑的多份状态）
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMHeteroConv(nn.Module):
    """单层 GNN-FiLM，支持多 relation（edge type）。"""

    def __init__(self, in_dim: int, out_dim: int, relations: Iterable[str]):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.relations: List[str] = list(relations)

        # 每种关系一个线性变换 W_k
        self.w_map = nn.ModuleDict()
        self.g_map = nn.ModuleDict()
        for r in self.relations:
            self.w_map[r] = nn.Linear(self.in_dim, self.out_dim, bias=False)
            # g(h_v) -> (omega, mu)
            self.g_map[r] = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)

    def forward(self, x: torch.Tensor, edges_by_type: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播。

        参数
        ----
        x: (N,in_dim) 或 (B,N,in_dim)
        edges_by_type: dict[etype] = edge_index (2,E)

        返回
        ----
        out: (N,out_dim) 或 (B,N,out_dim)
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        if x.dim() != 3:
            raise ValueError("x 必须是 (N,F) 或 (B,N,F)")

        B, N, _ = x.shape
        device = x.device
        out = torch.zeros((B, N, self.out_dim), dtype=x.dtype, device=device)

        for r in self.relations:
            if r not in edges_by_type:
                continue
            edge_index = edges_by_type[r]
            if edge_index.numel() == 0:
                continue
            src = edge_index[0].long()
            dst = edge_index[1].long()

            # W_k h_j
            x_src = x[:, src, :]  # (B,E,F)
            msg_lin = self.w_map[r](x_src)  # (B,E,out)

            # omega/mu 由“目标节点”特征生成（对应论文 omega_{k,v}, mu_{k,v}）
            x_dst = x[:, dst, :]  # (B,E,F)
            film = self.g_map[r](x_dst)  # (B,E,2*out)
            omega, mu = film.chunk(2, dim=-1)
            omega = torch.sigmoid(omega)  # 稳定性更好（论文未限定，可调）

            msg = F.relu(omega * msg_lin + mu)  # (B,E,out)

            # sum 聚合到 dst：scatter_add_ 沿 dim=1 累加
            index = dst.view(1, -1, 1).expand(B, -1, self.out_dim)
            out.scatter_add_(dim=1, index=index, src=msg)

        if squeeze:
            return out.squeeze(0)
        return out


class GNNFiLMEncoder(nn.Module):
    """两层 GNN-FiLM 编码器（论文案例：5->32->1）。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        relations: Iterable[str],
    ):
        super().__init__()
        rels = list(relations)
        self.conv1 = FiLMHeteroConv(in_dim=in_dim, out_dim=hidden_dim, relations=rels)
        self.conv2 = FiLMHeteroConv(in_dim=hidden_dim, out_dim=out_dim, relations=rels)

    def forward(self, x: torch.Tensor, edges_by_type: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.conv1(x, edges_by_type)
        x = self.conv2(x, edges_by_type)
        return x
