# -*- coding: utf-8 -*-
"""数据加载器。

支持从一个目录读取 nodes.csv / edges.csv / buildings.csv / meta.yaml。
用于：
- 合成数据（scripts/generate_synthetic_campus.py 生成）
- 用户接入真实数据（作者数据或自建数据）

详见 data/format_spec.md
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class NodeRecord:
    """基础设施组件节点记录。"""

    node_id: str
    node_type: str
    is_main: bool
    damage_state: int
    expected_repair_time: float
    in_bridge_area: bool
    cooling_target: bool


@dataclass(frozen=True)
class BuildingRecord:
    """建筑记录，用于计算社区功能 Q(t)。"""

    building_id: str
    area: float
    alpha: float
    beta_w: float
    beta_p: float
    beta_t: float
    pipeline_node: int
    substation_node: int
    in_bridge_area: bool


@dataclass
class DataBundle:
    """数据包（节点、边、建筑、元数据）。"""

    nodes: List[NodeRecord]
    edges: Dict[str, np.ndarray]  # edge_type -> (2, E) 的索引数组
    buildings: List[BuildingRecord]
    meta: Dict[str, Any]
    node_id_to_index: Dict[str, int]
    type_to_indices: Dict[str, List[int]]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)


def _read_meta(meta_path: str) -> Dict[str, Any]:
    """读取 meta.yaml（可选）。"""
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"meta.yaml 必须是 dict，当前类型: {type(obj)}")
    return obj


def load_data_bundle(data_dir: str) -> DataBundle:
    """从 data_dir 读取数据，返回 DataBundle。"""

    nodes_path = os.path.join(data_dir, "nodes.csv")
    edges_path = os.path.join(data_dir, "edges.csv")
    buildings_path = os.path.join(data_dir, "buildings.csv")
    meta_path = os.path.join(data_dir, "meta.yaml")

    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"找不到 nodes.csv: {nodes_path}")
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"找不到 edges.csv: {edges_path}")
    if not os.path.exists(buildings_path):
        raise FileNotFoundError(f"找不到 buildings.csv: {buildings_path}")

    # -------------------------
    # 1) 读取 nodes.csv
    # -------------------------
    node_df = pd.read_csv(nodes_path)
    required_cols = {"node_id", "node_type", "damage_state", "expected_repair_time"}
    missing = required_cols - set(node_df.columns)
    if missing:
        raise ValueError(f"nodes.csv 缺少列: {sorted(missing)}")

    # 可选列缺省处理
    for opt_col, default in [
        ("is_main", 0),
        ("in_bridge_area", 0),
        ("cooling_target", 0),
    ]:
        if opt_col not in node_df.columns:
            node_df[opt_col] = default

    node_df["node_id"] = node_df["node_id"].astype(str)
    node_df["node_type"] = node_df["node_type"].astype(str)

    node_id_to_index = {nid: i for i, nid in enumerate(node_df["node_id"].tolist())}

    nodes: List[NodeRecord] = []
    type_to_indices: Dict[str, List[int]] = {}

    for i, row in node_df.iterrows():
        node_type = str(row["node_type"])

        # 注意：CSV 可能有空值 NaN，因此这里需要稳健处理
        is_main = False
        if not pd.isna(row.get("is_main", 0)):
            is_main = bool(int(row.get("is_main", 0)))

        in_bridge_area = False
        if not pd.isna(row.get("in_bridge_area", 0)):
            in_bridge_area = bool(int(row.get("in_bridge_area", 0)))

        cooling_target = False
        if not pd.isna(row.get("cooling_target", 0)):
            cooling_target = bool(int(row.get("cooling_target", 0)))

        rec = NodeRecord(
            node_id=str(row["node_id"]),
            node_type=node_type,
            is_main=is_main,
            damage_state=int(row["damage_state"]),
            expected_repair_time=float(row["expected_repair_time"]),
            in_bridge_area=in_bridge_area,
            cooling_target=cooling_target,
        )
        nodes.append(rec)
        type_to_indices.setdefault(node_type, []).append(i)

    # -------------------------
    # 2) 读取 edges.csv
    # -------------------------
    edge_df = pd.read_csv(edges_path)
    required_edge_cols = {"src", "dst", "edge_type"}
    missing = required_edge_cols - set(edge_df.columns)
    if missing:
        raise ValueError(f"edges.csv 缺少列: {sorted(missing)}")

    # 边以“有向”的形式存储（无向边会添加双向）
    edges_tmp: Dict[str, List[List[int]]] = {}
    for _, row in edge_df.iterrows():
        src_id = str(row["src"])
        dst_id = str(row["dst"])
        etype = str(row["edge_type"])

        if src_id not in node_id_to_index or dst_id not in node_id_to_index:
            raise ValueError(f"edges.csv 引用了不存在的 node_id: {src_id}->{dst_id}")

        u = node_id_to_index[src_id]
        v = node_id_to_index[dst_id]

        # 默认认为 edges.csv 描述的是无向关系，因此加入双向
        edges_tmp.setdefault(etype, []).append([u, v])
        edges_tmp.setdefault(etype, []).append([v, u])

    # 自环（让 GNN 能在每层保留自身信息），单独作为一种边类型 self
    n = len(nodes)
    edges_tmp.setdefault("self", []).extend([[i, i] for i in range(n)])

    edges: Dict[str, np.ndarray] = {}
    for etype, pairs in edges_tmp.items():
        arr = np.array(pairs, dtype=np.int64).T  # (2, E)
        edges[etype] = arr

    # -------------------------
    # 3) 读取 buildings.csv
    # -------------------------
    bld_df = pd.read_csv(buildings_path)
    required_b_cols = {
        "building_id",
        "area",
        "alpha",
        "beta_w",
        "beta_p",
        "beta_t",
        "pipeline_node_id",
        "substation_node_id",
        "in_bridge_area",
    }
    missing = required_b_cols - set(bld_df.columns)
    if missing:
        raise ValueError(f"buildings.csv 缺少列: {sorted(missing)}")

    buildings: List[BuildingRecord] = []
    for _, row in bld_df.iterrows():
        pid = str(row["pipeline_node_id"])
        sid = str(row["substation_node_id"])
        if pid not in node_id_to_index:
            raise ValueError(f"buildings.csv pipeline_node_id 不存在: {pid}")
        if sid not in node_id_to_index:
            raise ValueError(f"buildings.csv substation_node_id 不存在: {sid}")

        in_bridge_area = False
        if not pd.isna(row.get("in_bridge_area", 0)):
            in_bridge_area = bool(int(row.get("in_bridge_area", 0)))

        buildings.append(
            BuildingRecord(
                building_id=str(row["building_id"]),
                area=float(row["area"]),
                alpha=float(row["alpha"]),
                beta_w=float(row["beta_w"]),
                beta_p=float(row["beta_p"]),
                beta_t=float(row["beta_t"]),
                pipeline_node=node_id_to_index[pid],
                substation_node=node_id_to_index[sid],
                in_bridge_area=in_bridge_area,
            )
        )

    meta = _read_meta(meta_path)

    return DataBundle(
        nodes=nodes,
        edges=edges,
        buildings=buildings,
        meta=meta,
        node_id_to_index=node_id_to_index,
        type_to_indices=type_to_indices,
    )
