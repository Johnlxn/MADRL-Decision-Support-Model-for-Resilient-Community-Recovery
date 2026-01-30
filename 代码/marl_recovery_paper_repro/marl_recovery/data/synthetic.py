# -*- coding: utf-8 -*-
"""合成校园案例生成器。

由于论文案例数据通常需要向作者索取，本工程提供一个“结构相似”的合成数据生成器，
用来验证算法流程与代码可运行性。

生成规模与论文案例一致：
- pipelines: 82
- substations: 13（其中 1 个主变）
- wells: 3
- bridges: 5
- buildings: 619

注意：合成数据的拓扑与损伤状态是随机生成的，因此数值曲线不应与论文完全一致。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import yaml


# ---------------------------
# 论文 Table 1 的期望修复时间（天）
# ---------------------------
_SUBSTATION_REPAIR_MEAN = {
    0: 0.0,  # intact
    1: 1.0,  # slight
    2: 3.0,  # moderate
    3: 7.0,  # extensive
    4: 30.0,  # complete
}

_BRIDGE_REPAIR_MEAN = {
    0: 0.0,
    1: 0.6,
    2: 2.5,
    3: 75.0,
    4: 230.0,
}


def _ensure_connected_gnm(n: int, m: int, seed: int) -> nx.Graph:
    """生成连通的 G(n, m) 随机图。"""
    rng = np.random.default_rng(seed)
    for _ in range(100):
        g = nx.gnm_random_graph(n, m, seed=int(rng.integers(0, 1_000_000)))
        if nx.is_connected(g):
            return g
    # 如果实在连不通，退化为先生成一棵树，再加边
    g = nx.random_tree(n, seed=seed)
    while g.number_of_edges() < m:
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        if u != v and not g.has_edge(u, v):
            g.add_edge(u, v)
    return g


def _sample_damage_state(rng: np.random.Generator, probs: List[float]) -> int:
    """按给定概率采样损伤等级（返回 0..len(probs)-1）。"""
    probs_arr = np.array(probs, dtype=float)
    probs_arr = probs_arr / probs_arr.sum()
    return int(rng.choice(len(probs_arr), p=probs_arr))


def _pipeline_expected_repair_time_from_pgv(rng: np.random.Generator, pgv_cm_s: float, length_km: float) -> Tuple[int, float, int, int]:
    """根据论文 Eq.(6) 与 Eq.(8) 生成管线的损伤与期望修复时间。

    返回：(damage_state, expected_repair_time, N_leak, N_break)
    """
    # RR = 0.0001 * PGV^2.25 （PGV 单位 cm/s）
    rr = 0.0001 * (pgv_cm_s ** 2.25)
    lam = rr * length_km

    n_damage = int(rng.poisson(lam))
    if n_damage <= 0:
        return 0, 0.0, 0, 0

    # 每处损伤 80% leak, 20% break
    is_break = rng.random(n_damage) < 0.2
    n_break = int(is_break.sum())
    n_leak = int(n_damage - n_break)

    # Ep = 0.5*Nl + 1*Nb （论文 Eq.(8)）
    ep = 0.5 * n_leak + 1.0 * n_break

    # damage_state: 1=leak（仅 leak）；2=break（包含 break）
    ds = 2 if n_break > 0 else 1
    return ds, float(ep), n_leak, n_break


def generate_synthetic_campus(
    out_dir: str,
    seed: int = 42,
    beta_w: float = 0.55,
    beta_p: float = 0.40,
    beta_t: float = 0.05,
) -> None:
    """生成合成校园数据到 out_dir。"""

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ---------------------------
    # 1) 生成 WDN：先生成“junction 图”，再做 line graph，使“管线”成为节点
    # ---------------------------
    n_junction = 40
    n_pipes = 82
    g_junc = _ensure_connected_gnm(n_junction, n_pipes, seed=seed)

    # 给每条“原始边(管线)”赋予一个 pipeline_id
    edge_list = list(g_junc.edges())
    if len(edge_list) != n_pipes:
        raise RuntimeError("WDN 生成失败：边数量不等于 82")

    # line graph：节点是原图的边
    g_pipe_line = nx.line_graph(g_junc)

    # 将 line graph 的节点（原边 (u,v)）映射到 pipeline 名称 P0..P81
    pipe_node_ids: Dict[Tuple[int, int], str] = {}
    for i, e in enumerate(edge_list):
        u, v = e
        # 无向边统一排序，避免 line_graph 表示不一致
        key = (u, v) if u <= v else (v, u)
        pipe_node_ids[key] = f"P{i}"

    # line graph 的节点可能以 (u,v) 或 (v,u) 表示，因此统一归一化
    def _norm_edge_node(x: Tuple[int, int]) -> Tuple[int, int]:
        a, b = x
        return (a, b) if a <= b else (b, a)

    pipe_nodes_order: List[str] = []
    for x in g_pipe_line.nodes():
        key = _norm_edge_node(x)
        if key not in pipe_node_ids:
            # 保险：如果出现未映射的边，说明 edge_list 与 line_graph 表示不一致
            raise RuntimeError(f"line_graph 节点未找到映射: {x}")
        pipe_nodes_order.append(pipe_node_ids[key])

    # 保证排序一致：P0..P81
    pipe_nodes_order = sorted(set(pipe_nodes_order), key=lambda s: int(s[1:]))

    # ---------------------------
    # 2) 生成 EPN：13 个变电站（S0 为主变）
    # ---------------------------
    n_sub = 13
    g_sub = nx.random_tree(n_sub, seed=int(rng.integers(0, 1_000_000)))
    # 额外加少量边增加冗余
    extra_edges = 6
    while extra_edges > 0:
        u = int(rng.integers(0, n_sub))
        v = int(rng.integers(0, n_sub))
        if u != v and not g_sub.has_edge(u, v):
            g_sub.add_edge(u, v)
            extra_edges -= 1

    sub_node_ids = [f"S{i}" for i in range(n_sub)]
    main_sub_id = "S0"

    # ---------------------------
    # 3) wells & bridges
    # ---------------------------
    well_ids = [f"W{i}" for i in range(3)]
    bridge_ids = [f"B{i}" for i in range(5)]

    # ---------------------------
    # 4) 生成 nodes.csv
    # ---------------------------
    rows_nodes = []

    # pipelines：按 Eq.(6)/(8) 采样期望修复时间
    # 为了让多数管线有一定概率受损，我们给 PGV 取 20~80 cm/s，长度 0.05~0.2 km
    pipe_damage_meta = {}
    for pid in pipe_nodes_order:
        pgv = float(rng.uniform(20.0, 80.0))
        length_km = float(rng.uniform(0.05, 0.2))
        ds, ep, nl, nb = _pipeline_expected_repair_time_from_pgv(rng, pgv_cm_s=pgv, length_km=length_km)
        rows_nodes.append(
            {
                "node_id": pid,
                "node_type": "pipeline",
                "is_main": 0,
                "damage_state": ds,
                "expected_repair_time": ep,
                "in_bridge_area": 0,
                "cooling_target": 0,
            }
        )
        pipe_damage_meta[pid] = {"pgv": pgv, "length_km": length_km, "n_leak": nl, "n_break": nb}

    # substations：采样损伤（更偏轻微），并赋予 Table 1 均值
    for sid in sub_node_ids:
        # 让主变更容易受损（模拟论文案例主变轻微损伤）
        if sid == main_sub_id:
            ds = _sample_damage_state(rng, probs=[0.4, 0.35, 0.15, 0.07, 0.03])
        else:
            ds = _sample_damage_state(rng, probs=[0.55, 0.25, 0.12, 0.06, 0.02])
        er = _SUBSTATION_REPAIR_MEAN[ds]
        # 论文案例：主变修复时间放大系数 1.3（论文 3.1）
        if sid == main_sub_id and er > 0:
            er *= 1.3

        rows_nodes.append(
            {
                "node_id": sid,
                "node_type": "substation",
                "is_main": 1 if sid == main_sub_id else 0,
                "damage_state": ds,
                "expected_repair_time": float(er),
                "in_bridge_area": 0,
                "cooling_target": 0,
            }
        )

    # wells：默认完好（也可扩展为可损伤组件）
    for wid in well_ids:
        rows_nodes.append(
            {
                "node_id": wid,
                "node_type": "well",
                "is_main": 0,
                "damage_state": 0,
                "expected_repair_time": 0.0,
                "in_bridge_area": 0,
                "cooling_target": 0,
            }
        )

    # bridges：采样损伤，Table 1 均值
    for bid in bridge_ids:
        ds = _sample_damage_state(rng, probs=[0.35, 0.25, 0.20, 0.12, 0.08])
        er = _BRIDGE_REPAIR_MEAN[ds]
        rows_nodes.append(
            {
                "node_id": bid,
                "node_type": "bridge",
                "is_main": 0,
                "damage_state": ds,
                "expected_repair_time": float(er),
                "in_bridge_area": 0,
                "cooling_target": 0,
            }
        )

    nodes_df = pd.DataFrame(rows_nodes)

    # 设置桥依赖区域：随机挑选一部分 pipeline/substation 作为“桥依赖区域组件”
    #（对应论文 3.1：河流与桥导致上游区域需要桥通行）
    comp_mask = nodes_df["node_type"].isin(["pipeline", "substation"])
    comp_indices = nodes_df[comp_mask].index.to_numpy()
    n_bridge_area = int(0.30 * len(comp_indices))
    chosen = rng.choice(comp_indices, size=n_bridge_area, replace=False)
    nodes_df.loc[chosen, "in_bridge_area"] = 1

    # 指定一个“冷却水管线”（用于主变互依赖）
    cooling_pipe = str(rng.choice(pipe_nodes_order))
    nodes_df.loc[nodes_df["node_id"] == cooling_pipe, "cooling_target"] = 1

    # ---------------------------
    # 5) 生成 edges.csv（论文 Fig.8 的 6 类边）
    # ---------------------------
    edges_rows = []

    # (1) pipe_pipe：来自 line graph
    # line graph 节点是原边，需要用我们自己的 P* id 映射
    # 这里通过 edge_list + line_graph 的边恢复 P* 之间的邻接
    # 先建立从原边(key)到 pid 的映射
    norm_to_pid = {}
    for e, pid in pipe_node_ids.items():
        norm_to_pid[e] = pid

    for u, v in g_pipe_line.edges():
        uu = _norm_edge_node(u)
        vv = _norm_edge_node(v)
        pu = norm_to_pid[uu]
        pv = norm_to_pid[vv]
        edges_rows.append({"src": pu, "dst": pv, "edge_type": "pipe_pipe"})

    # (2) sub_sub：变电站之间
    for u, v in g_sub.edges():
        edges_rows.append({"src": f"S{u}", "dst": f"S{v}", "edge_type": "sub_sub"})

    # wells：为每口井选择一个“最近变电站”供电，以及一个“最近管线”注水
    well_power_sub: Dict[str, str] = {}
    well_pipe: Dict[str, str] = {}

    dist_sub_ids = [sid for sid in sub_node_ids if sid != main_sub_id]

    for wid in well_ids:
        sid = str(rng.choice(dist_sub_ids))
        pid = str(rng.choice(pipe_nodes_order))
        well_power_sub[wid] = sid
        well_pipe[wid] = pid

        # (3) sub_well：井需要变电站供电
        edges_rows.append({"src": sid, "dst": wid, "edge_type": "sub_well"})

        # (4) well_pipe：井向管线注水（互依赖）
        edges_rows.append({"src": wid, "dst": pid, "edge_type": "well_pipe"})

    # (5) sub_pipe：主变对“冷却水管线”的依赖（也可扩展为更多耦合边）
    edges_rows.append({"src": main_sub_id, "dst": cooling_pipe, "edge_type": "sub_pipe"})

    # 额外：让每个配电站也随机连接若干管线，帮助 GNN 感知跨系统关系
    for sid in dist_sub_ids:
        for _ in range(2):
            pid = str(rng.choice(pipe_nodes_order))
            edges_rows.append({"src": sid, "dst": pid, "edge_type": "sub_pipe"})

    # (6) bridge_comp：桥与“桥依赖区域组件”的边
    bridge_area_nodes = nodes_df[nodes_df["in_bridge_area"] == 1]["node_id"].tolist()
    for bid in bridge_ids:
        for nid in bridge_area_nodes:
            edges_rows.append({"src": bid, "dst": nid, "edge_type": "bridge_comp"})

    edges_df = pd.DataFrame(edges_rows)

    # ---------------------------
    # 6) 生成 buildings.csv（619 栋建筑）
    # ---------------------------
    n_buildings = 619
    b_rows = []

    # 为建筑随机分配“最近管线/配电站”。
    # 注意：这里只是合成数据；真实案例应由 GIS/拓扑匹配确定。
    for i in range(n_buildings):
        pid = str(rng.choice(pipe_nodes_order))
        sid = str(rng.choice(dist_sub_ids))
        area = float(rng.uniform(500.0, 5000.0))
        alpha = 1.0

        # 建筑是否在桥依赖区域：如果它使用的管线或配电站在桥依赖区，则认为它也在该区
        in_bridge_area = int(
            nodes_df.loc[nodes_df["node_id"] == pid, "in_bridge_area"].iloc[0]
            or nodes_df.loc[nodes_df["node_id"] == sid, "in_bridge_area"].iloc[0]
        )

        b_rows.append(
            {
                "building_id": f"BL{i}",
                "area": area,
                "alpha": alpha,
                "beta_w": beta_w,
                "beta_p": beta_p,
                "beta_t": beta_t,
                "pipeline_node_id": pid,
                "substation_node_id": sid,
                "in_bridge_area": in_bridge_area,
            }
        )

    buildings_df = pd.DataFrame(b_rows)

    # ---------------------------
    # 7) meta.yaml：存一些互依赖映射
    # ---------------------------
    meta = {
        "main_substation_id": main_sub_id,
        "cooling_pipeline_id": cooling_pipe,
        "cooling_well_id": well_ids[0],  # 简化：选 W0 作为“冷却水相关井”
        "well_power_substation": well_power_sub,
        "well_injection_pipeline": well_pipe,
        "pipe_damage_meta": pipe_damage_meta,
    }

    # ---------------------------
    # 8) 保存文件
    # ---------------------------
    nodes_df.to_csv(os.path.join(out_dir, "nodes.csv"), index=False)
    edges_df.to_csv(os.path.join(out_dir, "edges.csv"), index=False)
    buildings_df.to_csv(os.path.join(out_dir, "buildings.csv"), index=False)
    with open(os.path.join(out_dir, "meta.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)
