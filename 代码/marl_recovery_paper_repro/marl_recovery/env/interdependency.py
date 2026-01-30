# -*- coding: utf-8 -*-
"""互依赖基础设施的功能判定。

论文 2.1 与 2.4.2 中给出了 WDN/EPN/TN 的互依赖逻辑：
- 井泵需要电（由附近变电站供电）
- 主变需要冷却水（由某条管线与井泵提供）
- 桥梁决定“桥依赖区域”组件是否可达（AC）

论文中通过图模型与路径存在性分析（NetworkX）来判断服务可达性。
本模块实现一个“可复现且可扩展”的版本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from marl_recovery.data.loader import DataBundle


@dataclass
class SystemIndicators:
    """计算得到的状态指标。"""

    fn: np.ndarray  # (N,) 组件功能指示（考虑互依赖）
    ac: np.ndarray  # (N,) 可达性（用于动作 mask）
    i_w: np.ndarray  # (Nb,) 建筑水可达性
    i_p: np.ndarray  # (Nb,) 建筑电可达性
    i_t: np.ndarray  # (Nb,) 建筑交通可达性


def _get_indices(bundle: DataBundle, node_type: str) -> List[int]:
    return bundle.type_to_indices.get(node_type, [])


def compute_accessibility(bundle: DataBundle, fn_bridge: np.ndarray) -> np.ndarray:
    """根据桥梁功能判定可达性 AC。

    论文假设：道路完好，仅桥梁损伤影响可达性。
    - 在 bridge-dependent 区域内的组件：AC = 1 当且仅当任意一座桥可用
    - 其他组件：AC=1
    - 桥梁自身始终可达（AC=1）
    """
    n = bundle.num_nodes
    ac = np.ones(n, dtype=np.float32)

    bridge_ok = bool(fn_bridge.sum() > 0.0)

    for i, node in enumerate(bundle.nodes):
        if node.node_type == "bridge":
            ac[i] = 1.0
            continue
        if node.in_bridge_area:
            ac[i] = 1.0 if bridge_ok else 0.0
        else:
            ac[i] = 1.0
    return ac


def compute_functionality_and_building_indicators(
    bundle: DataBundle,
    phys_ok: np.ndarray,
    max_iter: int = 10,
) -> SystemIndicators:
    """在给定“物理修复状态”下，计算：
    - 组件功能 FN（考虑互依赖）
    - 可达性 AC
    - 建筑 Iw/Ip/It

    参数
    ----
    bundle: 数据包
    phys_ok: (N,) 物理状态，1=完好或已修复完成；0=未完成修复

    返回
    ----
    SystemIndicators

    说明
    ----
    互依赖存在环（主变↔井泵↔管线），因此使用迭代求一个稳定状态。
    论文未显式说明采用哪种固定点，这里采用“从 phys_ok 出发”的方式（更接近工程直觉）。
    """
    n = bundle.num_nodes
    phys_ok = phys_ok.astype(np.float32)

    # 索引集合
    pipe_idx = _get_indices(bundle, "pipeline")
    sub_idx = _get_indices(bundle, "substation")
    well_idx = _get_indices(bundle, "well")
    bridge_idx = _get_indices(bundle, "bridge")

    # 主变索引
    main_sub = None
    for i in sub_idx:
        if bundle.nodes[i].is_main:
            main_sub = i
            break
    if main_sub is None and len(sub_idx) > 0:
        main_sub = sub_idx[0]

    # 冷却水相关节点（主变依赖）
    cooling_pipe = None
    for i in pipe_idx:
        if bundle.nodes[i].cooling_target:
            cooling_pipe = i
            break
    # 如果没有标注 cooling_target，则从 meta 读取
    if cooling_pipe is None:
        cid = bundle.meta.get("cooling_pipeline_id")
        if cid is not None and str(cid) in bundle.node_id_to_index:
            cooling_pipe = bundle.node_id_to_index[str(cid)]

    cooling_well = None
    wid = bundle.meta.get("cooling_well_id")
    if wid is not None and str(wid) in bundle.node_id_to_index:
        cooling_well = bundle.node_id_to_index[str(wid)]
    elif len(well_idx) > 0:
        cooling_well = well_idx[0]

    # 构建水系统图（pipelines + wells）
    g_water = nx.Graph()
    g_water.add_nodes_from(pipe_idx)
    g_water.add_nodes_from(well_idx)
    # pipe_pipe
    if "pipe_pipe" in bundle.edges:
        e = bundle.edges["pipe_pipe"]
        for u, v in e.T:
            u = int(u)
            v = int(v)
            if u in g_water and v in g_water:
                g_water.add_edge(u, v)
    # well_pipe
    if "well_pipe" in bundle.edges:
        e = bundle.edges["well_pipe"]
        for u, v in e.T:
            u = int(u)
            v = int(v)
            if u in g_water and v in g_water:
                g_water.add_edge(u, v)

    # 构建电系统图（substations）
    g_power = nx.Graph()
    g_power.add_nodes_from(sub_idx)
    if "sub_sub" in bundle.edges:
        e = bundle.edges["sub_sub"]
        for u, v in e.T:
            u = int(u)
            v = int(v)
            if u in g_power and v in g_power:
                g_power.add_edge(u, v)

    # well_power_substation 映射（优先用 meta）
    well_power_map: Dict[int, int] = {}
    meta_map = bundle.meta.get("well_power_substation", {})
    if isinstance(meta_map, dict):
        for w_id, s_id in meta_map.items():
            if str(w_id) in bundle.node_id_to_index and str(s_id) in bundle.node_id_to_index:
                well_power_map[bundle.node_id_to_index[str(w_id)]] = bundle.node_id_to_index[str(s_id)]

    # 如果 meta 没给，就从 sub_well 边中取一个邻居
    if len(well_power_map) == 0 and "sub_well" in bundle.edges:
        e = bundle.edges["sub_well"]
        for u, v in e.T:
            u = int(u)
            v = int(v)
            # 边可能是 sub->well 或 well->sub（因为 loader 转为双向），这里做判断
            if u in sub_idx and v in well_idx:
                well_power_map[v] = u

    # 初始化 FN=phys_ok
    fn = phys_ok.copy()

    for _ in range(max_iter):
        fn_prev = fn.copy()

        # 1) 桥梁功能（不依赖其他系统）
        fn_bridge = np.zeros(n, dtype=np.float32)
        for i in bridge_idx:
            fn_bridge[i] = phys_ok[i]

        # 2) 可达性 AC（依赖桥）
        ac = compute_accessibility(bundle, fn_bridge)

        # 3) 电系统：主变功能（依赖冷却水井与冷却水管线）
        fn_main = 0.0
        if main_sub is not None:
            if phys_ok[main_sub] > 0.0:
                # cooling_pipe / cooling_well 可能不存在（用户数据可能未指定），则考虑为不依赖
                ok_pipe = 1.0
                ok_well = 1.0
                if cooling_pipe is not None:
                    ok_pipe = fn_prev[cooling_pipe]
                if cooling_well is not None:
                    ok_well = fn_prev[cooling_well]
                fn_main = float(1.0 if (ok_pipe > 0.0 and ok_well > 0.0) else 0.0)

        # 4) 配电站 energization：物理完好且与主变连通，并且主变可用
        fn_sub = np.zeros(n, dtype=np.float32)
        energized = set()
        if main_sub is not None and fn_main > 0.0:
            # 在物理完好的子图中找与主变连通的节点
            active_subs = [i for i in sub_idx if phys_ok[i] > 0.0]
            g_act = g_power.subgraph(active_subs).copy()
            if main_sub in g_act:
                energized = nx.node_connected_component(g_act, main_sub)

        for i in sub_idx:
            if phys_ok[i] > 0.0 and i in energized and fn_main > 0.0:
                fn_sub[i] = 1.0
            else:
                fn_sub[i] = 0.0

        # 5) wells：物理完好且其供电变电站有电
        fn_well = np.zeros(n, dtype=np.float32)
        for i in well_idx:
            if phys_ok[i] <= 0.0:
                fn_well[i] = 0.0
                continue
            s = well_power_map.get(i, None)
            if s is None:
                # 若无映射，保守起见：没有电
                fn_well[i] = 0.0
            else:
                fn_well[i] = 1.0 if fn_sub[s] > 0.0 else 0.0

        # 6) 水系统：管线物理完好且与任一“可用井”连通
        fn_pipe = np.zeros(n, dtype=np.float32)
        # 只保留物理完好的管线与可用井
        active_water_nodes = [i for i in pipe_idx if phys_ok[i] > 0.0] + [i for i in well_idx if fn_well[i] > 0.0]
        g_act_w = g_water.subgraph(active_water_nodes).copy()

        # 从所有可用井出发，做可达性
        reachable_pipes = set()
        for w in well_idx:
            if fn_well[w] <= 0.0:
                continue
            if w not in g_act_w:
                continue
            # 与该井在同一连通分量的节点即为可达
            comp = nx.node_connected_component(g_act_w, w)
            for x in comp:
                if x in pipe_idx:
                    reachable_pipes.add(x)

        for i in pipe_idx:
            if phys_ok[i] > 0.0 and i in reachable_pipes:
                fn_pipe[i] = 1.0
            else:
                fn_pipe[i] = 0.0

        # 7) 汇总所有组件 FN
        fn_new = np.zeros(n, dtype=np.float32)
        # 默认：其他类型的组件（如果存在）直接等于 phys_ok
        fn_new[:] = phys_ok
        for i in pipe_idx:
            fn_new[i] = fn_pipe[i]
        for i in sub_idx:
            # 主变也属于 substation
            fn_new[i] = fn_sub[i]
        if main_sub is not None:
            fn_new[main_sub] = fn_main
        for i in well_idx:
            fn_new[i] = fn_well[i]
        for i in bridge_idx:
            fn_new[i] = fn_bridge[i]

        fn = fn_new

        # 迭代收敛判定
        if np.allclose(fn, fn_prev):
            break

    # ---------------------------
    # 建筑 Iw/Ip/It
    # ---------------------------
    nb = len(bundle.buildings)
    i_w = np.zeros(nb, dtype=np.float32)
    i_p = np.zeros(nb, dtype=np.float32)
    i_t = np.ones(nb, dtype=np.float32)

    # 桥是否通行：只要任意桥可用
    bridge_ok = False
    if len(bridge_idx) > 0:
        bridge_ok = bool(fn[bridge_idx].sum() > 0.0)

    for bi, b in enumerate(bundle.buildings):
        # 交通可达性
        if b.in_bridge_area:
            i_t[bi] = 1.0 if bridge_ok else 0.0
        else:
            i_t[bi] = 1.0

        # 水/电服务可达性
        i_w[bi] = 1.0 if fn[b.pipeline_node] > 0.0 else 0.0
        i_p[bi] = 1.0 if fn[b.substation_node] > 0.0 else 0.0

    # 最终可达性 AC：基于最终桥梁 FN
    fn_bridge_final = np.zeros(n, dtype=np.float32)
    for i in bridge_idx:
        fn_bridge_final[i] = fn[i]
    ac_final = compute_accessibility(bundle, fn_bridge_final)

    return SystemIndicators(fn=fn, ac=ac_final, i_w=i_w, i_p=i_p, i_t=i_t)
