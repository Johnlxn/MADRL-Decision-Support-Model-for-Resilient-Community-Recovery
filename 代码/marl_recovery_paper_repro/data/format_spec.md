# 数据格式规范（用于接入真实案例数据）

本工程的数据读取以“一个数据目录”为单位，例如：`data/tsinghua/`。

数据目录中至少包含三个文件：

- `nodes.csv`：组件节点表（管线/变电站/井/桥）
- `edges.csv`：异构图边表（包含边类型）
- `buildings.csv`：建筑属性与服务映射表

此外还可以包含：
- `meta.yaml`：可选，写入一些全局参数（RU 数量、控制时间 TLC、80% 阈值等）。

---

## 1) nodes.csv

字段说明（每行一个“组件节点”）：

| 字段 | 类型 | 说明 |
|---|---|---|
| node_id | str/int | 节点唯一 ID（可为数字或字符串） |
| node_type | str | `pipeline` / `substation` / `well` / `bridge` |
| is_main | int | 仅 substation 使用：1=主变电站，0=配电变电站（其他类型可留空） |
| damage_state | int | 损伤等级（整数编码），建议：0=完好；越大越严重。pipeline 可用 0/1/2 表示 intact/leak/break；substation/bridge 可用 0..4 表示 intact..complete |
| expected_repair_time | float | 期望修复时间 Er（单位：天），对应论文 Table 1 与 Eq.(8) 的“均值” |
| in_bridge_area | int | 1 表示位于“桥依赖区域”，此类组件可达性 AC 取决于桥是否通行 |
| cooling_target | int | 可选：1 表示该节点是“主变所需冷却水”关联的 pipeline（用于主变互依赖），其他为 0 |
| power_target | int | 可选：1 表示该节点是“冷却水/供水井需要的供电影响源”关键节点（可不填） |

> 注意：本工程假设“组件是否物理可用”是二值：完好或修复完成即 1，否则 0（与论文一致）。

---

## 2) edges.csv

字段说明（每行一条边，默认无向边；如需有向可在代码中扩展）：

| 字段 | 类型 | 说明 |
|---|---|---|
| src | str/int | 起点 node_id |
| dst | str/int | 终点 node_id |
| edge_type | str | 边类型（用于异构图 GNN-FiLM）：

支持的 edge_type 建议包括（对应论文 Fig.8 描述的 6 类边）：
- `pipe_pipe`
- `well_pipe`
- `sub_sub`
- `sub_well`
- `sub_pipe`
- `bridge_comp`

---

## 3) buildings.csv

字段说明（每行一个建筑）：

| 字段 | 类型 | 说明 |
|---|---|---|
| building_id | str/int | 建筑 ID |
| area | float | 建筑面积 S_i |
| alpha | float | 建筑权重 α_i（论文中案例简化为 1.0） |
| beta_w | float | 水系统重要度 β_w |
| beta_p | float | 电系统重要度 β_p |
| beta_t | float | 交通系统重要度 β_t |
| pipeline_node_id | str/int | 为该建筑供水的“管线节点”ID |
| substation_node_id | str/int | 为该建筑供电的“配电站节点”ID |
| in_bridge_area | int | 1 表示建筑在桥依赖区域，交通可达性 It 取决于桥是否通行 |

---

## 4) meta.yaml（可选）

示例：

```yaml
control_time: 100.0   # TLC
objective: resilience # resilience 或 time80
threshold_q: 0.8
ru:
  pipeline: 2
  substation: 1
  bridge: 1
  well: 0
```

如果缺省，则脚本会使用 `configs/*.yaml` 中的默认参数。
