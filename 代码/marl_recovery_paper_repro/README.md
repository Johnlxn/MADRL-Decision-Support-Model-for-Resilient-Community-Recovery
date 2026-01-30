# 论文复现代码（PyCharm 可直接运行）

本工程复现论文：**Multi-agent deep reinforcement learning based decision support model for resilient community post-hazard recovery**（Reliability Engineering & System Safety, 2024）中的核心方法：

- 社区功能度量 Q(t) 与韧性损失 RL（论文 Eq.(1)–(4)）
- 互依赖基础设施（WDN/EPN/TN）恢复仿真（图模型 + 路径存在性分析）
- 事件驱动（离散事件）恢复过程：执行联合动作后推进到“下一次修复完成”再决策（论文 2.5.1）
- 并行修复（同类型多个修复单元可同时修同一组件，线性加速）（论文 2.5.1 假设）
- 异构图 GNN-FiLM 编码（论文 Eq.(18)(19)）
- 多智能体**顺序更新** PPO（论文 Eq.(16)(17) + Algorithm 1 思路）
- 基线方法：Random / Importance-based / Rollout-SA（论文 3.2）

> 重要说明：论文案例（清华大学校园）原始数据在文中声明 **Data will be made available on request**，通常无法在公开网络直接下载。因此：
> - 本工程实现的是“方法/算法/流程级复现”（公式、网络结构、训练流程一致）。
> - 若要得到与论文图表完全一致的数值，需要使用作者提供的同一份案例数据。
> - 工程已提供“合成校园数据生成器”，可跑通训练与评估流程；你也可以按 `data/format_spec.md` 接入真实数据。

---

## 1. 环境准备

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

如果你安装 torch 有困难，请按你电脑系统选择官方安装方式（CPU/GPU）。

---

## 2. 一键跑通（合成数据）

### 2.1 生成合成校园网络

```bash
python scripts/generate_synthetic_campus.py --out data/synth_campus --seed 42
```

### 2.2 训练（目标1：最大化韧性/最小化 RL）

```bash
python scripts/train.py --config configs/train_resilience.yaml
```

### 2.3 评估（输出恢复曲线与指标）

```bash
python scripts/evaluate.py --config configs/train_resilience.yaml --ckpt outputs/resilience/best.pt
```

你会在 `outputs/...` 下看到：
- `curve_Q.png`：社区 Q(t) 恢复曲线
- `metrics.json`：RL、T80、最终 Q 等

---

## 3. 接入真实数据（作者数据或你自己的数据）

请查看：`data/format_spec.md`

将数据目录路径写入配置文件中的 `data_dir` 即可。

---

## 4. 代码结构

- `marl_recovery/env/`：恢复环境、互依赖求解、韧性计算
- `marl_recovery/graph/`：异构图数据结构、动作 mask
- `marl_recovery/models/`：GNN-FiLM、Actor/Critic
- `marl_recovery/algorithms/`：顺序更新多智能体 PPO、缓冲区、GAE
- `marl_recovery/baselines/`：三种基线策略
- `scripts/`：生成数据 / 训练 / 评估

---

## 5. 复现实验参数（对应论文 Table 2）

默认配置已提供：
- γ=0.999
- λ=0.95
- ϵ=0.2
- 轨迹长度 T=128
- GNN 两层卷积：5→32→1
- MLP：5 层，每层 128

你可以在 `configs/*.yaml` 中修改。

