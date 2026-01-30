# -*- coding: utf-8 -*-
"""marl_recovery

论文方法复现工程主包。

说明：本工程尽量按照论文中的公式与流程实现。
- 社区功能 Q(t) 与韧性损失 RL：论文 Eq.(1)~(4)
- 维修时间分布：论文 2.4.1（对数正态 + 截断）
- GNN-FiLM：论文 Eq.(18)(19)
- 顺序更新多智能体 PPO：论文 Eq.(16)(17) + Algorithm 1

作者数据若不可得，可先使用 scripts/generate_synthetic_campus.py 生成合成数据。
"""

__all__ = [
    "env",
    "graph",
    "models",
    "algorithms",
    "baselines",
    "data",
    "utils",
]
