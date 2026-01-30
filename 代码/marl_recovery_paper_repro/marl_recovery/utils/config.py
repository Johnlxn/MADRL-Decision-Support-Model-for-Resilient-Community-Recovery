# -*- coding: utf-8 -*-
"""配置与随机种子工具。

为了便于在 PyCharm 中运行，本工程使用 YAML 配置文件。

注意：如果你在没有 GPU 的机器上运行，torch 仍可以使用 CPU。
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import yaml

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def load_yaml(path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件内容必须是 dict，当前类型: {type(cfg)}")
    return cfg


def ensure_dir(path: str) -> None:
    """创建目录（若不存在）。"""
    os.makedirs(path, exist_ok=True)


def set_global_seeds(seed: int) -> None:
    """设置全局随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 让 cudnn 可复现（可能影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
