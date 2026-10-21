# -*- coding: utf-8 -*-
"""配置与随机种子工具。"""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f'YAML 顶层必须是 dict，当前为 {type(obj)}')
    return obj


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_global_seeds(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
