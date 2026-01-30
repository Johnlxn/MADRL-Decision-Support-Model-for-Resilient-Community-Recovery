# -*- coding: utf-8 -*-
"""简单的 IO 工具：保存/读取 JSON 等。"""

from __future__ import annotations

import json
from typing import Any, Dict


def save_json(path: str, obj: Dict[str, Any], indent: int = 2) -> None:
    """将字典保存为 JSON 文件（UTF-8）。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(path: str) -> Dict[str, Any]:
    """读取 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON 必须是 dict，当前类型: {type(obj)}")
    return obj
