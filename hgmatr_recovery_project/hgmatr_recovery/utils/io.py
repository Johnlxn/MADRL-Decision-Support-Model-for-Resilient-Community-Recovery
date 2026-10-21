# -*- coding: utf-8 -*-
"""I/O 工具。"""

from __future__ import annotations

import json
from typing import Any


def save_json(path: str, obj: Any) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
