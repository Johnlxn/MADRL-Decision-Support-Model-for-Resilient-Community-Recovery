# -*- coding: utf-8 -*-
"""生成合成校园数据。

用法：
python scripts/generate_synthetic_campus.py --out data/synth_campus --seed 42
"""

from __future__ import annotations
import os
import sys

# 让脚本在命令行与 PyCharm 中都能直接 import 本工程包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import argparse

from marl_recovery.data.synthetic import generate_synthetic_campus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="输出目录，例如 data/synth_campus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta_w", type=float, default=0.55)
    parser.add_argument("--beta_p", type=float, default=0.40)
    parser.add_argument("--beta_t", type=float, default=0.05)
    args = parser.parse_args()

    generate_synthetic_campus(
        out_dir=args.out,
        seed=args.seed,
        beta_w=args.beta_w,
        beta_p=args.beta_p,
        beta_t=args.beta_t,
    )
    print(f"已生成合成数据到: {args.out}")


if __name__ == "__main__":
    main()
