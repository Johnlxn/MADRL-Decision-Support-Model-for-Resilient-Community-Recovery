
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from hgmatr_recovery.data.synthetic import generate_synthetic_campus

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='data/synth_campus')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    out_dir = str(ROOT / args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    generate_synthetic_campus(out_dir=out_dir, seed=args.seed)
    print(f'已生成合成数据: {out_dir}')
if __name__ == '__main__':
    main()
