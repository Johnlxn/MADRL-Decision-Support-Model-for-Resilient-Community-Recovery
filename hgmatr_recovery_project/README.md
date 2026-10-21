
# HGMAT-R Recovery Project

这是一个**全新整理**的独立项目库，只保留：

- **HGMAT-R**  
  Heterogeneous Graph Memory-augmented Autoregressive Transformer for Recovery

不再保留旧的 GNN-FiLM / MAPPO / Seq-MAPPO 路线。

## 目录

- `hgmatr_recovery/`
  - `data/` 数据加载与合成数据生成
  - `env/` 事件驱动恢复环境
  - `models/hgt.py` 异构图编码器
  - `models/hgmatr.py` HGMAT-R 主网络
  - `algorithms/hgmatr_trainer.py` 训练器
- `scripts/train_hgmatr.py` 训练入口
- `scripts/smoke_test_hgmatr.py` 快速自检
- `configs/` 训练配置
- `data/synth_campus/` 示例合成数据

## 安装

```bash
pip install -r requirements.txt
```

如果你使用 GPU，建议按 PyTorch 官网安装与你 CUDA 版本匹配的 `torch`。

## 快速自检

```bash
python scripts/smoke_test_hgmatr.py
```

## 训练

正式配置：

```bash
python scripts/train_hgmatr.py --config configs/train_hgmatr_resilience.yaml --device cuda
```

快速配置：

```bash
python scripts/train_hgmatr.py --config configs/train_hgmatr_quick.yaml --device cuda
```

单卡更稳的 GPU 配置：

```bash
python scripts/train_hgmatr.py --config configs/train_hgmatr_gpu_single.yaml --device cuda
```
