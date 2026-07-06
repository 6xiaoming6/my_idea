# Imputation Benchmark

用于 TaxiBJ、BikeNYC 和 CHAP 的时空缺失值补全基准工程。所有上游模型源码、
数据适配、配置、运行入口和实验产物均按职责分区，根目录不存放临时测试结果。

## 目录结构

```text
imputation_benchmark/
├── baselines/              # 上游 baseline 源码（不修改模型结构）
├── configs/
│   ├── policies/           # 训练策略 JSON
│   ├── training/           # 正式训练生成配置
│   ├── testing/            # 集成测试专用配置
│   └── legacy_smoke/       # 只读归档的旧 smoke 配置
├── scripts/
│   ├── config/             # 配置生成
│   ├── data/               # 数据适配
│   ├── launch/             # 统一任务入口
│   └── train/              # 模型专用训练适配器
├── data/
│   ├── adapted/            # 正式适配数据
│   ├── testing/            # 集成测试小数据
│   └── legacy/             # 旧格式兼容数据
├── artifacts/
│   ├── runs/               # paper、TaxiBJ、集成测试运行记录
│   ├── selftests/          # 启动器自检记录
│   ├── legacy/             # 历史 smoke 只读归档
│   └── native_experiments/ # 上游模型临时原生输出
└── docs/                   # 协议、数据说明和上游原 README
```

## 常用入口

从项目目录 `my_idea/` 运行：

```bash
# 13 个主表 baseline × 3 数据集的一轮集成测试
python imputation_benchmark/scripts/launch/run_overview_baseline_1epoch.py --gpu 0

# 正式 baseline 矩阵
python imputation_benchmark/scripts/launch/run_all_baseline_train_2gpu.py --gpus 0 1

# TaxiBJ 单卡安全队列（不包含 CSDI/PriSTI）
python imputation_benchmark/scripts/launch/run_taxibj_full_baselines.py --gpu 0

# 单独训练一个模型
python imputation_benchmark/scripts/train/train_saits.py \
  --dataset TaxiBJ --mask fixed --rate 0.2 --channel 0 --gpu 0
```

正式输出仍统一写入项目级 `outputs/<dataset>/baseline/...`。启动器汇总、测试结果
和上游临时产物写入本目录的 `artifacts/`，不会重新散落到根目录。

详细说明见 [论文对比协议](docs/PAPER_BASELINE_PROTOCOL.md)、
[网格数据适配指南](docs/GRID_DATASET_GUIDE.md) 和
[训练脚本说明](scripts/train/README.md)。
