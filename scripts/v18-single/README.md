# V18 BARP-MoE 实验链路

V18 基于 V14 的稳定 Main Multi-Scale MoE，只替换原有绝对值 C2F 修正路径。
正式输出固定写入 `outputs/v18-single/`，不会覆盖 V14 或其他版本。

## 配置与训练协议

- TaxiBJ：batch 32，160 epoch，每 5 epoch 验证；fixed 不早停，random
  patience 10。
- BikeNYC：batch 16，140 epoch，每 2 epoch 验证；patience 12。
- CHAP：batch 32，150 epoch，每 5 epoch 验证，不早停。
- 所有训练由公共 `scripts/train.py` 完成：只保存验证 MAE 最优的
  `checkpoints/best.pt`，训练结束加载该参数测试。
- MAE、RMSE、MAPE 均按整个 split 的全部缺失元素累计误差后统一计算，不对
  不同大小 batch 的指标做简单平均；最佳 epoch 固定按全局验证 MAE 选择。
- 正式实验默认要求 Git 工作区干净；`--allow-dirty` 只用于调试。

## 0. 参数量

```bash
python scripts/v18-single/count_parameters.py
```

V18 总参数量应不超过 V14 的 1.05 倍。

## 1. 三数据集结构 Smoke

三个点均执行 2 epoch，并完整走 train/val/test：

```bash
python scripts/v18-single/run_smoke.py --gpu 0
```

只检查命令和配置：

```bash
python scripts/v18-single/run_smoke.py --gpu 0 --dry-run
```

## 2. 六点筛选

先提交代码、确认 `git status` 干净。为了保证 V14/V18 对比使用完全相同的
数据掩码、训练轮数、验证频率、早停策略和全局指标口径，先运行匹配的 V14
对照，再运行 V18：

```bash
python scripts/v18-single/run_v14_controls.py \
  --stage screening \
  --gpu 0

python scripts/v18-single/run_screening.py --gpu 0
```

汇总并与同点、同 seed 的 V14 结果配对：

```bash
python scripts/v18-single/summarize_v18.py \
  --stage screening \
  --name stage2_screening \
  --require-complete
```

## 3. 四个关键点三随机种子

```bash
python scripts/v18-single/run_v14_controls.py \
  --stage core4 \
  --gpu 0 \
  --seeds 42 2026 3407

python scripts/v18-single/run_multiseed.py \
  --gpu 0 \
  --points all \
  --seeds 42 2026 3407
```

```bash
python scripts/v18-single/summarize_v18.py \
  --stage core4 \
  --seeds 42 2026 3407 \
  --name stage3_core4_multiseed \
  --require-complete
```

汇总器仅在存在同点同 seed V14 结果时计算配对改善；缺少 V14 多种子结果时会
明确列出，不会误用 seed 42 替代。

## 4. 三数据集完整 24 点

按 fixed 后 random 的顺序运行。正式配对比较前，先产生同协议 V14 对照：

```bash
python scripts/v18-single/run_v14_controls.py \
  --stage full24 \
  --gpu 0 \
  --seeds 42 \
  --datasets all \
  --patterns fixed random \
  --rates 0.2 0.4 0.6 0.8

python scripts/v18-single/run_full_24.py \
  --gpu 0 \
  --seed 42 \
  --datasets all \
  --patterns fixed random \
  --rates 0.2 0.4 0.6 0.8
```

中断后执行相同命令，已完整保存 best checkpoint 和 Test 指标的点会自动跳过。
如需明确重跑，添加 `--force-rerun`。

```bash
python scripts/v18-single/summarize_v18.py \
  --stage full24 \
  --seeds 42 \
  --name stage4_full24_seed42 \
  --require-complete
```

## 单点与消融

```bash
python scripts/v18-single/train.py \
  --dataset TaxiBJ \
  --mask fixed \
  --rate 0.4 \
  --seed 42 \
  --gpu 0
```

当前实现的 7 项正式消融：

- `absolute_c2f`
- `unbounded_residual`
- `no_observed_utility`
- `no_reliability_filtering`
- `fine_only_residual`
- `fixed_budget`
- `no_sample_regret`

例如：

```bash
python scripts/v18-single/train.py \
  --dataset BikeNYC \
  --mask random \
  --rate 0.4 \
  --ablation no_observed_utility \
  --gpu 0
```

其中：

- `absolute_c2f` 使用独立的绝对值 Coarse-to-Fine 候选路径，并在相同
  Controller 预算下围绕 Base 进行受控组合，用来隔离“绝对重建”与
  “Base-relative 方向残差”的差异。它是机制级破坏性对照，不是 V14 全结构
  复刻，也不承诺 Full V18 的初始化回退或观测尺度硬界。
- `unbounded_residual` 把三尺度方向头的 `tanh` 输出替换成线性输出，用来验证
  方向硬界的必要性；该项不再承诺 Full V18 的残差硬上界。
- `fine_only_residual` 在构造期移除 Coarse/Mid DirectionHead，同时把对应辅助
  损失权重置零。

两项破坏性消融只由对应配置显式启用，不改变 Full V18 的默认安全路径。

## 输出完整性

正式完成的运行至少包含：

```text
checkpoints/best.pt
logs/train.log
logs/val.log
logs/test.log
logs/metrics.jsonl
config.json
```

汇总文件保存在：

```text
outputs/v18-single/summary/
```

自动跳过不仅检查文件是否存在，还核对 seed、数据/掩码 SHA256、模型、损失、
训练、验证和实验配置签名。2/5 epoch 调试结果或旧指标口径结果不会被当成正式
实验。汇总器同样只允许完全匹配协议的 V14/V18 结果配对。

本目录提供的是单卡正式训练入口。模型与 checkpoint 已做设备无关和
DataParallel/DDP wrapper 兼容，但当前没有提供 `torchrun`、DistributedSampler
和跨 rank 指标归并，因此不要直接把这些脚本当作完整 DDP 训练器运行。
