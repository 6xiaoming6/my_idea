# Version 2: Observation-Aware Multi-Scale MoE Imputer

This directory is a standalone PyTorch implementation of `design_v2.md`.

The current stage trains the main branch only. The v2-2 main branch uses two
parallel paths:

```text
fine/mid/coarse embeddings
  -> intra-scale top-k routed expert branch
  -> cross-scale shared expert branch
  -> fusion network -> x_hat_main
auxiliary placeholder -> delta_aux = 0
x_hat_final = x_hat_main
```

## Install

For CUDA 12.4, install PyTorch with the official CUDA 12.4 wheel first:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

TaxiBJ conversion needs pandas:

```powershell
pip install -e ".[taxibj]"
```

## Quick Check

```powershell
cd v2
python scripts/smoke_test.py
python scripts/train.py --config configs/smoke.json --synthetic
```

## 常用训练指令

### 合成数据快速验证

```powershell
# smoke test — 跑一个 epoch 验证 pipeline 是否通
python scripts/train.py -c configs/smoke.json --synthetic

# 合成数据完整训练（默认 20 epoch, lr=1e-4）
python scripts/train.py -c configs/default.json --synthetic
```

### 真实数据训练

```powershell
# TaxiBJ 训练（100 epoch, lr=1e-3, 余弦退火）
python scripts/train.py -c configs/taxibj.json `
    --train_npz data/TaxiBJ/taxibj_train.npz `
    --val_npz data/TaxiBJ/taxibj_val.npz

# BikeNYC 训练
python scripts/train.py -c configs/bikenyc.json `
    --train_npz data/BikeNYC/bikenyc_train.npz `
    --val_npz data/BikeNYC/bikenyc_val.npz
```

### 指定训练名称和跳过绘图

```powershell
# 给训练起个名字，方便区分不同实验
python scripts/train.py -c configs/taxibj.json `
    --train_npz data/TaxiBJ/taxibj_train.npz `
    --val_npz data/TaxiBJ/taxibj_val.npz `
    -n my_experiment

# 跳过曲线图（比如在无 GUI 的服务器上）
python scripts/train.py -c configs/taxibj.json `
    --train_npz data/TaxiBJ/taxibj_train.npz `
    --val_npz data/TaxiBJ/taxibj_val.npz `
    --no_plot
```

### 消融实验

```powershell
# 用 --override_config 覆盖默认配置中的特定开关
python scripts/train.py -c configs/default.json `
    --override_config configs/ablation_no_router.json `
    --synthetic

python scripts/train.py -c configs/default.json `
    --override_config configs/ablation_fine_only.json `
    --synthetic
```

### 从 Checkpoint 恢复曲线图

```powershell
# 如果训练完忘了画图，可以从已保存的 checkpoint 中提取数据重新绘制
python scripts/plot_from_checkpoints.py outputs/TaxiBJ/run_20260527_170032
```

### 输出目录结构

```
outputs/{dataset_name}/{name}_{timestamp}/
├── config.json          # 本次训练使用的完整配置
├── training_curves.png  # loss / MAE 曲线图
├── checkpoints/
│   ├── last.pt          # 最新 epoch
│   ├── best.pt          # 最佳 val_mae
│   └── epoch_N.pt       # 每 save_every 个 epoch 保存
└── logs/
    ├── train.log        # 配置 + 模型参数 + 每 epoch 训练指标
    └── val.log          # 每 epoch 验证指标
```

### NPZ 数据格式

```text
x_f_gt or x_f: [N,C,T,H,W] or [N,T,H,W,C]
m_f optional: [N,1,T,H,W] or [N,T,H,W,1]
```

若 `m_f` 缺失，训练时会根据 `data.mask` 配置自动生成 mask。中/粗尺度输入始终由观测值经 masked pooling 构建。

## TaxiBJ Preparation

From `v2`:

```powershell
python scripts/prepare_taxibj_npz.py --data_dir ..\data\TaxiBJ --out data\taxibj_windows.npz --window 12 --stride 1
```

Then split the generated NPZ as needed, or train directly for a first run:

```powershell
python scripts/train.py --config configs/default.json --train_npz data\taxibj_windows.npz
```

## Ablations

All configs are JSON. Use `--override_config` to patch the default:

```powershell
python scripts/train.py --config configs/default.json --override_config configs/ablation_fine_only.json --synthetic
python scripts/train.py --config configs/default.json --override_config configs/ablation_no_router.json --synthetic
python scripts/train.py --config configs/default.json --override_config configs/ablation_fixed_scale_experts.json --synthetic
python scripts/train.py --config configs/default.json --override_config configs/ablation_no_cross_scale.json --synthetic
python scripts/train.py --config configs/default.json --override_config configs/ablation_routed_only.json --synthetic
python scripts/train.py --config configs/default.json --override_config configs/ablation_shared_only.json --synthetic
```

Key switches live under `model.main`:

- `use_multiscale`
- `use_router`
- `share_experts`
- `top_k`
- `use_routed_branch`
- `use_shared_branch`

Missing patterns live under `data.mask`: `fixed` or `random`. Both are generated offline as CSV files before training.

## Main Class Names

The public code names are intentionally short:

- `DualBranchSTImputer`: full two-branch imputation model.
- `MultiScaleMoEBackbone`: observation-aware multi-scale MoE backbone.
- `ScaleTokenEncoder`: per-scale value/mask/position encoder.
- `QualityRouter`: observation-quality-aware expert router.
- `TopKRoutedExpertPool`: sparse top-k routed expert bank.
- `CrossScaleSharedExpert`: parallel cross-scale fusion expert.
- `NullResidualBranch`: current auxiliary placeholder.

Older verbose names, plus `OAMSBackbone`, are kept as aliases for compatibility.
