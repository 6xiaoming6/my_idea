# STMoE Imputer 5.0

这是一个面向时空网格流量数据补全任务的 PyTorch 项目。当前版本聚焦于 TaxiBJ 与 BikeNYC 两个数据集，在 fixed / random 两种离线缺失掩码策略下，评估多尺度、动态路由和双分支融合结构对缺失补全的作用。

项目当前主模型是 `DualBranchSTImputer`，核心骨干是 `MultiScaleMoEBackbone`。模型不会直接使用缺失位置真值作为输入；真实数据训练时，fine 观测输入由完整真值和离线 mask 相乘得到，mid / coarse 输入由 fine 观测值通过 masked pooling 构造。

## 当前模型结构

整体流程如下：

```text
完整时空窗口 x_f_gt
  + 离线 mask m_f
  -> fine 观测 x_f_obs
  -> masked pooling 得到 mid / coarse 观测

fine / mid / coarse 观测 + mask
  -> ScaleTokenEncoder
  -> 质量感知路由 QualityRouter
  -> Top-k Routed Expert Branch
  -> Cross-scale Shared Expert Branch
  -> Shared/Routed 分支融合
  -> 预测缺失位置 x_hat_final
```

### 多尺度输入

模型使用三种空间尺度：

- `fine`：原始网格分辨率。
- `mid`：由 fine 观测值按 `fine_to_mid=2` 做 masked pooling 得到。
- `coarse`：由 mid 继续 pooling 得到，对应 `fine_to_coarse=4`。

中粗尺度只从观测值构造，不从完整真值构造，因此不会泄漏缺失区域答案。

### ScaleTokenEncoder

每个尺度都有独立的编码器，将以下信息编码到同一隐藏维度：

- 流量值特征。
- mask 特征。
- 时间 / 空间位置嵌入。
- 尺度嵌入。

### Routed Expert Branch

路由分支包含：

- `QualityRouter`：根据当前尺度特征、观测质量统计和尺度嵌入生成专家权重。
- `TopKRoutedExpertPool`：对每个尺度选择 top-k 专家，得到 `z_f / z_m / z_c`。
- `ProgressiveRouteFusion`：将 coarse -> mid -> fine 逐级上采样并门控融合。

该分支主要学习尺度内和局部缺失模式相关的补全特征。

### Cross-scale Shared Expert Branch

共享分支包含：

- `ExpertEnhancedSharedInput`：控制共享分支是否使用 routed expert 输出增强输入。
- `GatedCrossScaleSharedExpert`：将 fine / mid / coarse 对齐到 fine 尺度。
- `ReliabilityAwareScaleGate`：结合各尺度观测统计和 reliability，动态调整尺度权重。

该分支主要学习跨尺度共享上下文和全局补全信息。

### 双分支融合

当 shared branch 和 routed branch 同时启用时，模型使用 `SharedRoutedResidualFusion` 融合两条分支。当前默认配置中：

- TaxiBJ 使用 `scale_mode=fine_mid`。
- BikeNYC 使用 `scale_mode=fine_mid_coarse`。
- `aux.enabled=false`，辅助分支目前是占位的 `NullResidualBranch`。
- 训练损失主要在缺失位置计算，同时包含跨尺度一致性、专家均衡、分支辅助和互补约束等项。

## 缺失掩码策略

当前只保留两种缺失模式：`fixed` 和 `random`。两者都离线生成 CSV，训练时只读取，不在线随机生成。

### fixed

同一数据集、同一缺失率下只生成一个空间 mask：

```text
data/TaxiBJ/fixed_mask/0.4/train.csv  -> 1 x H*W
data/TaxiBJ/fixed_mask/0.4/val.csv    -> 1 x H*W
data/TaxiBJ/fixed_mask/0.4/test.csv   -> 1 x H*W
```

加载时会广播到所有样本和所有时间步。同一数据集内部，train / val / test 使用相同 fixed mask。

### random

每个样本有独立空间 mask：

```text
data/TaxiBJ/random_mask/0.4/train.csv -> N_train x H*W
data/TaxiBJ/random_mask/0.4/val.csv   -> N_val x H*W
data/TaxiBJ/random_mask/0.4/test.csv  -> N_test x H*W
```

同一样本跨 epoch 不变，不同样本之间不同。random 的 train / val / test 使用不同 seed 偏移生成，避免不同 split 之间同索引样本复用相同 mask。

## 项目目录结构

```text
.
├── src/stmoe_imputer/       # 核心源码
├── configs/                 # 训练配置
├── scripts/                 # 当前主脚本入口
├── scripts/archive/         # 历史脚本和临时续跑脚本
├── data/                    # 本地数据集和离线 mask，不提交到 GitHub
├── outputs/                 # 训练输出，不提交到 GitHub
├── model_designs/           # 模型设计文档
├── changes/                 # 每次结构修改记录
├── experments_report/       # 实验分析报告
├── baselines/               # baseline 调研文档
└── README.md
```

### 源码目录

```text
src/stmoe_imputer/
├── data/
│   ├── masks.py             # fixed / random mask 生成与转换
│   ├── npz_dataset.py       # NPZ 数据集读取
│   ├── transforms.py        # masked pooling 与多尺度构造
│   ├── synthetic.py         # 合成数据集
│   └── build.py             # dataset / dataloader 构建
├── models/
│   ├── imputer.py           # DualBranchSTImputer
│   ├── main_branch.py       # MultiScaleMoEBackbone
│   ├── embedding.py         # ScaleTokenEncoder
│   ├── router.py            # QualityRouter
│   ├── experts.py           # Top-k expert pool
│   ├── fusion.py            # 跨尺度融合与双分支融合
│   └── stats.py             # 观测质量统计
├── engine.py                # train / evaluate loop
├── losses.py                # 补全损失和辅助损失
├── metrics.py               # MAE / RMSE / MAPE
├── config.py                # 配置读取与合并
└── utils/                   # seed、device、checkpoint、logger
```

### 配置目录

```text
configs/
├── datasets/
│   ├── taxibj.json          # TaxiBJ 默认训练配置
│   └── bikenyc.json         # BikeNYC 默认训练配置
├── presets/
│   ├── default.json         # 合成数据默认配置
│   └── smoke.json           # 快速 smoke test 配置
└── ablations/
    ├── ablation_fine_only.json
    ├── ablation_no_router.json
    ├── ablation_fixed_scale_experts.json
    ├── ablation_no_cross_scale.json
    ├── ablation_routed_only.json
    └── ablation_shared_only.json
```

`configs/ablations/` 中还有一些历史或扩展消融配置，当前批量脚本默认使用 6 个核心消融。

### 脚本目录

当前推荐使用的脚本保留在 `scripts/` 顶层：

```text
scripts/
├── train.py                 # 单次训练入口
├── evaluate.py              # checkpoint 评估入口
├── smoke_test.py            # 前向与 loss 快速检查
├── generate_fixed_masks.py  # fixed/random 离线 mask 生成
├── run_all_ablations.sh     # 单数据集/单缺失率 Full + 6 组消融
├── run_all_rates.sh         # 多缺失率、多 mask 模式批量训练
├── prepare_taxibj_npz.py    # TaxiBJ 原始数据转 NPZ
└── plot_from_checkpoints.py # 从 checkpoint 恢复训练曲线
```

`scripts/archive/` 只保留历史脚本和临时续跑脚本，新实验优先使用顶层脚本。

### 文档目录

```text
model_designs/               # 从 design_1 到 design_v2-5 的模型设计演进
changes/                     # 每次代码结构改动的说明
experments_report/           # 实验结果分析报告
baselines/                   # baseline 和相关方法调研
```

当前 mask 策略说明在：

```text
model_designs/fixed和random缺失模式策略.md
```

## 安装

建议先安装匹配 CUDA 的 PyTorch，再安装本项目：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

TaxiBJ 原始数据转换依赖 pandas：

```bash
pip install -e ".[taxibj]"
```

## 快速检查

```bash
python scripts/smoke_test.py
python scripts/train.py -c configs/presets/smoke.json --synthetic
```

## 真实数据训练

真实数据训练推荐使用批量脚本，因为它会自动生成 fixed/random 离线 mask，并把 `train_csv` / `val_csv` 写入临时 override 配置。

单数据集、单缺失率：

```bash
bash scripts/run_all_ablations.sh --dataset TaxiBJ --gpu 0 --mask_pattern random --mask_rate 0.4
bash scripts/run_all_ablations.sh --dataset BikeNYC --gpu 1 --mask_pattern fixed --mask_rate 0.4
```

全部缺失率：

```bash
bash scripts/run_all_rates.sh --dataset TaxiBJ --gpu 0 --mask_pattern random
bash scripts/run_all_rates.sh --dataset BikeNYC --gpu 1 --mask_pattern fixed
bash scripts/run_all_rates.sh --dataset all --gpu 0 --mask_pattern all
```

`run_all_rates.sh` 默认缺失率为：

```text
0.2, 0.4, 0.6, 0.8
```

## 单次训练

如果直接调用 `train.py` 训练真实数据，配置中必须包含离线 mask 路径：

```json
{
  "data": {
    "mask": {
      "pattern": "random",
      "missing_rate": 0.4,
      "train_csv": "data/TaxiBJ/random_mask/0.4/train.csv",
      "val_csv": "data/TaxiBJ/random_mask/0.4/val.csv"
    }
  }
}
```

示例：

```bash
python scripts/train.py \
  -c configs/datasets/taxibj.json \
  --override_config /path/to/mask_override.json \
  --train_npz data/TaxiBJ/taxibj_train.npz \
  --val_npz data/TaxiBJ/taxibj_val.npz \
  -n my_experiment \
  --no_plot
```

合成数据不需要 CSV：

```bash
python scripts/train.py -c configs/presets/default.json --synthetic
```

## 数据格式

NPZ 文件至少需要包含：

```text
x_f_gt or x_f: [N, C, T, H, W] 或 [N, T, H, W, C]
```

可选字段：

```text
m_f:     [N, 1, T, H, W]
x_m_obs, m_m
x_c_obs, m_c
r_m, r_c
```

如果 NPZ 中没有 `m_f`，真实数据训练必须通过 CSV 提供 mask。中尺度和粗尺度若没有预先存储，会由 `ensure_multiscale()` 根据 fine 观测值和 mask 自动构造。

## 输出目录

训练输出默认保存在：

```text
outputs/{dataset_name}/{run_name}_{mask_pattern}{rate}_{timestamp}/
├── config.json
├── logs/
│   ├── train.log
│   └── val.log
└── training_curves.png
```

当前 `train.py` 中 checkpoint 保存默认处于关闭状态，需要时可在 `scripts/train.py` 中重新启用相关代码。

## Git 与数据管理

`.gitignore` 只忽略根目录数据和输出：

```text
/data/
/outputs/
/runs/
/logs/
```

因此大数据文件、mask CSV、训练输出不会上传到 GitHub；而 `src/stmoe_imputer/data/` 是源码目录，会被 Git 正常管理。

## 常用类名

- `DualBranchSTImputer`：完整补全模型。
- `MultiScaleMoEBackbone`：多尺度 MoE 主干。
- `ScaleTokenEncoder`：每个尺度的值、mask、位置编码器。
- `QualityRouter`：观测质量感知专家路由器。
- `TopKRoutedExpertPool`：top-k 动态专家池。
- `GatedCrossScaleSharedExpert`：跨尺度共享专家。
- `ProgressiveRouteFusion`：routed 分支的逐级尺度融合。
- `SharedRoutedResidualFusion`：shared / routed 双分支融合。
