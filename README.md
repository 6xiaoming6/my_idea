# v24-COE：Temporal-Spatial Chain-of-Experts

最新 TaxiBJ 五组对照使用项目根目录中的 `scripts/v24/run_abcde.py`，按 A→B→C→D→E 在单卡上顺序运行，每组 70 epoch。配置位于 `configs/v24/abcde_experiments.json`，详见 [五组实验说明](scripts/v24/README_ABCDE.md)。`outputs/` 下的旧工作副本仅保留历史记录。

```bash
# 在项目根目录执行；默认使用 GPU 0，已有 GPU 计算任务时拒绝启动。
python scripts/v24/run_abcde.py --gpu 0
# 只检查计划，不启动训练：
python scripts/v24/run_abcde.py --dry-run
```

当前后续实验统一为 20 epoch、train/val/test 九类同分布混合 mask，共六组（硬路由基准、预热、分组输入、专家身份传递、固定路径、两层三专家），配置与运行顺序位于
[后续实验说明](scripts/v24/README_FOLLOWUP.md)，统一从 `scripts/v24/run_followup.py` 启动，仍按单卡顺序运行。

当前分支新增单尺度 TS-CoE：两轮共享时间/空间专家池，根据观测支撑和更新后的补全状态逐轮路由。第一版按方案 10.4/10.5，仅以隐藏目标 MAE 训练，`lambda_coe_balance=0`、`lambda_coe_mid=0`；保留 ST Gumbel-Softmax 梯度估计和轻量点级共享专家。先监控专家使用率、路径、梯度及缺失条件下的选择，不要求时间/空间专家平均分工；不构造中/粗尺度。

现另提供 **3 轮 / 4 类专家**与 **4 轮 / 6 类专家**候选，增加空洞时间、空洞空间、完整窗口时间注意力和局部时空联合专家，并提供轮数 × 专家池大小的九组消融。所有候选保持原始分辨率，详见 [深链候选说明](changes/v24_COE_deeper_candidates.md)。

```bash
python scripts/train.py -c configs/v24/smoke.json --synthetic --no_plot -n v24_smoke

# 3 轮候选；替换 chain4.json / 运行名即可运行 4 轮候选
python scripts/train.py -c configs/v24/smoke.json \
  --override_config configs/v24/candidates/chain3.json \
  --synthetic --no_plot -n v24_chain3_smoke
```

完整的可行性评价、方法边界、真实数据训练命令及消融配置见 [v24 实现说明](changes/v24_COE_implementation.md)。核心代码为 [temporal_spatial_coe.py](src/stmoe_imputer/models/temporal_spatial_coe.py)，设计来源为 [详细方案](model_designs/v24_Temporal-Spatial_Chain-of-Experts_详细方案.md)。

按方案第 13 节整理的顺序、重路由、状态传递与深度对照，及共享缺失 mask、离线路径干预和运行清单生成方法见 [v24 实验设计](changes/v24_COE_experiment_design.md)。第一版 `pilot/core` 保持无路由正则；新一轮默认运行 **4 轮、6 个路由专家、路由均衡损失权重 0.01**，配置为 [chain4_balance.json](configs/v24/experiments/chain4_balance.json)。继续使用单尺度、单种子 `7`、`80 epoch`；轮数 × 专家池网格暂缓。

v24 实验管理沿用 v23 的流程：每 2 轮完整验证，默认在 CPU 内存保留最佳权重，最后恢复它完整测试一次；重复运行跳过已完整完成的任务，未完成任务从头重跑。配置在 [experiments.json](configs/v24/experiments.json)，终端只显示训练进度条。

```bash
# 四轮六专家 + 路由损失，TaxiBJ 五种缺失结构各一组；只查看计划时加 --dry-run
python scripts/v24/run_experiments.py --study chain4 --gpu 0

# 原两轮无正则对照，显式选择 study
python scripts/v24/run_experiments.py --study pilot --gpu 0
python scripts/v24/run_experiments.py --study core --gpu 0
```

结果位于 `outputs/v24-COE/experiments/<chain4或pilot或core>/<指纹>/`，包含 `summary.csv`、`comparison.json`、`diagnostics.json` 及各任务完整日志。代码、数据或配置改变时使用新的指纹目录；旧 `section13/*/commands.sh` 不会自动升级，也不自动复用旧框架结果。新生成的计划命令会调用这一统一队列。需要离线路径分析时，将实验配置的 `training.save_best_checkpoint` 设为 `true`，每个任务只保留一个最佳 checkpoint。

下面保留原 v6 / main 基线说明；运行 v24 请使用 `configs/v24/`，原默认配置仍对应旧模型。

---

## 原 v6 基线

面向时空网格数据补全任务的 PyTorch 项目（第 6 版）。核心模型 `DualBranchSTImputer` 以 `MultiScaleMoEBackbone` 为骨干，在 TaxiBJ（出租车流量）、BikeNYC（共享单车）、CHAP（PM2.5 浓度）三个数据集上，评估 fixed / random 两种离线缺失掩码策略下的补全性能。

关键设计：多尺度表示（fine/mid/coarse）、质量感知稀疏路由（QualityRouter + TopKRoutedExpertPool）、可靠性感知跨尺度共享专家（GatedCrossScaleSharedExpert + ReliabilityAwareScaleGate）、共享-路由双分支残差融合（SharedRoutedResidualFusion）。模型仅以观测值作为输入，缺失位置真值不参与前向计算。

> **分支说明**：`main` 分支是项目的核心基线，包含经过完整实验验证的模型结构和训练流程。其他分支均在 `main` 的基础上尝试修改或优化模型结构，属于实验性探索，不代表最终方案。
>
> **分支命名规则**：
> - `single-v{i}` — 单分支架构第 i 个版本，不包含多模态辅助分支（`aux.enabled=false`）
> - `dual-v{j}` — 双分支架构第 j 个版本，在单分支基础上增加多模态辅助分支（`aux.enabled=true`）

---

## 项目结构

```
my_idea/
├── src/stmoe_imputer/          # 核心模型与训练源码
├── configs/                    # 训练配置
│   ├── datasets/               #   TaxiBJ / BikeNYC / CHAP 数据集配置
│   ├── presets/                #   合成数据默认 + smoke test 配置
│   └── policies/               #   训练策略（epochs、batch 等）
├── scripts/                    # 训练、评估、数据处理脚本
├── data/                       # 数据集与离线 mask（不提交 Git）
├── outputs/                    # 训练输出与实验索引（不提交 Git）
├── experments_report/          # 实验分析报告
├── model_designs/              # 模型设计演进文档
├── changes/                    # 代码结构改动记录
└── README.md
```

---

## 模型架构

### 总体数据流

```
NPZ 数据 → x_f_gt [B,C,T,H,W], m_f [B,1,T,H,W]

数据预处理 (transforms.py):
  x_f_obs = x_f_gt * m_f                              ← 仅观测值可见
  x_m_obs, m_m, r_m = masked_pool2d(x_f_obs, m_f, 2)  → [B,C,T,H/2,W/2]
  x_c_obs, m_c, r_c = masked_pool2d(x_m_obs, m_m, 2)  → [B,C,T,H/4,W/4]
  q_f, q_m, q_c = compute_observation_stats(m)         → [B,5] each

模型前向 (imputer.py → main_branch.py):

  x_f_obs,m_f        x_m_obs,m_m        x_c_obs,m_c
      │                   │                   │
  ┌───▼────┐         ┌───▼────┐         ┌───▼────┐
  │Embed_F │         │Embed_M │         │Embed_C │    ScaleTokenEncoder
  └───┬────┘         └───┬────┘         └───┬────┘      value+mask+scale+time+space
  h_f [B,64,T,H,W]  h_m [B,64,T,H/2,W/2] h_c [B,64,T,H/4,W/4]
      │                   │                   │
      ├───────────────────┼───────────────────┤
      │                   │                   │
  ┌───▼────┐         ┌───▼────┐         ┌───▼────┐
  │Router_F│         │Router_M│         │Router_C│    QualityRouter
  └───┬────┘         └───┬────┘         └───┬────┘     MLP(h_pool|q|scale_embed)
 gate_f[B,4]       gate_m[B,4]       gate_c[B,4]
      │                   │                   │
      └────────┬──────────┴──────────┬────────┘
               │                     │
        ┌──────▼──────┐              │
        │ ExpertPool  │ (4 experts,  │              TopKRoutedExpertPool
        │ top_k=2     │  3尺度共享)   │              STExpert = Conv3d+GELU+ResBlock
        └──────┬──────┘              │
      z_f,z_m,z_c                    │
               │                     │
        ┌──────▼──────┐              │
        │ Progressive │              │              c→m→f 渐进上采样+门控融合
        │ RouteFusion │              │
        └──────┬──────┘              │
          h_route                    │
               │                     │
               ├─────────────────────┤
               │                     │
        ┌──────▼──────────────────────▼──────┐
        │  GatedCrossScaleSharedExpert      │       可靠性感知尺度门控
        │  ├─ ReliabilityAwareScaleGate      │       MLP(209→128→3)→softmax
        │  └─ Conv1x1+2×ResBlock(concat)    │       加权融合 h_f,h_m_up,h_c_up
        └──────┬────────────────────────────┘
               │
          z_shared
               │
        ┌──────▼──────────────────────┐
        │  SharedRoutedResidualFusion │             双分支残差融合
        │  z_shared → 2×ResBlock → h_shared       │
        │  h_route → Conv+ResBlock → h_route_proj │  (+Dropout3d 0.1)
        │  h_main = h_shared + γ·h_route_proj     │  γ = sigmoid(trainable)
        └──────┬──────────────────────┘
               │
          h_main [B,64,T,H,W]
               │
        ┌──────┼──────────┬──────────┐
        ▼      ▼          ▼          ▼
    pred_head  shared_aux_head  route_aux_head      Conv3d×2
        │          │              │
  x_hat_main  x_hat_shared  x_hat_route
  [B,C,T,H,W]
```

### 关键模块

**ScaleTokenEncoder** — 多尺度时空嵌入
- `value_embed(x)` + `mask_embed(m)` + `scale_embed` + `time_embed` + `space_embed`
- 每个尺度独立参数，将 [B,C,T,H,W] 映射到 [B,64,T,H,W]

**QualityRouter** — 质量感知路由
- 输入：token 空间池化 [B,64] + 观测统计 q [B,5] + 尺度嵌入 [B,64]
- 输出：softmax(gate) [B,num_experts]
- `compute_observation_stats(m)` 返回 5 维统计量（缺失率、观测率、时间缺失分数、空间缺失分数、聚合可靠性）

**TopKRoutedExpertPool** — Top-K 稀疏专家池
- 4 个 STExpert（Conv3d→GroupNorm→GELU→ResidualSTBlock），3 尺度共享
- `top_k=2`：每样本激活 2/4 专家，加权组合输出

**ProgressiveRouteFusion** — 渐进路由融合
- Coarse(8×8) → Mid(16×16) → Fine(32×32) 逐级上采样
- 每级用 GatedFusion2 学习逐位置门控权重

**GatedCrossScaleSharedExpert** — 跨尺度共享专家
- `ReliabilityAwareScaleGate`：MLP(209→128→3) → softmax，综合 3 尺度特征+观测统计+可靠性评分，动态输出 [w_f, w_m, w_c]
- 加权拼接后经 Conv1x1+2×ResidualSTBlock → z_shared
- 默认 `shared_input_mode="pre"`：接收原始嵌入（非专家输出），与 Routed 分支互补

**SharedRoutedResidualFusion** — 双分支残差融合
- Shared：z_shared → 2×ResidualSTBlock → h_shared
- Routed：h_route → Conv3d(k1)+ResidualSTBlock+Dropout3d(0.1) → h_route_proj
- Fusion：`h_main = h_shared + sigmoid(γ) · h_route_proj`（γ 初始 sigmoid(-3)≈0.047，可学习）

### 损失函数

```python
L = SmoothL1(x_hat_main, x_gt)           # 主损失（仅 hidden 位置）
  + 0.10 × SmoothL1(x_hat_pooled, x_obs) # 跨尺度观测损失（mid+coarse）
  + 0.01 × Σ(gate_mean - 1/E)²          # 专家重要性均衡
  + 0.01 × Σ(load_mean - avg_load)²      # 专家负载均衡
  + 0.05 × SmoothL1(x_hat_shared, x_gt)  # 共享分支辅助
  + 0.10 × SmoothL1(x_hat_route, x_gt)   # 路由分支辅助
  + 0.003 × cos²(h_shared, h_route_proj) # 特征互补约束
```

### 默认超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| dim | 64 | 隐藏维度 |
| num_experts | 4 | 专家数（3 尺度共享） |
| top_k | 2 | 每 token 激活专家数 |
| c_in | 2 / 1 | TaxiBJ=2(in/out), BikeNYC/CHAP=1 |
| routing_mode | topk | 稀疏路由 |
| shared_input_mode | pre | 共享分支接收原始嵌入 |
| branch_fusion_mode | residual | h_shared + γ·h_route_proj |
| scale_mode | fine_mid_coarse | 三尺度全开 |
| route_gamma_init | -3.0 | γ 初始≈0.047 |
| route_dropout | 0.1 | 路由分支 Dropout3d |
| aux_branch | 关闭 | NullResidualBranch |

---

## 缺失掩码

支持 `fixed` 和 `random` 两种离线掩码，由 CSV 文件提供。

**fixed**：同一缺失率下所有样本共享同一个空间 mask，train/val/test 使用相同掩码。
**random**：每个样本有独立空间 mask，train/val/test 使用不同 seed 偏移生成。

```text
data/{dataset}/{fixed,random}_mask/{rate}/
├── train.csv    # fixed: 1×N | random: N_train×N
├── val.csv      # fixed: 1×N | random: N_val×N
└── test.csv     # fixed: 1×N | random: N_test×N
```

---

## 快速开始

```bash
# 安装
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .

# Smoke test（合成数据，快速验证前向+loss）
python scripts/train.py -c configs/presets/smoke.json --synthetic
```

---

## 真实数据训练

### 统一调度器（推荐）

```bash
# 单数据集、单模式
python scripts/run_experiments.py --dataset TaxiBJ --gpu 0 --mask-pattern fixed --mask-rate 0.4

# 全部数据集、全部模式和缺失率
python scripts/run_experiments.py --dataset all --gpu 0 --mask-pattern all --mask-rate all

# 使用训练策略（控制 epochs、batch、早停等）
python scripts/run_experiments.py \
  --dataset all --gpu 0 --mask-pattern all --mask-rate all \
  --experiments full \
  --training-policy configs/policies/full_model_paper.json
```

参数 `all` 展开：`--dataset all` → TaxiBJ, BikeNYC, CHAP；`--mask-pattern all` → fixed, random；`--mask-rate all` → 0.2, 0.4, 0.6, 0.8。

### 单次训练

```bash
python scripts/train.py \
  -c configs/datasets/taxibj.json \
  --train_npz data/TaxiBJ/taxibj_train.npz \
  --val_npz data/TaxiBJ/taxibj_val.npz \
  --test_npz data/TaxiBJ/taxibj_test.npz \
  -n my_experiment
```

配置中需包含离线 mask 路径（调度器自动生成）。合成数据不需 CSV：

```bash
python scripts/train.py -c configs/presets/default.json --synthetic
```

---

## 输出结构

```
outputs/{dataset}/{experiment_type}/{variant}/{mask}/rate{rate}/{timestamp}_seed{seed}_bs{bs}/
├── config.json
├── checkpoints/
│   └── best.pt
├── logs/
│   ├── train.log
│   ├── val.log
│   ├── test.log
│   └── metrics.jsonl
└── training_curves.png
```

`--name` 自动归类：`full` → `full/model`，`ablation_*` → `ablation/*`，`smoke_*` → `debug/*`。

训练终端只显示带 `loss / mae / rmse` 的训练进度条；每轮摘要和测试结果保存在 `logs/`，完整指标（含专家使用率、路径、梯度和缺失条件诊断）保存在 `metrics.jsonl`。无需额外参数，原有 `--quiet` 仍可兼容使用。

汇总索引：`outputs/summary/experiment_index.csv` 记录每次训练的 run_dir、数据集、mask、缺失率、best epoch、best val MAE、耗时、显存等。

---

## 数据格式

NPZ 文件需包含：`x_f_gt` 或 `x_f` [N,C,T,H,W] 或 [N,T,H,W,C]。可选：`m_f`, `x_m_obs/m_m`, `x_c_obs/m_c`, `r_m/r_c`。

中粗尺度若未预存，`ensure_multiscale()` 自动从 fine 观测值通过 masked pooling 构造。

---

## 源码结构

```
src/stmoe_imputer/
├── data/
│   ├── npz_dataset.py     # NPZ 数据集加载 + 离线 mask CSV
│   ├── transforms.py      # masked_pool2d_spatial, ensure_multiscale, ensure_observed
│   ├── masks.py           # mask 生成与转换
│   ├── synthetic.py       # 合成数据集
│   └── build.py           # Dataset/DataLoader 构建
├── models/
│   ├── imputer.py         # DualBranchSTImputer（顶层封装）
│   ├── main_branch.py     # MultiScaleMoEBackbone（核心骨干，forward 编排）
│   ├── embedding.py       # ScaleTokenEncoder
│   ├── router.py          # QualityRouter
│   ├── experts.py         # STExpert, TopKRoutedExpertPool
│   ├── fusion.py          # ProgressiveRouteFusion, GatedCrossScaleSharedExpert,
│   │                        SharedRoutedResidualFusion, ReliabilityAwareScaleGate,
│   │                        AdaptiveBranchGate, ExpertEnhancedSharedInput
│   ├── blocks.py          # ResidualSTBlock
│   ├── stats.py           # compute_observation_stats
│   ├── scale_utils.py     # build_scale_active_mask
│   └── aux_branch.py      # NullResidualBranch
├── engine.py              # train_one_epoch, evaluate, build_optimizer/scheduler
├── losses.py              # compute_main_stage_loss, masked_loss, cross_scale_loss
├── metrics.py             # masked_metrics (MAE/RMSE/MAPE)
├── config.py              # 配置加载与深度合并
└── utils/
    ├── checkpoint.py      # save/load checkpoint
    ├── device.py          # get_device
    ├── seed.py            # set_seed
    └── train_logger.py    # TrainLogger（epoch/test 日志 + metrics.jsonl）
```

---

## 常用类名速查

| 类名 | 职责 |
|------|------|
| `DualBranchSTImputer` | 顶层模型，组合主分支+辅助分支 |
| `MultiScaleMoEBackbone` | 多尺度 MoE 骨干，编排完整前向 |
| `ScaleTokenEncoder` | 单尺度时空嵌入 |
| `QualityRouter` | 质量感知专家路由 |
| `TopKRoutedExpertPool` | Top-K 稀疏专家池 |
| `STExpert` | 单个专家（Conv3d+ResBlock） |
| `GatedCrossScaleSharedExpert` | 跨尺度共享专家+可靠性门控 |
| `ProgressiveRouteFusion` | 渐进路由融合 |
| `SharedRoutedResidualFusion` | 共享-路由残差融合 |
| `ReliabilityAwareScaleGate` | 可靠性感知尺度门控 |
| `ExpertEnhancedSharedInput` | 专家增强共享输入适配器 |
| `ResidualSTBlock` | 时空残差块（Conv3d×2+GroupNorm+GELU） |
