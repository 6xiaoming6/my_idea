# 第12次改动：Frequency Multi-Resolution MoE

- 开发分支：v12-single
- 修改前提交：4a4933b
- 修改后提交：`a4bc2ff5cb48123729976c820c3611c03c80b475`
- 设计文档：`model_designs/v12-single.md`
- 独立输出目录：`outputs/v12-single/`

## 1. 修改目标

基于稳定 main 主干独立新增 v12-single，不继承、覆盖或删除 v7-v11 的代码和实验内容。模型保留多分辨率输入、MoE Router、共享/路由双分支和原训练损失，在每个 fine/mid/coarse 尺度内部增加稳定的趋势—残差伪频域分解：低频专家负责平滑趋势，高频专家负责局部细节与动态残差。

## 2. 实际结构

每个尺度执行：

1. `ScaleTokenEncoder` 生成尺度特征 `h_s`。
2. `FrequencyDecomposition` 使用 `3×3×3` 滑动平均得到 `h_low`，并计算 `h_high=h-h_low`。
3. 低频 Router 在 3 个趋势专家中 Top-1：
   - `SmoothTrendExpert`
   - `TemporalTrendExpert`
   - `CoarseContextExpert`
4. 高频 Router 在 3 个细节专家中 Top-1：
   - `LocalDetailExpert`
   - `DynamicDetailExpert`
   - `BoundaryExpert`
5. `FrequencyGate` 根据质量统计、低频能量和高频能量生成样本级 `g_high`：

   `z = z_low + sigmoid(eta) * g_high * z_high`

6. 三个尺度的频率特征继续进入原 `ProgressiveRouteFusion`，再与共享跨尺度分支做原有残差融合。

三尺度共享同一组低频/高频专家、Router 和频率门控参数；尺度 embedding 仍作为 Router 输入，用于区分 fine/mid/coarse。

## 3. 对设计文档的合理修正

### 3.1 复制边界代替零填充

文档伪代码直接使用带 padding 的 `AvgPool3d`。普通平均池化的零填充会在网格边缘人为降低趋势值，从而把边界差异错误识别为高频。实际实现先做 replicate padding，再做无 padding 平均池化，保证常量输入在边界仍得到零高频残差。

### 3.2 高频初始有效系数

主配置使用 `high_eta_init=-3`，即 `sigmoid(-3)=0.0474`；frequency gate 最后一层零初始化后 `g_high=0.5`，所以训练开始时高频有效系数约为 `0.0237`。这比完全关闭高频更利于高频专家获得梯度，同时仍保持低频主导。

### 3.3 high-only 消融

`high_only` 消融直接输出 `z_high`，不再乘小 eta。否则该消融测试的是“几乎为零的高频分支”，不能公平判断高频专家自身能力。

### 3.4 FFT 与回退

v12 第一阶段不实现 FFT。设置 `use_fft=true` 会明确报错，避免静默执行错误实验。`frequency_mode=none` 或 `main_fallback` 配置会回退原始 homogeneous Top-K 主干。

## 4. 新增代码

- `src/stmoe_imputer/models/v_single/frequency_decomposition.py`
  - `FrequencyDecomposition`
- `src/stmoe_imputer/models/v_single/frequency_experts.py`
  - 3 个低频专家
  - 3 个高频专家
  - `RoutedFrequencyExpertPool`
- `src/stmoe_imputer/models/v_single/v12_frequency_mr_moe.py`
  - `FrequencyGate`
  - `FrequencyMultiResolutionExpertPool`
- `src/stmoe_imputer/models/v_single/__init__.py`

所有频率专家采用残差结构，最后一层卷积零初始化。初始时低频专家近似输出 `h_low`，高频专家近似输出 `h_high`，不同随机 Router 选择不会在训练开始时造成大幅输出跳变。

## 5. 主干兼容修改

- `MultiScaleMoEBackbone` 新增 `expert_pool_type="frequency_mr"` 工厂路径。
- 默认 `expert_pool_type="homogeneous"`，原 main 行为保持不变。
- 频率主实验向原损失系统返回 6 维组合 gate：低频和高频 Router 各占总概率质量的 0.5。
- 低频和高频各 Top-1，组合 selected mask 每个样本共选择 2/6 个专家，可继续使用原 importance/load balance loss。
- 可学习 `high_eta_logit` 进入 scalar 优化器参数组。
- `scripts/run_experiments.py` 增加通用 `--model-config-dir` 和 `--full-name`，支持独立版本配置与消融输出命名。

## 6. 日志指标

训练、验证和测试日志新增每个尺度的：

- `frequency_{scale}_high_gate_mean/std`
- `frequency_{scale}_high_coefficient_mean/std`
- `frequency_{scale}_low_energy_mean/std`
- `frequency_{scale}_high_energy_mean/std`
- `frequency_{scale}_high_energy_ratio_mean/std`
- `frequency_{scale}_input_low_energy_mean/std`
- `frequency_{scale}_input_high_energy_mean/std`
- `frequency_{scale}_{low|high}_router_entropy`
- `frequency_{scale}_{low|high}_expert_{i}_weight`
- `frequency_{scale}_{low|high}_expert_{i}_usage`
- `frequency_eta_high`

这些指标用于判断高频是否被过度放大、不同数据集的频率能量差异，以及低频/高频专家是否出现路由塌缩。

## 7. 配置与脚本

### 7.1 正式配置

- `configs/v12-single/taxibj.json`
- `configs/v12-single/bikenyc.json`
- `configs/v12-single/chap.json`

正式配置均使用 avg-residual、3+3 专家、低/高频各 Top-1、可学习 eta 和三尺度模式。数据、优化器与原损失权重不变。

### 7.2 消融配置

- `main_fallback.json`：原 main homogeneous Top-K。
- `low_only.json`：只保留趋势分支。
- `high_only.json`：只保留细节分支。
- `eta_fixed_0.05.json`：固定高频全局系数为 0.05。

未创建可执行 `fft_rfft` 配置，因为 v12 第一阶段明确不实现 FFT，生成该配置会造成不可复现实验。

### 7.3 独立运行脚本

- `scripts/v12-single/train.py`：单任务或 `--ablation` 消融。
- `scripts/v12-single/run_validation_matrix.py`：5 个代表性点的 1/5 epoch 验证。
- `scripts/v12-single/run_full_experiments.py`：正式实验，固定先全部 fixed 后全部 random。

## 8. 验证结果

- Python 静态编译：通过。
- 全部 v12 JSON 解析：通过。
- 主实验及 4 个消融配置前向/反向：通过。
- `main_fallback`：确认进入原 homogeneous 4 专家 Top-2 路径。
- 5 点验证矩阵 dry-run：通过。
- 全量脚本 fixed→random 顺序 dry-run：通过。
- 合成数据完整 smoke：完成训练、验证、最佳检查点保存、重新加载最佳模型和最终测试。
- smoke 日志已确认包含高频门控、eta、低/高频能量、Router entropy 和专家 usage。

Smoke 输出：

`outputs/v12-single/unknown/debug/smoke_v12_single/random/rate0.45/20260711_103527_seed7_bs2/`

## 9. 运行命令

5 点 1 epoch 验证：

```bash
python scripts/v12-single/run_validation_matrix.py --gpu 0 --epochs 1
```

单个消融：

```bash
python scripts/v12-single/train.py \
  --dataset TaxiBJ \
  --mask-pattern fixed \
  --mask-rate 0.2 \
  --gpu 0 \
  --quick 5 \
  --ablation low_only
```

单卡完整实验，先 fixed 后 random：

```bash
python scripts/v12-single/run_full_experiments.py --gpu 0
```

所有正式输出进入 `outputs/v12-single/`，不会覆盖旧版本结果。
