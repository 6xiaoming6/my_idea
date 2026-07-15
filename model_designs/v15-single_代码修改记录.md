# v15-single 代码修改记录

## 1. 版本信息

- 版本：第15版（`v15-single`）
- 模型名称：CBRP-MoE（Compact Base-Anchored Residual Pyramid Mixture-of-Experts）
- 开发分支：`v15-single`
- 开发基础：V14 的 main 多分辨率 MoE Backbone
- 修改前提交：`4544e22`
- 修改后提交：尚未提交
- 输出根目录：`outputs/v15-single`

本次实现依据：

```text
model_designs/v15-single_有效简洁的基线锚定残差金字塔MoE详细修改说明.md
```

V14 的代码、配置、脚本和实验输出均被保留，没有覆盖或删除。

## 2. 实际结构修改

V15 完整复用 main Backbone，只新增两个轻量模块：

```text
main MultiScaleMoEBackbone
  -> x_base, z_f, z_m, z_c, h_main, scale_gate

CompactResidualPyramid(z_c -> z_m -> z_f + h_main)
  -> delta_raw

ResidualBudgetController(8维观测条件)
  -> beta

x_final = x_base + beta * scale_ref * tanh(delta_raw)
```

其中：

- `scale_ref` 是从 `x_base` detach 后按样本、按通道计算的 RMS；
- `beta` 的范围为 `[0, beta_max]`；
- `tanh(delta_raw)` 的范围为 `[-1, 1]`；
- 因而逐元素满足 `|effective_delta| <= beta_max * scale_ref`；
- Residual Head 最后一层零初始化，模型初始化时严格满足 `x_final == x_base`。

V15 不再调用 V14 的 DifficultyConditionEncoder、Geometry、ObservedConsistency、三层 alpha、完整值 C2F 重建或 CorrectionAdapter。

以 TaxiBJ 配置统计参数量：

| 模型 | 总参数量 | 相对 main 新增参数 |
|---|---:|---:|
| main | 4,682,755 | 0 |
| V14 | 4,947,169 | 264,415 |
| V15 | 5,428,455 | 745,701 |

V15 的“Compact”指更少的预测路径、门控和中间语义，而不是参数量小于 V14。三个 `ResidualSTBlock(64)` 使其参数量高于 V14，但仍未复制 main Encoder、Router 或 ExpertPool；正式实验后可再决定是否做 bottleneck/depthwise 轻量化消融，第一版不偏离设计文档规定的结构。

## 3. Loss 修改

保留 main 原有损失，并为 V15 增加：

```text
0.50 * L_v15_base
+ 0.10 * L_v15_delta
+ 0.10 * L_v15_safe
```

- `L_v15_base`：保持 `x_base` 独立补全能力；
- `L_v15_delta`：直接监督 `effective_delta` 拟合 `target-x_base.detach()`；
- `L_v15_safe`：按样本惩罚 final MAE 高于 base MAE 的情况。

V14/V15 loss 通过 `v14_enabled`、`v15_enabled` 显式区分。V15 输出 `x_hat_base` 时不会误触发 V14 的 mid/coarse/regret/gate loss。

## 4. 训练与日志

独立配置：

```text
configs/v15-single/taxibj.json
configs/v15-single/bikenyc.json
configs/v15-single/chap.json
configs/v15-single/smoke.json
```

核心消融配置位于 `configs/v15-single/ablations/`，已经支持：

```text
no_pyramid
fixed_budget
unbounded_residual
no_base_loss
no_delta_loss
no_safety_loss
```

正式训练预算与 V14 对齐：

| 数据集 | Epoch | val_epoch | lr_main | lr_v15 |
|---|---:|---:|---:|---:|
| TaxiBJ | 160 | 5 | 0.001 | 0.001 |
| BikeNYC | 140 | 2 | 0.001 | 0.001 |
| CHAP | 150 | 5 | 0.001 | 0.001 |

训练仍采用定期验证、仅覆盖保存最佳 `best.pt`、训练结束加载最佳权重并测试一次的统一流程。

新增诊断包括：

```text
v15_beta_mean/std/min/max
v15_scale_ref_mean
v15_raw_delta_rms_mean
v15_direction_rms_mean
v15_effective_delta_rms_mean
v15_effective_relative_rms_mean
v15_base_hidden_mae
v15_final_hidden_mae
v15_final_vs_base_improvement
v15_sample_non_regression_violation_rate
```

`effective_relative_rms` 实际按 `RMS(effective_delta / scale_ref)` 计算，而不是 `RMS(effective_delta) / mean(scale_ref)`。前者在多通道尺度不一致时仍能严格验证其不超过 `beta_max`。

## 5. 训练入口

单点训练：

```bash
python scripts/v15-single/train.py \
  --dataset TaxiBJ \
  --mask fixed \
  --rate 0.2 \
  --gpu 0
```

单点消融通过 `--ablation` 选择，例如：

```bash
python scripts/v15-single/train.py \
  --dataset BikeNYC \
  --mask fixed \
  --rate 0.6 \
  --gpu 0 \
  --ablation fixed_budget
```

六点验证矩阵：

```bash
python scripts/v15-single/run_validation_matrix.py --gpu 0 --epochs 5
```

24 点完整实验（固定缺失先跑，随机缺失后跑）：

```bash
python scripts/v15-single/run_full_experiments.py --gpu 0
```

不写 `--dry-run` 时执行真实训练。

该入口会自动审计每个正式组合。只有同时满足“正式 epoch 跑满、存在非空 `best.pt`、存在非空 `test.log`、`metrics.jsonl` 中包含有限 test MAE”的结果才会被标记为完成并跳过；失败、中断、只跑部分 epoch 或尚未测试的组合会重新执行。使用 `--force-rerun` 可以显式重跑已经完成的组合。

## 6. 验证结果

已通过标准库 `unittest` 的 18 项 V14/V15 回归测试，其中 V15 测试 10 项：

- 三个数据集输入/输出 shape；
- V15 关闭时严格等价 main；
- Residual Head 零初始化时严格等价 main；
- 有效残差逐元素上界；
- hidden target 无 Forward 泄漏；
- V15 全部新模块梯度有限且可达；
- 大半精度数值下 RMS/scale 统计有限；
- 三套正式配置与 Registry 正确；
- No Pyramid 与 Fixed Budget 核心消融开关可执行；
- V14/main Registry 和旧测试保持兼容。

同时使用三套正式配置和完整维度分别完成了 TaxiBJ `[1,2,12,32,32]`、BikeNYC `[1,2,12,24,12]`、CHAP `[1,1,7,32,32]` 的 CPU Forward，输出均有限且 shape 正确。

此外已经完成一次 CPU synthetic 的完整闭环：

```text
1 epoch train
-> 1次 validation
-> 保存 best.pt
-> 加载 best.pt
-> 1次 test
```

该 smoke 未出现 NaN/Inf，测试 MAE 为 `0.607448`，并确认 V15 诊断项进入 `metrics.jsonl`。这只证明工程流程跑通，不代表三个真实数据集的正式性能。

## 7. 新增和修改文件

新增模型文件：

```text
src/stmoe_imputer/models/v_single/compact_residual_pyramid.py
src/stmoe_imputer/models/v_single/residual_budget.py
src/stmoe_imputer/models/v_single/v15_compact_residual_moe.py
```

修改公共接口：

```text
src/stmoe_imputer/models/v_single/__init__.py
src/stmoe_imputer/models/registry.py
src/stmoe_imputer/losses.py
src/stmoe_imputer/engine.py
```

新增测试：

```text
tests/_v15_utils.py
tests/test_v15_config.py
tests/test_v15_shapes.py
tests/test_v15_main_equivalence.py
tests/test_v15_bounded_residual.py
tests/test_v15_no_target_leakage.py
tests/test_v15_gradient_flow.py
```
