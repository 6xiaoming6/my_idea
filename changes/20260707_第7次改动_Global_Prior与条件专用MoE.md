# 第7次改动：Global Prior + Conditional Specialized MoE

## 1. 修改信息

- 修改时间：2026-07-07 13:58:57
- 开发分支：`v7-single`
- 修改前提交：`4a4933b5f4e8768084c9adb1ba955f3eda080d9b`
- 修改后提交：`b7d098e2334ffdf73053cda48fba2503e1f2a2ae`
- 对应设计文档：`model_designs/v7-single.md`
- 版本目标：在不改动数据、mask、训练主循环和评价指标的前提下，将原有 shared/routed 结构重构为论文语义更清晰的“全局先验 + 条件专家” MoE。

## 2. 修改文件

本次修改和新增：

- `src/stmoe_imputer/models/experts.py`
- `src/stmoe_imputer/models/fusion.py`
- `src/stmoe_imputer/models/main_branch.py`
- `src/stmoe_imputer/losses.py`
- `configs/v7-single/taxibj.json`
- `configs/v7-single/bikenyc.json`
- `configs/v7-single/chap.json`
- `scripts/v7-single/train.py`
- `scripts/v7-single/run_full_experiments.py`
- `scripts/run_experiments.py`
- `changes/20260707_第7次改动_Global_Prior与条件专用MoE.md`

## 3. 第七版模型结构

修改后的主路径为：

```text
x_f_gt, m_f
  -> Observed Builder + Masked Multi-scale Pooling
  -> ScaleTokenEncoder(F/M/C)
  -> QualityRouter(F/M/C)
  -> ConditionalRoutedExpertPool + Top-K
  -> ProgressiveRouteFusion
  -> h_special

ScaleTokenEncoder(F/M/C)
  -> GlobalPriorExpert
  -> h_prior

h_prior, h_special
  -> PriorSpecializedFusion
  -> Prediction Head
  -> x_hat
```

结构仍保留三条硬约束：

- Shared Expert 以 `GlobalPriorExpert` 形式保留。
- Routed Experts 以 `ConditionalRoutedExpertPool` 形式保留。
- Router + Top-K MoE 的动态路由保持不变。

## 4. 具体代码改动

### 4.1 GlobalPriorExpert

新增：

```python
class GlobalPriorExpert(GatedCrossScaleSharedExpert):
    pass
```

第七版第一阶段完整复用已验证稳定的 `GatedCrossScaleSharedExpert`，因此不改变：

- fine/mid/coarse 特征的尺度对齐方式；
- reliability-aware scale gate；
- active scale mask；
- 两层时空残差块；
- 输出 shape。

新的 `h_prior` 语义表示跨尺度、跨区域和跨缺失率的稳定全局时空先验。

### 4.2 ConditionalRoutedExpertPool

新增：

```python
class ConditionalRoutedExpertPool(TopKRoutedExpertPool):
    pass
```

该类完整复用原有 Top-K 专家选择和加权求和逻辑，不改动专家数量、路由权重、top-k 执行方式和 tensor shape。新语义表示这些专家根据观测质量动态建模局部、样本特定和高难度模式。

### 4.3 PriorSpecializedFusion

新增先验—专用残差融合：

```text
h_prior_refined = ResidualSTBlock(h_prior)
h_special_proj  = Dropout3d(ResidualSTBlock(Conv1x1(h_special)))
alpha           = sigmoid(route_gamma)
h_main          = h_prior_refined + alpha * h_special_proj
```

配置中 `route_gamma_init=-3.0`，所以：

```text
alpha_init = sigmoid(-3.0) = 0.04742587
```

训练初期以全局先验为主，条件专家只作为小幅残差修正，用于降低重构后的训练风险。

### 4.4 主模型配置化切换

`MultiScaleMoEBackbone` 新增：

```text
model_variant
shared_expert_type
routed_expert_type
fusion_type
```

当配置为：

```json
{
  "name": "v7-single",
  "shared_expert_type": "global_prior",
  "routed_expert_type": "conditional_specialized",
  "fusion_type": "prior_specialized"
}
```

模型使用第七版结构。不提供这些字段时，默认仍构建 main 的旧类。

### 4.5 兼容性保留

原有下列类没有删除：

- `GatedCrossScaleSharedExpert`
- `TopKRoutedExpertPool`
- `SharedRoutedResidualFusion`

旧配置仍使用 `cross_scale_shared_expert` 和 `branch_fusion` 成员名，保留旧 checkpoint 的 state-dict 键。第七版才使用 `global_prior_expert` 和 `prior_specialized_fusion` 成员名。

输出同时保留旧字段和新语义字段：

| 新字段 | 旧兼容字段 |
|---|---|
| `h_prior` | `z_shared` |
| `h_prior_refined` | `h_shared` |
| `h_special` | `h_route` |
| `h_special_proj` | `h_route_proj` |
| `x_hat_prior` | `x_hat_shared` |
| `x_hat_specialized` | `x_hat_route` |

## 5. Loss 计算调整

遵循设计文档“Loss 完全保持 main”的要求，本次没有修改任何 loss 权重、warmup、优化器或总损失公式。

总损失仍为：

```text
L = L_main
  + lambda_cross * L_cross
  + lambda_importance_balance * L_importance
  + lambda_load_balance * L_load
  + lambda_fusion_entropy * L_fusion_entropy
  + lambda_branch_entropy * L_branch_entropy
  + lambda_shared_aux * L_prior_aux
  + lambda_route_aux * L_specialized_aux
  + lambda_complementary * L_complementary
  + lambda_final * L_final
```

只调整了 loss 的语义读取顺序：

- 先读取 `x_hat_prior`，缺失时回退到 `x_hat_shared`。
- 先读取 `x_hat_specialized`，缺失时回退到 `x_hat_route`。
- 互补损失先读取 `h_prior_refined` 和 `h_special_proj`，并保留旧字段回退。
- 日志新增 `l_prior_aux` 和 `l_specialized_aux`，同时保留 `l_shared_aux` 和 `l_route_aux`。

因此第七版与 main 的损失目标一致，不引入额外实验变量。

## 6. 三数据集配置

新增三个第七版配置补丁：

```text
configs/v7-single/taxibj.json
configs/v7-single/bikenyc.json
configs/v7-single/chap.json
```

它们通过 `--override_config` 与原数据集配置合并，不复制数据路径、batch size、epoch、学习率和 loss 权重。三个配置都固定：

```json
"output_dir": "outputs/v7-single"
```

因此正式实验输出与 main 隔离，目录格式为：

```text
outputs/v7-single/{dataset}/{experiment_type}/{variant}/{mask}/rate{rate}/{run_id}/
```

第七版专用训练入口位于 `scripts/v7-single/train.py`。该脚本只负责选择数据集配置和第七版模型配置，训练、验证、最佳 checkpoint 保存及最终测试仍调用公共 `scripts/train.py`，避免复制训练逻辑。示例：

```bash
python scripts/v7-single/train.py \
  --dataset TaxiBJ \
  --train_npz <TaxiBJ train.npz> \
  --val_npz <TaxiBJ val.npz> \
  --test_npz <TaxiBJ test.npz> \
  --name full
```

BikeNYC 和 CHAP 分别使用 `--dataset BikeNYC` 和 `--dataset CHAP`。新增的第七版训练脚本和输出均不得散放到公共 `scripts/` 根目录或旧版 `outputs/` 路径。

### 6.1 单卡完整 fixed + random 实验

新增：

```text
scripts/v7-single/run_full_experiments.py
```

该脚本默认使用 GPU0，执行顺序固定为：

```text
阶段 1：TaxiBJ/BikeNYC/CHAP × fixed × 0.2/0.4/0.6/0.8
  -> fixed 的 12 次训练全部结束
阶段 2：TaxiBJ/BikeNYC/CHAP × random × 0.2/0.4/0.6/0.8
  -> random 的 12 次训练全部结束
```

每个组合只运行 `full`，不执行消融模型。训练策略复用 `configs/policies/full_model_paper.json`：TaxiBJ、BikeNYC 和 CHAP 的最大 epoch 分别为 160、140 和 150，保留各自的验证间隔、early stopping、最佳 checkpoint 覆盖保存及最终测试。

运行命令：

```bash
python scripts/v7-single/run_full_experiments.py
```

运行前检查而不启动训练：

```bash
python scripts/v7-single/run_full_experiments.py --dry-run
```

公共 `scripts/run_experiments.py` 只新增通用的 `--model-config-dir` 参数，用于在数据集配置和训练策略之外合并版本模型配置；未提供该参数时，原有 main/ablation 实验行为保持不变。

## 7. 参数量变化

`PriorSpecializedFusion` 的 prior refine 从原融合器的两个 `ResidualSTBlock` 收缩为设计文档要求的一个，因此三个数据集均减少 `239,363` 个参数：

| 数据集 | main 参数量 | v7-single 参数量 | 变化 |
|---|---:|---:|---:|
| TaxiBJ | 4,682,755 | 4,443,392 | -239,363 |
| BikeNYC | 4,620,931 | 4,381,568 | -239,363 |
| CHAP_Beijing | 4,677,472 | 4,438,109 | -239,363 |

## 8. 验证结果

### 8.1 静态检查

- `python -m compileall -q src scripts`：通过。
- 三个 JSON 配置格式检查：通过。
- `git diff --check`：通过。

### 8.2 前向与反向验证

使用小尺度合成 tensor 验证以下路径：

| 模式 | 前向 | Loss | backward |
|---|---|---|---|
| v7-single full | 通过 | 有限值 | 通过 |
| shared_only | 通过 | 有限值 | 通过 |
| routed_only | 通过 | 有限值 | 通过 |
| no_router | 通过 | 有限值 | 通过 |

旧 main 配置构建的 state-dict 仍包含 `cross_scale_shared_expert` 和 `branch_fusion` 键，旧模型回退路径通过检查。

### 8.3 1-epoch 完整 smoke

使用项目原生 `scripts/train.py` 完成：

```text
1 epoch train
  -> 1 validation
  -> save checkpoints/best.pt
  -> reload best.pt
  -> final test
```

结果：

- Train loss：`0.544068`
- Validation MAE：`0.727459`
- Validation RMSE：`0.904487`
- Test MAE：`0.725457`
- Test RMSE：`0.901541`
- best epoch：`1`
- checkpoint、`train.log`、`val.log`、`test.log` 和 `metrics.jsonl` 均成功生成。

smoke 只用于验证工程链路，不代表正式实验效果。

## 9. 阶段性结论

第七版已按设计文档完成为“Global Prior Shared Expert + Conditional Specialized Routed Experts + Prior-Specialized Residual Fusion”的结构重构。数据、mask、训练主循环、评价指标和总损失权重未变，并保留了 main 类和旧 checkpoint 参数键的回退能力。

当前验证只能证明代码可完整训练，不能证明第七版精度优于 main。后续需要按设计文档在 TaxiBJ random 0.6、CHAP random 0.8 和 BikeNYC fixed 0.6 上先进行重点对比，再决定是否展开全量实验。
