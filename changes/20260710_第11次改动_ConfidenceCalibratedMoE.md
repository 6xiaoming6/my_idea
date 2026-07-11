# 第11次改动：Confidence-Calibrated MoE

- 开发分支：v11-single
- 修改前提交：4a4933b
- 修改后提交：`bba608226fa236b710f4f21d820122bfa7ded09c`
- 设计文档：`model_designs/v11-single.md`
- 独立输出目录：`outputs/v11-single/`

## 1. 修改目标

在不覆盖 v7、v8、v9、v10 版本内容的前提下，基于稳定 main 主干新增 v11-single。模型保留多分辨率输入与 MoE，同时让每个专家在输出时空特征的同时估计 sample-level confidence，最终专家融合权重由 Router 偏好与专家自身置信度共同决定。

## 2. 实际结构

每个尺度执行以下流程：

1. `QualityRouter` 同时返回原始 logits 和 softmax router weight。
2. 4 个原始 `STExpert` 分别生成专家输出。
3. 每个专家对应一个 `ExpertConfidenceHead`，输入专家输出的全局池化、尺度输入特征的全局池化和观测比例，输出 `[B,1]` 置信度。
4. `CalibratedWeightComposer` 在 logit 空间执行：

   `calibrated_logits = router_logits + beta_conf * log(confidence)`

5. v11 主配置使用 soft calibrated routing；`topk_calibrated` 消融在校准后执行 Top-K。
6. fine/mid/coarse 三个尺度继续进入原有路由融合分支和共享跨尺度分支，预测头、主损失和训练策略不变。

第一阶段只记录各尺度置信度，不把置信度加入 scale gate，避免同时修改专家融合和尺度融合后无法定位收益来源。

## 3. 对设计文档的合理修正

设计文档建议 `confidence_beta_init=0`，同时将 confidence head 最后一层零初始化。若严格照做，则初始置信度全部为 0.5，beta 为 0，校准权重完全不依赖 confidence，confidence head 和 beta 都无法获得有效学习信号。

本次将正式配置的 `confidence_beta_init` 调整为 0.05，`confidence_beta_max` 保持 0.5。由于 confidence head 零初始化后所有专家初始置信度完全相同，softmax 对统一 logit 平移不敏感，所以初始校准权重仍与 Router 权重数值一致；与此同时，0.05 的小系数允许 confidence head 从第一步开始获得非零梯度。实测初始校准权重与 Router 权重最大绝对误差约为 `5.96e-08`。

`CalibratedWeightComposer` 使用有界 sigmoid 参数化，保证有效 beta 始终处于 `[0, beta_max]`。配置中保留 `confidence_enabled=false` 作为严格 gate-only 回退。

文档中的 pixel-level confidence 和 calibrated scale fusion 均未在第一阶段实现；如果配置错误地开启 `use_calibrated_scale_fusion=true`，代码会明确报错，而不是静默忽略。

## 4. 代码改动

### 4.1 新增模块

- `src/stmoe_imputer/models/v_single/confidence_heads.py`
  - `ExpertConfidenceHead`
  - `CalibratedWeightComposer`
- `src/stmoe_imputer/models/v_single/v11_confidence_calibrated_moe.py`
  - `ConfidenceCalibratedExpertPool`
- `src/stmoe_imputer/models/v_single/__init__.py`

### 4.2 主干兼容接入

- `QualityRouter.forward(return_logits=False)`：默认接口保持不变；v11 可选同时取得 weight 和 logits。
- `MultiScaleMoEBackbone` 新增 `expert_pool_type=confidence_calibrated` 工厂分支。
- 原始 `TopKRoutedExpertPool` 未修改，默认 `expert_pool_type=homogeneous` 时行为保持不变。
- 置信度关闭且使用 Top-K 时，v11 专家池与原始专家池已验证为逐元素完全一致，最大绝对误差为 0。
- `scripts/run_experiments.py` 新增通用 `--model-config-dir`，可以按数据集合并版本模型配置。

## 5. 日志指标

训练、验证和测试的 `metrics.jsonl` 新增：

- `confidence_mean`、`confidence_std`
- `confidence_fine/mid/coarse_mean`
- `confidence_fine/mid/coarse_std`
- `confidence_expert_0/1/2/3_mean`
- `confidence_missing_rate`（fine 尺度实际缺失率）
- `confidence_fine/mid/coarse_missing_rate`
- `confidence_multiscale_missing_rate_mean`
- `confidence_beta`
- `router_weight_entropy`
- `calibrated_weight_entropy`
- `confidence_weight_shift_l1`

上述指标用于判断置信度是否塌缩、专家间是否形成可靠性差异，以及校准权重相对原 Router 权重产生了多大变化。

## 6. 配置与运行脚本

### 6.1 正式配置

- `configs/v11-single/taxibj.json`
- `configs/v11-single/bikenyc.json`
- `configs/v11-single/chap.json`

正式配置使用 4 个原始同构 ST 专家、soft calibrated routing、sample-level confidence、`beta_init=0.05`，不修改原损失函数。

### 6.2 消融配置

- `gate_only_fallback.json`：原始 homogeneous Top-K 主干。
- `topk_calibrated.json`：校准后执行 Top-K。
- `no_confidence.json`：保持 soft routing，但关闭置信度修正。
- `confidence_no_mask.json`：置信度头不输入 mask 观测比例。
- `confidence_no_input_feature.json`：置信度头不输入专家处理前特征。

未创建“scale confidence off”重复消融，因为 v11 第一阶段主配置本身已经关闭 calibrated scale fusion；创建同配置的实验不会提供有效对照。

### 6.3 独立脚本

- `scripts/v11-single/train.py`：运行单个主实验或通过 `--ablation` 运行 v11 消融。
- `scripts/v11-single/run_validation_matrix.py`：运行 5 个代表性点的 1/5 epoch 验证。
- `scripts/v11-single/run_full_experiments.py`：完整运行三数据集、两种缺失模式和全部缺失率，固定按 fixed 后 random 的顺序执行。

## 7. 验证结果

- `python -m compileall`：通过。
- 全部 v11 JSON 配置解析：通过。
- 5 点验证矩阵 dry-run：通过。
- 全量脚本 fixed→random 顺序 dry-run：通过。
- soft calibrated 和 top-k calibrated 独立前向/反向：通过。
- 5 个消融配置前向/反向：全部通过。
- gate-only 回退与原 `TopKRoutedExpertPool` 数值等价：最大绝对误差 0。
- 合成数据完整 smoke：完成 1 epoch 训练、验证、最佳检查点保存、最佳模型加载和最终测试。
- smoke 日志已确认包含 confidence、router entropy、calibrated entropy 和 weight shift 指标。

Smoke 输出：

`outputs/v11-single/unknown/debug/smoke_v11_single_final/random/rate0.45/20260710_130951_seed7_bs2/`

## 8. 运行命令

单个 1 epoch 验证：

```bash
python scripts/v11-single/train.py \
  --dataset TaxiBJ \
  --mask-pattern fixed \
  --mask-rate 0.2 \
  --gpu 0 \
  --quick 1
```

五点验证矩阵：

```bash
python scripts/v11-single/run_validation_matrix.py --gpu 0 --epochs 1
```

单个消融实验：

```bash
python scripts/v11-single/train.py \
  --dataset TaxiBJ \
  --mask-pattern fixed \
  --mask-rate 0.2 \
  --gpu 0 \
  --quick 5 \
  --ablation no_confidence
```

完整实验，单卡先 fixed 后 random：

```bash
python scripts/v11-single/run_full_experiments.py --gpu 0
```

正式输出全部进入 `outputs/v11-single/`，不会覆盖旧版本实验目录。
