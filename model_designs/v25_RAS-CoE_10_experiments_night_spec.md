# v25 RAS-CoE 十组探索实验：Codex 实现与整夜运行规范

> **用途**：直接交给 Codex。要求 Codex 在当前 `v24-COE` 最新代码基础上，实现并组织 10 组 RAS-CoE 对比实验，确保可以连续无人值守运行，并在全部实验结束后自动生成统一汇总结果。  
> **仓库**：`6xiaoming6/my_idea`  
> **基线分支**：`v24-COE`  
> **实验日期**：2026-09-28 ~ 2026-09-29  
> **运行目标**：今晚 23:00 至次日 09:00，约 10 小时，完成 10 组探索性实验。  
> **预计预算**：约 1 小时 / 组。  
> **核心研究问题**：验证 `Repair-Aware Selective Chain-of-Experts (RAS-CoE)` 是否真正能够减少多轮补全过程中的 over-repair，并判断各组件的独立贡献。

---

# 1. 本轮实验的总体目标

本轮不是最终论文实验，也不是大规模调参。

本轮只回答一个核心问题：

> **将多轮 CoE 从“无条件状态更新”改造成“候选修复 → 选择性接受 → 隐状态提交 → 修复反馈 → 重新路由”的闭环机制，是否真的有价值？**

因此，本轮十组实验必须围绕以下四个机制问题展开：

1. **Naive Feedback 是否会传播错误修复？**
2. **仅对输出值做 acceptance 是否不够，是否必须控制 latent state？**
3. **利用 pseudo-missing GT 显式监督 acceptance 是否有额外收益？**
4. **将 rejection 状态反馈给下一轮 Router 是否能形成真正闭环？**

另外再验证：

5. **RAS 是否只对当前 C2 partner routing 有效，还是对普通 Native Top-2 也有效？**
6. **Acceptance loss 权重是否对结果高度敏感？**

---

# 2. 本轮实验统一协议

所有 10 组实验必须共享以下训练协议。

除每组明确列出的差异项以外，其他配置必须完全一致。

```text
Dataset              = TaxiBJ
Train mask           = 九种 mixed masks
Validation mask      = 九种 mixed masks
Test mask            = 九种 mixed masks
Missing rate         = 0.4
Seed                 = 7
Global batch size    = 16
Epoch                = 80
Validation frequency = every 2 epochs
Early stopping       = False
Optimizer            = AdamW
Main LR              = 1e-3
Weight decay         = 1e-4
Grad clip norm       = 1.0
AMP                  = True
Scheduler            = cosine
Scheduler total      = 80
eta_min              = 1e-6
Best model criterion = minimum validation MAE
Save best checkpoint = True
Full checkpoint      = True
Final test           = once, using best validation checkpoint
```

注意：

> **本轮必须使用 cosine80，而不是 cosine120。**

原因：

- 今晚属于机制探索；
- 单组预算约 1 小时；
- 80 epoch 已足够判断方向；
- 后续若方法成立，再统一做 120 epoch 正式实验。

---

# 3. 数据与 mask 协议

延续当前最新 v24-COE 的 TaxiBJ clean split。

训练、验证和测试均使用九种缺失类型：

```text
random_point
node_outage
temporal_gap
spatial_region
spatiotemporal_block
stripe
moving_region
multi_block
composite
```

缺失率：

```text
0.4
```

训练：

```text
resample_each_epoch = true
```

验证/测试：

```text
resample_each_epoch = false
```

即：

> 训练 mask 每轮重采样，验证和测试固定。

不允许不同实验使用不同 mask seed。

---

# 4. Backbone 统一设置

本轮默认采用当前 C2 对应的强结构作为主要 baseline：

```text
4 rounds
8 experts
shared expert pool
hard Top-2
partner_residual
corrected partner fusion
partner auxiliary head supervision
```

专家池：

```json
[
  "T",
  "S",
  "TD",
  "SD",
  "TA",
  "ST",
  "TL",
  "SL"
]
```

公共 CoE 配置建议：

```json
{
  "num_steps": 4,

  "expert_pool": [
    "T",
    "S",
    "TD",
    "SD",
    "TA",
    "ST",
    "TL",
    "SL"
  ],

  "routing_mode": "hard",
  "top_k": 2,

  "router_state": "dynamic",
  "expert_state": "dynamic",

  "use_shared": true,
  "use_routed": true,
  "expert_sharing": "shared",

  "temperature": 1.0,

  "routing_warmup_epochs": 3,
  "routing_transition_epochs": 3,
  "uniform_mix_start": 0.5,
  "sampling_temperature_start": 2.0,

  "router_hidden_dim": 64
}
```

---

# 5. RAS-CoE 新结构前提

Codex 在运行本轮实验前，必须先完成上一份 `v25_RAS-CoE_implementation_spec.md` 中的核心实现：

```text
Repair Proposal
Position-wise Repair Acceptance
Latent Selective Commit
Acceptance Auxiliary Supervision
Repair Rejection Feedback
Next-round Router Feedback
```

新增配置至少包括：

```json
"repair_acceptance": "none",
"repair_feedback_to_router": false,
"repair_accept_aux_head_only": true,
"repair_accept_init_prob": 0.99
```

Loss 配置至少支持：

```json
"lambda_coe_accept": 0.0,
"lambda_coe_monotonic": 0.0,
"repair_accept_margin": 0.0
```

本轮统一：

```json
"lambda_coe_monotonic": 0.0
```

不要在今晚加入 monotonic loss。

---

# 6. 十组实验总览

| 编号 | 名称 | 主要目的 |
|---|---|---|
| E0 | C2 Baseline Reproduction | 新代码是否保持旧最佳 baseline |
| E1 | Naive Completion Feedback | 无控制反馈是否传播错误 |
| E2 | Legacy Point Acceptance | 只 gate completion 是否足够 |
| E3 | Latent Selective Commit | 控制 hidden update 本身是否有效 |
| E4 | Supervised Selective Commit | pseudo-missing acceptance supervision 是否必要 |
| E5 | Full RAS-CoE | rejection → router feedback 是否带来闭环收益 |
| E6 | Full RAS λ=0.03 | 弱 acceptance supervision |
| E7 | Full RAS λ=0.30 | 强 acceptance supervision |
| E8 | Native Top-2 Baseline | 普通共享 Top-2 基线 |
| E9 | Native Top-2 + Full RAS | RAS 是否独立于 partner routing 也成立 |

---

# 7. E0：C2 Baseline Reproduction

## 目的

验证新代码在关闭所有新功能时能够复现当前 C2。

这组是今晚最重要的 regression baseline。

## 配置

```json
{
  "pair_mode": "partner_residual",
  "partner_fusion": "corrected",
  "partner_aux_head_only": true,

  "completion_feedback": false,

  "acceptance": "none",

  "repair_acceptance": "none",
  "repair_feedback_to_router": false,
  "repair_accept_aux_head_only": true,

  "lambda_coe_accept": 0.0,
  "lambda_coe_monotonic": 0.0
}
```

## 预期

历史参考：

```text
Best Val MAE ≈ 21.242
Test MAE     ≈ 19.932
Test RMSE    ≈ 34.161
```

不要求 bit-level 完全一致，但如果：

```text
Test MAE > 20.5
```

或明显偏离历史结果，则标记：

```text
REGRESSION_WARNING
```

---

# 8. E1：Naive Completion Feedback

## 目的

验证：

> 单纯把前一轮 completion 反馈给下一轮，不做可靠性控制，是否会传播错误修复。

## 相对 E0 唯一核心变化

```json
"completion_feedback": true
```

仍然：

```json
{
  "acceptance": "none",
  "repair_acceptance": "none",
  "repair_feedback_to_router": false,
  "lambda_coe_accept": 0.0
}
```

## 研究意义

如果：

```text
E1 worse than E0
```

则支持：

> Feedback 并非天然有效；错误修复若直接反馈会影响下一轮状态和路由。

---

# 9. E2：Legacy Point Acceptance

## 目的

验证当前已有 acceptance gate：

> 只控制 completion，不控制 hidden state，是否足够。

## 配置

```json
{
  "pair_mode": "partner_residual",
  "partner_fusion": "corrected",
  "partner_aux_head_only": true,

  "completion_feedback": true,

  "acceptance": "point",

  "repair_acceptance": "none",
  "repair_feedback_to_router": false,

  "lambda_coe_accept": 0.0,
  "lambda_coe_monotonic": 0.0
}
```

## 机制

当前 legacy acceptance：

```text
hidden = hidden + update
↓
candidate completion
↓
old completion + gate * proposal
```

即：

```text
completion 被 gate
hidden 未被 gate
```

这组主要用于和 E3 比较。

---

# 10. E3：Latent Selective Commit

## 目的

验证：

> 不使用显式 acceptance 标签，仅允许模型对 latent update 做选择性提交，本身是否有价值。

## 配置

```json
{
  "pair_mode": "partner_residual",
  "partner_fusion": "corrected",
  "partner_aux_head_only": true,

  "completion_feedback": true,

  "acceptance": "none",

  "repair_acceptance": "latent_point",
  "repair_feedback_to_router": false,
  "repair_accept_aux_head_only": true,
  "repair_accept_init_prob": 0.99,

  "lambda_coe_accept": 0.0,
  "lambda_coe_monotonic": 0.0
}
```

## 机制

```text
expert update
↓
candidate hidden
↓
candidate prediction
↓
repair gate
↓
hidden = hidden_old + accept * update
↓
accepted prediction
↓
next round
```

## 关键比较

```text
E2 vs E3
```

如果：

```text
E3 < E2
```

则说明：

> 只 gate completion 不够；真正控制 latent state 才能减少错误传播。

---

# 11. E4：Supervised Selective Commit

## 目的

验证：

> 利用 pseudo-missing GT 明确监督“Candidate 是否优于旧状态”，是否比只依赖最终 reconstruction loss 更有效。

## 配置

与 E3 相同，但：

```json
"lambda_coe_accept": 0.1
```

并保持：

```json
"repair_feedback_to_router": false
```

## Acceptance label

训练阶段：

\[
label =
\mathbb{I}
[
E(candidate,GT)
<
E(old,GT)
]
\]

target 只能用于：

```text
acceptance loss
diagnostic metrics
```

不得进入：

```text
Gate input
Router input
Expert input
Inference
```

## 关键比较

```text
E3 vs E4
```

若：

```text
E4 < E3
```

则支持：

> 最终 reconstruction objective 不足以独立学习可靠 acceptance，pseudo-missing 提供了直接 repair decision supervision。

---

# 12. E5：Full RAS-CoE

## 目的

这是今晚最核心的新模型。

验证：

> 将上一轮的 rejection 状态显式反馈给下一轮 Router，是否能够让下一轮专家协作针对“尚未修好的区域”重新调整。

## 配置

```json
{
  "pair_mode": "partner_residual",
  "partner_fusion": "corrected",
  "partner_aux_head_only": true,

  "completion_feedback": true,

  "acceptance": "none",

  "repair_acceptance": "latent_point",
  "repair_feedback_to_router": true,
  "repair_accept_aux_head_only": true,
  "repair_accept_init_prob": 0.99,

  "lambda_coe_accept": 0.1,
  "lambda_coe_monotonic": 0.0
}
```

## 数据流

```text
Accepted Repair State
        ↓
Router
        ↓
Expert Collaboration
        ↓
Repair Proposal
        ↓
Acceptance
        ↓
Selective Latent Commit
        ↓
Accepted Completion
        ↓
Rejection Map
        ↓
Next-round Router
```

## 关键比较

```text
E4 vs E5
```

这组对比只回答：

> rejection feedback 是否真正有价值。

---

# 13. E6：Full RAS，λ_accept = 0.03

## 目的

检查新方法是否依赖于特定 acceptance loss 权重。

## 配置

完全等于 E5，但：

```json
"lambda_coe_accept": 0.03
```

---

# 14. E7：Full RAS，λ_accept = 0.30

## 目的

测试较强 acceptance supervision 是否会：

```text
提高 acceptance 分类
但损害 reconstruction
```

## 配置

完全等于 E5，但：

```json
"lambda_coe_accept": 0.30
```

---

# 15. E8：Native Top-2 Baseline

## 目的

建立一个不依赖 partner routing 的普通 CoE / Top-2 baseline。

验证 RAS 是否具有更普遍的价值。

## 配置

```json
{
  "pair_mode": "native",

  "partner_fusion": "individual",
  "partner_aux_head_only": false,

  "completion_feedback": false,

  "acceptance": "none",

  "repair_acceptance": "none",
  "repair_feedback_to_router": false,

  "lambda_coe_accept": 0.0,
  "lambda_coe_monotonic": 0.0
}
```

其余保持：

```text
4 rounds
8 experts
shared pool
hard Top-2
```

---

# 16. E9：Native Top-2 + Full RAS

## 目的

这是今晚第二个非常重要的实验。

验证：

> RAS 是否只是 C2/partner routing 的附属技巧，还是可以独立改善普通 Native Top-2 CoE。

## 配置

基于 E8：

```json
{
  "completion_feedback": true,

  "acceptance": "none",

  "repair_acceptance": "latent_point",
  "repair_feedback_to_router": true,
  "repair_accept_aux_head_only": true,
  "repair_accept_init_prob": 0.99,

  "lambda_coe_accept": 0.1,
  "lambda_coe_monotonic": 0.0
}
```

## 关键比较

```text
E8 vs E9
```

如果：

```text
E9 < E8
```

同时：

```text
E5 < E0
```

则是非常重要的证据：

> RAS 是一个独立于具体 pair routing 策略的 repair mechanism。

---

# 17. 十组实验建议命名

统一使用：

```text
v25_ras_e0_c2_baseline
v25_ras_e1_naive_feedback
v25_ras_e2_legacy_point
v25_ras_e3_latent_unsup
v25_ras_e4_latent_sup
v25_ras_e5_full
v25_ras_e6_full_l003
v25_ras_e7_full_l030
v25_ras_e8_native_baseline
v25_ras_e9_native_full
```

不要让名字包含时间戳之外的随机字符串作为唯一识别方式。

---

# 18. 推荐配置文件布局

新增：

```text
configs/v25/ras_night/
```

包含：

```text
base_taxibj.json

e0_c2_baseline.json
e1_naive_feedback.json
e2_legacy_point.json
e3_latent_unsup.json
e4_latent_sup.json
e5_full.json
e6_full_l003.json
e7_full_l030.json
e8_native_baseline.json
e9_native_full.json

experiments.json
```

其中：

```text
base_taxibj.json
```

保存所有公共设置。

每个实验 JSON 只 override 差异项。

不要复制十份完整配置并手工修改大量重复字段，以免配置漂移。

---

# 19. 建议新增实验 Runner

新增：

```text
scripts/v25/run_ras_night.py
```

要求：

```bash
python scripts/v25/run_ras_night.py
```

即可顺序运行十组实验。

支持：

```bash
python scripts/v25/run_ras_night.py --start E0
python scripts/v25/run_ras_night.py --start E5
python scripts/v25/run_ras_night.py --only E3 E4 E5
```

---

# 20. Runner 必须支持断点续跑

每组实验开始前检查：

```text
receipt.json
summary.json
final test record
```

如果该组已完整完成：

```text
SKIP_COMPLETED
```

如果存在目录但没有完整结束标志：

```text
INCOMPLETE
```

默认行为：

```text
重新运行该组
```

不要因为某组失败导致后面 9 组都停止。

---

# 21. 单组失败处理

Runner 必须：

```python
try:
    run_experiment(...)
except Exception:
    save_failure_receipt(...)
    continue
```

记录：

```text
experiment_id
timestamp
exception type
traceback
config path
git commit
git dirty status
```

然后继续下一组。

---

# 22. 运行顺序

默认：

```text
E0
E1
E2
E3
E4
E5
E6
E7
E8
E9
```

不要打乱。

原因：

前 6 组形成最重要的机制链。

---

# 23. 如果时间不足的优先级

最高优先级：

```text
E0
E1
E2
E3
E4
E5
E8
E9
```

次优先级：

```text
E6
E7
```

Runner 可以支持：

```text
priority
```

但今晚默认仍全部运行。

---

# 24. 必须记录的基础指标

每组至少：

```text
best_epoch
best_val_mae
test_mae
test_rmse
test_wape
parameter_count
completed_epochs
training_time
avg_epoch_time
forward_latency
peak_memory
```

MAPE 仍然保存，但不作为核心结论。

---

# 25. 必须记录的逐轮 Repair 指标

每组至少输出：

```text
coe_initial_mae

coe_step1_mae
coe_step2_mae
coe_step3_mae
coe_step4_mae
```

要求：

> 使用全数据精确 numerator/count 累积。

不要简单平均 batch MAE。

---

# 26. Candidate Harm 指标

每轮：

```text
coe_step1_candidate_harm_rate
coe_step2_candidate_harm_rate
coe_step3_candidate_harm_rate
coe_step4_candidate_harm_rate
```

定义：

```text
candidate_error > old_error
```

---

# 27. Accepted Harm 指标

每轮：

```text
coe_step1_accepted_harm_rate
...
```

定义：

```text
accepted_error > old_error
```

---

# 28. Over-repair Prevention Rate

每轮：

\[
PreventionRate
=
1-
\frac{AcceptedHarmCount}
{CandidateHarmCount}
\]

输出：

```text
coe_step1_overrepair_prevention_rate
...
```

如果：

```text
CandidateHarmCount == 0
```

输出：

```text
NaN / null
```

不要假设为 1。

---

# 29. Acceptance 统计

每轮：

```text
coe_step1_accept_mean
coe_step1_accept_std
coe_step1_pred_accept_rate
coe_step1_oracle_accept_rate
```

以及：

```text
accept_accuracy
accept_precision
accept_recall
accept_f1
```

---

# 30. Monotonicity 指标

每轮：

```text
coe_step1_nonworse_sample_rate
coe_step2_nonworse_sample_rate
coe_step3_nonworse_sample_rate
coe_step4_nonworse_sample_rate
```

定义：

```text
sample MAE_r <= sample MAE_(r-1)
```

另外：

```text
coe_all_steps_monotonic_sample_rate
```

定义：

\[
E_1\le E_0,
E_2\le E_1,
E_3\le E_2,
E_4\le E_3
\]

同时成立的 sample 比例。

---

# 31. Oracle Selective Repair

仅用于 validation/test diagnostic。

每轮：

```text
oracle_completion =
candidate if candidate_error < old_error
else old
```

记录：

```text
coe_step1_oracle_selective_mae
...
```

以及：

```text
coe_step1_acceptance_oracle_gap
```

其中：

```text
acceptance_oracle_gap
=
actual accepted MAE
-
oracle selective MAE
```

不能用该指标选 checkpoint。

---

# 32. Router 相关指标

保留当前：

```text
route_entropy
candidate_importance_entropy
pair_vs_top2_disagreement_rate
partner_fusion_weight
expert usage
pair path frequency
```

新增对 Full RAS：

```text
repair rejection vs router change
```

最简单可先输出：

```text
step r mean rejection
step r+1 pair distribution
```

不要求第一晚实现复杂统计相关分析。

---

# 33. 九类缺失单独统计

每组测试结果必须输出：

```text
random_point MAE
node_outage MAE
temporal_gap MAE
spatial_region MAE
spatiotemporal_block MAE
stripe MAE
moving_region MAE
multi_block MAE
composite MAE
```

尤其关注：

```text
node_outage
spatial_region
spatiotemporal_block
stripe
moving_region
multi_block
composite
```

---

# 34. 自动汇总 CSV

全部结束后生成：

```text
outputs/v25-RAS/night_20260928/summary.csv
```

列至少包含：

```text
experiment_id
name
status

best_epoch
best_val_mae
test_mae
test_rmse

initial_mae
step1_mae
step2_mae
step3_mae
step4_mae

step1_candidate_harm_rate
step2_candidate_harm_rate
step3_candidate_harm_rate
step4_candidate_harm_rate

step1_accepted_harm_rate
step2_accepted_harm_rate
step3_accepted_harm_rate
step4_accepted_harm_rate

step1_overrepair_prevention_rate
step2_overrepair_prevention_rate
step3_overrepair_prevention_rate
step4_overrepair_prevention_rate

step1_accept_rate
step2_accept_rate
step3_accept_rate
step4_accept_rate

accept_accuracy
accept_precision
accept_recall
accept_f1

all_steps_monotonic_sample_rate

params
train_time_min
forward_latency_ms
peak_memory_gb
```

---

# 35. 自动生成 comparison.md

全部结束后生成：

```text
outputs/v25-RAS/night_20260928/comparison.md
```

必须包含以下六部分。

---

## Part 1：主结果

表格：

| Exp | Val MAE | Test MAE | RMSE | Best Epoch |
|---|---:|---:|---:|---:|

---

## Part 2：逐轮 MAE

| Exp | Initial | Step1 | Step2 | Step3 | Step4 |
|---|---:|---:|---:|---:|---:|

---

## Part 3：Over-repair

| Exp | S1 Candidate Harm | S1 Accepted Harm | ... |
|---|---:|---:|---:|

---

## Part 4：Acceptance Quality

| Exp | Accept Rate | Oracle Rate | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|

---

## Part 5：九类缺失

输出 10×9 对比表。

---

## Part 6：关键 Pairwise Differences

自动计算：

```text
E1 - E0
E2 - E1
E3 - E2
E4 - E3
E5 - E4

E6 - E5
E7 - E5

E9 - E8

E5 - E0
E9 - E8
```

对：

```text
Val MAE
Test MAE
RMSE
monotonic rate
accepted harm rate
```

计算差值。

不要自动写“显著优于”之类统计结论，因为只有 seed 7。

只写：

```text
lower / higher / difference
```

---

# 36. 自动结论标签

可以生成机械标签，但不要自动写论文结论。

例如：

```text
REGRESSION_OK
REGRESSION_WARNING

NAIVE_FEEDBACK_HELPED
NAIVE_FEEDBACK_HURT

LATENT_COMMIT_HELPED
LATENT_COMMIT_HURT

ACCEPT_SUPERVISION_HELPED
ACCEPT_SUPERVISION_HURT

ROUTER_FEEDBACK_HELPED
ROUTER_FEEDBACK_HURT

RAS_GENERALIZES_TO_NATIVE
RAS_NOT_CONFIRMED_ON_NATIVE
```

判定只根据 test MAE 数值方向。

---

# 37. 今晚最关键的五个比较

明早分析时优先看：

---

## Comparison A

```text
E0 vs E1
```

问题：

> naive feedback 会不会变差？

---

## Comparison B

```text
E2 vs E3
```

问题：

> latent selective commit 是否优于只 gate completion？

---

## Comparison C

```text
E3 vs E4
```

问题：

> explicit acceptance supervision 是否有用？

---

## Comparison D

```text
E4 vs E5
```

问题：

> rejection feedback 是否真正形成有价值的闭环？

---

## Comparison E

```text
E8 vs E9
```

问题：

> RAS 是否独立于 partner routing 也成立？

---

# 38. 理想实验趋势

理想趋势示例：

```text
E0 baseline                19.93

E1 naive feedback          20.8
E2 legacy point            20.4
E3 latent selective        20.0
E4 supervised latent       19.5
E5 full RAS                19.2

E6 lambda=.03              19.3
E7 lambda=.30              19.6

E8 native baseline         21.3
E9 native + RAS            20.5
```

这只是示意，不是期望 Codex 硬编码判断目标。

---

# 39. 机制上比最终 MAE 更重要的现象

即使 E5 最终只比 E0 小幅提升，也必须重点观察：

```text
candidate_harm_rate
accepted_harm_rate
```

例如：

```text
Step4 candidate harm = 35%
Step4 accepted harm  = 12%
```

这比单纯：

```text
MAE -0.2
```

更能证明：

> RAS 确实阻止了有害修改。

---

# 40. 不允许做的事情

Codex 今晚实现和运行时不要：

```text
× 修改数据 split
× 修改 mask family
× 修改 missing rate
× 换 seed
× 改 expert pool
× 改 num_steps
× 开 multiscale
× 加 monotonic loss
× 改 optimizer
× 改 scheduler
× 改 lr
× 自动调参
× 根据测试集结果中途修改后续实验
```

十组必须预先冻结。

---

# 41. 不允许测试集参与模型选择

checkpoint：

```text
只根据 validation MAE
```

测试集：

```text
只在 best validation checkpoint 上最终运行一次
```

不允许：

```text
test every epoch
```

也不允许：

```text
选择 test 最优 epoch
```

---

# 42. Acceptance GT 防泄漏要求

必须增加自动测试：

```text
same x_f_obs
same m_f
different x_f_gt
```

model forward 输出必须完全一样。

允许不同的只有：

```text
acceptance auxiliary loss
evaluation oracle metrics
```

---

# 43. Smoke Test 要求

正式十组之前：

```text
每组至少跑 1 batch forward
```

检查：

```text
finite loss
finite grad
finite gate
finite route probs
output shape correct
```

不要求每组单独训练多个 epoch。

---

# 44. 全量测试建议

至少运行：

```bash
pytest tests/test_v24_coe.py
pytest tests/test_v24_partner.py
pytest tests/test_v24_partner_residual4.py
pytest tests/test_v24_team_accept_v4.py
pytest tests/test_v24_ddp.py
pytest tests/test_v24_amp_dispatch.py
pytest tests/test_v25_repair_acceptance.py
```

通过后再开始夜间队列。

---

# 45. 输出目录

统一：

```text
outputs/v25-RAS/night_20260928/
```

单组：

```text
E0_c2_baseline/
E1_naive_feedback/
E2_legacy_point/
E3_latent_unsup/
E4_latent_sup/
E5_full/
E6_full_l003/
E7_full_l030/
E8_native_baseline/
E9_native_full/
```

---

# 46. 每组必须保留

```text
config.json
receipt.json

logs/
    train.log
    val.log
    test.log
    metrics.jsonl

checkpoints/
    best.pt

analysis/
    repair_metrics.json
    routing_metrics.json
    family_metrics.json
```

---

# 47. Night-run 总状态文件

新增：

```text
outputs/v25-RAS/night_20260928/night_status.json
```

记录：

```json
{
  "started_at": "...",
  "finished_at": "...",
  "experiments_total": 10,
  "experiments_completed": 0,
  "experiments_failed": 0,
  "current_experiment": "E0",
  "results": {
    "E0": "pending",
    "E1": "pending"
  }
}
```

每组结束后立即更新。

这样即使进程中断，也能看到已经跑到哪里。

---

# 48. Codex 不需要做自动停止

即使某组明显失败：

```text
也跑满 80 epoch
```

因为今晚的目标之一就是比较收敛行为。

除非：

```text
NaN
Inf
OOM
RuntimeError
```

否则不要提前停止。

---

# 49. OOM 处理

如果新 RAS 因双 decode 导致 OOM：

第一选择：

```text
降低 per-GPU batch
保持 global batch 不变
通过 gradient accumulation 补偿
```

不要直接改变：

```text
global batch size
```

如果项目当前没有 accumulation 支持，则优先保持原 batch 并检查临时 tensor 是否未释放。

不要擅自将不同实验使用不同 global batch。

---

# 50. 时间记录

记录：

```text
setup time
train time
validation time
final test time
total wall time
```

至少保证：

```text
total wall time
avg epoch time
```

有值。

---

# 51. 最终 Codex 交付

代码实现完成后，Codex 必须向用户汇报：

1. 新增/修改文件；
2. 十组实验配置文件；
3. Runner 入口；
4. 所有 unit tests 是否通过；
5. smoke test 是否通过；
6. 夜间运行命令；
7. 输出目录；
8. 如何断点续跑；
9. 如何只重跑某一个实验；
10. 自动汇总文件位置。

---

# 52. 推荐最终运行命令

建议提供：

```bash
python scripts/v25/run_ras_night.py
```

如果需要 GPU 参数：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python scripts/v25/run_ras_night.py
```

若使用 torchrun，则由现有项目规范决定。

不要强行引入与当前训练入口不一致的启动方式。

---

# 53. 明早最先看什么

顺序：

```text
1. E0 是否复现 C2
2. E5 是否优于 E0
3. E9 是否优于 E8
4. E2 vs E3
5. E3 vs E4
6. E4 vs E5
7. accepted_harm_rate 是否明显低于 candidate_harm_rate
8. Step3/Step4 是否更少反弹
9. lambda sensitivity
10. structured masks 是否获益
```

---

# 54. 本轮实验成功的最低标准

不要求全部成立。

如果至少出现：

```text
E5 <= E0
```

并且：

```text
accepted_harm_rate
显著低于
candidate_harm_rate
```

同时：

```text
Step3 / Step4 nonworse rate 提升
```

则：

> RAS 值得继续深入。

如果再出现：

```text
E9 < E8
```

则更强：

> RAS 很可能不是 partner routing 专属技巧，而是更普遍的多轮补全机制。

---

# 55. 若 E5 不如 E0，也不要立即放弃

必须进一步区分：

```text
E3
E4
E5
```

---

## E3 好，E4 好，E5 差

说明：

```text
Selective Commit 有效
Router feedback 有害
```

下一版保留 gate，删除 rejection feedback。

---

## E3 差，E4 好

说明：

```text
Gate 必须显式监督
```

---

## E3/E4 都差

重点检查：

```text
latent commit 设计
acceptance feature
accepted decode
gate collapse
```

---

## E5 最终 MAE 无提升，但 harm rate 明显降低

说明机制有效，但：

```text
Gate 过于保守
丢失了 beneficial proposal
```

下一步研究：

```text
soft target
margin
acceptance calibration
```

而不是立即否定整个方向。

---

# 56. 本轮不做统计显著性结论

全部：

```text
seed = 7
```

因此明早报告必须写：

> 单 seed 探索性结果，仅用于机制筛选，不能据此声称统计显著性或稳定优势。

后续正式实验再跑：

```text
seed = 7
seed = 17
seed = 27
```

---

# 57. 研究逻辑总图

```text
E0 C2 Baseline
│
├── E1 Naive Feedback
│
│   └── 检查“无控制反馈”是否传播错误
│
├── E2 Legacy Completion Gate
│
│   └── 只 gate value
│
├── E3 Latent Selective Commit
│
│   └── gate latent update
│
├── E4 + Acceptance Supervision
│
│   └── 学习何时接受
│
├── E5 + Rejection Feedback
│
│   └── Full RAS closed loop
│
├── E6 λ=0.03
│
└── E7 λ=0.30


E8 Native Top-2
│
└── E9 Native Top-2 + Full RAS
```

---

# 58. 最终研究目标

今晚真正想获得的证据链是：

```text
多轮专家会产生有害 Proposal
        ↓
Naive Feedback 会传播错误
        ↓
仅 gate 输出无法阻止 latent 污染
        ↓
Selective Latent Commit 能阻止部分有害更新
        ↓
pseudo-missing supervision 提高 repair decision 质量
        ↓
rejection feedback 使下一轮 Router 面向剩余修复难点重路由
        ↓
多轮补全更加稳定
        ↓
最终 MAE / RMSE 改善
```

如果这条链成立，

RAS-CoE 的贡献就不再是：

> “把 CoE 用到了时空补全”

而是：

> **针对时空补全中修复可靠性随位置和轮次变化的问题，将传统无条件专家迭代改造成可拒绝有害更新、可反馈剩余修复状态的闭环选择性专家链。**

---

# 59. Codex 执行要求总结

Codex 需要完成：

```text
[1] RAS 核心结构实现
[2] 十组配置
[3] 夜间串行 Runner
[4] 断点续跑
[5] 单组失败继续后续
[6] 精确 repair metrics
[7] 九类缺失统计
[8] 自动 summary.csv
[9] 自动 comparison.md
[10] regression/unit/smoke tests
```

今晚运行前必须确保：

```text
E0 new-feature-off 与旧 C2 行为一致
```

然后冻结代码。

**开始跑十组实验之后，不要再根据中途结果修改模型或超参数。**
