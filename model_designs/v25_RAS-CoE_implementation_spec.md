# v25 RAS-CoE：Repair-Aware Selective Chain-of-Experts 实现设计文档

> **用途**：直接交给 Codex，基于当前 `v24-COE` 分支实现下一版结构。  
> **仓库**：`6xiaoming6/my_idea`  
> **基线分支**：`v24-COE`  
> **设计日期**：2026-09-28  
> **核心目标**：不要再把创新描述为“把 CoE 搬到时空补全”。新结构应把时空补全建模为一个**动态、闭环、可拒绝有害修改的逐轮修复过程**。

---

# 1. 新版本的核心定位

建议新方法暂定名：

## RAS-CoE：Repair-Aware Selective Chain-of-Experts

核心命题：

> **时空数据补全不是一次性专家选择问题，而是一个随着修复结果不断变化的序贯修复决策问题。**

传统多轮 CoE 的基本过程是：

```text
state -> route -> experts -> residual update -> next state
```

本项目的新结构应改造成：

```text
current repair state
        ↓
route experts
        ↓
experts propose an update
        ↓
decode candidate repair
        ↓
position-wise acceptance
        ↓
selectively commit latent update
        ↓
accepted completion
        ↓
repair feedback
        ↓
next-round routing
```

即：

\[
\text{Route}
\rightarrow
\text{Propose}
\rightarrow
\text{Evaluate}
\rightarrow
\text{Accept/Reject}
\rightarrow
\text{Feedback}
\rightarrow
\text{Re-route}
\]

这里真正新增的重点不是“多轮”，而是：

1. **专家的输出只是 Proposal，不再无条件写入状态；**
2. **每个时空位置可以独立决定接受多少本轮修改；**
3. **被拒绝的修改不能继续污染下一轮 hidden state；**
4. **下一轮 Router 显式知道上一轮哪些区域没有接受修复；**
5. **训练阶段利用 pseudo-missing 的真实值直接监督“候选修改是否比上一轮更好”，但推理阶段不使用真实值。**

---

# 2. 当前 v24-COE 的代码基础

当前核心文件：

```text
src/stmoe_imputer/models/temporal_spatial_coe.py
src/stmoe_imputer/models/coe_router.py
src/stmoe_imputer/losses.py
src/stmoe_imputer/engine.py
src/stmoe_imputer/models/imputer.py
src/stmoe_imputer/routing_metrics.py
scripts/train.py
```

当前主模型：

```python
TemporalSpatialCoE
```

注册为：

```text
architecture = "v24_ts_coe"
```

目前已经支持：

- 多轮 `num_steps`
- 共享/独立专家池
- `T / S / TD / SD / TA / ST / TL / SL`
- hard Top-k
- pair routing
- partner routing
- `partner_residual`
- corrected partner fusion
- completion feedback
- legacy acceptance gate
- 逐轮 candidate/completion 保存
- 路由与接受行为诊断

因此**不要复制一份新的 7 万行 `temporal_spatial_coe.py`**。

新版本优先在现有实现上做严格向后兼容的增量修改。

---

# 3. 当前 acceptance 为什么还不够

当前代码已经存在：

```python
acceptance in {"none", "point", "window"}
```

并且有：

```python
self.acceptance_head
```

当前 forward 中的关键流程大致为：

```python
hidden = hidden + update
candidate_prediction = self.decoder(hidden)

old_completion = completion

acceptance_weight = sigmoid(acceptance_head(...))

completion = old_completion + acceptance_weight * proposal
```

这个机制存在三个关键问题。

---

## 3.1 Hidden 已经无条件被修改

当前逻辑先执行：

```python
hidden = hidden + update
```

然后才计算 acceptance。

因此即使：

```text
acceptance_weight ≈ 0
```

表示当前 Proposal 应当被拒绝，

**expert update 仍然已经永久进入 hidden state。**

下一轮 Router 和 Expert 仍会读取这个被污染的 hidden。

这与“拒绝本轮修复”的语义不一致。

---

## 3.2 Acceptance 没有直接学习“这次修改是否变好”

当前 acceptance 主要依赖最终任务 loss 反向传播。

它没有明确的监督：

```text
candidate 是否比 old completion 更接近真实值？
```

但训练阶段的 pseudo-missing 位置实际上拥有真实值，可以构造明确标签。

因此当前 gate 没有充分利用补全任务天然提供的训练信号。

---

## 3.3 下一轮 Router 没有显式知道哪些 Proposal 被拒绝

当前 Router 会看到：

- hidden
- completion
- support
- change
- missing fraction

但没有明确的：

```text
previous rejection / previous acceptance
```

信号。

因此它并没有真正形成：

```text
“上一轮没有修好的区域 -> 下一轮换一种专家策略”
```

这样的闭环。

---

# 4. 本次实现范围

## 必须实现

第一版只实现以下四件事：

1. **Repair Proposal**
2. **Position-wise Latent Acceptance**
3. **Oracle-derived Acceptance Supervision**
4. **Repair Feedback to Next Router**

---

## 暂时不要实现

这一版禁止同时加入以下内容：

```text
× 多尺度输入
× Fine/Mid/Coarse 专家
× 新增更多专家
× 新的 28-pair Router
× Risk Estimator
× 动态停止轮数
× 新的 Attention Backbone
× 新的数据增强
× 新的缺失模式
× 蒸馏
× 分阶段训练
```

原因：

> 第一版必须能够清楚回答一个问题：  
> **选择性接受机制本身能否解决多轮 CoE 的 over-repair，并使逐轮修复更稳定？**

不要一次改变太多变量。

---

# 5. 推荐基线

建议基于当前表现最有价值的 C2 类结构继续：

```text
4 rounds
8 experts
shared expert pool
hard Top-2
partner_residual
corrected partner fusion
partner scoring head auxiliary supervision
```

专家池：

```python
[
    "T",
    "S",
    "TD",
    "SD",
    "TA",
    "ST",
    "TL",
    "SL",
]
```

主要配置：

```json
{
  "num_steps": 4,
  "expert_pool": ["T", "S", "TD", "SD", "TA", "ST", "TL", "SL"],
  "routing_mode": "hard",
  "top_k": 2,
  "pair_mode": "partner_residual",
  "partner_fusion": "corrected",
  "partner_aux_head_only": true,
  "expert_sharing": "shared",
  "router_state": "dynamic",
  "expert_state": "dynamic"
}
```

注意：

当前 C2 类实验中 `completion_feedback=false`。

新结构需要把它改成：

```json
"completion_feedback": true
```

但反馈的不是无条件 Candidate，而是**经过选择性接受之后的 Repair State**。

这也能够直接解释之前 naive feedback 可能变差的现象：

> 不是 feedback 本身没有意义，而是“把所有修复结果无条件反馈”可能把错误修复继续传播到后续轮次。

---

# 6. 新增配置接口

不要破坏已有：

```python
acceptance = "none" / "point" / "window"
```

旧实验必须还能完全复现。

建议增加新的独立配置：

```json
"repair_acceptance": "none",
"repair_feedback_to_router": false,
"repair_accept_init_prob": 0.99
```

支持：

```text
repair_acceptance:
    none
    latent_point
```

第一版只实现 `latent_point`。

---

## 配置合法性约束

当：

```json
"repair_acceptance": "latent_point"
```

时必须满足：

```text
acceptance == "none"
router_state == "dynamic"
expert_state == "dynamic"
completion_feedback == true
```

如果不满足，初始化时直接 `ValueError`。

当：

```json
"repair_feedback_to_router": true
```

时第一版要求：

```text
router_features == "legacy"
repair_acceptance == "latent_point"
```

不要在第一版处理 grouped router 的维度兼容。

---

# 7. 新模块：RepairAcceptanceGate

建议新增文件：

```text
src/stmoe_imputer/models/repair_gate.py
```

新增：

```python
class RepairAcceptanceGate(nn.Module):
    ...
```

---

# 8. Gate 的输入

每一轮 \(r\) 定义：

```text
hidden_before
update
old_completion
candidate_completion
previous_change
original_mask
support
```

形状：

```text
hidden_before:
[B, D, T, H, W]

update:
[B, D, T, H, W]

old_completion:
[B, C, T, H, W]

candidate_completion:
[B, C, T, H, W]

previous_change:
[B, C, T, H, W]

original_mask:
[B, C, T, H, W]

support:
[B, 10*C, T, H, W]
```

内部构造：

```python
proposal = candidate_completion - old_completion
proposal_abs = proposal.abs()
```

Gate 输入：

```python
gate_features = torch.cat(
    [
        hidden_before,
        update,
        old_completion,
        candidate_completion,
        proposal,
        proposal_abs,
        previous_change,
        original_mask,
        support,
    ],
    dim=1,
)
```

输入通道数：

\[
2D + 16C
\]

因为：

```text
old_completion       C
candidate_completion C
proposal             C
proposal_abs         C
previous_change      C
original_mask        C
support             10C

total = 16C
```

---

# 9. Gate 输出

不要再输出 `C` 个 acceptance channel。

新机制输出：

```text
[B, 1, T, H, W]
```

即：

> 每一个时空位置对应一个 latent-state acceptance score。

原因：

当前 hidden state 是所有变量共享的：

```text
[B, D, T, H, W]
```

如果每个变量产生不同 gate，很难定义怎样一致地作用于 hidden。

所以第一版采用：

> **point-wise acceptance，而不是 variable-wise acceptance。**

---

# 10. Gate 网络结构

推荐保持简单：

```python
self.net = nn.Sequential(
    nn.Conv3d(input_dim, head_dim, kernel_size=1),
    nn.GELU(),
    nn.Conv3d(head_dim, 1, kernel_size=1),
)
```

其中：

```python
head_dim = max(16, dim // 2)
```

第一版不要加入：

```text
× attention
× 3×3 convolution
× transformer
```

否则难以判断收益来自哪里。

---

# 11. Gate 初始化

需要尽量接近旧模型的“全部接受”。

配置：

```json
"repair_accept_init_prob": 0.99
```

最后一层：

```python
nn.init.zeros_(last.weight)

bias = log(p / (1 - p))
```

当：

```text
p = 0.99
```

时：

```text
bias ≈ 4.595
```

所以初始阶段：

```text
acceptance ≈ 0.99
```

即新模型一开始近似原始 CoE，然后逐渐学会拒绝有害 update。

---

# 12. 最核心的 forward 改造

当前代码：

```python
update = ...
hidden = hidden + update

candidate_prediction = decoder(hidden)
...
```

必须改成下面的逻辑。

---

## 12.1 保存旧状态

每轮开始：

```python
hidden_before = hidden
old_completion = completion
previous_change = change
```

---

## 12.2 Expert 只生成 Proposal

保持当前 shared/routed expert 逻辑不变：

```python
update = shared_update + routed_update
```

此时不能马上覆盖：

```python
hidden
```

而是：

```python
candidate_hidden = hidden_before + update
```

---

## 12.3 解码 Candidate

```python
candidate_prediction = self.decoder(candidate_hidden)

candidate_completion = torch.where(
    observed,
    x_input,
    candidate_prediction,
)
```

然后：

```python
proposal = candidate_completion - old_completion
```

---

## 12.4 预测 acceptance

```python
acceptance_logits = self.repair_acceptance_gate(
    hidden_before=hidden_before,
    update=update,
    old_completion=old_completion,
    candidate_completion=candidate_completion,
    previous_change=previous_change,
    original_mask=original_mask,
    support=support,
)
```

得到：

```python
acceptance_weight = torch.sigmoid(acceptance_logits)
```

形状：

```text
[B, 1, T, H, W]
```

---

# 13. Latent Selective Commit

这是新结构最关键的一步。

定义 point-level missing：

```python
missing = ~observed

point_missing = missing.any(dim=1, keepdim=True)
```

对于完全 observed 的位置：

```text
hidden update 正常允许
```

对于存在缺失变量的位置：

```text
由 acceptance gate 控制
```

因此：

```python
commit_weight = torch.where(
    point_missing,
    acceptance_weight,
    torch.ones_like(acceptance_weight),
)
```

然后：

```python
hidden = hidden_before + commit_weight * update
```

这一步必须发生在真正写回 hidden 之前。

---

# 14. 为什么一定要 gate hidden

如果只执行：

```python
completion = old + a * proposal
```

但仍然：

```python
hidden = hidden_before + update
```

那么：

```text
value 上拒绝了本轮修改
latent state 中却仍然接受了本轮修改
```

下一轮仍会读到错误 hidden。

这不是完整的 Selective Repair。

新结构必须保证：

> **Rejected Proposal 不继续污染后续链。**

---

# 15. Accepted Prediction 必须重新 decode

由于：

```python
hidden != candidate_hidden
```

所以不能继续使用：

```python
candidate_prediction
```

作为真正的 accepted prediction。

需要：

```python
accepted_prediction = self.decoder(hidden)
```

然后：

```python
completion = torch.where(
    observed,
    x_input,
    accepted_prediction,
)
```

最终：

```python
prediction = accepted_prediction
```

这样：

```text
hidden
prediction
completion
```

三者保持一致。

虽然每轮多一次 decoder forward，但当前 decoder 很轻：

```text
LayerNorm
1×1 Conv
GELU
1×1 Conv
```

因此优先保证语义正确，不要为了省这一点计算量制造 latent/value 不一致。

---

# 16. 下一轮的 Repair Feedback

下一轮必须读取 accepted state，而不是 Candidate。

因此：

```python
change = torch.where(
    observed,
    0,
    (completion - old_completion).abs(),
)
```

下一轮当前已有代码中的：

```python
_router_features(
    hidden,
    completion,
    ...,
    change,
)
```

应该读取：

```text
accepted hidden
accepted completion
accepted change
```

这已经形成第一层闭环。

---

# 17. 显式 Rejection Feedback

为了让 Router 明确知道上一轮哪些区域没有接受本次修复，再增加：

```python
repair_rejection = torch.where(
    point_missing,
    1.0 - acceptance_weight,
    0.0,
)
```

形状：

```text
[B, 1, T, H, W]
```

第一轮之前：

```python
previous_rejection = zeros
```

---

# 18. 修改 `_router_features`

当前：

```python
_router_features(
    hidden,
    values,
    support_summary,
    change,
    missing,
    pattern_summary=None,
)
```

增加：

```python
repair_signal: torch.Tensor | None = None
```

当：

```text
repair_feedback_to_router == true
```

时，额外加入两个标量：

```python
repair_global = repair_signal.mean(dim=(2, 3, 4))
repair_missing = self._missing_pool(repair_signal, point_missing)
```

然后 append：

```python
[
    repair_global,
    repair_missing,
]
```

因此 Router input dimension：

```python
router_input_dim += 2
```

第一轮：

```python
repair_signal = zeros
```

第二轮开始：

```python
repair_signal = previous_rejection
```

这意味着 Router 可以学习：

```text
上一轮大量被拒绝
-> 当前修复困难
-> 应改变下一轮专家策略
```

---

# 19. 为什么 Rejection Feedback 只加两个统计量

第一版不要把完整 rejection map 直接送入一个新 CNN Router。

原因：

1. 当前 Router 是 sample/window-level Router；
2. 当前专家选择也是 window-level；
3. 两个 pooled features 足以让 Router 感知“上一轮修复是否被大量拒绝”；
4. 改动最小；
5. 更适合消融；
6. 不会把 Router 复杂度突然放大。

后续如果这一机制有效，再研究 local routing。

---

# 20. Acceptance 的训练监督

这是新结构的第二个关键点。

训练数据中的 missing 是人工 mask 的 pseudo-missing。

因此在训练阶段：

```text
x_f_gt
```

是真实已知的。

但：

> **target 只能进入 loss，绝对不能进入 model forward 或 Gate input。**

---

# 21. Acceptance Oracle Label

需要在：

```text
src/stmoe_imputer/losses.py
```

新增：

```python
compute_repair_acceptance_loss(...)
```

对于第 \(r\) 轮：

```text
old_completion
candidate_completion
target
supervision mask Q
```

---

## 21.1 Supervision Mask

复用当前：

```python
supervision_mask(...)
```

得到：

```text
Q:
[B, C, T, H, W]
```

---

## 21.2 聚合到 point-level

因为 acceptance gate 是：

```text
[B, 1, T, H, W]
```

所以 error 同样按 point 聚合。

```python
q = selected.float()

count = q.sum(dim=1, keepdim=True)

valid_point = count > 0
```

旧状态 MAE：

```python
old_error = (
    (old_completion - target).abs() * q
).sum(dim=1, keepdim=True) / count.clamp_min(1)
```

Candidate MAE：

```python
candidate_error = (
    (candidate_completion - target).abs() * q
).sum(dim=1, keepdim=True) / count.clamp_min(1)
```

---

# 22. Label 定义

定义：

```python
improvement = old_error - candidate_error
```

如果：

```text
improvement > margin
```

说明：

```text
Candidate 更好
```

label：

```text
1
```

如果：

```text
improvement < -margin
```

label：

```text
0
```

如果：

```text
|improvement| <= margin
```

可以忽略该位置。

配置：

```json
"repair_accept_margin": 0.0
```

第一版默认 0。

未来若标签过于抖动，可以再增加 margin。

---

# 23. Label 必须 detach

标签构造：

```python
with torch.no_grad():
    ...
```

或者显式：

```python
old_error = old_error.detach()
candidate_error = candidate_error.detach()
```

绝对不能让：

```text
label generation
```

本身对 Candidate/Backbone 产生梯度。

---

# 24. Auxiliary Acceptance Logits

推荐采用和当前 C2 的：

```text
partner_aux_head_only
```

相同思想。

主 forward：

```python
acceptance_logits = gate(gate_features)
```

用于真正控制：

```text
hidden commit
```

另外：

```python
acceptance_aux_logits = gate(gate_features.detach())
```

使用**同一个 gate 参数**，但 input detach。

Acceptance classification loss 使用：

```python
acceptance_aux_logits
```

这样：

```text
Acceptance BCE
-> 只直接训练 Gate
-> 不通过 oracle label 直接修改 Backbone
```

而最终任务 loss：

```text
L_main
```

仍然可以通过真正的 acceptance gate 对 Backbone 和 Gate 正常反向传播。

配置：

```json
"repair_accept_aux_head_only": true
```

第一版建议固定开启。

---

# 25. Acceptance Loss

建议使用 class-balanced BCE。

不要简单：

```python
BCE(...).mean()
```

因为不同轮次中：

```text
beneficial candidate
harmful candidate
```

比例可能严重不平衡。

实现：

```python
positive = valid & (label == 1)
negative = valid & (label == 0)
```

如果正负都有：

```python
loss_pos = BCEWithLogits(logits[positive], 1).mean()
loss_neg = BCEWithLogits(logits[negative], 0).mean()

loss = 0.5 * (loss_pos + loss_neg)
```

如果只有一类：

```python
loss = BCEWithLogits(valid_logits, valid_labels).mean()
```

没有 valid point：

```python
loss = differentiable_zero
```

---

# 26. 总 Loss

当前 CoE：

\[
L =
L_{main}
+
\lambda_{mid}L_{mid}
+
\lambda_{balance}L_{balance}
+
\lambda_zL_z
\]

新增：

\[
L =
L_{main}
+
\lambda_{balance}L_{balance}
+
\lambda_{accept}L_{accept}
+
...
\]

第一轮实验建议：

```json
"lambda_coe_accept": 0.1
```

当前：

```json
"lambda_coe_mid": 0.0
```

继续保持。

不要同时打开大量新的 regularizer。

---

# 27. 可选 Monotonic Repair Loss

可以实现，但**第一轮主实验默认关闭**。

配置：

```json
"lambda_coe_monotonic": 0.0
```

定义：

\[
L_{mono}^{r}
=
\max(
0,
E(\hat X^{r},Y)
-
E(\hat X^{r-1},Y)
)
\]

也就是：

> 只惩罚 accepted completion 比上一轮更差的位置。

实现 point-level：

```python
accepted_error = ...
old_error = ...

regret = F.relu(
    accepted_error - old_error - tolerance
)
```

配置：

```json
"repair_monotonic_tolerance": 0.0
```

如果后续发现 gate 虽然分类准确，但逐轮 MAE 仍明显反弹，再打开：

```text
lambda_coe_monotonic = 0.02 / 0.05 / 0.1
```

第一版不要让它和 acceptance supervision 混在一起。

---

# 28. 修改 `compute_coe_loss`

当前签名：

```python
compute_coe_loss(outputs, batch, cfg)
```

建议改为：

```python
compute_coe_loss(
    outputs,
    batch,
    cfg,
    epoch: int | None = None,
)
```

并在：

```python
compute_main_stage_loss(...)
```

中继续传入现有 `epoch`。

即：

```python
return compute_coe_loss(
    outputs,
    batch,
    cfg,
    epoch=epoch,
)
```

即使第一版暂时不用 epoch，也为后续 acceptance warmup 保留接口。

---

# 29. `coe` 输出新增字段

在：

```python
TemporalSpatialCoE.forward()
```

输出中增加：

```python
"repair_acceptance_logits"
"repair_acceptance_aux_logits"
"repair_acceptance_weights"
"repair_rejection_maps"
"repair_commit_weights"
```

建议形状：

```text
repair_acceptance_logits:
[B, R, 1, T, H, W]

repair_acceptance_aux_logits:
[B, R, 1, T, H, W]

repair_acceptance_weights:
[B, R, 1, T, H, W]

repair_rejection_maps:
[B, R, 1, T, H, W]

repair_commit_weights:
[B, R, 1, T, H, W]
```

其中：

```text
R = num_steps
```

---

# 30. 保留已有字段

以下现有字段继续保留：

```python
"predictions"
"completions"
"candidate_predictions"
"candidate_completions"
"candidate_changes"
"changes"
"route_logits"
"route_probs"
"route_weights"
"selected_experts"
...
```

这样现有：

```text
_CoEQualityMetrics
routing metrics
experiment report
```

仍然可以复用。

---

# 31. 新增 diagnostics

每一步增加：

```text
step1_repair_accept_mean
step1_repair_accept_std
step1_repair_reject_mean
step1_repair_proposal_abs_mean
step1_repair_commit_below_05_rate

step2_...
step3_...
step4_...
```

全局：

```text
repair_accept_mean
repair_reject_mean
```

注意只统计 missing point 时，优先在 evaluation metrics 中使用精确 hidden count。

---

# 32. 扩展 `_CoEQualityMetrics`

当前已经统计：

```text
candidate_harm
candidate_benefit
accepted_harm
accepted_benefit
```

这部分非常有价值，保留。

另外新增：

---

## 32.1 每轮真实 MAE

```text
coe_initial_mae

coe_step1_mae
coe_step2_mae
coe_step3_mae
coe_step4_mae
```

必须基于：

```text
supervision_mask
```

精确累积 numerator/count。

不要对 batch MAE 再平均。

---

## 32.2 Candidate Harm Rate

不仅记录 harm magnitude，还记录：

```text
candidate_harm_rate
```

定义：

```text
candidate_error > old_error
```

的 supervised point 比例。

---

## 32.3 Accepted Harm Rate

```text
accepted_harm_rate
```

定义：

```text
accepted_error > old_error
```

比例。

这是证明 Selective Repair 是否有效的关键指标。

理想现象：

```text
candidate_harm_rate 较高
accepted_harm_rate 显著下降
```

即：

> 专家确实会提出错误修改，但 Gate 把其中相当一部分挡住。

---

# 33. Oracle Acceptance Rate

Evaluation 中可以利用 target 做诊断，但绝不能参与推理。

定义：

```text
oracle_accept = candidate_error < old_error
```

统计：

```text
coe_step{r}_oracle_accept_rate
```

同时统计模型：

```text
coe_step{r}_pred_accept_rate
```

其中：

```python
pred_accept = acceptance_weight > 0.5
```

---

# 34. Gate Classification Metrics

至少输出：

```text
accept_accuracy
accept_precision
accept_recall
accept_f1
```

按所有 supervised point 精确累计 TP/FP/TN/FN。

不强制做 AUROC。

---

# 35. Over-repair Prevention Rate

这是论文非常值得报告的指标。

定义：

```text
H = candidate 本来会让误差变大
```

即：

```python
candidate_error > old_error
```

如果最终：

```python
accepted_error <= candidate_error
```

说明 Gate 至少降低了 Candidate 的伤害。

更严格定义建议：

```python
candidate_harm = candidate_error > old_error
accepted_harm = accepted_error > old_error
```

然后：

\[
PreventionRate
=
1 -
\frac{\# accepted\_harm}
{\# candidate\_harm}
\]

仅在：

```text
candidate_harm_count > 0
```

时统计。

输出：

```text
coe_step{r}_overrepair_prevention_rate
```

---

# 36. Oracle Selective Repair Lower Bound

为了知道 Gate 还有多少提升空间，可以构造纯诊断 Oracle：

```python
oracle_point = torch.where(
    candidate_error < old_error,
    candidate,
    old,
)
```

注意：

> 只做 evaluation diagnostic，不进入正式预测。

记录：

```text
coe_step{r}_oracle_selective_mae
```

然后：

```text
actual accepted MAE - oracle selective MAE
```

就是 Gate gap。

输出：

```text
coe_step{r}_acceptance_oracle_gap
```

---

# 37. 最重要的机制指标：Monotonic Improvement

逐 sample 统计：

```text
MAE_r <= MAE_{r-1}
```

得到：

```text
coe_step1_nonworse_sample_rate
coe_step2_nonworse_sample_rate
coe_step3_nonworse_sample_rate
coe_step4_nonworse_sample_rate
```

以及：

```text
all_steps_monotonic_sample_rate
```

后者表示一个 sample 是否满足：

\[
E_1 \le E_0,
E_2 \le E_1,
E_3 \le E_2,
E_4 \le E_3
\]

这是未来论文最直观的机制证据之一。

---

# 38. 新 forward 伪代码

Codex 实现时应尽量按下面逻辑改。

```python
hidden = initial_hidden
completion = initial_completion
change = zeros_like(completion)
previous_rejection = zeros([B, 1, T, H, W])

for step in range(num_steps):

    # -------------------------------------------------
    # 1. Router reads accepted repair state
    # -------------------------------------------------

    router_features = self._router_features(
        hidden=hidden,
        values=completion,
        support_summary=support_summary,
        change=change,
        missing=missing,
        repair_signal=(
            previous_rejection
            if self.repair_feedback_to_router
            else None
        ),
    )

    route = route_experts(router_features)

    # -------------------------------------------------
    # 2. Experts produce an update proposal
    # -------------------------------------------------

    hidden_before = hidden
    old_completion = completion
    previous_change = change

    update = shared_update + routed_update

    candidate_hidden = hidden_before + update

    candidate_prediction = self.decoder(candidate_hidden)

    candidate_completion = torch.where(
        observed,
        x_input,
        candidate_prediction,
    )

    # -------------------------------------------------
    # 3. Repair acceptance
    # -------------------------------------------------

    if self.repair_acceptance == "latent_point":

        gate_features = ...

        acceptance_logits = self.repair_acceptance_gate(
            gate_features
        )

        acceptance_aux_logits = self.repair_acceptance_gate(
            gate_features.detach()
        )

        acceptance_weight = torch.sigmoid(
            acceptance_logits
        )

        point_missing = missing.any(
            dim=1,
            keepdim=True
        )

        commit_weight = torch.where(
            point_missing,
            acceptance_weight,
            torch.ones_like(
                acceptance_weight
            ),
        )

        # ---------------------------------------------
        # 4. Selectively commit expert update
        # ---------------------------------------------

        hidden = (
            hidden_before
            + commit_weight * update
        )

        accepted_prediction = self.decoder(hidden)

        completion = torch.where(
            observed,
            x_input,
            accepted_prediction,
        )

        prediction = accepted_prediction

        previous_rejection = torch.where(
            point_missing,
            1.0 - acceptance_weight,
            torch.zeros_like(
                acceptance_weight
            ),
        )

    else:

        # Exact original path
        hidden = candidate_hidden
        prediction = candidate_prediction
        completion = candidate_completion
        previous_rejection = zeros

    # -------------------------------------------------
    # 5. Accepted repair becomes next state
    # -------------------------------------------------

    change = torch.where(
        observed,
        torch.zeros_like(completion),
        (completion - old_completion).abs(),
    )

    save_histories(...)
```

---

# 39. 非常重要：关闭新功能时必须严格保持旧行为

当：

```json
"repair_acceptance": "none",
"repair_feedback_to_router": false
```

时：

> 同 seed、同配置下，forward 输出、参数数量、随机数消耗顺序和当前 v24 基线尽量保持一致。

至少要求：

```text
x_hat_main allclose
route_logits allclose
route_weights allclose
selected_experts equal
```

新模块只有开启时才实例化。

不要让新模块初始化改变旧模型 RNG 顺序。

---

# 40. 新模块初始化顺序

这是当前仓库很重视的问题。

当前已有多项测试保证：

```text
新增可选结构
不能改变 base model common parameters 的初始化
```

因此：

> `RepairAcceptanceGate` 必须在所有原有公共模块完成初始化后再创建。

或者：

```text
repair_acceptance == none
```

时根本不创建。

务必增加 equivalence test。

---

# 41. Config 建议

建议新增：

```text
configs/v25/
```

如果不想引入新目录，也可以：

```text
configs/v24/repair_acceptance/
```

但从研究版本管理角度，更推荐：

```text
configs/v25/
```

模型 registry 第一版仍可继续使用：

```text
architecture = "v24_ts_coe"
```

只通过新 flags 开启。

不要复制整套 backbone。

后续结构稳定后再增加 alias：

```text
"v25_ras_coe"
```

---

# 42. 推荐完整配置

第一版主配置建议：

```json
{
  "model": {
    "architecture": "v24_ts_coe",
    "c_in": 2,

    "main": {
      "dim": 64,
      "max_t": 12,
      "use_multiscale": false
    },

    "coe": {
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

      "pair_mode": "partner_residual",
      "partner_fusion": "corrected",
      "partner_aux_head_only": true,

      "completion_feedback": true,

      "acceptance": "none",

      "repair_acceptance": "latent_point",
      "repair_feedback_to_router": true,
      "repair_accept_aux_head_only": true,
      "repair_accept_init_prob": 0.99
    },

    "aux": {
      "enabled": false
    }
  },

  "loss": {
    "type": "l1",

    "lambda_coe_mid": 0.0,
    "lambda_coe_balance": 0.01,
    "lambda_coe_z": 0.0,

    "lambda_coe_accept": 0.1,
    "lambda_coe_monotonic": 0.0,

    "repair_accept_margin": 0.0,
    "repair_monotonic_tolerance": 0.0
  },

  "train": {
    "epochs": 120,
    "val_epoch": 2,

    "lr_main": 0.001,
    "weight_decay": 0.0001,
    "grad_clip_norm": 1.0,

    "amp": true,

    "scheduler": {
      "type": "cosine",
      "total_epochs": 120,
      "eta_min": 0.000001
    },

    "early_stopping": {
      "enabled": false
    }
  }
}
```

---

# 43. 第一轮实验不要直接跑 120 epoch

开发流程：

## Step A：unit tests

全部通过。

## Step B：smoke test

小 tensor：

```text
B=2
C=2
T=4
H=6
W=6
dim=16
num_steps=2
```

完成：

```text
forward
backward
AMP
DDP basic check
```

## Step C：短实验

TaxiBJ：

```text
20 epoch
seed=7
```

只检查：

```text
loss 正常
gate 不 NaN
gate 有梯度
accepted_harm_rate 是否低于 candidate_harm_rate
```

## Step D：探索实验

```text
80 epoch
seed=7
```

## Step E：正式实验

结构确定后：

```text
120 epoch
seeds = [7, 17, 27]
```

然后 BikeNYC 同协议复核。

---

# 44. 必须做的 6 组消融

为了最终能够证明创新，建议一次性准备以下配置。

---

## A0：Current C2 Baseline

```text
completion_feedback = false
repair_acceptance = none
repair_feedback_to_router = false
```

这是当前 strongest routing/fusion reference。

---

## A1：Naive Feedback

```text
completion_feedback = true
repair_acceptance = none
repair_feedback_to_router = false
```

证明：

> 单纯把前一轮 completion 喂给下一轮是否有效。

---

## A2：Legacy Acceptance

使用当前已有：

```text
acceptance = point
repair_acceptance = none
```

用于说明：

> 单纯输出端 interpolation gate 是否足够。

---

## A3：Latent Selective Commit

```text
repair_acceptance = latent_point
lambda_coe_accept = 0
repair_feedback_to_router = false
```

只让最终 task loss 自己学习 acceptance。

---

## A4：Supervised Selective Commit

```text
repair_acceptance = latent_point
lambda_coe_accept = 0.1
repair_feedback_to_router = false
```

验证：

> pseudo-missing oracle supervision 是否必要。

---

## A5：Full RAS-CoE

```text
repair_acceptance = latent_point
lambda_coe_accept = 0.1
repair_feedback_to_router = true
```

这是最终新结构。

---

# 45. 后续可选 A6

只有 A5 已经有效时再做：

```text
A6 = A5 + monotonic loss
```

例如：

```json
"lambda_coe_monotonic": 0.05
```

不要第一轮就混进去。

---

# 46. 论文级实验假设

新结构真正需要验证的不是：

```text
“加了 gate，最终 MAE 降低”
```

而是下面四个机制假设。

---

## H1：Candidate 确实存在 over-repair

需要看到：

```text
candidate_harm_rate > 0
```

且后轮尤其明显。

---

## H2：Gate 能识别有害 Proposal

需要看到：

```text
acceptance F1 > random / majority baseline
```

并且：

```text
harmful candidate 的 acceptance 显著低于 beneficial candidate
```

---

## H3：Selective Commit 能降低 over-repair

最重要：

```text
accepted_harm_rate
<
candidate_harm_rate
```

而且差异明显。

---

## H4：修复反馈能改善下一轮决策

Full RAS-CoE 应相对 A4：

```text
最终 MAE 更好
或
逐轮 monotonic rate 更高
或
结构化缺失族明显改善
```

并且 Router path 应出现与上一轮 rejection 状态有关的变化。

---

# 47. 最终理想的逐轮结果

原始模型可能出现：

```text
Initial : 60
Step 1  : 31
Step 2  : 23
Step 3  : 20
Step 4  : 22
```

新模型理想：

```text
Initial : 60
Step 1  : 31
Step 2  : 23
Step 3  : 20
Step 4  : 19.8
```

或者至少：

```text
Initial : 60
Step 1  : 31
Step 2  : 23
Step 3  : 20
Step 4  : 20.1
```

即：

> 后续轮次即使没有明显收益，也尽量不要破坏已经修好的区域。

---

# 48. 结构化缺失必须单独报告

当前数据已经有九类 mask：

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

新方法特别应该关注：

```text
node_outage
spatial_region
spatiotemporal_block
stripe
moving_region
multi_block
composite
```

原因：

> over-repair 和 repair state evolution 在结构化缺失中更有意义。

不要只报告总 MAE。

---

# 49. Unit Tests

新增：

```text
tests/test_v25_repair_acceptance.py
```

至少实现以下测试。

---

## test_1: disabled_equivalence

关闭：

```text
repair_acceptance
repair_feedback
```

同 seed 与旧模型：

```text
forward allclose
route weights allclose
```

---

## test_2: gate_shape_and_range

检查：

```text
acceptance logits
[B,R,1,T,H,W]

weights ∈ (0,1)
```

---

## test_3: full_accept_matches_candidate

把 gate bias 强制：

```text
+40
```

则：

```text
accepted hidden ≈ candidate hidden
accepted completion ≈ candidate completion
```

---

## test_4: full_reject_preserves_missing_hidden_update

把 gate bias：

```text
-40
```

对于 missing point：

```text
hidden_after ≈ hidden_before
```

observed point 仍允许正常 update。

---

## test_5: no_target_leakage

构造两个 batch：

```text
x_f_obs 相同
m_f 相同
x_f_gt 不同
```

model forward 输出必须完全相同。

只有：

```text
acceptance supervision loss
```

可以不同。

---

## test_6: oracle_label_correctness

手工构造：

```text
old error = 3
candidate error = 1
-> label 1

old error = 1
candidate error = 4
-> label 0
```

检查 point aggregation。

---

## test_7: acceptance_aux_only_gradient

开启：

```text
repair_accept_aux_head_only = true
```

只反传 acceptance BCE：

```text
Gate 参数有梯度
Backbone / Expert 参数没有来自该 auxiliary branch 的梯度
```

---

## test_8: router_feedback_changes_router_input

固定 hidden/completion/change，

只修改：

```text
previous_rejection
```

检查：

```text
router feature
```

发生变化。

---

## test_9: empty_supervision_safe

没有 pseudo-missing label 时：

```text
acceptance loss = differentiable 0
无 NaN
```

---

## test_10: AMP

CUDA 可用时验证：

```text
forward/backward
finite
```

---

# 50. Existing Tests

实现完成后至少运行：

```bash
pytest tests/test_v24_coe.py
pytest tests/test_v24_partner.py
pytest tests/test_v24_partner_residual4.py
pytest tests/test_v24_team_accept_v4.py
pytest tests/test_v24_ddp.py
pytest tests/test_v24_amp_dispatch.py
pytest tests/test_v25_repair_acceptance.py
```

最终最好：

```bash
pytest tests
```

---

# 51. 不允许出现的数据泄漏

以下内容绝对不能进入：

```text
Router input
Acceptance Gate input
Expert input
Repair feedback
Inference
```

禁止：

```text
x_f_gt
target_mask 中的真实值
old_error
candidate_error
oracle label
oracle best pair
```

它们只能存在于：

```text
loss
evaluation diagnostics
offline analysis
```

---

# 52. DDP 注意事项

Acceptance loss 使用的是 local batch supervised point。

如果只是作为训练 loss：

```text
各 rank 正常 backward
DDP 自动同步参数梯度
```

即可。

日志若要精确：

```text
TP
FP
TN
FN
absolute error numerator
count
harm count
```

必须做：

```text
sum-reduction
```

不要：

```text
先计算每卡 ratio
再平均 ratio
```

否则双卡结果不精确。

---

# 53. AMP 注意事项

以下建议用 FP32：

```text
acceptance BCE
oracle label error comparison
metric accumulation
```

Gate forward 本身可以使用 AMP。

error comparison：

```python
old_error = old.float()
candidate_error = candidate.float()
target = target.float()
```

避免 FP16/BF16 近似导致标签翻转。

---

# 54. 推理行为

测试阶段完全不使用 target。

流程：

```text
Round r

Router
↓
Top-2 / Partner Collaboration
↓
Candidate Update
↓
Candidate Decode
↓
Repair Acceptance Gate
↓
Selective Hidden Commit
↓
Accepted Decode
↓
Accepted Completion
↓
Rejection Feedback
↓
Round r+1
```

这才是最终可部署路径。

---

# 55. 计算量记录

新结构每轮额外：

```text
1 个轻量 1×1 Repair Gate
1 次 accepted hidden decode
```

因此实验报告必须新增：

```text
params
train time
forward latency
peak memory
```

目标不是要求零开销，而是证明：

> 提升不是靠大规模扩大 Backbone。

---

# 56. 不要把 Gate 变成“自动作弊”

如果发现模型：

```text
acceptance -> 全部接近 0
```

但 final MAE 看起来没有崩，

必须检查：

```text
initial completion 是否本来就很强
```

同时报告：

```text
accept mean
oracle accept rate
candidate benefit rate
missed benefit rate
```

不能只看最终 MAE。

---

# 57. Acceptance Collapse 判断

建议判定：

```text
mean acceptance < 0.05
```

或：

```text
mean acceptance > 0.95
```

持续大量 epoch 时，记录为可能 collapse。

但不要训练时强行人为限制。

第一版先观察。

---

# 58. 未来论文中的方法叙述

不要写：

> We introduce CoE into spatiotemporal imputation.

应该写成：

> Existing expert-routing approaches generally treat each refinement step as an unconditional state update. However, in spatiotemporal imputation, the reliability of a repair is spatially heterogeneous: an expert update may correct one missing region while corrupting another region that has already been well reconstructed. We therefore formulate multi-round imputation as a selective repair process. At each round, routed experts first generate a repair proposal; a repair-aware acceptance module then decides, at each spatiotemporal position, how much of the latent update should be committed. Only the accepted state is propagated to subsequent rounds, while rejection statistics are fed back to the next router to adapt expert collaboration to the remaining repair difficulty.

中文核心表述：

> 传统多轮专家链通常默认每轮专家更新都应被写入后续状态，但时空补全中的不同缺失位置具有显著不同的修复可靠性，同一次专家更新可能改善部分区域，却破坏已经较准确的区域。为此，本研究将多轮补全重新建模为“候选修复—选择性接受—状态反馈”的闭环过程：专家首先产生候选修复，位置级接受模块判断各时空位置的更新是否值得提交，只有被接受的隐状态更新才进入下一轮，并将上一轮拒绝信息反馈给 Router，使后续专家选择针对尚未有效修复的区域动态调整。

---

# 59. 论文中真正要证明的创新不是 Gate 本身

不要最后只得到：

```text
“我们加了一个 sigmoid gate”
```

真正需要通过实验支持的是：

> **多轮补全存在 proposal-specific over-repair；  
> 无条件状态更新会把错误修复继续传播；  
> 通过显式学习“当前 Proposal 是否比旧状态更好”，可以把专家链从无条件迭代改造成选择性修复过程；  
> 被接受后的 repair state 与 rejection feedback 再改变下一轮专家协作。**

这是完整的方法机制。

---

# 60. 第一版 Codex 实现顺序

严格按以下顺序执行。

## Phase 1

新增：

```text
RepairAcceptanceGate
```

但不接 loss。

完成：

```text
latent selective commit
accepted decode
```

测试通过。

---

## Phase 2

新增：

```text
acceptance oracle label
acceptance BCE
aux-head-only gradient
```

测试通过。

---

## Phase 3

新增：

```text
repair_rejection
router feedback
```

测试通过。

---

## Phase 4

扩展：

```text
_CoEQualityMetrics
```

加入：

```text
step MAE
harm rate
accept metrics
oracle gap
monotonic sample rate
```

---

## Phase 5

新增 6 组配置：

```text
A0
A1
A2
A3
A4
A5
```

先进行：

```text
20 epoch pilot
```

不要直接开始正式长实验。

---

# 61. Codex 最终交付要求

Codex 完成后必须给出：

1. 修改文件列表；
2. 每个文件修改内容摘要；
3. 新增配置字段说明；
4. forward 新旧数据流对比；
5. 所有新 loss 公式；
6. unit test 结果；
7. 原 v24 regression test 结果；
8. 参数量变化；
9. smoke test 的 acceptance 统计；
10. 是否发现任何 backward compatibility 问题；
11. 不要在未说明的情况下修改已有实验结果文件；
12. 不要自动覆盖当前 `v24-COE` 的历史输出。

---

# 62. 第一阶段成功标准

只有同时满足下面条件，才值得继续做多尺度。

## 性能

```text
A5 final MAE <= A0
```

最好有明确改善。

---

## 机制

至少满足：

```text
accepted_harm_rate
<
candidate_harm_rate
```

并且不是微小差异。

---

## 多轮

后两轮：

```text
nonworse sample rate
```

明显高于当前 baseline。

---

## Gate

```text
acceptance 不坍缩
F1 有意义
```

---

## 反馈

A5 至少在：

```text
final MAE
monotonicity
structured missing
```

其中一项明显优于 A4。

---

# 63. 如果第一阶段失败，如何判断失败原因

## 情况 1：Gate classification 很差

说明：

```text
当前 observable features 无法判断 Proposal 是否值得接受
```

下一步研究 Gate input。

不要先加多尺度。

---

## 情况 2：Gate F1 很好，但最终 MAE 不提升

说明：

```text
binary accept/reject 不是主要瓶颈
或 latent commit 与 value error 不完全一致
```

再考虑：

```text
soft improvement target
regret loss
```

---

## 情况 3：A4 好，A5 变差

说明：

```text
显式 rejection feedback 干扰 Router
```

则保留 Selective Commit，
删除 Router feedback。

论文主线仍然成立。

---

## 情况 4：A3 就明显有效

说明：

```text
结构本身已经有价值
```

Acceptance oracle supervision 可作为增强而不是必要组件。

---

## 情况 5：A2 与 A4 差不多

需要检查：

```text
hidden selective commit
```

是否真正带来了区别。

重点比较：

```text
下一轮 route
hidden state
accepted harm rate
```

---

# 64. 后续第二创新：多尺度

只有本版本稳定以后，再做：

```text
Repair-State-Aware Multi-Scale Evidence
```

届时不是简单：

```text
Fine Expert
Mid Expert
Coarse Expert
```

而应让：

```text
当前 Repair State
```

决定下一轮更需要：

```text
coarse structural evidence
mid regional evidence
fine local evidence
```

这一部分不要和本次代码改动混合。

---

# 65. 最终一句话

本次改造的目标不是：

> “给 v24 加一个 acceptance gate”。

而是把当前结构从：

```text
Router
→ Experts
→ unconditional update
→ next round
```

真正改造成：

```text
Repair State
→ Expert Collaboration
→ Repair Proposal
→ Position-wise Acceptance
→ Selective Latent Commit
→ Accepted Repair State
→ Rejection Feedback
→ Re-routing
```

如果这条链能够在实验中证明：

```text
有害 Candidate 很常见
↓
Gate 能识别它
↓
被接受后的伤害显著减少
↓
后续轮次更稳定
↓
最终补全更好
```

那么这一部分就不再是“把 CoE 简单搬到时空数据补全”，而会变成一个明确针对**时空缺失逐轮修复特性**重新设计的专家链机制。
