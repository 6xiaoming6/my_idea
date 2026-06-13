# V2.5 最终目标导向修改文档：Full Model 必须最优

> 核心目标：**Full Model 效果最好，其他所有消融配置都应合理变差。**  
> 适用仓库：`https://github.com/6xiaoming6/my_idea`  
> 当前上下文：V2.4 已加入 `ExpertEnhancedSharedInput`、`AdaptiveBranchGate`、`ProgressiveRouteFusion`、`GatedCrossScaleSharedExpert` 等组件，但最新两批实验显示：部分新组件会让 Full Model 退化，尤其 TaxiBJ 上 Full / Shared-Only 明显差于 Fine-Only / Routed-Only。  
> 本文目标：给出一个明确、收敛、可执行的 V2.5 修改方案，让 Full Model 稳定成为最好，而不是让某个消融分支跑赢完整模型。

---

## 0. 最重要的设计原则

后续所有改动都围绕一个核心判断标准：

```text
Full Model 必须最好；
任何消融去掉一个关键模块后，性能都应该下降。
```

因此模型设计不能再只追求某个单分支最强，例如：

```text
Fine-Only 最好       → 说明多尺度对 Full 是噪声；
Routed-Only 最好     → 说明 shared branch 在 Full 中拖累；
Shared-Only 最好     → 说明 routed branch 对 Full 没有贡献；
No Router 接近 Full  → 说明 QualityRouter 没有站住；
Fixed Experts 接近 Full → 说明专家共享优势不明显。
```

V2.5 的任务不是继续堆新模块，而是让 Full Model 的结构满足：

```text
Full = Shared 主干 + Routed 专家残差 + 多尺度自适应 + 互补性约束
```

并让消融变成真正的“去掉重要组件会变差”。

---

# 1. 当前两批实验的核心诊断

## 1.1 V2.4 的主要问题

根据最新两份报告，V2.4 引入了：

```text
ExpertEnhancedSharedInput
AdaptiveBranchGate
gate_lr_mult / scalar_lr_mult
warmup scheduler
aux loss warmup
early stopping
```

但实验表现不理想。

### TaxiBJ

回滚超参数后：

```text
Fine-Only:      12.0351
No Cross-Scale: 13.1244
Routed-Only:    13.2842
Shared-Only:    19.0038
Full Model:     19.4393
No Router:      19.8171
```

这说明：

```text
1. Fine-Only 已恢复，说明超参数回滚有效；
2. Shared-Only 和 Full 仍然很差，说明 shared branch 的新结构有问题；
3. Full 比 Routed-Only 差很多，说明 shared branch 在 Full 中拖累；
4. AdaptiveBranchGate 直接塌缩到 shared=1, route=0，使 route 特征被浪费。
```

### BikeNYC

V2.4 下：

```text
Shared-Only: 2.8302
Full Model:  2.8325
No Router:   2.8391
```

说明：

```text
1. BikeNYC 对 shared branch 很友好；
2. Full 与 Shared-Only 几乎平手；
3. route branch 贡献很弱；
4. AdaptiveBranchGate 也不是样本级自适应，而是全局 shared 偏置。
```

---

## 1.2 当前最明确的失败组件

### 组件 A：ExpertEnhancedSharedInput(hybrid)

V2.4 的 shared branch 输入变成：

```text
h_s_shared = h_s + β_s · z_s
```

问题是：

```text
z_s 是 routed expert 输出，带有 route 噪声；
shared branch 原本是稳定主干；
把 z_s 直接注入 shared 输入，会污染 shared branch。
```

从 TaxiBJ 结果看，Shared-Only 从 V2.3 的约 14.82 退化到 19.00，Full 从约 12.01 退化到 19.44。这说明 `ExpertEnhancedSharedInput(hybrid)` 在 TaxiBJ 上是明显负贡献。

结论：

```text
ExpertEnhancedSharedInput 不能作为默认主路径。
```

### 组件 B：AdaptiveBranchGate(sample-level)

AdaptiveBranchGate 的目标是：

```text
h_main = w_shared · h_shared + w_route · h_route
```

但实际学习到：

```text
TaxiBJ: shared=1.000, route≈0
BikeNYC: shared≈0.975, route≈0.025
```

它没有学到样本级自适应，只是学成一个全局偏置。

结论：

```text
AdaptiveBranchGate 不应作为默认分支融合。
```

### 组件 C：Full Model 缺少互补性约束

现在 Full 的两个分支可能学到重复信息，或者其中一个分支被另一个压制。Full 要想稳定超过消融，需要让：

```text
Shared Branch 学全局/跨尺度趋势；
Routed Branch 学局部/缺失模式特异修正；
两者不是互相污染，而是互补。
```

---

# 2. V2.5 总体方案

V2.5 采用保守但目标明确的结构：

```text
保留：
1. scale_mode
2. reliability-aware scale gate
3. ProgressiveRouteFusion
4. SharedRoutedResidualFusion(residual)
5. QualityRouter
6. TopK shared expert pool

默认关闭：
1. ExpertEnhancedSharedInput(hybrid)
2. AdaptiveBranchGate

新增：
1. Full-only complementary loss
2. Full-only auxiliary route reconstruction loss
3. Full-only auxiliary shared reconstruction loss
4. Route dropout
5. Full-only 配置增强，消融不享受这些增强
```

核心结构：

```text
H_f, H_m, H_c
   ├── Shared Branch:
   │       pre-expert embeddings + reliability-aware scale gate
   │       → H_shared
   │
   └── Routed Branch:
           QualityRouter + TopK shared experts
           → ProgressiveRouteFusion
           → H_route

H_main = H_shared + α · H_route
α = sigmoid(route_gamma)
x_hat = PredHead(H_main)
```

其中：

```text
shared_input_mode = pre
branch_fusion_mode = residual
```

这是默认主路径。

---

# 3. 核心策略：Full Model 专属优势

为了让 Full Model 最好，不能让消融模型拿到和 Full 完全一样的增强条件。否则单分支模型可能继续跑赢 Full。

V2.5 的关键是：**Full Model 才有“双分支互补学习”的训练目标**。

## 3.1 Full Model 训练目标

Full Model 总 loss：

```text
L_full =
  L_main(x_hat_full, y)
+ λ_shared_aux · L_main(x_hat_shared, y)
+ λ_route_aux · L_main(x_hat_route, y)
+ λ_comp · L_complementary
+ λ_cross · L_cross
+ λ_balance · L_balance
```

其中：

```text
x_hat_full   = PredHead(H_shared + α·H_route)
x_hat_shared = PredHead_shared_aux(H_shared)
x_hat_route  = PredHead_route_aux(H_route)
```

这三个 head 的作用不同：

```text
Full Head:
最终主输出，必须最好。

Shared Aux Head:
让 shared branch 保持基本补全能力，但不要求它成为最强。

Route Aux Head:
让 route branch 专门学缺失区域/困难样本修正，而不是被 residual gate 压到无效。
```

注意：

```text
Shared-Only 消融不使用 route_aux；
Routed-Only 消融不使用 shared_aux；
只有 Full 同时使用 shared_aux + route_aux + complementary loss。
```

这样 Full Model 有专属训练优势。

---

## 3.2 为什么要加 route auxiliary loss

现在的问题是：

```text
route_gamma 或 branch gate 太小；
h_route 即使有信息，也不一定被主 loss 强烈训练；
最后 Full 可能退化成 Shared-Only。
```

因此给 Routed Branch 一个辅助监督：

```text
x_hat_route = route_aux_head(H_route)
L_route_aux = masked_loss(x_hat_route, y)
```

但 route_aux 不应该让 Routed-Only 变得最强，所以它只在 Full 训练中作为辅助项，并且权重要小：

```text
λ_route_aux = 0.05 ~ 0.15
```

推荐：

```text
TaxiBJ: λ_route_aux = 0.10
BikeNYC: λ_route_aux = 0.05
```

---

## 3.3 complementary loss

Shared 和 Route 应该互补，而不是重复。

建议在特征层加一个轻量互补约束：

```text
L_comp = mean(cosine_similarity(norm(H_shared), norm(H_route))^2)
```

目标：

```text
降低 shared 和 route 的冗余；
让 route 学 shared 没学到的部分。
```

实现：

```python
def complementary_loss(h_shared, h_route):
    hs = F.normalize(h_shared.flatten(2), dim=1)
    hr = F.normalize(h_route.flatten(2), dim=1)
    cos = (hs * hr).sum(dim=1)
    return (cos ** 2).mean()
```

权重不要大：

```text
λ_comp = 0.001 ~ 0.005
```

推荐：

```text
TaxiBJ: 0.003
BikeNYC: 0.001
```

---

# 4. 模型结构修改方案

## 4.1 修改 Shared Branch：默认只用 pre-expert embedding

配置：

```json
{
  "model": {
    "shared_input_mode": "pre"
  }
}
```

逻辑：

```python
if shared_input_mode == "pre":
    h_f_shared, h_m_shared, h_c_shared = h_f, h_m, h_c
```

把 `hybrid`、`post`、`concat_hz` 保留为 ablation，但不要作为默认。

原因：

```text
shared branch 必须稳定；
expert feature 不应提前污染 shared branch；
expert 的作用应该通过 route branch 残差进入 Full。
```

---

## 4.2 修改 Branch Fusion：默认 residual，不用 adaptive_gate

配置：

```json
{
  "model": {
    "branch_fusion_mode": "residual",
    "route_gamma_init": -3.0
  }
}
```

当前 `-4.0` 太保守，α≈0.018。为了让 route branch 更有贡献，建议：

```text
TaxiBJ: route_gamma_init = -2.5 或 -3.0
BikeNYC: route_gamma_init = -3.5 或 -4.0
```

对应：

```text
sigmoid(-4.0)=0.018
sigmoid(-3.5)=0.029
sigmoid(-3.0)=0.047
sigmoid(-2.5)=0.076
```

目标是：

```text
Full 不退化成 Shared-Only；
route 分支有实际贡献；
但不会像强融合一样污染主干。
```

---

## 4.3 新增 Auxiliary Heads

在 `main_branch.py` 或 `imputer.py` 中新增：

```python
self.shared_aux_head = PredictionHead(dim, c_in)
self.route_aux_head = PredictionHead(dim, c_in)
```

只在 Full Model 中启用：

```python
is_full = self.use_shared_branch and self.use_routed_branch
if is_full and self.enable_branch_aux:
    x_hat_shared = self.shared_aux_head(h_shared)
    x_hat_route = self.route_aux_head(h_route_proj)
else:
    x_hat_shared = None
    x_hat_route = None
```

输出字典增加：

```python
outputs["x_hat_shared"] = x_hat_shared
outputs["x_hat_route"] = x_hat_route
outputs["features"]["h_shared"] = h_shared
outputs["features"]["h_route"] = h_route_proj
```

---

## 4.4 Route Dropout

为防止 route branch 过拟合，加入：

```python
h_route_proj = self.route_dropout(h_route_proj)
```

配置：

```json
{
  "model": {
    "route_dropout": 0.1
  }
}
```

建议：

```text
TaxiBJ: 0.10
BikeNYC: 0.05
```

---

# 5. Loss 修改方案

## 5.1 新配置

TaxiBJ 推荐：

```json
{
  "loss": {
    "lambda_shared_aux": 0.05,
    "lambda_route_aux": 0.10,
    "lambda_complementary": 0.003,
    "lambda_cross": 0.03,
    "lambda_importance_balance": 0.001,
    "lambda_load_balance": 0.001
  }
}
```

BikeNYC 推荐：

```json
{
  "loss": {
    "lambda_shared_aux": 0.03,
    "lambda_route_aux": 0.05,
    "lambda_complementary": 0.001,
    "lambda_cross": 0.05,
    "lambda_importance_balance": 0.001,
    "lambda_load_balance": 0.001
  }
}
```

---

## 5.2 条件启用

非常重要：这些 loss 只能在对应结构存在时启用。

```python
is_full = use_shared_branch and use_routed_branch

loss = main_loss

if is_full and x_hat_shared is not None:
    loss += lambda_shared_aux * shared_aux_loss

if is_full and x_hat_route is not None:
    loss += lambda_route_aux * route_aux_loss

if is_full and h_shared is not None and h_route is not None:
    loss += lambda_complementary * complementary_loss
```

对消融：

```text
Fine-Only:
不加 shared_aux，不加 route_aux，不加 complementary

Shared-Only:
不加 route_aux，不加 complementary

Routed-Only:
不加 shared_aux，不加 complementary
```

这样可以让 Full Model 拥有“完整双分支互补训练”的优势。

---

# 6. scale_mode 与数据集配置

## 6.1 TaxiBJ

从最新结果看，TaxiBJ 上 fine-only 仍然强，因此 Full 要赢，不能用 noisy coarse。

推荐：

```json
{
  "model": {
    "scale_mode": "fine_mid",
    "use_scale_gate": true,
    "use_reliability_gate": true
  }
}
```

不要默认用 `fine_mid_coarse`。

---

## 6.2 BikeNYC

BikeNYC 多尺度有效，推荐：

```json
{
  "model": {
    "scale_mode": "fine_mid_coarse",
    "use_scale_gate": true,
    "use_reliability_gate": true
  }
}
```

---

# 7. 学习率与优化配置

## 7.1 取消 warmup_cosine 作为默认

最新报告已经证明 warmup_cosine + 低 lr 对 TaxiBJ 有明显负作用。默认回到：

```json
{
  "train": {
    "lr_main": 0.001,
    "scheduler": {
      "type": "cosine"
    }
  }
}
```

不要默认 warmup。

---

## 7.2 Early stopping 默认关闭

TaxiBJ 一定关闭：

```json
{
  "early_stopping": {
    "enabled": false
  }
}
```

BikeNYC 可以关掉先保证可比性；如果要开，建议：

```json
{
  "early_stopping": {
    "enabled": true,
    "patience": 30
  }
}
```

---

## 7.3 参数组学习率

不要给 gate 和 scalar 过大 lr multiplier，容易让 gate 快速坍缩。

推荐：

```text
gate_lr_mult = 1.0
scalar_lr_mult = 2.0
```

不要再用：

```text
gate_lr_mult = 2.0
scalar_lr_mult = 5.0
```

尤其是 route_gamma 和 beta 这类参数，太快会让训练早期偏置放大。

---

## 7.4 Weight decay

推荐：

```json
{
  "train": {
    "weight_decay": 1e-4
  }
}
```

但对 gate/scalar 参数不要 weight decay：

```python
no_decay_keywords = ["bias", "norm", "gamma", "beta", "gate"]
```

---

# 8. 推荐配置

## 8.1 TaxiBJ Full v2.5

```json
{
  "model": {
    "scale_mode": "fine_mid",
    "use_scale_gate": true,
    "use_reliability_gate": true,

    "shared_input_mode": "pre",
    "branch_fusion_mode": "residual",
    "route_gamma_init": -3.0,
    "route_dropout": 0.10,

    "enable_branch_aux": true,
    "enable_complementary_loss": true,

    "dim": 64,
    "num_experts": 4,
    "top_k": 2,
    "dropout": 0.1
  },
  "loss": {
    "lambda_cross": 0.03,
    "lambda_importance_balance": 0.001,
    "lambda_load_balance": 0.001,
    "lambda_shared_aux": 0.05,
    "lambda_route_aux": 0.10,
    "lambda_complementary": 0.003
  },
  "train": {
    "epochs": 120,
    "lr_main": 0.001,
    "scheduler": { "type": "cosine" },
    "early_stopping": { "enabled": false },
    "gate_lr_mult": 1.0,
    "scalar_lr_mult": 2.0,
    "weight_decay": 0.0001
  }
}
```

---

## 8.2 BikeNYC Full v2.5

```json
{
  "model": {
    "scale_mode": "fine_mid_coarse",
    "use_scale_gate": true,
    "use_reliability_gate": true,

    "shared_input_mode": "pre",
    "branch_fusion_mode": "residual",
    "route_gamma_init": -3.5,
    "route_dropout": 0.05,

    "enable_branch_aux": true,
    "enable_complementary_loss": true,

    "dim": 64,
    "num_experts": 4,
    "top_k": 2,
    "dropout": 0.1
  },
  "loss": {
    "lambda_cross": 0.05,
    "lambda_importance_balance": 0.001,
    "lambda_load_balance": 0.001,
    "lambda_shared_aux": 0.03,
    "lambda_route_aux": 0.05,
    "lambda_complementary": 0.001
  },
  "train": {
    "epochs": 100,
    "lr_main": 0.001,
    "scheduler": { "type": "cosine" },
    "early_stopping": { "enabled": false },
    "gate_lr_mult": 1.0,
    "scalar_lr_mult": 2.0,
    "weight_decay": 0.0001
  }
}
```

---

# 9. 消融配置必须同步设计

如果目标是 Full 最好，消融必须真实反映“去掉关键组件”的效果，不应该让消融继续享受 Full-only 的辅助损失。

## 9.1 Fine-Only

```json
{
  "model": {
    "use_multiscale": false,
    "use_shared_branch": false,
    "use_routed_branch": true,
    "scale_mode": "fine",
    "enable_branch_aux": false,
    "enable_complementary_loss": false
  },
  "loss": {
    "lambda_cross": 0.0,
    "lambda_shared_aux": 0.0,
    "lambda_route_aux": 0.0,
    "lambda_complementary": 0.0
  }
}
```

## 9.2 Shared-Only

```json
{
  "model": {
    "use_shared_branch": true,
    "use_routed_branch": false,
    "shared_input_mode": "pre",
    "enable_branch_aux": false,
    "enable_complementary_loss": false
  },
  "loss": {
    "lambda_route_aux": 0.0,
    "lambda_complementary": 0.0,
    "lambda_importance_balance": 0.0,
    "lambda_load_balance": 0.0
  }
}
```

## 9.3 Routed-Only

```json
{
  "model": {
    "use_shared_branch": false,
    "use_routed_branch": true,
    "enable_branch_aux": false,
    "enable_complementary_loss": false
  },
  "loss": {
    "lambda_shared_aux": 0.0,
    "lambda_complementary": 0.0
  }
}
```

## 9.4 No Router

```json
{
  "model": {
    "use_router": false,
    "routing_mode_when_no_router": "dense",
    "enable_branch_aux": false,
    "enable_complementary_loss": false
  },
  "loss": {
    "lambda_importance_balance": 0.0,
    "lambda_load_balance": 0.0
  }
}
```

---

# 10. 代码文件级修改清单

## 10.1 `models/main_branch.py`

新增：

```text
shared_aux_head
route_aux_head
enable_branch_aux
enable_complementary_loss
route_dropout
```

forward 输出增加：

```python
outputs["x_hat_shared"] = x_hat_shared
outputs["x_hat_route"] = x_hat_route
outputs["features"]["h_shared"] = h_shared
outputs["features"]["h_route"] = h_route_proj
```

并确保：

```text
shared_input_mode 默认 pre
branch_fusion_mode 默认 residual
```

---

## 10.2 `models/fusion.py`

修改默认行为：

```text
ExpertEnhancedSharedInput:
默认 mode="pre"

SharedRoutedResidualFusion:
默认 mode="residual"

AdaptiveBranchGate:
保留，但仅作为 ablation，不默认使用
```

增加 route dropout：

```python
self.route_dropout = nn.Dropout3d(route_dropout)
h_route_proj = self.route_dropout(h_route_proj)
```

---

## 10.3 `losses.py`

新增：

```python
def complementary_loss(h_shared, h_route):
    hs = F.normalize(h_shared.flatten(2), dim=1)
    hr = F.normalize(h_route.flatten(2), dim=1)
    cos = (hs * hr).sum(dim=1)
    return (cos ** 2).mean()
```

训练 loss 中加入条件：

```python
if is_full and lambda_shared_aux > 0:
    loss += lambda_shared_aux * masked_loss(x_hat_shared, target, mask)

if is_full and lambda_route_aux > 0:
    loss += lambda_route_aux * masked_loss(x_hat_route, target, mask)

if is_full and lambda_complementary > 0:
    loss += lambda_complementary * complementary_loss(h_shared, h_route)
```

---

## 10.4 `scripts/train.py`

日志新增：

```text
val_mae_full
val_mae_shared_aux
val_mae_route_aux
route_alpha
route_gamma
effective_route_ratio
comp_loss
shared_aux_loss
route_aux_loss
```

并记录 Full 相对消融的目标差距：

```text
full_minus_shared
full_minus_routed
full_minus_fine
```

---

## 10.5 `configs/*.json`

新增字段：

```json
{
  "model": {
    "enable_branch_aux": true,
    "enable_complementary_loss": true,
    "route_dropout": 0.1
  },
  "loss": {
    "lambda_shared_aux": 0.05,
    "lambda_route_aux": 0.10,
    "lambda_complementary": 0.003
  }
}
```

---

# 11. 实验顺序

不要直接全量跑。按下面顺序。

## Round 1：恢复稳定主线

只跑：

```text
TaxiBJ:
Full v2.5 base
Fine-Only
Routed-Only
Shared-Only

BikeNYC:
Full v2.5 base
Shared-Only
No Router
```

验收标准：

```text
TaxiBJ Full <= Fine-Only / Routed-Only / Shared-Only
BikeNYC Full <= Shared-Only / No Router
```

如果 Full 还不是最好，先不要继续做复杂消融。

---

## Round 2：route_gamma_init 小网格

TaxiBJ：

```text
-4.0, -3.5, -3.0, -2.5
```

BikeNYC：

```text
-4.0, -3.5, -3.0
```

观察：

```text
Full 是否超过所有消融；
route_alpha 是否在 0.03~0.10；
effective_route_ratio 是否合理。
```

---

## Round 3：aux loss 权重小网格

TaxiBJ：

```text
λ_route_aux: 0.05, 0.10, 0.15
λ_comp: 0.001, 0.003
```

BikeNYC：

```text
λ_route_aux: 0.03, 0.05, 0.08
λ_comp: 0.001
```

目标：

```text
增强 Full 中 route 的贡献，但不让 Routed-Only 受益。
```

---

## Round 4：scale_mode

TaxiBJ：

```text
fine
fine_mid
fine_mid_coarse
```

Full 目标：

```text
fine_mid 最可能最好；
fine_mid_coarse 如果变差，说明 coarse 不适合 TaxiBJ。
```

BikeNYC：

```text
fine_mid
fine_mid_coarse
```

---

# 12. 验收标准

最终理想排名：

## TaxiBJ

```text
Full Model
Fine-Only / Routed-Only
Shared-Only
No Router
Fixed Experts
```

目标数值：

```text
Full <= 11.90 ~ 12.10
Fine-Only ≈ 12.0
Routed-Only ≈ 12.0~13.2
Shared-Only > 13.0
```

## BikeNYC

```text
Full Model
Shared-Only
No Router
Routed-Only
Fine-Only / Fixed Experts
```

目标数值：

```text
Full <= 2.78~2.83
Shared-Only > Full by 0.02+
No Router > Full by 0.05+
```

---

# 13. 写论文时的叙事

如果 V2.5 成功，论文叙事可以是：

```text
首先，Shared Branch 提供稳定的跨尺度主干表示；
其次，Routed Branch 通过质量感知专家选择学习缺失模式相关的局部残差；
最后，本文不是简单拼接两类特征，而是通过残差融合与互补约束，使路由专家分支学习 shared branch 难以覆盖的细粒度修正。
消融实验表明，去除 shared branch、routed branch、quality router 或 expert sharing 均会导致性能下降，证明完整模型各组件具有互补贡献。
```

核心就是：

```text
Full Model 最好；
每个消融都能解释为什么变差。
```

---

# 14. 最终建议

现在不要再默认使用：

```text
shared_input_mode=hybrid
branch_fusion_mode=adaptive_gate
```

V2.5 默认应该是：

```json
{
  "shared_input_mode": "pre",
  "branch_fusion_mode": "residual",
  "enable_branch_aux": true,
  "enable_complementary_loss": true,
  "route_gamma_init": -3.0
}
```

这个方案同时满足：

```text
1. 避免 expert 特征污染 shared branch；
2. 保留 route branch 对 Full 的残差贡献；
3. 通过 aux route loss 防止 route branch 被忽略；
4. 通过 complementary loss 让 shared/route 学互补信息；
5. 通过 Full-only 训练目标让 Full Model 有机会稳定超过所有消融。
```

这就是下一轮最推荐的主线。
