# v2.4 修改设计文档：Adaptive Branch Gate + Expert-Enhanced Shared + 超参数系统化调优

> 面向当前仓库：`6xiaoming6/my_idea`。  
> 当前基础版本：`DualBranchSTImputer + GatedCrossScaleSharedExpert + ProgressiveRouteFusion(scale_mode-aware) + SharedRoutedResidualFusion`。  
> 目标：在 `scale_mode + reliability-aware scale gate` 已经明显有效的基础上，进一步解决 **TaxiBJ 上 Shared Branch 拖累 Full Model**、**route_gamma 过小导致路由分支贡献不足**、**不同数据集超参数差异较大** 等问题。

---

## 0. 当前实验结论摘要

### 0.1 TaxiBJ 最新结论

当前 TaxiBJ 使用：

```text
scale_mode = fine_mid
coarse disabled
```

最新结果：

| 模型 | MAE | RMSE | 结论 |
|---|---:|---:|---|
| Routed-Only | **11.8989** | 19.1720 | 当前最优 |
| Full Model | 12.0095 | 19.4836 | 接近最优，但被 Shared Branch 略拖累 |
| No Cross-Scale | 12.0326 | 19.4393 | 与 Routed-Only 等价附近 |
| Fine-Only | 12.1452 | 19.6721 | 稳定但不如 fine+mid route |
| No Router | 12.8881 | 21.1402 | 明显改善，但仍弱于 router |
| Shared-Only | 14.8235 | 24.5479 | 明显退化 |
| Fixed Experts | 17.5096 | 29.5590 | 很差 |

关键结论：

```text
1. 去掉 coarse 是 TaxiBJ 的决定性改进；
2. TaxiBJ 最强路径已经从 Fine-Only 变成 Routed-Only(fine_mid)；
3. Shared-Only 反而很差，说明当前 Shared Branch 的 pre-expert embedding 特征质量不够；
4. Full Model 比 Routed-Only 略差，说明 shared 分支在 Full 中可能有轻微拖累；
5. route_gamma 只有约 0.032，路由残差在 Full 中仍然参与过弱。
```

---

### 0.2 BikeNYC 最新结论

当前 BikeNYC 使用：

```text
scale_mode = fine_mid_coarse
```

最新结果：

| 模型 | MAE | RMSE | 结论 |
|---|---:|---:|---|
| Full Model | **2.7808** | 8.0640 | 当前最优，新纪录 |
| No Router | 2.8297 | 8.2149 | 第二，scale gate 很强 |
| Shared-Only | 2.8477 | 8.4086 | 稳定强基线 |
| Routed-Only | 2.9405 | 8.7104 | 中等 |
| Fine-Only | 3.0848 | 9.2675 | 最差之一 |

关键结论：

```text
1. BikeNYC 仍然需要三尺度；
2. Full Model 是最优；
3. scale_gate 学到 fine=0.69, mid=0.12, coarse=0.19，coarse 比 mid 更有用；
4. No Router 已经接近 Shared-Only，说明 scale gate 提供了很强的质量感知；
5. route_gamma 约 0.021，路由残差贡献仍然很小。
```

---

## 1. 当前主要问题

### 1.1 问题 A：TaxiBJ 上 Shared Branch 明显拖累

TaxiBJ：

```text
Routed-Only: 11.8989
Full:        12.0095
Shared-Only: 14.8235
```

这说明 TaxiBJ 的有效信息主要来自：

```text
Z_f, Z_m -> ProgressiveRouteFusion -> h_route
```

而不是：

```text
H_f, H_m -> GatedCrossScaleSharedExpert -> h_shared
```

当前 shared branch 使用的是 **pre-expert embedding**：

```text
h_f, h_m, h_c
```

而 routed branch 使用的是 **post-expert feature**：

```text
z_f, z_m, z_c
```

TaxiBJ 上 post-expert 路由特征明显更强，因此下一步应该让 shared branch 也能使用 expert-enhanced features。

---

### 1.2 问题 B：route_gamma 过于保守

当前：

```text
route_gamma_init = -4
alpha = sigmoid(route_gamma)
alpha_init ≈ 0.018
```

最新训练后：

```text
TaxiBJ alpha ≈ 0.032
BikeNYC alpha ≈ 0.021
```

说明路由分支在 Full Model 中仍然只作为极小残差参与。

但是 TaxiBJ 的 Routed-Only 已经是最优，这说明 TaxiBJ 上应该让 route branch 更强，而不是一直保持 2%～3% 的残差。

---

### 1.3 问题 C：不同数据集需要不同融合偏置

当前最优结构是：

```text
TaxiBJ: Routed-Only / route-dominant
BikeNYC: Full / shared+routed
```

所以不能继续只用：

```text
h_main = h_shared + small_alpha * h_route
```

这个设计天然偏向 shared branch，不适合 TaxiBJ。

应该改成：

```text
h_main = w_shared * h_shared + w_route * h_route
```

让模型根据数据集和样本自动决定 shared 与 routed 的权重。

---

### 1.4 问题 D：超参数仍然比较粗

目前配置中仍有一些风险：

```text
1. TaxiBJ lr=1e-3 可能略高，Full/Shared 稳定性一般；
2. balance loss 对不同配置的强度还需要细化；
3. route_gamma 使用单个初值 -4，不适合所有数据集；
4. scale_gate 只在样本级，不能对局部缺失区域自适应；
5. early stopping 和 scheduler 没有针对不同数据集单独优化。
```

---

## 2. v2.4 总体目标

v2.4 不建议推翻当前结构，而是在当前有效的 v2.3 上做三类增强：

```text
1. Expert-Enhanced Shared Branch
   让 shared branch 也能使用 post-expert 特征，解决 TaxiBJ Shared-Only 弱的问题。

2. Adaptive Branch Gate
   用自适应 shared/routed 分支门控替代固定 small residual。
   TaxiBJ 可以 route-dominant，BikeNYC 可以 shared-dominant。

3. Training Hyperparameter Refinement
   针对 TaxiBJ / BikeNYC 分别设置 lr、loss 权重、gamma/gate 初值、warmup、early stopping。
```

最终推荐命名：

```text
v2.4: Dataset-Adaptive Expert-Enhanced Shared-Routed MoE
```

---

## 3. 修改一：Expert-Enhanced Shared Branch

### 3.1 当前 shared branch

当前 shared branch 输入：

```text
h_f, h_m, h_c
```

输出：

```text
z_shared = GatedCrossScaleSharedExpert(h_f, h_m, h_c, q, r)
```

问题：

```text
h_f/h_m/h_c 只是 embedding 后的特征，未经过专家加工；
TaxiBJ 上 Routed-Only 远强于 Shared-Only，说明 expert 后特征更有效。
```

---

### 3.2 新方案：shared 输入融合 pre-expert + post-expert

新增配置：

```json
{
  "model": {
    "main": {
      "shared_input_mode": "hybrid"
    }
  }
}
```

可选值：

| shared_input_mode | shared branch 输入 | 用途 |
|---|---|---|
| `pre` | `h_f,h_m,h_c` | 当前默认，保留对照 |
| `post` | `z_f,z_m,z_c` | 让 shared 直接用专家后特征 |
| `hybrid` | `h_s + beta_s * z_s` | 推荐主方案 |
| `concat_hz` | `concat(h_s,z_s)->1x1Conv` | 更强但参数更多 |

---

### 3.3 推荐实现：hybrid mode

公式：

```text
h_f_shared = h_f + beta_f * z_f
h_m_shared = h_m + beta_m * z_m
h_c_shared = h_c + beta_c * z_c
```

其中：

```text
beta_f, beta_m, beta_c 是可学习参数，建议初始化为 0.1 或 0.0。
```

第一版建议：

```text
beta_init = 0.1
```

原因：

```text
1. 不会完全破坏原 shared branch；
2. 允许 shared branch 逐渐吸收专家后特征；
3. 对 TaxiBJ 可能明显改善 Shared-Only / Full；
4. 对 BikeNYC 风险较小。
```

---

### 3.4 代码修改建议

在 `models/fusion.py` 新增模块：

```python
class ExpertEnhancedSharedInput(nn.Module):
    def __init__(self, dim, mode="hybrid", beta_init=0.1):
        super().__init__()
        self.mode = mode
        self.beta = nn.Parameter(torch.ones(3) * beta_init)

        if mode == "concat_hz":
            self.proj_f = nn.Conv3d(dim * 2, dim, kernel_size=1)
            self.proj_m = nn.Conv3d(dim * 2, dim, kernel_size=1)
            self.proj_c = nn.Conv3d(dim * 2, dim, kernel_size=1)

    def forward(self, h_f, h_m, h_c, z_f=None, z_m=None, z_c=None):
        if self.mode == "pre" or z_f is None:
            return h_f, h_m, h_c

        if self.mode == "post":
            return z_f, z_m, z_c

        if self.mode == "hybrid":
            b = torch.sigmoid(self.beta)  # safer than raw beta
            return (
                h_f + b[0] * z_f,
                h_m + b[1] * z_m,
                h_c + b[2] * z_c,
            )

        if self.mode == "concat_hz":
            return (
                self.proj_f(torch.cat([h_f, z_f], dim=1)),
                self.proj_m(torch.cat([h_m, z_m], dim=1)),
                self.proj_c(torch.cat([h_c, z_c], dim=1)),
            )

        raise ValueError(self.mode)
```

在 `MultiScaleMoEBackbone.__init__()` 中加入：

```python
self.shared_input_adapter = ExpertEnhancedSharedInput(
    dim=dim,
    mode=main_cfg.get("shared_input_mode", "hybrid"),
    beta_init=main_cfg.get("shared_expert_beta_init", 0.1),
)
```

在 `forward()` 中，先计算 routed experts：

```python
z_f, z_m, z_c = ...
```

再构造 shared 输入：

```python
h_f_shared, h_m_shared, h_c_shared = self.shared_input_adapter(
    h_f, h_m, h_c,
    z_f=z_f.detach() if detach_shared_expert_input else z_f,
    z_m=z_m.detach() if detach_shared_expert_input else z_m,
    z_c=z_c.detach() if detach_shared_expert_input else z_c,
)
```

然后送入：

```python
z_shared, scale_gate = self.cross_scale_shared_expert(
    h_f=h_f_shared,
    h_m=h_m_shared,
    h_c=h_c_shared,
    ...
)
```

---

### 3.5 是否 detach z_s？

建议加配置：

```json
"detach_shared_expert_input": false
```

两种选择：

| detach | 含义 | 建议 |
|---|---|---|
| `false` | shared loss 会反向影响 routed experts | 默认，可能更强 |
| `true` | shared 只使用专家特征，不反向干扰专家 | 如果训练不稳定再试 |

第一版先用：

```text
detach_shared_expert_input = false
```

---

## 4. 修改二：Adaptive Branch Gate

### 4.1 当前分支融合

当前：

```text
h_main = h_shared + alpha * h_route_proj
alpha = sigmoid(route_gamma)
```

问题：

```text
1. 结构天然偏 shared branch；
2. TaxiBJ 最优是 Routed-Only，但 Full 中 route_alpha 只有 0.032；
3. BikeNYC 适合 shared+routed，但 route_alpha 也只有 0.021；
4. 这个 scalar alpha 无法根据样本缺失模式改变。
```

---

### 4.2 新方案：二路分支门控

改成：

```text
[w_shared, w_route] = BranchGate(h_shared, h_route, q_f, scale_gate, dataset_embed)

h_main = w_shared * h_shared + w_route * h_route_proj
```

输出形状：

```text
branch_gate: [B, 2]
```

含义：

```text
branch_gate[:,0] = shared 权重
branch_gate[:,1] = routed 权重
```

---

### 4.3 BranchGate 输入

建议输入：

```text
pool(h_shared): [B,D]
pool(h_route):  [B,D]
q_f:            [B,Q]
scale_gate:     [B,3]
route_alpha_old / route_gamma: optional
```

拼接：

```text
[B, 2D + Q + 3]
```

---

### 4.4 BranchGate 结构

在 `models/fusion.py` 新增：

```python
class AdaptiveBranchGate(nn.Module):
    def __init__(self, dim, q_dim=5, hidden_dim=128, init_mode="shared_bias"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2 + q_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )
        self.init_mode = init_mode
        self.reset_parameters()

    def reset_parameters(self):
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        if self.init_mode == "shared_bias":
            last.bias.data = torch.tensor([1.0, -1.0])
        elif self.init_mode == "route_bias":
            last.bias.data = torch.tensor([-1.0, 1.0])
        elif self.init_mode == "balanced":
            last.bias.data.zero_()

    def forward(self, h_shared, h_route, q_f, scale_gate):
        p_s = h_shared.mean(dim=(2, 3, 4))
        p_r = h_route.mean(dim=(2, 3, 4))
        x = torch.cat([p_s, p_r, q_f, scale_gate], dim=-1)
        logits = self.mlp(x)
        gate = torch.softmax(logits, dim=-1)
        return gate
```

---

### 4.5 修改 SharedRoutedResidualFusion

当前类：

```text
SharedRoutedResidualFusion
```

建议扩展为支持三种模式：

```json
"branch_fusion_mode": "residual" | "adaptive_gate" | "routed_primary"
```

#### residual 当前模式

```text
h_main = h_shared + sigmoid(gamma) * h_route
```

保留，用于对照。

#### adaptive_gate 推荐新模式

```text
branch_gate = AdaptiveBranchGate(...)
h_main = w_shared * h_shared + w_route * h_route
```

#### routed_primary TaxiBJ 专用对照

```text
h_main = h_route + sigmoid(shared_gamma) * h_shared
```

用于验证：TaxiBJ 是否应该以 routed branch 为主。

---

### 4.6 推荐配置

TaxiBJ：

```json
{
  "branch_fusion_mode": "adaptive_gate",
  "branch_gate_init": "route_bias"
}
```

BikeNYC：

```json
{
  "branch_fusion_mode": "adaptive_gate",
  "branch_gate_init": "shared_bias"
}
```

如果你希望写论文时统一配置，可以用：

```json
{
  "branch_fusion_mode": "adaptive_gate",
  "branch_gate_init": "balanced"
}
```

但是实验初期，建议先用数据集特定 bias 找上限。

---

## 5. 修改三：位置级/patch级 scale gate（P1，可后续实现）

当前 scale_gate 是样本级：

```text
[B,3]
```

它已经有效，但表达力有限。后续可以改成低分辨率 patch gate：

```text
[B,3,T,H/4,W/4] -> upsample to fine
```

不过这一步改动较大，不建议和 v2.4 主改动同时做。

建议放到 v2.5。

当前 v2.4 先做：

```text
Expert-Enhanced Shared + Adaptive Branch Gate + 超参数优化
```

---

## 6. 超参数修改方案

### 6.1 学习率建议

当前多用：

```text
lr_main = 1e-3
cosine eta_min=1e-6
```

建议改成带 warmup 的 cosine：

```json
"scheduler": {
  "type": "warmup_cosine",
  "warmup_epochs": 5,
  "eta_min": 1e-6
}
```

#### TaxiBJ 推荐

```json
"lr_main": 8e-4
```

原因：TaxiBJ 对结构和初始化更敏感，略降学习率更稳。

#### BikeNYC 推荐

```json
"lr_main": 1e-3
```

原因：BikeNYC 当前训练比较稳定。

---

### 6.2 参数组学习率

建议引入 param groups：

| 参数组 | lr multiplier | weight_decay | 原因 |
|---|---:|---:|---|
| encoder / experts / pred_head | 1.0 | 1e-4 | 主体参数 |
| scale_gate / branch_gate | 2.0 | 0 | gate 需要更快适配 |
| route_gamma / branch bias / beta | 5.0 | 0 | 标量门控需要更快脱离初值 |
| norm / bias / embedding | 1.0 | 0 | 常规不做 weight decay |

实现建议在 `engine.py` 或 `optim.py` 中新增：

```python
def build_optimizer(model, cfg):
    ...
```

不要再直接：

```python
AdamW(model.parameters(), lr=...)
```

---

### 6.3 route_gamma / branch gate 调参

当前 route_gamma 训练后仍然很小。建议跑：

| 数据集 | route_gamma_init / gate init |
|---|---|
| TaxiBJ | route_bias / routed_primary / gamma_init=-2 |
| BikeNYC | shared_bias / balanced / gamma_init=-4 or -3 |

如果继续使用 residual 模式，可以跑：

```text
TaxiBJ: -4, -2, -1
BikeNYC: -4, -3, -2
```

但是我更建议直接转向：

```text
adaptive_gate
```

因为它可以自动学习 shared/routed 权重。

---

### 6.4 loss 权重建议

#### TaxiBJ

```json
"loss": {
  "lambda_cross": 0.01,
  "lambda_importance_balance": 0.0005,
  "lambda_load_balance": 0.0005,
  "lambda_route_norm": 0.0,
  "lambda_branch_entropy": 0.0
}
```

说明：

```text
TaxiBJ routed branch 已经很强，不要用过强 cross/balance 约束干扰。
```

#### BikeNYC

```json
"loss": {
  "lambda_cross": 0.05,
  "lambda_importance_balance": 0.005,
  "lambda_load_balance": 0.005,
  "lambda_route_norm": 0.0,
  "lambda_branch_entropy": 0.0
}
```

说明：

```text
当前 0.1 / 0.01 能跑，但可略降，避免过约束，并观察是否更稳。
```

---

### 6.5 Loss warmup

建议所有 auxiliary loss 不要从 epoch 0 开始满权重：

```python
def ramp_weight(base_weight, epoch, warmup_epochs=10):
    return base_weight * min(1.0, epoch / warmup_epochs)
```

应用到：

```text
lambda_cross
lambda_importance_balance
lambda_load_balance
```

主 loss 始终完整启用。

---

### 6.6 Batch size 和 epoch

#### TaxiBJ

当前 batch_size=8。建议尝试：

```text
batch_size = 8 或 16
```

如果显存允许，优先 16，能降低训练波动。

Epoch：

```text
100~120
```

但必须加 early stopping：

```json
"early_stopping": {
  "enabled": true,
  "monitor": "val_mae",
  "patience": 20,
  "mode": "min"
}
```

#### BikeNYC

```text
batch_size = 16 或 32
100 epochs 足够
patience = 15
```

---

## 7. 推荐配置文件

### 7.1 TaxiBJ v2.4 full config

```json
{
  "model": {
    "main": {
      "scale_mode": "fine_mid",
      "use_scale_gate": true,
      "use_reliability_gate": true,
      "shared_input_mode": "hybrid",
      "shared_expert_beta_init": 0.1,
      "detach_shared_expert_input": false,
      "branch_fusion_mode": "adaptive_gate",
      "branch_gate_init": "route_bias",
      "dropout": 0.05,
      "routing_mode": "topk",
      "routing_mode_when_no_router": "dense"
    }
  },
  "loss": {
    "lambda_cross": 0.01,
    "lambda_importance_balance": 0.0005,
    "lambda_load_balance": 0.0005,
    "lambda_fusion_entropy": 0.0,
    "lambda_branch_entropy": 0.0
  },
  "train": {
    "epochs": 120,
    "lr_main": 0.0008,
    "weight_decay": 0.0001,
    "grad_clip_norm": 1.0,
    "amp": true,
    "scheduler": {
      "type": "warmup_cosine",
      "warmup_epochs": 5,
      "eta_min": 1e-6
    },
    "early_stopping": {
      "enabled": true,
      "monitor": "val_mae",
      "patience": 20,
      "mode": "min"
    }
  }
}
```

---

### 7.2 BikeNYC v2.4 full config

```json
{
  "model": {
    "main": {
      "scale_mode": "fine_mid_coarse",
      "use_scale_gate": true,
      "use_reliability_gate": true,
      "shared_input_mode": "hybrid",
      "shared_expert_beta_init": 0.1,
      "detach_shared_expert_input": false,
      "branch_fusion_mode": "adaptive_gate",
      "branch_gate_init": "shared_bias",
      "dropout": 0.0,
      "routing_mode": "topk",
      "routing_mode_when_no_router": "dense"
    }
  },
  "loss": {
    "lambda_cross": 0.05,
    "lambda_importance_balance": 0.005,
    "lambda_load_balance": 0.005,
    "lambda_fusion_entropy": 0.0,
    "lambda_branch_entropy": 0.0
  },
  "train": {
    "epochs": 100,
    "lr_main": 0.001,
    "weight_decay": 0.0001,
    "grad_clip_norm": 1.0,
    "amp": true,
    "scheduler": {
      "type": "warmup_cosine",
      "warmup_epochs": 5,
      "eta_min": 1e-6
    },
    "early_stopping": {
      "enabled": true,
      "monitor": "val_mae",
      "patience": 15,
      "mode": "min"
    }
  }
}
```

---

## 8. 日志新增项

必须新增：

```text
shared_input_beta_f
shared_input_beta_m
shared_input_beta_c
branch_gate_shared_mean
branch_gate_route_mean
branch_gate_shared_std
branch_gate_route_std
effective_shared_norm
effective_route_norm
effective_route_ratio
lr_group_main
lr_group_gate
lr_group_scalar
```

解释：

```text
branch_gate_route_mean 如果 TaxiBJ 仍然很低，说明模型没有真正使用 routed branch；
shared_input_beta 如果变大，说明 shared branch 确实在吸收 post-expert feature；
effective_route_ratio 可以判断 routed branch 的真实贡献，而不只看 gate/alpha。
```

---

## 9. 实验路线

### Round 1：验证 shared input mode

只跑 TaxiBJ：

```text
Full + shared_input_mode=pre
Full + shared_input_mode=post
Full + shared_input_mode=hybrid
Full + shared_input_mode=concat_hz
```

目标：

```text
让 Full 超过 Routed-Only 11.90，或者至少不低于 12.0。
```

预期：

```text
hybrid 最稳；
post 可能在 TaxiBJ 更强；
concat_hz 可能更强但过拟合风险更高。
```

---

### Round 2：验证 Adaptive Branch Gate

TaxiBJ：

```text
Routed-Only baseline
Full residual gamma=-4
Full adaptive_gate route_bias
Full adaptive_gate balanced
Full routed_primary
```

BikeNYC：

```text
Full residual gamma=-4
Full adaptive_gate shared_bias
Full adaptive_gate balanced
```

目标：

```text
TaxiBJ: Full >= Routed-Only
BikeNYC: Full 保持 2.78 附近或更好
```

---

### Round 3：超参数小网格

TaxiBJ：

```text
lr: 1e-3, 8e-4, 5e-4
lambda_cross: 0.0, 0.01, 0.03
lambda_balance: 0.0, 5e-4, 1e-3
```

BikeNYC：

```text
lr: 1e-3, 8e-4
lambda_cross: 0.03, 0.05, 0.1
lambda_balance: 0.001, 0.005, 0.01
```

不要全部组合全跑，先用 one-factor-at-a-time：一次只动一个参数。

---

### Round 4：最终多 seed

只对每个数据集 3 个模型跑 3 seeds：

TaxiBJ：

```text
Fine-Only
Routed-Only
Best Full v2.4
```

BikeNYC：

```text
Shared-Only
No Router
Best Full v2.4
```

报告：

```text
mean ± std
```

---

## 10. 文件级修改清单

### 10.1 `models/fusion.py`

新增：

```text
ExpertEnhancedSharedInput
AdaptiveBranchGate
```

修改：

```text
SharedRoutedResidualFusion 支持 residual / adaptive_gate / routed_primary
```

---

### 10.2 `models/main_branch.py`

修改：

```text
1. 读取 shared_input_mode / shared_expert_beta_init / detach_shared_expert_input；
2. routed experts 计算后，再构造 shared branch 输入；
3. shared branch 输入从 h_s 改成 h_s_shared；
4. branch fusion 接收 q_f 和 scale_gate，输出 branch_gate；
5. outputs['gates'] 增加 branch_gate；
6. outputs['diagnostics'] 增加 beta、branch_gate、effective_route_ratio。
```

---

### 10.3 `engine.py` / `utils/train_logger.py`

新增日志：

```text
branch_gate_shared_mean
branch_gate_route_mean
shared_input_beta_f/m/c
effective_route_ratio
lr_group_*
```

---

### 10.4 `optim.py` 或 `engine.py`

新增：

```text
build_optimizer_with_param_groups
```

实现 gate/scalar 参数更高 lr、norm/bias 无 weight decay。

---

### 10.5 `configs/*.json`

新增：

```json
{
  "shared_input_mode": "hybrid",
  "shared_expert_beta_init": 0.1,
  "detach_shared_expert_input": false,
  "branch_fusion_mode": "adaptive_gate",
  "branch_gate_init": "balanced",
  "scheduler": {
    "type": "warmup_cosine",
    "warmup_epochs": 5
  },
  "early_stopping": {
    "enabled": true,
    "patience": 15
  }
}
```

---

## 11. 预期结果

### 11.1 TaxiBJ

目标：

```text
Best Full v2.4 <= 11.90
```

次级目标：

```text
Full v2.4 明显优于 Fine-Only 12.15
Full v2.4 不低于 Routed-Only
Shared-Only 通过 expert-enhanced shared 从 14.82 降到 12.x
```

如果达到：

```text
Full <= Routed-Only
```

说明 adaptive branch gate 成功让模型在 TaxiBJ 上自动转向 route-dominant。

---

### 11.2 BikeNYC

目标：

```text
Best Full v2.4 <= 2.7808
```

次级目标：

```text
No Router 不超过 Full
Shared-Only 仍稳定
branch_gate_shared_mean > branch_gate_route_mean
```

如果 BikeNYC 仍然保持 Full 最好，说明 adaptive gate 没有破坏当前强结构。

---

## 12. 风险与回滚方案

### 风险 A：Expert-enhanced shared 导致过拟合

回滚：

```text
shared_input_mode = pre
```

或：

```text
detach_shared_expert_input = true
```

---

### 风险 B：Adaptive branch gate 不稳定

回滚：

```text
branch_fusion_mode = residual
route_gamma_init = -4
```

或者使用：

```text
branch_gate_init = shared_bias
```

---

### 风险 C：TaxiBJ route-dominant 后 BikeNYC 下降

解决：

```text
TaxiBJ: branch_gate_init=route_bias
BikeNYC: branch_gate_init=shared_bias
```

论文里可以解释为数据集自适应初始化，也可以后续改成根据数据统计自动设 bias。

---

## 13. 最终建议

下一步不要继续扩大专家数量或加复杂辅助分支。现在最有价值的是：

```text
1. 修复 shared branch 输入质量问题：让 shared branch 使用 post-expert/hybrid features；
2. 修复分支融合偏置问题：用 adaptive branch gate 代替固定 small residual；
3. 系统化调参：warmup cosine、param groups、loss warmup、dataset-specific loss weights；
4. 多 seed 验证：证明 11.90 和 2.78 不是随机结果。
```

最终 v2.4 推荐主结构：

```text
H_f,H_m,H_c
  ├── Routed Experts -> Z_f,Z_m,Z_c -> ProgressiveRouteFusion -> H_route
  └── ExpertEnhancedSharedInput(H,Z) -> ReliabilityAwareScaleGate -> GatedCrossScaleSharedExpert -> H_shared

AdaptiveBranchGate(H_shared,H_route,q_f,scale_gate)
  -> w_shared,w_route

H_main = w_shared * H_shared + w_route * H_route
X_hat = PredHead(H_main)
```

这个版本能统一解释当前两个数据集的差异：

```text
TaxiBJ: gate 应该学到 route-dominant；
BikeNYC: gate 应该学到 shared+route 或 shared-dominant；
scale_gate 已经负责不同尺度选择；
branch_gate 负责 shared/routed 分支选择。
```
