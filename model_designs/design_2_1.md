# 修改版设计文档：并行式尺度内路由专家 + 跨尺度共享专家的多尺度时空补全主分支

## 0. 文档定位

本文档基于原 v2 设计文档进行修改，重点修正 **主分支多尺度 MoE 的数据流**。

原 v2 文档中的总体方向是：

> 并行双分支框架下，主分支负责多尺度时空补全，辅助分支暂时占位，当前阶段只训练和验证主分支。

这个方向保持不变。

本次修改的核心是：

> 主分支内部不再是“多尺度特征经过路由专家后，再输入跨尺度专家”；而是改为 **路由专家分支和共享专家分支并行**。  
> 路由专家分支负责提取每个尺度内部的特征；共享专家分支直接输入完整多尺度数据，负责尺度间融合；最后将两类特征拼接，通过融合网络得到最细尺度补全结果。

---

# 1. 最新主分支核心思想

## 1.1 主分支整体目标

给定三种尺度的时空观测数据：

```text
fine-scale:   X_f_obs, M_f
mid-scale:    X_m_obs, M_m
coarse-scale: X_c_obs, M_c
```

主分支输出最细尺度补全结果：

```text
x_hat_main: [B, C, T, H, W]
```

当前阶段辅助分支输出 0，因此：

```text
x_hat_final = x_hat_main
```

最终补全结果：

```text
x_comp = M_f * X_f_gt + (1 - M_f) * x_hat_main
```

---

## 1.2 当前修改后的 MoE 设计重点

新的主分支包含两个并行模块：

```text
1. 尺度内路由专家分支 Intra-scale Routed Expert Branch
2. 跨尺度共享专家分支 Cross-scale Shared Expert Branch
```

二者是并行关系：

```text
多尺度输入 / 多尺度 embedding
        ├── 尺度内路由专家分支
        │       每个尺度经过自己的 router
        │       从 4 个同构路由专家中选择 top-k 个专家
        │       得到每个尺度的尺度内特征
        │
        └── 跨尺度共享专家分支
                输入完整多尺度特征
                学习 fine / mid / coarse 之间的尺度间关系
                得到尺度间融合特征
```

然后：

```text
尺度内特征 + 尺度间特征
        ↓
Concat
        ↓
Fusion Network
        ↓
Prediction Head
        ↓
x_hat_main
```

---

# 2. 与 v2 原设计的区别

## 2.1 v2 原设计理解

原设计更接近：

```text
H_f, H_m, H_c
        ↓
Per-scale Router + Shared Expert Pool
        ↓
E_f, E_m, E_c
        ↓
Cross-scale Shared Expert
        ↓
Fusion
        ↓
x_hat_main
```

也就是说，跨尺度专家输入的是路由专家处理后的特征。

---

## 2.2 当前最新设计

现在修改为：

```text
H_f, H_m, H_c
        ├── Per-scale Router + Routed Expert Pool
        │       ↓
        │     Z_f, Z_m, Z_c
        │
        └── Cross-scale Shared Expert
                ↓
              Z_shared

Concat(Z_f, Up(Z_m), Up(Z_c), Z_shared)
        ↓
Fusion Network
        ↓
x_hat_main
```

也就是说：

> 路由专家分支和共享专家分支都直接基于原始多尺度输入的 embedding 特征进行处理；它们是并行的，不是串行的。

---

# 3. 总体框架图

```text
输入：
X_f_obs, M_f
X_m_obs, M_m
X_c_obs, M_c

        │
        ▼
┌──────────────────────────────────────┐
│          Multi-scale Embedding        │
│                                      │
│  H_f: [B,D,T,H,W]                    │
│  H_m: [B,D,T,H/2,W/2]                │
│  H_c: [B,D,T,H/4,W/4]                │
└──────────────────┬───────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌──────────────────────┐   ┌──────────────────────────────┐
│ 尺度内路由专家分支     │   │ 跨尺度共享专家分支             │
│ Intra-scale Routed   │   │ Cross-scale Shared Expert     │
│ Expert Branch        │   │                              │
│                      │   │ 输入：H_f, H_m, H_c           │
│ H_f -> Router_f      │   │ 对齐到 fine 尺度              │
│     -> Top-k Experts │   │ Concat(H_f,H_m_up,H_c_up)     │
│     -> Z_f           │   │     -> Shared Expert          │
│                      │   │     -> Z_shared               │
│ H_m -> Router_m      │   │                              │
│     -> Top-k Experts │   └───────────────┬──────────────┘
│     -> Z_m           │                   │
│                      │                   │
│ H_c -> Router_c      │                   │
│     -> Top-k Experts │                   │
│     -> Z_c           │                   │
└──────────┬───────────┘                   │
           │                               │
           ▼                               ▼
     Z_f, Z_m, Z_c                    Z_shared
           │                               │
           └───────────────┬───────────────┘
                           ▼
        Concat(Z_f, Up(Z_m), Up(Z_c), Z_shared)
                           ↓
                    Fusion Network
                           ↓
                       H_main
                           ↓
                    Prediction Head
                           ↓
                      x_hat_main
```

---

# 4. 输入数据形状定义

以 TaxiBJ 为例：

```text
B = batch size
C = 2                      # inflow / outflow
T = 12                     # 时间窗口长度
H = 32
W = 32
D = 64                     # hidden dimension
K = 4                      # 路由专家数量
top_k = 2                  # 每个尺度激活专家数量
Q = 5                      # observation statistics 维度
```

输入数据：

```text
fine:
X_f_obs: [B, 2, T, 32, 32]
M_f:     [B, 1, T, 32, 32]

mid:
X_m_obs: [B, 2, T, 16, 16]
M_m:     [B, 1, T, 16, 16]

coarse:
X_c_obs: [B, 2, T, 8, 8]
M_c:     [B, 1, T, 8, 8]
```

其中：

```text
X_f_obs = X_f_gt * M_f
X_m_obs, M_m = MaskedPool(X_f_obs, M_f)
X_c_obs, M_c = MaskedPool(X_m_obs, M_m)
```

---

# 5. 多尺度数据构造

## 5.1 fine-scale 输入

完整数据窗口：

```text
X_f_gt: [B, C, T, H, W]
```

mask：

```text
M_f: [B, 1, T, H, W]
```

观测输入：

```text
X_f_obs = X_f_gt * M_f
```

形状：

```text
X_f_obs: [B, C, T, H, W]
```

---

## 5.2 mid-scale 输入

由 fine-scale 观测数据通过 masked pooling 构造：

```text
X_m_obs, M_m = MaskedPool(X_f_obs, M_f, scale=2)
```

形状：

```text
X_m_obs: [B, C, T, H/2, W/2]
M_m:     [B, 1, T, H/2, W/2]
```

TaxiBJ 示例：

```text
X_m_obs: [B, 2, T, 16, 16]
M_m:     [B, 1, T, 16, 16]
```

---

## 5.3 coarse-scale 输入

由 mid-scale 观测数据继续 masked pooling 构造：

```text
X_c_obs, M_c = MaskedPool(X_m_obs, M_m, scale=2)
```

形状：

```text
X_c_obs: [B, C, T, H/4, W/4]
M_c:     [B, 1, T, H/4, W/4]
```

TaxiBJ 示例：

```text
X_c_obs: [B, 2, T, 8, 8]
M_c:     [B, 1, T, 8, 8]
```

---

## 5.4 Masked Pooling 公式

推荐第一版使用 masked average pooling：

```text
X_down = SumPool(X * M) / (SumPool(M) + eps)
M_down = 1(SumPool(M) > 0)
```

其中：

- `X` 是输入观测值；
- `M` 是对应 mask；
- `SumPool(M)` 表示该粗格子由多少个观测细格子构成；
- 一个粗格子只要至少有一个观测值，就认为该粗格子可观测。

注意：

> 中粗尺度输入只能由观测值构造，不能从完整 ground truth 构造，否则会数据泄漏。

---

# 6. Multi-scale Embedding

## 6.1 Embedding 目的

不同尺度的输入空间尺寸不同，不能直接送入统一模块。因此先将它们映射到统一 hidden dimension `D`。

每个尺度使用独立 embedding：

```text
Embed_f
Embed_m
Embed_c
```

每个 embedding 包含：

```text
value embedding
mask embedding
scale embedding
time embedding
spatial embedding
```

---

## 6.2 Embedding 公式

对于尺度 `s ∈ {f, m, c}`：

```text
H_s = ValueEmbed_s(X_s_obs)
    + MaskEmbed_s(M_s)
    + ScaleEmbed_s
    + TimeEmbed
    + SpaceEmbed_s
```

---

## 6.3 Embedding 输出形状

```text
H_f: [B, D, T, H,   W  ]
H_m: [B, D, T, H/2, W/2]
H_c: [B, D, T, H/4, W/4]
```

TaxiBJ 示例，`D=64`：

```text
H_f: [B, 64, T, 32, 32]
H_m: [B, 64, T, 16, 16]
H_c: [B, 64, T, 8, 8]
```

这三个 `H_f / H_m / H_c` 会同时送入两个并行分支：

```text
1. 尺度内路由专家分支
2. 跨尺度共享专家分支
```

---

# 7. 分支 A：尺度内路由专家分支

## 7.1 分支功能

尺度内路由专家分支负责：

> 对每个尺度的数据分别进行尺度内模式建模。每个尺度通过自己的路由网络，从 4 个同构路由专家中选择 top-k 个专家激活，并将专家输出加权融合，得到该尺度的尺度内特征。

这里的“尺度内”指：

```text
fine 内部的局部细节和时空变化
mid 内部的区域模式
coarse 内部的全局趋势
```

---

## 7.2 路由专家池

专家池包含 4 个同构专家：

```text
Expert_1
Expert_2
Expert_3
Expert_4
```

记作：

```text
K = 4
```

每个专家结构相同，例如：

```text
Conv3D(D -> D, kernel_size=3, padding=1)
GroupNorm
GELU
ResBlock3D(D)
```

专家输入输出形状保持一致：

```text
Expert_k(H_s): [B, D, T, H_s, W_s]
```

其中：

```text
H_s, W_s 分别是当前尺度的空间尺寸。
```

---

## 7.3 每个尺度独立 Router

三个尺度各自有一个 router：

```text
Router_f
Router_m
Router_c
```

每个 router 输入三部分：

```text
global_pool(H_s): [B, D]
q_s:              [B, Q]
scale_emb_s:      [B, D]
```

拼接后：

```text
router_input_s: [B, 2D + Q]
```

其中 `q_s` 是 observation statistics：

```text
q_s = [
    missing_rate_s,
    observed_ratio_s,
    temporal_missing_score_s,
    spatial_missing_score_s,
    aggregation_reliability_s
]
```

当 `D=64, Q=5` 时：

```text
router_input_s: [B, 133]
```

Router 输出：

```text
gate_s: [B, K]
```

当 `K=4` 时：

```text
gate_s: [B, 4]
```

---

## 7.4 Top-k 激活策略

当前设计中：

```text
top_k = 2
```

也就是说，每个尺度每个样本只激活 4 个专家中的 2 个。

对于尺度 `s`：

```text
gate_s = softmax(router_s(...))        # [B, 4]
top_values_s, top_indices_s = TopK(gate_s, k=2)
```

然后对 top-k 权重重新归一化：

```text
top_weights_s = top_values_s / sum(top_values_s)
```

最终输出：

```text
Z_s = Σ_{k in TopK(gate_s)} top_weights_s,k * Expert_k(H_s)
```

---

## 7.5 fine-scale 路由专家处理

输入：

```text
H_f: [B, D, T, 32, 32]
```

Router 输出：

```text
gate_f: [B, 4]
```

Top-2：

```text
top_indices_f: [B, 2]
top_weights_f: [B, 2]
```

专家输出：

```text
Expert_k(H_f): [B, D, T, 32, 32]
```

Top-2 加权融合后：

```text
Z_f: [B, D, T, 32, 32]
```

---

## 7.6 mid-scale 路由专家处理

输入：

```text
H_m: [B, D, T, 16, 16]
```

Router 输出：

```text
gate_m: [B, 4]
```

Top-2 加权融合后：

```text
Z_m: [B, D, T, 16, 16]
```

为了后续和 fine 对齐，上采样：

```text
Z_m_up: [B, D, T, 32, 32]
```

---

## 7.7 coarse-scale 路由专家处理

输入：

```text
H_c: [B, D, T, 8, 8]
```

Router 输出：

```text
gate_c: [B, 4]
```

Top-2 加权融合后：

```text
Z_c: [B, D, T, 8, 8]
```

为了后续和 fine 对齐，上采样：

```text
Z_c_up: [B, D, T, 32, 32]
```

---

## 7.8 尺度内路由专家分支输出

最终得到三种尺度内特征：

```text
Z_f:    [B, D, T, 32, 32]
Z_m_up: [B, D, T, 32, 32]
Z_c_up: [B, D, T, 32, 32]
```

也可以拼接成：

```text
Z_route_all = Concat(Z_f, Z_m_up, Z_c_up)
```

形状：

```text
Z_route_all: [B, 3D, T, 32, 32]
```

当 `D=64` 时：

```text
Z_route_all: [B, 192, T, 32, 32]
```

---

# 8. 分支 B：跨尺度共享专家分支

## 8.1 分支功能

跨尺度共享专家分支负责：

> 直接输入完整的多尺度特征，学习 fine / mid / coarse 之间的尺度间关系，输出尺度间融合特征。

它和尺度内路由专家分支是并行的。

它不是 4 个路由专家中的一个，而是单独的共享专家模块。

---

## 8.2 共享专家输入

输入原始多尺度 embedding：

```text
H_f: [B, D, T, 32, 32]
H_m: [B, D, T, 16, 16]
H_c: [B, D, T, 8, 8]
```

先将 `H_m` 和 `H_c` 上采样到 fine 尺度：

```text
H_m_up: [B, D, T, 32, 32]
H_c_up: [B, D, T, 32, 32]
```

然后拼接：

```text
H_multi = Concat(H_f, H_m_up, H_c_up)
```

形状：

```text
H_multi: [B, 3D, T, 32, 32]
```

当 `D=64` 时：

```text
H_multi: [B, 192, T, 32, 32]
```

---

## 8.3 跨尺度共享专家结构

第一版建议使用轻量结构：

```text
CrossScaleSharedExpert:
Conv3D(3D -> D, kernel_size=1)
ResBlock3D(D)
ResBlock3D(D)
```

输入：

```text
H_multi: [B, 3D, T, 32, 32]
```

输出：

```text
Z_shared: [B, D, T, 32, 32]
```

当 `D=64` 时：

```text
Z_shared: [B, 64, T, 32, 32]
```

---

## 8.4 跨尺度共享专家的作用

它主要学习：

```text
fine 局部细节和 mid 区域趋势之间的关系
mid 区域趋势和 coarse 全局趋势之间的关系
coarse 全局模式对 fine 缺失区域的补充作用
不同尺度之间是否存在冲突、互补或一致性
```

因此它输出的是：

```text
尺度间特征 Z_shared
```

而不是某个单一尺度的特征。

---

# 9. 路由特征与共享特征融合

## 9.1 融合输入

当前有四个 fine-resolution 特征：

```text
Z_f:      [B, D, T, 32, 32]
Z_m_up:   [B, D, T, 32, 32]
Z_c_up:   [B, D, T, 32, 32]
Z_shared: [B, D, T, 32, 32]
```

拼接：

```text
F_fuse = Concat(Z_f, Z_m_up, Z_c_up, Z_shared)
```

形状：

```text
F_fuse: [B, 4D, T, 32, 32]
```

当 `D=64` 时：

```text
F_fuse: [B, 256, T, 32, 32]
```

---

## 9.2 Fusion Network

第一版建议：

```text
FusionNetwork:
Conv3D(4D -> D, kernel_size=1)
ResBlock3D(D)
ResBlock3D(D)
```

输出：

```text
H_main: [B, D, T, 32, 32]
```

当 `D=64` 时：

```text
H_main: [B, 64, T, 32, 32]
```

这里：

```text
h_st_aux = H_main
```

后续辅助分支可以接收 `h_st_aux`，但当前阶段辅助分支不启用。

---

## 9.3 Prediction Head

预测头：

```text
PredictionHead:
Conv3D(D -> D/2, kernel_size=3, padding=1)
GELU
Conv3D(D/2 -> C, kernel_size=1)
```

输入：

```text
H_main: [B, D, T, 32, 32]
```

输出：

```text
x_hat_main: [B, C, T, 32, 32]
```

TaxiBJ 示例：

```text
x_hat_main: [B, 2, T, 32, 32]
```

---

# 10. 当前阶段最终输出

由于当前阶段辅助分支占位输出 0：

```text
delta_aux = 0
```

所以：

```text
x_hat_final = x_hat_main
```

最终补全：

```text
x_comp = M_f * X_f_gt + (1 - M_f) * x_hat_main
```

训练 loss 只在 `M_f = 0` 的位置计算。

---

# 11. 完整数据流形状总表

以 TaxiBJ、`D=64`、`K=4`、`top_k=2`、`T=12` 为例：

| 阶段 | 输出形状 |
|---|---|
| `X_f_obs` | `[B, 2, 12, 32, 32]` |
| `M_f` | `[B, 1, 12, 32, 32]` |
| `X_m_obs` | `[B, 2, 12, 16, 16]` |
| `M_m` | `[B, 1, 12, 16, 16]` |
| `X_c_obs` | `[B, 2, 12, 8, 8]` |
| `M_c` | `[B, 1, 12, 8, 8]` |
| `H_f = Embed_f(X_f_obs, M_f)` | `[B, 64, 12, 32, 32]` |
| `H_m = Embed_m(X_m_obs, M_m)` | `[B, 64, 12, 16, 16]` |
| `H_c = Embed_c(X_c_obs, M_c)` | `[B, 64, 12, 8, 8]` |
| `q_f, q_m, q_c` | `[B, 5]` |
| `gate_f, gate_m, gate_c` | `[B, 4]` |
| `TopK(gate_f)` | indices `[B, 2]`, weights `[B, 2]` |
| `TopK(gate_m)` | indices `[B, 2]`, weights `[B, 2]` |
| `TopK(gate_c)` | indices `[B, 2]`, weights `[B, 2]` |
| `Z_f = RoutedExperts(H_f)` | `[B, 64, 12, 32, 32]` |
| `Z_m = RoutedExperts(H_m)` | `[B, 64, 12, 16, 16]` |
| `Z_c = RoutedExperts(H_c)` | `[B, 64, 12, 8, 8]` |
| `Z_m_up` | `[B, 64, 12, 32, 32]` |
| `Z_c_up` | `[B, 64, 12, 32, 32]` |
| `H_m_up` | `[B, 64, 12, 32, 32]` |
| `H_c_up` | `[B, 64, 12, 32, 32]` |
| `H_multi = Concat(H_f, H_m_up, H_c_up)` | `[B, 192, 12, 32, 32]` |
| `Z_shared = CrossScaleSharedExpert(H_multi)` | `[B, 64, 12, 32, 32]` |
| `F_fuse = Concat(Z_f, Z_m_up, Z_c_up, Z_shared)` | `[B, 256, 12, 32, 32]` |
| `H_main = FusionNetwork(F_fuse)` | `[B, 64, 12, 32, 32]` |
| `x_hat_main = PredictionHead(H_main)` | `[B, 2, 12, 32, 32]` |
| `delta_aux` 当前阶段 | `[B, 2, 12, 32, 32]`，全 0 |
| `x_hat_final` 当前阶段 | `[B, 2, 12, 32, 32]` |
| `x_comp` | `[B, 2, 12, 32, 32]` |

---

# 12. 损失函数设计

当前阶段只训练主分支，辅助分支不启用。

## 12.1 主补全损失

```text
L_main = Loss((1 - M_f) * x_hat_main, (1 - M_f) * X_f_gt)
```

推荐：

```text
SmoothL1Loss 或 MAE
```

---

## 12.2 Cross-scale Consistency Loss

为了让 fine-scale 预测结果在聚合后和中粗尺度观测保持一致，可以加入：

```text
x_hat_m = Pool(x_hat_main)
x_hat_c = Pool(x_hat_m)
```

然后：

```text
L_cross = Loss(M_m * x_hat_m, M_m * X_m_obs)
        + Loss(M_c * x_hat_c, M_c * X_c_obs)
```

注意：

```text
X_m_obs, X_c_obs 都是由观测值构造的，不会造成 ground truth 泄漏。
```

---

## 12.3 Router Balance Loss

因为使用 top-k 路由，需要避免所有样本都选同几个专家。

将三个尺度的 gate 合并：

```text
gate_all = Concat(gate_f, gate_m, gate_c)
```

形状：

```text
gate_all: [3B, K]
```

计算专家平均使用率：

```text
usage_k = mean(gate_all[:, k])
```

期望：

```text
usage_k ≈ 1 / K
```

loss：

```text
L_balance = Σ_k (usage_k - 1/K)^2
```

---

## 12.4 当前阶段总损失

```text
L_total = L_main
        + lambda_cross * L_cross
        + lambda_balance * L_balance
```

推荐初值：

```text
lambda_cross   = 0.1
lambda_balance = 0.01
```

第一阶段如果训练不稳定，可以先用：

```text
L_total = L_main + lambda_balance * L_balance
```

等主干能正常收敛后再加入 `L_cross`。

---

# 13. PyTorch 伪代码

## 13.1 Top-k Routed Expert Pool

```python
class TopKRoutedExpertPool(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            Conv3DExpert(dim) for _ in range(num_experts)
        ])

    def forward(self, h, gate):
        """
        h:    [B, D, T, H, W]
        gate: [B, K]

        return:
            z: [B, D, T, H, W]
        """
        B = h.shape[0]

        top_values, top_indices = torch.topk(gate, k=self.top_k, dim=-1)
        top_weights = top_values / (top_values.sum(dim=-1, keepdim=True) + 1e-6)

        # 为了代码简单，第一版可以先计算所有专家输出，再只选 top-k 加权
        expert_outputs = [expert(h) for expert in self.experts]
        # list of [B,D,T,H,W]

        z = torch.zeros_like(expert_outputs[0])

        for i in range(self.top_k):
            idx_i = top_indices[:, i]      # [B]
            w_i = top_weights[:, i]        # [B]

            for k in range(self.num_experts):
                selected = (idx_i == k).float().view(B, 1, 1, 1, 1)
                weight = w_i.view(B, 1, 1, 1, 1)
                z = z + selected * weight * expert_outputs[k]

        return z, top_indices, top_weights
```

---

## 13.2 Cross-scale Shared Expert

```python
class CrossScaleSharedExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim * 3, dim, kernel_size=1),
            ResBlock3D(dim),
            ResBlock3D(dim),
        )

    def forward(self, h_f, h_m, h_c):
        """
        h_f: [B,D,T,H,W]
        h_m: [B,D,T,H/2,W/2]
        h_c: [B,D,T,H/4,W/4]
        """
        B, D, T, H, W = h_f.shape

        h_m_up = F.interpolate(
            h_m, size=(T, H, W), mode="trilinear", align_corners=False
        )
        h_c_up = F.interpolate(
            h_c, size=(T, H, W), mode="trilinear", align_corners=False
        )

        h_multi = torch.cat([h_f, h_m_up, h_c_up], dim=1)
        z_shared = self.net(h_multi)

        return z_shared, h_m_up, h_c_up
```

---

## 13.3 修改后的主分支 forward

```python
class ParallelRoutedMoEWithSharedExpertImputer(nn.Module):
    def __init__(self, c_in, dim=64, num_experts=4, top_k=2):
        super().__init__()

        self.embed_f = ScaleEmbedding(c_in, dim, max_t=24, h=32, w=32)
        self.embed_m = ScaleEmbedding(c_in, dim, max_t=24, h=16, w=16)
        self.embed_c = ScaleEmbedding(c_in, dim, max_t=24, h=8,  w=8)

        self.router_f = ObservationAwareRouter(dim, q_dim=5, num_experts=num_experts)
        self.router_m = ObservationAwareRouter(dim, q_dim=5, num_experts=num_experts)
        self.router_c = ObservationAwareRouter(dim, q_dim=5, num_experts=num_experts)

        self.routed_expert_pool = TopKRoutedExpertPool(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k
        )

        self.cross_scale_shared_expert = CrossScaleSharedExpert(dim)

        self.fusion = nn.Sequential(
            nn.Conv3d(dim * 4, dim, kernel_size=1),
            ResBlock3D(dim),
            ResBlock3D(dim),
        )

        self.pred_head = nn.Sequential(
            nn.Conv3d(dim, dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(dim // 2, c_in, kernel_size=1),
        )

    def forward(self, x_f, m_f, x_m, m_m, x_c, m_c):
        # 1. Multi-scale embedding
        h_f = self.embed_f(x_f, m_f)  # [B,D,T,H,W]
        h_m = self.embed_m(x_m, m_m)  # [B,D,T,H/2,W/2]
        h_c = self.embed_c(x_c, m_c)  # [B,D,T,H/4,W/4]

        # 2. Observation statistics
        q_f = compute_observation_stats(m_f)
        q_m = compute_observation_stats(m_m)
        q_c = compute_observation_stats(m_c)

        # 3. Router gate
        gate_f = self.router_f(h_f, q_f, self.embed_f.scale_embed_vec())
        gate_m = self.router_m(h_m, q_m, self.embed_m.scale_embed_vec())
        gate_c = self.router_c(h_c, q_c, self.embed_c.scale_embed_vec())

        # 4. Intra-scale routed expert branch
        z_f, top_idx_f, top_w_f = self.routed_expert_pool(h_f, gate_f)
        z_m, top_idx_m, top_w_m = self.routed_expert_pool(h_m, gate_m)
        z_c, top_idx_c, top_w_c = self.routed_expert_pool(h_c, gate_c)

        # 5. Cross-scale shared expert branch, parallel to routed experts
        z_shared, h_m_up, h_c_up = self.cross_scale_shared_expert(h_f, h_m, h_c)

        # 6. Align routed expert outputs
        B, D, T, H, W = z_f.shape
        z_m_up = F.interpolate(z_m, size=(T, H, W), mode="trilinear", align_corners=False)
        z_c_up = F.interpolate(z_c, size=(T, H, W), mode="trilinear", align_corners=False)

        # 7. Fuse intra-scale and inter-scale features
        fuse_in = torch.cat([z_f, z_m_up, z_c_up, z_shared], dim=1)
        h_main = self.fusion(fuse_in)

        # 8. Prediction
        x_hat_main = self.pred_head(h_main)

        return {
            "x_hat_main": x_hat_main,
            "h_st_aux": h_main,
            "gates": {
                "fine": gate_f,
                "mid": gate_m,
                "coarse": gate_c,
            },
            "topk": {
                "fine_indices": top_idx_f,
                "fine_weights": top_w_f,
                "mid_indices": top_idx_m,
                "mid_weights": top_w_m,
                "coarse_indices": top_idx_c,
                "coarse_weights": top_w_c,
            },
            "features": {
                "z_f": z_f,
                "z_m": z_m,
                "z_c": z_c,
                "z_shared": z_shared,
                "h_main": h_main,
            }
        }
```

---

# 14. 辅助分支占位设计

当前阶段辅助分支不影响主分支。

```text
delta_aux = 0
x_hat_final = x_hat_main
```

后续辅助分支接入时输入：

```text
h_st_aux: [B, D, T, H, W]
F_text:   [B, D] 或 [B, T, D]
F_img:    [B, D] 或 [B, H, W, D]
M_f:      [B, 1, T, H, W]
```

注意：

> 辅助分支不输入 `x_hat_main`。

后续输出：

```text
delta_aux: [B, C, T, H, W]
```

最终：

```text
x_hat_final = x_hat_main + alpha * delta_aux
```

当前：

```text
alpha = 0 或 aux_enabled = False
```

---

# 15. 推荐消融实验

为了证明当前主分支设计有效，建议至少做：

| 方法 | 多尺度输入 | Top-k 路由专家 | 跨尺度共享专家 | 说明 |
|---|---:|---:|---:|---|
| Fine-only | 否 | 否 | 否 | 只用最细尺度 |
| Multi-scale Concat | 是 | 否 | 否 | 普通多尺度拼接 |
| Fixed Scale Experts | 是 | 否 | 否 | 一尺度一专家 |
| Routed Experts only | 是 | 是 | 否 | 只有尺度内路由专家 |
| Shared Expert only | 是 | 否 | 是 | 只有跨尺度共享专家 |
| Full Main Branch | 是 | 是 | 是 | 当前完整主分支 |

其中最关键的是：

```text
Routed Experts only vs Full Main Branch
Shared Expert only vs Full Main Branch
Multi-scale Concat vs Full Main Branch
```

这样才能证明：

```text
1. 多尺度输入有用；
2. Top-k 路由专家有用；
3. 跨尺度共享专家有用；
4. 路由专家和共享专家并行融合比单独使用更有效。
```

---

# 16. 当前最终一句话版本

> 当前主分支采用并行式多尺度 MoE 结构：每个尺度的数据经过独立 router，从 4 个同构路由专家中选择 top-k 个专家提取尺度内特征；同时，完整多尺度特征直接输入跨尺度共享专家以建模尺度间关系。最后将尺度内路由特征和尺度间共享特征拼接，通过融合网络输出最细尺度补全结果。辅助分支当前仅占位输出 0，不影响主分支，后续可接入多模态残差修正。
