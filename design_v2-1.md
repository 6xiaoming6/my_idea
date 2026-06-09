# 最新模型设计文档：并行双分支框架下的多尺度 MoE 主分支设计

## 0. 文档定位

本文档基于当前最新 idea，重点设计 **主分支的多尺度时空补全模块**。当前阶段先不展开多模态辅助分支的具体实现，只保留辅助分支接口，使后续可以方便接入文本、图像等多模态信息。

当前模型整体定位为：

> **并行双分支时空数据补全框架。主分支负责多尺度时空补全，辅助分支负责后续多模态残差修正。当前实验阶段只启用主分支，辅助分支输出 0，不影响主分支结果。**

---

# 1. 最新整体设计思想

## 1.1 为什么采用并行双分支

整体模型分成两个并行分支：

1. **主分支 Main Branch**
   - 输入多尺度时空数据；
   - 使用 Observation-Aware Multi-Scale MoE 进行补全；
   - 输出基础补全结果；
   - 当前阶段重点实现和实验验证。

2. **辅助分支 Auxiliary Branch**
   - 后续输入文本、图像等多模态信息；
   - 融合多模态特征；
   - 输出残差修正项；
   - 当前阶段只做占位，输出 0，不影响主分支。

这样设计的好处是：

- 主分支可以先独立训练、独立评估；
- 不会因为多模态数据尚未准备好而阻塞实验；
- 后续加入辅助分支时，不需要大改主分支；
- 论文叙事上可以先证明多尺度 MoE 主干有效，再证明多模态辅助修正有效。

---

## 1.2 当前阶段只关注主分支

当前阶段模型实际执行逻辑是：

```text
多尺度时空数据
    ↓
主分支 Observation-Aware Multi-Scale MoE
    ↓
x_hat_main

辅助分支 Placeholder
    ↓
delta_aux = 0

最终输出：
x_hat_final = x_hat_main + delta_aux = x_hat_main
```

也就是说：

```text
当前阶段：
x_hat_final = x_hat_main
```

辅助分支只是保留接口，暂时不参与训练，也不影响结果。

---

# 2. 总体模型结构

## 2.1 整体框架图

```text
                           ┌──────────────────────────────┐
                           │         输入完整数据          │
                           │       X_f_gt, M_f             │
                           └───────────────┬──────────────┘
                                           │
                                           │  masked pooling
                                           ▼
                  ┌─────────────────────────────────────────────┐
                  │             多尺度输入构造                  │
                  │                                             │
                  │ fine:   X_f_obs, M_f                        │
                  │ mid:    X_m_obs, M_m                        │
                  │ coarse: X_c_obs, M_c                        │
                  └─────────────────┬───────────────────────────┘
                                    │
                 ┌──────────────────┴───────────────────┐
                 │                                      │
                 ▼                                      ▼
┌──────────────────────────────────────┐   ┌─────────────────────────────────┐
│ 主分支 Main Branch                    │   │ 辅助分支 Auxiliary Branch        │
│ Observation-Aware Multi-Scale MoE     │   │ Placeholder at current stage     │
│                                      │   │                                 │
│ Multi-scale Embedding                │   │ text / image inputs             │
│ Per-scale Router                     │   │ multimodal fusion later         │
│ Shared Expert Pool                   │   │                                 │
│ Cross-scale Shared Expert            │   │ current output: delta_aux = 0   │
│ Prediction Head                      │   │                                 │
│                                      │   │                                 │
│ output: x_hat_main, h_st_aux         │   │ output: delta_aux               │
└──────────────────┬───────────────────┘   └────────────────┬────────────────┘
                   │                                        │
                   └──────────────────┬─────────────────────┘
                                      ▼
                            x_hat_final = x_hat_main + alpha * delta_aux

当前阶段：
delta_aux = 0
alpha = 0 或 aux_enabled=False

所以：
x_hat_final = x_hat_main

最终补全：
x_comp = M_f * X_f_gt + (1 - M_f) * x_hat_final
```

---

## 2.2 当前阶段有效路径

当前阶段实际有效路径只有主分支：

```text
X_f_obs, M_f
X_m_obs, M_m
X_c_obs, M_c
        ↓
Multi-scale Embedding
        ↓
Observation-Aware Routers
        ↓
Shared Expert Pool
        ↓
Cross-scale Shared Expert
        ↓
Prediction Head
        ↓
x_hat_main
        ↓
x_hat_final = x_hat_main
```

---

# 3. 任务定义

## 3.1 原始完整数据

以 TaxiBJ 为例，完整数据形状为：

```text
X_all: [T_total, C, H, W]
```

其中：

```text
C = 2，表示 inflow / outflow
H = 32
W = 32
```

切成窗口后：

```text
X_f_gt: [B, C, T, H, W]
```

例如：

```text
X_f_gt: [B, 2, 12, 32, 32]
```

其中：

- `B`：batch size；
- `C`：变量通道数；
- `T`：输入窗口长度；
- `H, W`：最细尺度空间网格大小。

---

## 3.2 人工构造缺失 mask

训练时使用完整数据作为 ground truth，然后人工构造缺失 mask：

```text
M_f: [B, 1, T, H, W]
```

其中：

```text
M_f = 1 表示观测到
M_f = 0 表示缺失
```

最细尺度观测输入为：

```text
X_f_obs = X_f_gt * M_f
```

形状：

```text
X_f_obs: [B, C, T, H, W]
```

---

## 3.3 多尺度补全目标

给定：

```text
fine-scale observed data:
X_f_obs, M_f

mid-scale observed data:
X_m_obs, M_m

coarse-scale observed data:
X_c_obs, M_c
```

模型输出最细尺度补全结果：

```text
x_hat_main: [B, C, T, H, W]
```

当前阶段最终预测：

```text
x_hat_final = x_hat_main
```

最终完整补全结果：

```text
x_comp = M_f * X_f_gt + (1 - M_f) * x_hat_final
```

训练 loss 只在 `M_f = 0` 的位置计算。

---

# 4. 多尺度数据构造

## 4.1 为什么从观测数据构造多尺度

当前数据本身是完整真实数据，但补全任务需要模拟缺失。标准做法是：

1. 完整数据作为 ground truth；
2. 人工 mask 得到观测数据；
3. 只从观测数据构造中粗尺度；
4. 用完整 ground truth 监督缺失位置预测。

注意：

> **中尺度和粗尺度输入必须从观测值构造，不能从完整 ground truth 构造，否则会发生数据泄漏。**

错误做法：

```text
X_m = Pool(X_f_gt)
```

正确做法：

```text
X_m_obs = MaskedPool(X_f_gt, M_f)
```

也就是只用观测到的位置进行聚合。

---

## 4.2 fine-scale 数据

```text
X_f_gt:  [B, C, T, H, W]
M_f:     [B, 1, T, H, W]
X_f_obs: [B, C, T, H, W]
```

计算：

```text
X_f_obs = X_f_gt * M_f
```

以 TaxiBJ 为例：

```text
X_f_obs: [B, 2, T, 32, 32]
M_f:     [B, 1, T, 32, 32]
```

---

## 4.3 mid-scale 数据

从 fine-scale 观测数据构造：

```text
X_m_obs, M_m = MaskedPool(X_f_obs, M_f, scale=2)
```

形状：

```text
X_m_obs: [B, C, T, H/2, W/2]
M_m:     [B, 1, T, H/2, W/2]
```

以 TaxiBJ 为例：

```text
X_m_obs: [B, 2, T, 16, 16]
M_m:     [B, 1, T, 16, 16]
```

---

## 4.4 coarse-scale 数据

从 mid-scale 观测数据继续构造：

```text
X_c_obs, M_c = MaskedPool(X_m_obs, M_m, scale=2)
```

形状：

```text
X_c_obs: [B, C, T, H/4, W/4]
M_c:     [B, 1, T, H/4, W/4]
```

以 TaxiBJ 为例：

```text
X_c_obs: [B, 2, T, 8, 8]
M_c:     [B, 1, T, 8, 8]
```

---

## 4.5 Masked Average Pooling

第一版建议使用 **masked average pooling**，公式为：

```text
X_down = SumPool(X * M) / (SumPool(M) + eps)
M_down = 1(SumPool(M) > 0)
```

含义：

- 每个 coarse cell 只根据观测到的 fine cells 计算；
- 如果一个区域内至少有一个观测值，则该 coarse cell 视为可观测；
- 如果一个区域内完全没有观测值，则该 coarse cell 仍然是缺失。

优点：

- 数值尺度稳定；
- 不会引入 ground truth 泄漏；
- fine / mid / coarse 的数值范围相对接近；
- 训练更稳定。

---

## 4.6 是否使用 sum pooling

交通流量从物理意义上更适合 sum pooling，因为粗区域流量可以看成细区域流量之和。

但是第一版建议先使用 masked average pooling。原因：

- 主分支当前利用中粗尺度作为上下文，不直接评估中粗尺度预测；
- masked average pooling 训练更稳定；
- 不同尺度数值范围更接近；
- 更适合先验证 MoE 结构有效性。

后续可以做消融：

```text
masked average pooling vs masked sum pooling
```

---

# 5. 主分支：Observation-Aware Multi-Scale MoE Imputer

## 5.1 主分支总体目标

主分支输入三种尺度的观测数据和 mask：

```text
X_f_obs, M_f
X_m_obs, M_m
X_c_obs, M_c
```

输出：

```text
x_hat_main: [B, C, T, H, W]
h_st_aux:   [B, D, T, H, W]
```

其中：

- `x_hat_main` 是主分支基础补全结果；
- `h_st_aux` 是主分支中间高分辨率时空特征，供后续辅助分支使用；
- 当前阶段辅助分支不影响主分支，所以 `h_st_aux` 只是预留接口。

---

## 5.2 主分支结构概览

```text
Input:
X_f_obs, M_f
X_m_obs, M_m
X_c_obs, M_c

Step 1: Multi-scale Embedding
H_f^0, H_m^0, H_c^0

Step 2: Observation Statistics
q_f, q_m, q_c

Step 3: Per-scale Observation-Aware Router
gate_f, gate_m, gate_c

Step 4: Shared Expert Pool
E_f, E_m, E_c

Step 5: Align to fine scale
E_m_up, E_c_up

Step 6: Cross-scale Shared Expert
H_cross

Step 7: Fusion and Prediction Head
H_main
x_hat_main

Output:
x_hat_main, h_st_aux
```

---

# 6. Multi-scale Embedding

## 6.1 为什么需要 embedding

不同尺度的数据有不同空间分辨率：

```text
fine:   [B, C, T, H, W]
mid:    [B, C, T, H/2, W/2]
coarse: [B, C, T, H/4, W/4]
```

如果直接输入 MoE，模型很难区分：

- 当前特征来自哪个尺度；
- 当前位置是否观测；
- 当前时间位置；
- 当前空间位置；
- 该尺度数据是否可靠。

因此需要统一编码成隐藏维度 `D`。

---

## 6.2 每个尺度的 embedding 组成

每个尺度的 embedding 包含：

```text
value embedding
mask embedding
scale embedding
time embedding
spatial embedding
```

公式表示：

```text
H_s^0 = ValueEmbed_s(X_s_obs)
      + MaskEmbed_s(M_s)
      + ScaleEmbed(s)
      + TimeEmbed(t)
      + SpaceEmbed_s(h, w)
```

其中：

```text
s ∈ {fine, mid, coarse}
```

输出形状：

```text
H_f^0: [B, D, T, H,   W  ]
H_m^0: [B, D, T, H/2, W/2]
H_c^0: [B, D, T, H/4, W/4]
```

---

## 6.3 Value Embedding

第一版可以使用简单的 3D 卷积：

```text
ValueEmbed_s:
Conv3D(C -> D, kernel_size=3, padding=1)
Norm
ReLU/GELU
```

输入：

```text
X_s_obs: [B, C, T, H_s, W_s]
```

输出：

```text
V_s: [B, D, T, H_s, W_s]
```

---

## 6.4 Mask Embedding

mask 也应该输入模型，因为模型必须知道哪些位置是真实观测，哪些位置是缺失。

```text
MaskEmbed_s:
Conv3D(1 -> D, kernel_size=3, padding=1)
```

输入：

```text
M_s: [B, 1, T, H_s, W_s]
```

输出：

```text
MEmb_s: [B, D, T, H_s, W_s]
```

---

## 6.5 Scale Embedding

每个尺度有一个可学习向量：

```text
ScaleEmbed(fine):   [D]
ScaleEmbed(mid):    [D]
ScaleEmbed(coarse): [D]
```

广播到对应时空形状：

```text
ScaleEmb_s: [1, D, 1, 1, 1]
```

加入到 `H_s^0` 中。

作用：

> 让模型知道当前输入属于哪个尺度。

---

## 6.6 Time Embedding

可以使用可学习时间位置编码：

```text
TimeEmbed: [T_max, D]
```

对于窗口长度 `T`，取前 `T` 个：

```text
TimeEmb: [1, D, T, 1, 1]
```

作用：

> 表示当前时间步在窗口内的位置。

如果后续使用真实时间信息，例如小时、星期、节假日，也可以扩展成：

```text
hour embedding
weekday embedding
holiday embedding
```

---

## 6.7 Spatial Embedding

每个尺度使用独立空间 embedding：

```text
SpaceEmbed_f: [D, H, W]
SpaceEmbed_m: [D, H/2, W/2]
SpaceEmbed_c: [D, H/4, W/4]
```

广播后：

```text
SpaceEmb_s: [1, D, 1, H_s, W_s]
```

作用：

> 表示空间网格位置，帮助模型学习固定区域差异。

---

# 7. Observation Statistics：观测质量统计

## 7.1 为什么需要观测统计

借鉴 quality-aware routing 思路，router 不应该只根据特征内容路由，还应该知道当前输入的观测质量。

对于时空补全，所谓“质量”可以理解为：

```text
缺失率
观测比例
时间连续缺失程度
空间块缺失程度
尺度聚合可靠性
```

这些统计量作为 router 的额外输入，使专家选择更加合理。

---

## 7.2 每个尺度的统计向量

对每个尺度 `s` 构造：

```text
q_s: [B, Q]
```

建议第一版包含：

```text
q_s = [
    missing_rate_s,
    observed_ratio_s,
    temporal_missing_score_s,
    spatial_missing_score_s,
    aggregation_reliability_s
]
```

即：

```text
Q = 5
```

---

## 7.3 missing_rate_s

```text
missing_rate_s = mean(1 - M_s)
```

形状：

```text
[B, 1]
```

表示当前样本当前尺度整体缺失比例。

---

## 7.4 observed_ratio_s

```text
observed_ratio_s = mean(M_s)
```

形状：

```text
[B, 1]
```

理论上：

```text
observed_ratio_s = 1 - missing_rate_s
```

但保留两个量可以让 router 更容易学习。

---

## 7.5 temporal_missing_score_s

用于描述时间连续缺失程度。

第一版可以简化为：

```text
temporal_missing_score_s = mean over spatial locations of longest missing run / T
```

如果实现复杂，也可以先用更简单版本：

```text
temporal_missing_score_s = mean_t(1 - any_observed_at_time_t)
```

含义：

> 某些时间步是否整体观测很少或完全缺失。

---

## 7.6 spatial_missing_score_s

用于描述空间块缺失程度。

第一版可以简化为：

```text
spatial_missing_score_s = avg_pool(1 - M_s) 的最大值
```

例如用 `4×4` 空间窗口计算局部缺失密度：

```text
spatial_missing_score_s = max(AvgPool2D(1 - M_s))
```

含义：

> 当前样本中是否存在大面积空间块缺失。

---

## 7.7 aggregation_reliability_s

对于 fine-scale：

```text
aggregation_reliability_f = observed_ratio_f
```

对于 mid/coarse：

```text
aggregation_reliability_s = 平均每个 coarse cell 由多少观测 fine cells 聚合得到
```

第一版简化：

```text
aggregation_reliability_s = observed_ratio_s
```

后续可以在 masked pooling 时记录每个 coarse cell 的 `m_sum`，再计算更精细的可靠性。

---

# 8. Per-scale Observation-Aware Router

## 8.1 Router 设计目标

每个尺度有自己的 router：

```text
Router_f
Router_m
Router_c
```

但三个尺度共享同一个专家池。

这样可以避免：

```text
fine 只能用 fine expert
mid 只能用 mid expert
coarse 只能用 coarse expert
```

这种死板设计。

正确设计是：

```text
fine   -> Router_f -> shared expert pool
mid    -> Router_m -> shared expert pool
coarse -> Router_c -> shared expert pool
```

不同尺度可以根据自己的特征和缺失状态动态选择专家。

---

## 8.2 Router 输入

每个尺度 router 输入三部分：

```text
1. 当前尺度全局特征 global_pool(H_s^0)
2. 当前尺度观测统计 q_s
3. 当前尺度 scale embedding
```

其中：

```text
global_pool(H_s^0): [B, D]
q_s:                [B, Q]
scale_emb_s:         [B, D]
```

拼接：

```text
router_input_s: [B, 2D + Q]
```

---

## 8.3 Router 输出

假设专家数量为 `K`：

```text
gate_s = softmax(MLP(router_input_s))
```

形状：

```text
gate_s: [B, K]
```

其中：

```text
K = 4 或 8
```

第一版建议：

```text
K = 4
```

---

## 8.4 Softmax 全专家版本

第一版建议先做 softmax 全专家融合，训练更稳定：

```text
E_s = Σ_k gate_s,k * Expert_k(H_s^0)
```

优点：

- 实现简单；
- 梯度稳定；
- 所有专家都有训练信号；
- 适合第一阶段验证。

---

## 8.5 Top-k 稀疏 MoE 版本

第二版再做 top-k：

```text
topk = 2
只激活 gate_s 最大的两个专家
```

形式：

```text
E_s = Σ_{k in TopK(gate_s)} gate_s,k * Expert_k(H_s^0)
```

优点：

- 更符合 sparse MoE 思想；
- 有动态选择能力；
- 可解释性更强。

缺点：

- 更容易 expert collapse；
- 需要 gate balance loss；
- 调参难度更高。

---

# 9. Shared Expert Pool

## 9.1 专家池设计目标

专家池是共享的：

```text
Expert_1
Expert_2
Expert_3
Expert_4
```

所有尺度都可以调用同一组专家。

这样专家学习的是不同类型的补全模式，而不是固定尺度：

```text
某些专家可能更偏局部细节
某些专家可能更偏平滑趋势
某些专家可能更偏时间变化
某些专家可能更偏大范围缺失恢复
```

这种设计比“一尺度一专家”更灵活。

---

## 9.2 第一版专家结构：同构专家

第一版建议所有专家使用同构结构：

```text
Expert_k:
Conv3D(D -> D, kernel_size=3, padding=1)
GroupNorm / LayerNorm
GELU
Conv3D(D -> D, kernel_size=3, padding=1)
Residual
```

输入：

```text
H_s^0: [B, D, T, H_s, W_s]
```

输出：

```text
Expert_k(H_s^0): [B, D, T, H_s, W_s]
```

优点：

- 实现简单；
- 容易调试；
- 参数量可控；
- 有利于先验证 router 是否有效。

---

## 9.3 第二版专家结构：轻微异构专家

后续可以改成轻微异构：

```text
Expert_1: local spatial expert
Expert_2: temporal expert
Expert_3: dilated spatial-temporal expert
Expert_4: global/context expert
```

但是第一版不建议直接上异构专家，否则实验变量太多，不容易判断提升来自哪里。

---

## 9.4 Expert 输出融合

对于每个尺度：

```text
O_s,k = Expert_k(H_s^0)
```

然后使用 gate 加权：

```text
E_s = Σ_k gate_s,k * O_s,k
```

输出：

```text
E_f: [B, D, T, H,   W  ]
E_m: [B, D, T, H/2, W/2]
E_c: [B, D, T, H/4, W/4]
```

---

# 10. Cross-scale Shared Expert

## 10.1 为什么需要跨尺度专家

Shared Expert Pool 主要在每个尺度内部做动态建模，但多尺度补全还需要显式建模尺度间关系：

```text
fine ↔ mid ↔ coarse
```

普通 concat 只能粗糙融合，不一定能学习稳定的尺度间关系。

因此设计一个 Cross-scale Shared Expert。

---

## 10.2 尺度对齐

先将中粗尺度特征上采样到最细尺度：

```text
E_m_up = Upsample(E_m) -> [B, D, T, H, W]
E_c_up = Upsample(E_c) -> [B, D, T, H, W]
```

然后拼接：

```text
E_all = Concat(E_f, E_m_up, E_c_up)
```

形状：

```text
E_all: [B, 3D, T, H, W]
```

---

## 10.3 第一版 Cross-scale Expert

第一版用轻量 3D Conv：

```text
CrossScaleExpert:
Conv3D(3D -> D, kernel_size=1)
ResBlock3D(D -> D)
ResBlock3D(D -> D)
```

输出：

```text
H_cross: [B, D, T, H, W]
```

---

## 10.4 Cross-scale Expert 的作用

它负责：

```text
1. 融合 fine 的局部细节
2. 融合 mid 的区域趋势
3. 融合 coarse 的全局趋势
4. 显式建模不同尺度之间的补全关系
```

这比直接：

```text
concat + conv
```

更有结构意义。

---

# 11. Fusion and Prediction Head

## 11.1 融合输入

将以下特征拼接：

```text
E_f
E_m_up
E_c_up
H_cross
```

形状：

```text
FusionInput: [B, 4D, T, H, W]
```

---

## 11.2 融合网络

第一版建议：

```text
Conv3D(4D -> D, kernel_size=1)
ResBlock3D(D -> D)
ResBlock3D(D -> D)
```

输出：

```text
H_main: [B, D, T, H, W]
```

这里：

```text
h_st_aux = H_main
```

---

## 11.3 预测头

```text
PredictionHead:
Conv3D(D -> D/2, kernel_size=3, padding=1)
GELU
Conv3D(D/2 -> C, kernel_size=1)
```

输出：

```text
x_hat_main: [B, C, T, H, W]
```

---

# 12. Auxiliary Branch Placeholder

## 12.1 当前阶段辅助分支定位

当前阶段辅助分支只保留接口，不参与模型效果。

它的职责是：

```text
后续融合文本、图像等多模态信息
输出残差修正项 delta_aux
修正主分支结果
```

但是当前阶段：

```text
delta_aux = 0
```

所以：

```text
x_hat_final = x_hat_main
```

---

## 12.2 后续辅助分支输入接口

后续辅助分支输入：

```text
h_st_aux: [B, D, T, H, W]
F_text:   [B, D] 或 [B, T, D]
F_img:    [B, D] 或 [B, H, W, D]
M_f:      [B, 1, T, H, W]
```

注意：

> 辅助分支不输入 `x_hat_main`。

原因：

- 避免辅助分支过度依赖主分支输出；
- 保持两个分支并行；
- 辅助分支只根据主分支中间特征和多模态信息学习修正；
- 主分支输出不被辅助分支反向影响结构设计。

---

## 12.3 当前 Placeholder 输出

当前阶段：

```text
delta_aux = zeros_like(x_hat_main)
```

形状：

```text
delta_aux: [B, C, T, H, W]
```

最终：

```text
x_hat_final = x_hat_main + alpha * delta_aux
```

其中当前设置：

```text
alpha = 0
```

或者：

```text
aux_enabled = False
```

即：

```text
x_hat_final = x_hat_main
```

---

## 12.4 后续接入多模态时的接口不变

后续只需要把 placeholder 替换成：

```text
MultimodalAuxBranch(
    h_st_aux,
    f_text,
    f_img,
    m_f
) -> delta_aux
```

然后设置：

```text
aux_enabled = True
alpha > 0
```

即可启用辅助分支。

---

# 13. Final Output

## 13.1 当前阶段

```text
delta_aux = 0
x_hat_final = x_hat_main
```

最终补全：

```text
x_comp = M_f * X_f_gt + (1 - M_f) * x_hat_final
```

---

## 13.2 后续完整阶段

```text
delta_aux = AuxBranch(h_st_aux, text, image, M_f)
x_hat_final = x_hat_main + alpha * delta_aux
x_comp = M_f * X_f_gt + (1 - M_f) * x_hat_final
```

---

# 14. 损失函数设计

当前阶段只训练主分支，因此重点是：

```text
L_main
L_cross
L_balance
```

辅助分支相关 loss 暂时不启用。

---

## 14.1 主补全损失 L_main

只在 fine-scale 缺失位置计算：

```text
L_main = Loss((1 - M_f) * x_hat_main, (1 - M_f) * X_f_gt)
```

推荐使用：

```text
MAE / SmoothL1Loss
```

第一版建议：

```text
SmoothL1Loss
```

---

## 14.2 跨尺度一致性损失 L_cross

为了让输出的 fine-scale 结果和中粗尺度观测保持一致，引入 cross-scale consistency loss。

先下采样预测结果：

```text
x_hat_m = Masked/AvgPool(x_hat_main)
x_hat_c = Masked/AvgPool(x_hat_m)
```

然后与输入的中粗尺度观测进行约束：

```text
L_cross =
Loss(M_m * x_hat_m, M_m * X_m_obs)
+
Loss(M_c * x_hat_c, M_c * X_c_obs)
```

注意：

- `X_m_obs` 和 `X_c_obs` 是由观测值构造的；
- 这个 loss 只在中粗尺度可观测位置计算；
- 它不会引入 ground truth 泄漏。

---

## 14.3 Gate Balance Loss

为了避免所有样本都路由到同一个专家，加入专家均衡正则。

假设三个尺度的 gate 分别是：

```text
gate_f: [B, K]
gate_m: [B, K]
gate_c: [B, K]
```

可以合并：

```text
gate_all: [3B, K]
```

计算每个专家平均使用率：

```text
usage_k = mean(gate_all[:, k])
```

期望每个专家使用率接近：

```text
1 / K
```

loss：

```text
L_balance = Σ_k (usage_k - 1/K)^2
```

作用：

- 防止 expert collapse；
- 鼓励不同专家都被训练；
- 增强 MoE 的稳定性。

---

## 14.4 当前阶段总损失

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

如果训练不稳定，可以先设置：

```text
lambda_cross   = 0
lambda_balance = 0.01
```

先跑通主分支，再加入 cross loss。

---

## 14.5 后续辅助分支损失

当前阶段不用。

后续启用辅助分支后再加入：

```text
L_final = Loss((1 - M_f) * x_hat_final, (1 - M_f) * X_f_gt)
L_delta = ||(1 - M_f) * delta_aux||_1
```

完整 loss 后续可以是：

```text
L_total = L_final
        + lambda_main * L_main
        + lambda_cross * L_cross
        + lambda_balance * L_balance
        + lambda_delta * L_delta
```

---

# 15. PyTorch 伪代码

## 15.1 Masked Pooling

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_avg_pool2d_spatial(x, mask, kernel_size=2, eps=1e-6):
    """
    x:    [B, C, T, H, W]
    mask: [B, 1, T, H, W]

    return:
        x_down:    [B, C, T, H/k, W/k]
        mask_down: [B, 1, T, H/k, W/k]
    """

    B, C, T, H, W = x.shape
    k = kernel_size

    x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    m_2d = mask.permute(0, 2, 1, 3, 4).reshape(B * T, 1, H, W)

    x_sum = F.avg_pool2d(x_2d * m_2d, kernel_size=k, stride=k) * (k * k)
    m_sum = F.avg_pool2d(m_2d, kernel_size=k, stride=k) * (k * k)

    x_down = x_sum / (m_sum + eps)
    m_down = (m_sum > 0).float()

    H2, W2 = x_down.shape[-2], x_down.shape[-1]

    x_down = x_down.reshape(B, T, C, H2, W2).permute(0, 2, 1, 3, 4)
    m_down = m_down.reshape(B, T, 1, H2, W2).permute(0, 2, 1, 3, 4)

    return x_down, m_down
```

---

## 15.2 ResBlock3D

```python
class ResBlock3D(nn.Module):
    def __init__(self, dim, num_groups=8):
        super().__init__()

        self.conv1 = nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, dim)

        self.conv2 = nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, dim)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.gelu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = F.gelu(out)

        return out
```

---

## 15.3 MultiScaleEmbedding

```python
class ScaleEmbedding(nn.Module):
    def __init__(self, c_in, dim, max_t, h, w):
        super().__init__()

        self.value_embed = nn.Sequential(
            nn.Conv3d(c_in, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
            nn.GELU()
        )

        self.mask_embed = nn.Conv3d(1, dim, kernel_size=3, padding=1)

        self.scale_embed = nn.Parameter(torch.randn(1, dim, 1, 1, 1))

        self.time_embed = nn.Parameter(torch.randn(1, dim, max_t, 1, 1))
        self.space_embed = nn.Parameter(torch.randn(1, dim, 1, h, w))

    def forward(self, x, mask):
        """
        x:    [B, C, T, H, W]
        mask: [B, 1, T, H, W]
        """

        B, C, T, H, W = x.shape

        v = self.value_embed(x)
        m = self.mask_embed(mask)

        t_emb = self.time_embed[:, :, :T, :, :]
        s_emb = self.space_embed[:, :, :, :H, :W]

        h = v + m + self.scale_embed + t_emb + s_emb

        return h
```

---

## 15.4 Observation Statistics

```python
def compute_observation_stats(mask):
    """
    mask: [B, 1, T, H, W]

    return:
        q: [B, Q]
    """

    B, _, T, H, W = mask.shape

    observed_ratio = mask.mean(dim=(1, 2, 3, 4), keepdim=False).view(B, 1)
    missing_rate = 1.0 - observed_ratio

    # temporal missing score:
    # ratio of timesteps with very low observation
    obs_per_t = mask.mean(dim=(1, 3, 4))  # [B, T]
    temporal_missing_score = (obs_per_t < 0.1).float().mean(dim=1, keepdim=True)

    # spatial block missing score:
    # use simple average pooling on missing mask and take max
    missing = 1.0 - mask
    missing_2d = missing.permute(0, 2, 1, 3, 4).reshape(B * T, 1, H, W)

    if H >= 4 and W >= 4:
        block = F.avg_pool2d(missing_2d, kernel_size=4, stride=1, padding=0)
        block_score = block.amax(dim=(1, 2, 3)).reshape(B, T).mean(dim=1, keepdim=True)
    else:
        block_score = missing_rate

    aggregation_reliability = observed_ratio

    q = torch.cat([
        missing_rate,
        observed_ratio,
        temporal_missing_score,
        block_score,
        aggregation_reliability
    ], dim=1)

    return q
```

---

## 15.5 Router

```python
class ObservationAwareRouter(nn.Module):
    def __init__(self, dim, q_dim, num_experts):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim + q_dim + dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_experts)
        )

    def forward(self, h, q, scale_embed_vec):
        """
        h: [B, D, T, H, W]
        q: [B, Q]
        scale_embed_vec: [B, D]

        return:
            gate: [B, K]
        """

        pooled = h.mean(dim=(2, 3, 4))  # [B, D]

        inp = torch.cat([pooled, q, scale_embed_vec], dim=1)
        logits = self.net(inp)
        gate = torch.softmax(logits, dim=-1)

        return gate
```

---

## 15.6 Expert

```python
class Conv3DExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            ResBlock3D(dim)
        )

    def forward(self, x):
        return self.block(x)
```

---

## 15.7 Shared Expert Pool

```python
class SharedExpertPool(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()

        self.experts = nn.ModuleList([
            Conv3DExpert(dim) for _ in range(num_experts)
        ])

    def forward(self, h, gate):
        """
        h:    [B, D, T, H, W]
        gate: [B, K]

        return:
            out: [B, D, T, H, W]
        """

        expert_outputs = []

        for expert in self.experts:
            expert_outputs.append(expert(h))

        # [K, B, D, T, H, W]
        expert_outputs = torch.stack(expert_outputs, dim=0)

        # gate: [B, K] -> [K, B, 1, 1, 1, 1]
        weights = gate.transpose(0, 1).view(
            gate.shape[1], gate.shape[0], 1, 1, 1, 1
        )

        out = (expert_outputs * weights).sum(dim=0)

        return out
```

---

## 15.8 CrossScaleExpert

```python
class CrossScaleExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(dim * 3, dim, kernel_size=1),
            ResBlock3D(dim),
            ResBlock3D(dim)
        )

    def forward(self, e_f, e_m, e_c):
        """
        e_f: [B, D, T, H, W]
        e_m: [B, D, T, H/2, W/2]
        e_c: [B, D, T, H/4, W/4]
        """

        B, D, T, H, W = e_f.shape

        e_m_up = F.interpolate(
            e_m,
            size=(T, H, W),
            mode="trilinear",
            align_corners=False
        )

        e_c_up = F.interpolate(
            e_c,
            size=(T, H, W),
            mode="trilinear",
            align_corners=False
        )

        x = torch.cat([e_f, e_m_up, e_c_up], dim=1)
        h_cross = self.net(x)

        return h_cross, e_m_up, e_c_up
```

---

## 15.9 Main Branch

```python
class ObservationAwareMultiScaleMoEImputer(nn.Module):
    def __init__(
        self,
        c_in,
        dim=64,
        num_experts=4,
        max_t=24,
        h=32,
        w=32,
        q_dim=5
    ):
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts

        self.embed_f = ScaleEmbedding(c_in, dim, max_t, h, w)
        self.embed_m = ScaleEmbedding(c_in, dim, max_t, h // 2, w // 2)
        self.embed_c = ScaleEmbedding(c_in, dim, max_t, h // 4, w // 4)

        self.router_f = ObservationAwareRouter(dim, q_dim, num_experts)
        self.router_m = ObservationAwareRouter(dim, q_dim, num_experts)
        self.router_c = ObservationAwareRouter(dim, q_dim, num_experts)

        self.expert_pool = SharedExpertPool(dim, num_experts)

        self.cross_scale_expert = CrossScaleExpert(dim)

        self.fusion = nn.Sequential(
            nn.Conv3d(dim * 4, dim, kernel_size=1),
            ResBlock3D(dim),
            ResBlock3D(dim)
        )

        self.pred_head = nn.Sequential(
            nn.Conv3d(dim, dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(dim // 2, c_in, kernel_size=1)
        )

    def get_scale_embed_vec(self, embed_module, batch_size):
        """
        Convert scale embedding parameter [1,D,1,1,1] to [B,D]
        """
        return embed_module.scale_embed.view(1, self.dim).expand(batch_size, self.dim)

    def forward(self, x_f, m_f, x_m, m_m, x_c, m_c):
        """
        x_f: [B,C,T,H,W]
        m_f: [B,1,T,H,W]

        x_m: [B,C,T,H/2,W/2]
        m_m: [B,1,T,H/2,W/2]

        x_c: [B,C,T,H/4,W/4]
        m_c: [B,1,T,H/4,W/4]
        """

        B = x_f.shape[0]

        # 1. embedding
        h_f = self.embed_f(x_f, m_f)
        h_m = self.embed_m(x_m, m_m)
        h_c = self.embed_c(x_c, m_c)

        # 2. observation stats
        q_f = compute_observation_stats(m_f)
        q_m = compute_observation_stats(m_m)
        q_c = compute_observation_stats(m_c)

        # 3. scale embedding vector for router
        se_f = self.get_scale_embed_vec(self.embed_f, B)
        se_m = self.get_scale_embed_vec(self.embed_m, B)
        se_c = self.get_scale_embed_vec(self.embed_c, B)

        # 4. router
        gate_f = self.router_f(h_f, q_f, se_f)
        gate_m = self.router_m(h_m, q_m, se_m)
        gate_c = self.router_c(h_c, q_c, se_c)

        # 5. shared expert pool
        e_f = self.expert_pool(h_f, gate_f)
        e_m = self.expert_pool(h_m, gate_m)
        e_c = self.expert_pool(h_c, gate_c)

        # 6. cross-scale expert
        h_cross, e_m_up, e_c_up = self.cross_scale_expert(e_f, e_m, e_c)

        # 7. fusion
        fusion_in = torch.cat([e_f, e_m_up, e_c_up, h_cross], dim=1)
        h_main = self.fusion(fusion_in)

        # 8. prediction
        x_hat_main = self.pred_head(h_main)

        return {
            "x_hat_main": x_hat_main,
            "h_st_aux": h_main,
            "gates": {
                "fine": gate_f,
                "mid": gate_m,
                "coarse": gate_c
            },
            "features": {
                "e_f": e_f,
                "e_m": e_m,
                "e_c": e_c,
                "h_cross": h_cross
            }
        }
```

---

## 15.10 Auxiliary Placeholder

```python
class AuxiliaryPlaceholder(nn.Module):
    def __init__(self, c_out):
        super().__init__()
        self.c_out = c_out

    def forward(self, h_st_aux, **kwargs):
        """
        Current stage:
            return zero residual.

        h_st_aux: [B,D,T,H,W]
        """

        B, D, T, H, W = h_st_aux.shape
        return h_st_aux.new_zeros(B, self.c_out, T, H, W)
```

---

## 15.11 Full Model

```python
class ParallelTwoBranchImputer(nn.Module):
    def __init__(
        self,
        main_branch,
        aux_branch,
        aux_enabled=False,
        alpha_init=0.0
    ):
        super().__init__()

        self.main_branch = main_branch
        self.aux_branch = aux_branch
        self.aux_enabled = aux_enabled

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, batch):
        """
        batch:
            x_f_gt: optional, for completion output
            x_f_obs, m_f
            x_m_obs, m_m
            x_c_obs, m_c
        """

        x_f_obs = batch["x_f_obs"]
        m_f = batch["m_f"]

        main_outputs = self.main_branch(
            x_f=batch["x_f_obs"],
            m_f=batch["m_f"],
            x_m=batch["x_m_obs"],
            m_m=batch["m_m"],
            x_c=batch["x_c_obs"],
            m_c=batch["m_c"]
        )

        x_hat_main = main_outputs["x_hat_main"]
        h_st_aux = main_outputs["h_st_aux"]

        if self.aux_enabled:
            delta_aux = self.aux_branch(
                h_st_aux=h_st_aux,
                mask_f=m_f
            )
            x_hat_final = x_hat_main + self.alpha * delta_aux
        else:
            delta_aux = torch.zeros_like(x_hat_main)
            x_hat_final = x_hat_main

        if "x_f_gt" in batch:
            x_comp = m_f * batch["x_f_gt"] + (1.0 - m_f) * x_hat_final
        else:
            x_comp = m_f * x_f_obs + (1.0 - m_f) * x_hat_final

        outputs = {
            "x_hat_main": x_hat_main,
            "h_st_aux": h_st_aux,
            "delta_aux": delta_aux,
            "x_hat_final": x_hat_final,
            "x_comp": x_comp,
        }

        outputs.update(main_outputs)

        return outputs
```

---

# 16. Loss 伪代码

## 16.1 Masked Loss

```python
def masked_loss(pred, target, mask, loss_type="smooth_l1"):
    """
    pred:   [B,C,T,H,W]
    target: [B,C,T,H,W]
    mask:   [B,1,T,H,W]
    """

    missing = 1.0 - mask

    pred_m = pred * missing
    target_m = target * missing

    if loss_type == "l1":
        loss = F.l1_loss(pred_m, target_m, reduction="sum")
    elif loss_type == "mse":
        loss = F.mse_loss(pred_m, target_m, reduction="sum")
    else:
        loss = F.smooth_l1_loss(pred_m, target_m, reduction="sum")

    denom = missing.sum().clamp(min=1.0)

    return loss / denom
```

---

## 16.2 Cross-scale Loss

```python
def cross_scale_loss(x_hat_main, x_m_obs, m_m, x_c_obs, m_c):
    """
    x_hat_main: [B,C,T,H,W]
    x_m_obs:    [B,C,T,H/2,W/2]
    m_m:        [B,1,T,H/2,W/2]
    x_c_obs:    [B,C,T,H/4,W/4]
    m_c:        [B,1,T,H/4,W/4]
    """

    ones_f = torch.ones(
        x_hat_main.shape[0],
        1,
        x_hat_main.shape[2],
        x_hat_main.shape[3],
        x_hat_main.shape[4],
        device=x_hat_main.device,
        dtype=x_hat_main.dtype
    )

    x_hat_m, _ = masked_avg_pool2d_spatial(x_hat_main, ones_f, kernel_size=2)

    ones_m = torch.ones_like(m_m)
    x_hat_c, _ = masked_avg_pool2d_spatial(x_hat_m, ones_m, kernel_size=2)

    l_m = F.smooth_l1_loss(x_hat_m * m_m, x_m_obs * m_m, reduction="sum") / m_m.sum().clamp(min=1.0)
    l_c = F.smooth_l1_loss(x_hat_c * m_c, x_c_obs * m_c, reduction="sum") / m_c.sum().clamp(min=1.0)

    return l_m + l_c
```

---

## 16.3 Gate Balance Loss

```python
def gate_balance_loss(gates):
    """
    gates:
        {
            "fine":   [B,K],
            "mid":    [B,K],
            "coarse": [B,K]
        }
    """

    gate_all = torch.cat(
        [gates["fine"], gates["mid"], gates["coarse"]],
        dim=0
    )  # [3B,K]

    usage = gate_all.mean(dim=0)  # [K]
    K = gate_all.shape[1]
    target = torch.ones_like(usage) / K

    loss = ((usage - target) ** 2).sum()

    return loss
```

---

## 16.4 当前阶段总 Loss

```python
def compute_main_stage_loss(outputs, batch, lambdas):
    x_f_gt = batch["x_f_gt"]
    m_f = batch["m_f"]

    l_main = masked_loss(
        pred=outputs["x_hat_main"],
        target=x_f_gt,
        mask=m_f,
        loss_type="smooth_l1"
    )

    l_cross = cross_scale_loss(
        x_hat_main=outputs["x_hat_main"],
        x_m_obs=batch["x_m_obs"],
        m_m=batch["m_m"],
        x_c_obs=batch["x_c_obs"],
        m_c=batch["m_c"]
    )

    l_balance = gate_balance_loss(outputs["gates"])

    loss = (
        l_main
        + lambdas["cross"] * l_cross
        + lambdas["balance"] * l_balance
    )

    loss_dict = {
        "loss": loss,
        "l_main": l_main,
        "l_cross": l_cross,
        "l_balance": l_balance
    }

    return loss, loss_dict
```

推荐：

```python
lambdas = {
    "cross": 0.1,
    "balance": 0.01
}
```

---

# 17. 训练流程

## 17.1 当前阶段训练流程

```text
1. 读取完整数据 X_f_gt
2. 构造缺失 mask M_f
3. 得到 X_f_obs = X_f_gt * M_f
4. 用 masked pooling 构造 X_m_obs, M_m
5. 用 masked pooling 构造 X_c_obs, M_c
6. 输入主分支
7. 得到 x_hat_main
8. 辅助分支输出 delta_aux = 0
9. x_hat_final = x_hat_main
10. 用 L_main + L_cross + L_balance 训练
```

---

## 17.2 后续加入辅助分支流程

```text
1. 加载已经训练好的主分支
2. 保留主分支输出 h_st_aux
3. 辅助分支输入 h_st_aux + text + image + M_f
4. 输出 delta_aux
5. x_hat_final = x_hat_main + alpha * delta_aux
6. 训练辅助分支
7. 最后端到端微调
```

---

# 18. Baseline 和消融实验建议

为了证明主分支 MoE 有效，当前阶段建议做这些实验。

## 18.1 基础 baseline

```text
Mean Fill
Historical Average
Linear Interpolation
Conv3D U-Net
SAITS
```

---

## 18.2 多尺度结构消融

```text
Fine-only
Multi-scale Concat Fusion
Fixed Scale Experts
MoE w/o Router
MoE w/o Cross-scale Expert
Full Main Branch
```

---

## 18.3 消融表格建议

| 方法 | 多尺度输入 | 共享专家池 | 动态 Router | 跨尺度专家 |
|---|---:|---:|---:|---:|
| Fine-only | 否 | 否 | 否 | 否 |
| Multi-scale Concat | 是 | 否 | 否 | 否 |
| Fixed Scale Experts | 是 | 否 | 否 | 否 |
| MoE w/o Router | 是 | 是 | 否 | 是 |
| MoE w/o Cross-scale Expert | 是 | 是 | 是 | 否 |
| Full Main Branch | 是 | 是 | 是 | 是 |

---

# 19. 缺失模式设计

至少测试三类：

```text
Random Missing
Spatial Block Missing
Spatio-temporal Block Missing
```

缺失率：

```text
20%, 40%, 60%, 80%
```

你的模型理论上应该在下面场景优势更明显：

```text
高缺失率
空间块缺失
时空块缺失
```

如果只在 random 20% 上测试，多尺度 MoE 不一定能明显拉开差距。

---

# 20. 推荐实现顺序

## Step 1：数据处理

```text
TAXIBJ.grid -> taxibj_flow.npy
切窗口
构造 mask
构造 fine/mid/coarse
```

## Step 2：主分支最小版本

```text
Multi-scale Embedding
Shared Expert Pool
Router
Cross-scale Expert
Prediction Head
```

先用：

```text
K = 4
D = 64
softmax all experts
T = 12
```

## Step 3：训练主分支

```text
L_main
L_balance
```

先不加 cross loss。

## Step 4：加入 cross loss

```text
L_main + 0.1 * L_cross + 0.01 * L_balance
```

## Step 5：做消融实验

```text
Fine-only
Multi-scale Concat
Fixed Scale Experts
w/o Router
w/o Cross-scale Expert
Full
```

## Step 6：后续再加入辅助分支

```text
AuxiliaryPlaceholder -> Real Multimodal Auxiliary Branch
```

---

# 21. 关键设计总结

当前最新版本的核心点是：

```text
1. 双分支是并行结构；
2. 当前阶段只训练主分支；
3. 辅助分支只是占位，输出 0；
4. 主分支输入 fine / mid / coarse 多尺度观测数据；
5. 多尺度数据从观测值 masked pooling 构造；
6. 每个尺度有自己的 observation-aware router；
7. 三个尺度共享同一个 expert pool；
8. 专家不再和尺度固定绑定；
9. 使用 cross-scale shared expert 建模尺度间关系；
10. 输出最细尺度补全结果；
11. 使用 fine-scale missing loss、cross-scale consistency loss、gate balance loss；
12. 后续辅助分支只接收 h_st_aux、多模态特征和 mask，不接收 x_hat_main；
13. 后续辅助分支输出 delta_aux 作为残差修正。
```

---

# 22. 最终一句话版本

> 本模型采用并行双分支结构。当前阶段主分支通过 observation-aware router 和共享专家池动态处理 fine、mid、coarse 多尺度时空输入，并使用跨尺度共享专家建模尺度间关系，输出最细尺度补全结果；辅助分支当前仅作为占位模块，输出 0，不影响主分支，后续可接入文本和图像多模态特征并输出残差修正项。
