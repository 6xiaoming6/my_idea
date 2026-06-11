# my_idea 当前代码详细改进方案

> 面向当前仓库：`https://github.com/6xiaoming6/my_idea`  
> 目标：在现有 `DualBranchSTImputer / MultiScaleMoEBackbone` 基础上，让“多尺度 + MoE 路由专家”真正站住，而不是继续被 `Shared-Only` 压制。  
> 当前实验依据：更新后的 V2 消融报告显示，当前代码已接入 `ProgressiveScaleGatedFusion`，但 `Shared-Only` 仍是两个数据集上最稳定配置；TaxiBJ 上 `Fine-Only` 最好，BikeNYC 上 `Shared-Only` 最好，Full Model 仍未超过 Shared-Only。

---

## 0. 总体判断

当前代码已经不是早期的简单 concat 融合版本，而是：

```text
fine/mid/coarse embeddings
    ├── Intra-scale routed expert branch
    ├── Cross-scale shared expert branch
    └── ProgressiveScaleGatedFusion
            → x_hat_main
```

这个方向没有错，但当前实验说明：

```text
Shared-Only > Full Model > Routed-Only
```

尤其 TaxiBJ 上，`Routed-Only / No Router / Fixed Experts` 泛化明显崩溃，说明当前主要瓶颈已经不是“融合模块不够复杂”，而是：

1. `Routed Branch` 自身泛化差；
2. `No Router` 消融实现可能不是“所有专家均匀融合”，而是“uniform gate 后仍 top-k”，语义不干净；
3. 消融时关闭分支后仍将 zero feature 输入同一个 `ProgressiveScaleGatedFusion`，导致消融解释不够干净；
4. `QualityRouter` 是全局样本级 router，太粗；
5. 缺少 reliability map，模型不知道中粗尺度信息是否可靠；
6. 负载均衡只约束 soft gate 平均使用率，没有约束 top-k 实际选择次数；
7. TaxiBJ 的 `32 → 16 → 8` 三尺度策略可能过度压缩，coarse 尺度可能引入噪声。

因此下一步建议不是继续堆 fusion，而是把主结构改成：

```text
Shared Branch 作为主干
+ Routed Branch 作为残差增强
+ Reliability-aware gate
+ 更干净的消融实现
+ 更细粒度的 Router
```

---

# 1. 当前代码中最需要优先处理的问题

## 1.1 `No Router` 消融语义有问题

当前 `QualityRouter` 关闭时，会返回 uniform gate：

```python
if not self.use_router:
    return uniform_gate(...)
```

但是 `TopKRoutedExpertPool.forward()` 仍然会执行：

```python
top_values, top_indices = torch.topk(gate, k=self.top_k, dim=-1)
```

如果 gate 是：

```text
[0.25, 0.25, 0.25, 0.25]
```

那么 `topk(k=2)` 通常会固定选其中两个专家。这意味着 `No Router` 不是：

```text
4 个专家全部均匀融合
```

而可能变成：

```text
固定选 2 个专家融合
```

这会严重影响 No Router 消融，尤其可以解释 TaxiBJ 上新版 No Router 从 12.49 退化到 20.59 的异常现象。

### 修改建议

给 `TopKRoutedExpertPool.forward()` 增加 `routing_mode`：

```python
class TopKRoutedExpertPool(nn.Module):
    def forward(self, h, gate, routing_mode: str = "topk"):
        expert_outputs = torch.stack([expert(h) for expert in self.experts], dim=1)
        # expert_outputs: [B, K, D, T, H, W]

        if routing_mode == "dense":
            weights = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            z = (weights[:, :, None, None, None, None] * expert_outputs).sum(dim=1)
            top_indices = torch.arange(self.num_experts, device=h.device).view(1, -1).expand(h.shape[0], -1)
            top_weights = weights
            selected_mask = torch.ones_like(weights)
            return z, top_indices, top_weights, selected_mask

        if routing_mode == "topk":
            top_values, top_indices = torch.topk(gate, k=self.top_k, dim=-1)
            top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            selected_mask = torch.zeros_like(gate)
            selected_mask.scatter_(1, top_indices, 1.0)
            z = torch.zeros_like(expert_outputs[:, 0])
            for slot in range(self.top_k):
                idx = top_indices[:, slot]
                w = top_weights[:, slot].view(h.shape[0], 1, 1, 1, 1)
                for k in range(self.num_experts):
                    mask = (idx == k).to(h.dtype).view(h.shape[0], 1, 1, 1, 1)
                    z = z + mask * w * expert_outputs[:, k]
            return z, top_indices, top_weights, selected_mask
```

然后配置语义改成：

```json
{
  "model": {
    "main": {
      "use_router": false,
      "routing_mode_when_no_router": "dense"
    }
  }
}
```

也就是：

```text
No Router = 所有专家等权参与，而不是 top-k 固定选择。
```

---

## 1.2 关闭分支后不应该统一用 zero feature 进同一个 fusion

当前逻辑大概是：

```python
if not use_routed_branch:
    z_f = zeros
    z_m = zeros
    z_c = zeros

if not use_shared_branch:
    z_shared = zeros

fusion_outputs = progressive_fusion(z_f, z_m, z_c, z_shared)
```

这个实现能跑，但是消融语义不干净。

例如 `Shared-Only` 本来应该是：

```text
h_main = SharedBranch 输出
```

但现在实际是：

```text
z_f/z_m/z_c = 0
z_shared = 有效
ProgressiveScaleGatedFusion(0,0,0,z_shared)
```

模型还要额外学习 gate 如何忽略 zero routed branch。虽然实验上 Shared-Only 仍然强，但这会让不同消融不够纯粹。

### 修改建议

将 forward 拆成干净路径：

```python
if use_shared_branch and not use_routed_branch:
    h_main = self.shared_refine(z_shared)

elif use_routed_branch and not use_shared_branch:
    route_outputs = self.route_fusion(z_f, z_m, z_c)
    h_main = route_outputs["h_route"]

elif use_shared_branch and use_routed_branch:
    h_shared = self.shared_refine(z_shared)
    route_outputs = self.route_fusion(z_f, z_m, z_c)
    h_route = route_outputs["h_route"]
    gamma = torch.sigmoid(self.route_gamma)
    h_main = h_shared + gamma * self.route_proj(h_route)
else:
    raise ValueError("At least one of shared/routed branch must be enabled.")
```

这样每个消融的含义更清楚：

| 配置 | 实际含义 |
|---|---|
| Shared-Only | 只用共享分支输出 |
| Routed-Only | 只用路由分支输出 |
| Full | Shared 主干 + Routed 残差增强 |

---

# 2. 最推荐的结构改动：Shared 主干 + Routed 残差增强

当前 Full Model 的问题是：

```text
Shared-Only 很强，加入 Routed Branch 后反而下降。
```

所以不应该再让 Routed Branch 和 Shared Branch 并列竞争，而应该让 Shared Branch 作为主干，Routed Branch 只做残差增强。

## 2.1 新主分支结构

```text
Input:
X_f, M_f
X_m, M_m
X_c, M_c

Embedding:
H_f, H_m, H_c

Shared Branch:
H_shared = CrossScaleSharedExpert(H_f, H_m, H_c)
H_shared = SharedRefine(H_shared)

Routed Branch:
Z_f = RoutedExperts(H_f, gate_f)
Z_m = RoutedExperts(H_m, gate_m)
Z_c = RoutedExperts(H_c, gate_c)
H_route = ProgressiveRouteFusion(Z_f, Z_m, Z_c)

Final:
H_main = H_shared + gamma * RouteProj(H_route)
x_hat_main = PredHead(H_main)
```

其中：

```python
self.route_gamma = nn.Parameter(torch.tensor(-4.0))
gamma = torch.sigmoid(self.route_gamma)
```

`gamma≈0.018`，初始化时模型几乎等价于 Shared-Only。只有 Routed Branch 真的有价值时，训练才会逐渐增大 gamma。

## 2.2 为什么这个改动最重要

这个方案能直接解决你当前实验中的核心矛盾：

```text
Shared-Only 是最强主干，但 Full 被 Routed Branch 拖累。
```

残差式路由的好处是：

1. 训练初始不破坏 Shared-Only；
2. Routed Branch 没用时 gamma 会保持很小；
3. Routed Branch 有用时才作为增强项加入；
4. 可以通过 gamma 解释路由分支贡献；
5. 消融更干净。

## 2.3 推荐新增模块

### SharedRefine

```python
self.shared_refine = nn.Sequential(
    ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout),
    ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout),
)
```

### RouteFusionOnly

把你现在的 `ProgressiveScaleGatedFusion` 拆成两个版本：

```text
ProgressiveScaleGatedFusion: 融合 Z_f/Z_m/Z_c/Z_shared
ProgressiveRouteFusion: 只融合 Z_f/Z_m/Z_c
```

推荐新建：

```python
class ProgressiveRouteFusion(nn.Module):
    def __init__(self, dim, num_groups=8, dropout=0.0):
        super().__init__()
        self.up_c_to_m = LearnableUpsample3D(dim, num_groups, dropout)
        self.fuse_m_c = GatedFusion2(dim, num_groups, dropout)
        self.up_mc_to_f = LearnableUpsample3D(dim, num_groups, dropout)
        self.fuse_f_mc = GatedFusion2(dim, num_groups, dropout)

    def forward(self, z_f, z_m, z_c):
        _, _, t, h, w = z_f.shape
        _, _, _, hm, wm = z_m.shape
        z_c_to_m = self.up_c_to_m(z_c, target_size=(t, hm, wm))
        z_mc, gate_16 = self.fuse_m_c(z_m, z_c_to_m)
        z_mc_to_f = self.up_mc_to_f(z_mc, target_size=(t, h, w))
        h_route, gate_32_route = self.fuse_f_mc(z_f, z_mc_to_f)
        return {
            "h_route": h_route,
            "z_mc": z_mc,
            "z_mc_to_f": z_mc_to_f,
            "gate_16": gate_16,
            "gate_32_route": gate_32_route,
        }
```

---

# 3. Reliability-aware 多尺度建模

当前中粗尺度是从缺失观测数据 masked pooling 得到的。问题是：

```text
一个 mid/coarse cell 可能由很多 observed fine cells 聚合而来；
另一个 mid/coarse cell 可能只由 1 个 observed cell 聚合而来。
```

这两者可靠性不同，但当前模型没有显式知道。

## 3.1 数据构造时保留 reliability map

建议 `masked_pool2d_spatial` 返回：

```text
x_down: [B,C,T,H/k,W/k]
m_down: [B,1,T,H/k,W/k]
r_down: [B,1,T,H/k,W/k]
```

其中：

```python
r_down = observed_count / (kernel_size * kernel_size)
```

含义：

```text
r_down = 1.0  → 该粗格子由完整观测聚合而来
r_down = 0.25 → 该粗格子只由 25% 观测聚合而来
r_down = 0.0  → 完全没有观测
```

## 3.2 将 reliability 输入 Shared Branch

当前 `CrossScaleSharedExpert` 输入：

```python
cat([h_f, h_m_up, h_c_up], dim=1)
```

建议改成：

```python
cat([
    h_f,
    h_m_up,
    h_c_up,
    mask_embed_f,
    mask_embed_m_up,
    mask_embed_c_up,
    rel_embed_m_up,
    rel_embed_c_up,
], dim=1)
```

第一版可以轻量实现：

```python
self.quality_proj = nn.Conv3d(5, dim, kernel_size=1)
quality_feat = self.quality_proj(torch.cat([m_f, m_m_up, m_c_up, r_m_up, r_c_up], dim=1))
z_shared = self.net(torch.cat([h_f, h_m_up, h_c_up, quality_feat], dim=1))
```

然后把第一层 `Conv3d(dim*3, dim)` 改成 `Conv3d(dim*4, dim)`。

## 3.3 将 reliability 输入 gate

对于 route fusion gate：

```text
Gate_16 input:
Concat(Z_m, Z_c_to_m, M_m, M_c_to_m, R_m, R_c_to_m)

Gate_32 input:
Concat(Z_f, Z_mc_to_f, M_f, R_m_to_f, R_c_to_f)
```

这样模型可以学会：

```text
低可靠 coarse 少用；
高可靠 coarse 多用；
高缺失区域多依赖 mid/coarse；
低缺失区域多依赖 fine。
```

这对 TaxiBJ 尤其重要，因为 TaxiBJ 的 `32→8` coarse 信息可能损失较大，必须让模型知道哪些 coarse 特征可信。

---

# 4. Router 改进：从 sample-level 到 patch-level

当前 `QualityRouter` 是：

```python
pooled = h.mean(dim=(2, 3, 4))
logits = MLP([pooled, q, scale_embed])
gate = softmax(logits)
```

也就是：

```text
gate_s: [B, K]
```

这个粒度太粗。一个样本里所有时间、所有空间位置共享同一组专家权重，不适合补全任务。

## 4.1 第一阶段：Time-level Router

先改成时间级：

```text
gate_s: [B, T, K]
```

实现方式：

```python
pooled_t = h.mean(dim=(3, 4)).transpose(1, 2)  # [B,T,D]
q_t = timewise_mask_stats(mask)               # [B,T,Q]
logits = self.net(torch.cat([pooled_t, q_t, scale_embed_t], dim=-1))
gate = torch.softmax(logits, dim=-1)
```

适合处理：

```text
某些时间步缺失严重，某些时间步观测充分。
```

## 4.2 第二阶段：Patch-level Router

再改成局部 patch 路由：

```text
gate_s: [B, K, T, H_s/r, W_s/r]
```

例如：

```text
fine 32×32 → router map 8×8
mid 16×16 → router map 4×4
coarse 8×8 → router map 2×2
```

推荐实现：

```python
class PatchQualityRouter(nn.Module):
    def __init__(self, dim, num_experts, patch_size=4):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size))
        self.net = nn.Sequential(
            nn.Conv3d(dim + 2, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(dim, num_experts, kernel_size=1),
        )

    def forward(self, h, mask, reliability=None):
        h_p = self.pool(h)       # [B,D,T,Hp,Wp]
        m_p = self.pool(mask)    # [B,1,T,Hp,Wp]
        if reliability is None:
            r_p = m_p
        else:
            r_p = self.pool(reliability)
        logits = self.net(torch.cat([h_p, m_p, r_p], dim=1))
        return torch.softmax(logits, dim=1)  # [B,K,T,Hp,Wp]
```

然后将 patch gate 上采样回专家输出大小，或者在 patch 分辨率上做专家加权后再上采样。

## 4.3 Router warm-up

不要一开始就 hard top-k。建议训练策略：

```text
0%~20% epoch: dense soft routing，所有专家参与
20%~50% epoch: temperature softmax，tau 从 2 降到 1
50%~100% epoch: hard top-k
```

配置示例：

```json
{
  "model": {
    "main": {
      "routing_schedule": {
        "warmup_dense_ratio": 0.2,
        "soft_topk_ratio": 0.3,
        "final_mode": "hard_topk"
      }
    }
  }
}
```

---

# 5. 负载均衡损失升级

当前 `gate_balance_loss` 只做：

```python
gate_all = cat([gate_f, gate_m, gate_c], dim=0)
usage = gate_all.mean(dim=0)
loss = ((usage - 1/K) ** 2).sum()
```

这只是 soft gate importance balance，不是 top-k load balance。

## 5.1 增加 load balance

专家负载应该同时看：

```text
importance_k: 专家获得的 gate 权重总量
load_k: 专家被 top-k 选中的次数
```

建议：

```python
def moe_balance_loss(gates, selected_masks, top_k: int):
    gate_all = torch.cat([gates["fine"], gates["mid"], gates["coarse"]], dim=0)
    mask_all = torch.cat([selected_masks["fine"], selected_masks["mid"], selected_masks["coarse"]], dim=0)

    k = gate_all.shape[1]
    importance = gate_all.mean(dim=0)
    load = mask_all.mean(dim=0)

    target_importance = torch.full_like(importance, 1.0 / k)
    target_load = torch.full_like(load, top_k / k)

    l_importance = ((importance - target_importance) ** 2).sum()
    l_load = ((load - target_load) ** 2).sum()
    return l_importance + l_load
```

## 5.2 注意 No Router 模式

如果 `use_router=False` 且使用 dense all experts，则不应该计算 load balance，或者设为 0：

```python
if not use_router or routing_mode == "dense":
    l_balance = 0
```

否则 No Router 消融会被不必要的 balance loss 干扰。

---

# 6. 多尺度策略改进：不要强制所有数据集都用三尺度

当前实验已经说明：

```text
TaxiBJ: Fine-Only 最好
BikeNYC: Shared-Only 最好
```

这说明不同数据集对多尺度的需求不同。TaxiBJ 的 `32×32 → 8×8` 可能损失太多细节。

## 6.1 增加 scale_mode 配置

建议支持：

```json
{
  "data": {
    "scales": {
      "scale_mode": "fine_mid_coarse"
    }
  }
}
```

可选：

```text
fine_only
fine_mid
fine_mid_coarse
```

## 6.2 需要跑的关键实验

TaxiBJ 必跑：

```text
Fine-Only
Fine + Mid
Fine + Mid + Coarse
```

BikeNYC 必跑：

```text
Fine-Only
Fine + Mid
Fine + Mid + Coarse
```

判断逻辑：

```text
如果 TaxiBJ: Fine + Mid > Fine + Mid + Coarse
说明 coarse 8×8 有害。

如果 TaxiBJ: Fine-Only > Fine + Mid
说明 TaxiBJ 主要依赖细粒度局部模式。

如果 BikeNYC: Fine + Mid + Coarse 最好
说明低分辨率区域趋势对 BikeNYC 有价值。
```

## 6.3 pooling mode 消融

对于交通流量数据，coarse 既可以做 average，也可以做 sum。

建议实验：

```text
masked average pooling
masked sum pooling
masked average + reliability
```

如果你最终用 average，需要在论文中解释：

> 中粗尺度作为上下文特征，而不是严格物理总流量，因此采用 masked average 保持数值稳定，并用 reliability 描述聚合可信度。

---

# 7. CrossScaleSharedExpert 也可以升级

当前 `CrossScaleSharedExpert` 内部仍是：

```text
h_m_up = interpolate(h_m)
h_c_up = interpolate(h_c)
cat([h_f,h_m_up,h_c_up])
Conv3d(dim*3, dim)
2×ResidualSTBlock
```

虽然最后 fusion 已经升级，但 shared branch 内部仍然是直接上采样拼接。

## 7.1 建议升级成 HierarchicalSharedExpert

```text
h_c 8×8
  → LearnableUpsample 8→16
  → GatedFuse with h_m
  → h_mc 16×16

h_mc 16×16
  → LearnableUpsample 16→32
  → GatedFuse with h_f
  → z_shared 32×32
```

也就是把你设计的 progressive fusion 思想也用到 shared branch 内部。

不过这个优先级低于“Shared 主干 + Routed 残差”，因为现在 shared branch 已经很强，不要先动最稳定的部分。建议放到第二轮。

---

# 8. 训练与实验流程改进

## 8.1 多 seed

当前 Full Model 存在随机性，建议每个关键配置至少 3 seed：

```text
42, 2024, 2026
```

报告：

```text
mean ± std
```

## 8.2 按缺失模式分别评估

不要只看 mixed 40%。你的模型理论上应该在下面场景更有优势：

```text
spatial block missing
spatiotemporal block missing
high missing ratio
```

建议每个数据集报告：

| Missing Type | Fine-Only | Shared-Only | Full | Improved Full |
|---|---:|---:|---:|---:|
| Random 20% |  |  |  |  |
| Random 40% |  |  |  |  |
| Random 60% |  |  |  |  |
| Spatial Block 40% |  |  |  |  |
| Spatiotemporal Block 40% |  |  |  |  |

## 8.3 Gate 可视化

你现在已经输出：

```text
fusion_16: [B,2,T,H/2,W/2]
fusion_32: [B,3,T,H,W]
```

必须利用起来。

建议可视化：

```text
1. gate_16 的 mid/coarse 平均权重
2. gate_32 的 fine/mid-coarse/shared 平均权重
3. 不同缺失模式下 gate 权重变化
4. TaxiBJ Full 中 gate 是否错误偏向 routed 分支
5. Shared-Only 中 gate 是否稳定给 shared 分支
```

如果发现 Full 中 gate_32 给 routed 分支高权重，而 Routed-Only 泛化差，就能解释 Full 为什么不如 Shared-Only。

## 8.4 记录专家使用率

每个 epoch 记录：

```text
fine expert load:   [e1,e2,e3,e4]
mid expert load:    [e1,e2,e3,e4]
coarse expert load: [e1,e2,e3,e4]
importance:         [e1,e2,e3,e4]
gamma:              scalar
fusion_16 mean:     [mid, coarse]
fusion_32 mean:     [fine, mc, shared]
```

这些日志后面可以直接写论文分析。

---

# 9. 推荐下一轮实验路线

不要一次改太多。建议分 5 轮。

## Round 0：先修正消融语义

目的：确认当前异常不是实现问题。

```text
TaxiBJ:
1. No Router + dense all experts
2. No Router + current top-k behavior
3. Shared-Only clean forward
4. Shared-Only current zero-branch fusion
```

如果 `No Router + dense all experts` 恢复到 12~14，而 current top-k 是 20，则说明之前 No Router 消融不可信。

## Round 1：Shared 主干 + Routed 残差

配置：

```text
1. Shared-Only
2. Full current
3. Shared + gamma * Routed, gamma init = 0
4. Shared + gamma * Routed, gamma init = -4 sigmoid
5. Shared + gamma * Routed, gamma init = 0.1
```

核心目标：

```text
Shared + gamma * Routed >= Shared-Only
```

只要不低于 Shared-Only，说明 Routed Branch 至少不会拖累。

## Round 2：多尺度数量消融

```text
Fine-Only
Fine + Mid
Fine + Mid + Coarse
```

两个数据集都跑。

目标：确定 TaxiBJ 是否应该禁用 coarse。

## Round 3：Reliability-aware gate

```text
Shared-Only
Shared-Only + reliability
Shared + Routed Residual
Shared + Routed Residual + reliability
```

目标：判断 reliability 是否能缓解 TaxiBJ coarse 噪声。

## Round 4：Router 升级

```text
Sample-level Router
Time-level Router
Patch-level Router
```

建议先在 BikeNYC 上试，因为 BikeNYC 多尺度有效且训练更稳定；TaxiBJ 后续再跑。

---

# 10. 代码修改清单

## 10.1 `src/stmoe_imputer/models/experts.py`

修改：

- `TopKRoutedExpertPool.forward()` 增加 `routing_mode`；
- 支持 `dense`、`topk`、`soft_topk`；
- 返回 `selected_mask`；
- 修正 No Router 消融。

新增返回：

```python
return z, top_indices, top_weights, selected_mask
```

---

## 10.2 `src/stmoe_imputer/models/fusion.py`

保留现有：

```text
LearnableUpsample3D
GatedFusion2
GatedFusion3
ProgressiveScaleGatedFusion
```

新增：

```text
ProgressiveRouteFusion
SharedRoutedResidualFusion
ReliabilityAwareGatedFusion2
ReliabilityAwareGatedFusion3
```

第一步只做 `ProgressiveRouteFusion` 和 `SharedRoutedResidualFusion`。

---

## 10.3 `src/stmoe_imputer/models/main_branch.py`

核心改动：

1. 不再无条件调用 `progressive_fusion(z_f,z_m,z_c,z_shared)`；
2. 按分支开关走不同 forward；
3. Full 模式改为：

```python
h_shared = self.shared_refine(z_shared)
h_route = self.route_fusion(z_f, z_m, z_c)["h_route"]
h_main = h_shared + sigmoid(self.route_gamma) * self.route_proj(h_route)
```

4. 输出字典增加：

```python
"route_gamma": gamma.detach(),
"selected_masks": {...},
"routing_mode": ..., 
"branch_mode": ...,
```

---

## 10.4 `src/stmoe_imputer/losses.py`

修改：

- `gate_balance_loss` 改为 `moe_balance_loss`；
- 同时使用 gate importance 和 selected_mask load；
- No Router dense 模式下禁用 balance loss；
- 可选增加 fusion gate entropy 正则。

建议新增：

```python
def fusion_entropy_loss(fusion_gate):
    entropy = -(fusion_gate * (fusion_gate.clamp_min(1e-8)).log()).sum(dim=1).mean()
    return entropy
```

注意：

```text
如果希望 gate 更明确，可以最小化 entropy；
如果希望早期训练不塌缩，可以最大化 entropy 或 warm-up 后再关闭。
```

---

## 10.5 `src/stmoe_imputer/data/transforms.py`

修改：

- `masked_pool2d_spatial` 返回 reliability；
- 数据 batch 中增加 `r_m`、`r_c`；
- 后续传入模型。

---

## 10.6 `configs/*.json`

新增配置：

```json
{
  "model": {
    "main": {
      "branch_fusion_mode": "shared_plus_routed_residual",
      "route_gamma_init": -4.0,
      "routing_mode": "topk",
      "routing_mode_when_no_router": "dense",
      "router_granularity": "sample",
      "use_reliability": false,
      "scale_mode": "fine_mid_coarse"
    }
  },
  "loss": {
    "lambda_importance_balance": 0.01,
    "lambda_load_balance": 0.01,
    "lambda_fusion_entropy": 0.0
  }
}
```

新增消融 config：

```text
ablation_shared_clean.json
ablation_no_router_dense.json
ablation_no_router_topk.json
ablation_shared_plus_routed_residual.json
ablation_fine_mid_only.json
ablation_reliability_shared.json
ablation_patch_router.json
```

---

# 11. 最终推荐模型版本

短期最推荐的模型不是当前 Full，而是：

```text
Reliability-Aware Shared Backbone with Residual Routed Experts
```

数据流：

```text
X_f,M_f / X_m,M_m,R_m / X_c,M_c,R_c
        ↓
ScaleTokenEncoder
        ↓
H_f,H_m,H_c
        ↓
CrossScaleSharedExpert + reliability
        ↓
H_shared

H_f,H_m,H_c
        ↓
QualityRouter / PatchRouter
        ↓
Top-k shared routed experts
        ↓
Z_f,Z_m,Z_c
        ↓
ProgressiveRouteFusion
        ↓
H_route

Final:
H_main = H_shared + gamma * H_route
x_hat_main = PredictionHead(H_main)
```

这个版本的优势：

1. 保留当前最稳定的 Shared-Only 作为主干；
2. 路由专家只做残差增强，不会强行拖累主干；
3. gamma 可解释路由分支是否真的有贡献；
4. reliability 解决中粗尺度可信度问题；
5. patch/time router 后续可以逐步替换 sample router；
6. 消融语义更干净，论文更好写。

---

# 12. 论文叙事建议

你当前最稳的论文叙事不应该是：

```text
我提出了一个 Full MoE，它一定最好。
```

而应该是：

```text
多尺度补全中，跨尺度共享建模是稳定有效的主干；
但直接并联尺度内路由专家会引入噪声，尤其在高分辨率细粒度场景中容易过拟合。
因此，我们进一步将路由专家从并列主分支改为残差增强分支，
并通过可靠性建模和局部路由机制，使模型能够在不同缺失模式下自适应利用尺度内专家信息。
```

这种说法更符合你目前实验事实，也更像真实科研迭代。

---

# 13. 最短执行清单

如果只做最关键的 5 件事，建议按这个顺序：

```text
1. 修复 No Router：uniform gate 时 dense all experts，不再 top-k。
2. 改干净 forward：Shared-Only / Routed-Only / Full 走不同路径，不再全靠 zero branch。
3. Full 改成 H_main = H_shared + gamma * H_route。
4. 增加 fine+mid 消融，判断 TaxiBJ 是否该禁用 coarse。
5. 记录并可视化 fusion_16 / fusion_32 / expert load / gamma。
```

这 5 步做完，你的下一轮实验就会清楚很多，也能直接判断路由专家还有没有继续投入的价值。
