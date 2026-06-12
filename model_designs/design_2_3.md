# v2.3 修改设计文档：scale_mode + reliability-aware scale gate

> 目标：在当前 `DualBranchSTImputer + ProgressiveRouteFusion + SharedRoutedResidualFusion` 基础上，优先加入 **尺度选择机制（scale_mode）** 与 **可靠性感知尺度门控（reliability-aware scale gate）**。  
> 适用代码结构：`models/main_branch.py`、`models/fusion.py`、`models/experts.py`、`models/stats.py`、`losses.py`、`configs/*.json`。  
> 当前核心问题：BikeNYC 上多尺度有效，TaxiBJ 上三尺度可能引入噪声；因此需要让模型显式支持不同尺度组合，并让 shared branch 自动判断 fine/mid/coarse 的可信度。

---

## 1. 修改动机

当前实验已经说明：

```text
BikeNYC:
Full Model 最优或接近最优，多尺度有效。

TaxiBJ:
Fine-Only 仍然最强，三尺度 Full Model 不稳定，多尺度可能带来噪声。
```

这说明当前固定使用：

```text
fine + mid + coarse
```

不是所有数据集都合适。

尤其 TaxiBJ 是 `32×32` 网格，降采样到：

```text
mid:    16×16
coarse: 8×8
```

后可能损失较多局部细节；而 BikeNYC 是 `24×12`，本身空间分辨率更低，区域趋势更明显，因此中粗尺度更有用。

所以 v2.3 的修改目标是：

```text
1. 支持 scale_mode，让模型可以只用 fine、fine+mid 或 fine+mid+coarse；
2. 在 shared branch 中加入 reliability-aware scale gate；
3. 让模型根据数据集和缺失模式自动降低低可靠尺度的权重；
4. 减少 TaxiBJ 上 coarse 尺度噪声对 Full Model 的干扰；
5. 保持 BikeNYC 上多尺度收益。
```

---

## 2. 当前模型与 v2.3 修改位置

当前模型主干可以抽象为：

```text
H_f, H_m, H_c
   ├── Shared Branch → h_shared
   └── Routed Branch → h_route

h_main = h_shared + sigmoid(route_gamma) * h_route
```

v2.3 不推翻这个结构，只修改两个关键点。

### 2.1 新增 scale_mode

新增配置字段：

```json
{
  "model": {
    "scale_mode": "fine_mid_coarse"
  }
}
```

可选值：

```text
fine
fine_mid
fine_mid_coarse
```

含义：

| scale_mode | 使用尺度 | 适合场景 |
|---|---|---|
| `fine` | 只用 fine | TaxiBJ 等细粒度局部模式强的数据 |
| `fine_mid` | 用 fine + mid | 高分辨率数据，避免 coarse 过粗 |
| `fine_mid_coarse` | 用 fine + mid + coarse | BikeNYC 等多尺度区域趋势明显的数据 |

后续可以扩展：

```text
auto
```

让模型通过 gate 自动学习是否使用 mid/coarse，但第一版建议先显式配置，方便做消融。

### 2.2 新增 reliability-aware scale gate

当前 `CrossScaleSharedExpert` 大概率是：

```text
Concat(H_f, Up(H_m), Up(H_c))
    → Conv1×1
    → ResBlocks
    → Z_shared
```

v2.3 改成：

```text
ScaleGate(H_f, H_m, H_c, q_f, q_m, q_c, R_m, R_c)
    → scale_weight: [B, 3]

H_f_weighted = w_f * H_f
H_m_weighted = w_m * Up(H_m)
H_c_weighted = w_c * Up(H_c)

Concat(H_f_weighted, H_m_weighted, H_c_weighted)
    → CrossScaleSharedExpert
    → Z_shared
```

其中 `R_m / R_c` 表示中粗尺度聚合可靠性。模型可以根据：

```text
missing_rate
observed_ratio
aggregation_reliability
```

动态决定：

```text
当前位置或当前样本应该更相信 fine、mid 还是 coarse。
```

第一版先实现 **样本级 scale gate**：

```text
scale_weight: [B, 3]
```

后续再扩展为 **位置级 scale gate**：

```text
scale_weight: [B, 3, T, H, W]
```

---

## 3. 数据侧修改：保留多尺度 reliability map

你现在构造多尺度数据时应该已经做了 masked pooling。建议统一输出：

```python
x_m, m_m, r_m = masked_avg_pool2d_spatial(x_f_obs, m_f, scale=2)
x_c, m_c, r_c = masked_avg_pool2d_spatial(x_m, m_m, scale=2)
```

其中：

```text
x_m: [B, C, T, H/2, W/2]
m_m: [B, 1, T, H/2, W/2]
r_m: [B, 1, T, H/2, W/2]

x_c: [B, C, T, H/4, W/4]
m_c: [B, 1, T, H/4, W/4]
r_c: [B, 1, T, H/4, W/4]
```

`r_m` 和 `r_c` 的定义：

```text
r_down = observed_count_in_pooling_window / pooling_window_size
```

例如 `2×2` pooling：

```text
r = 0.00 表示这个 coarse/mid cell 没有任何观测值；
r = 0.25 表示只有 1/4 的细格子被观测；
r = 1.00 表示 4/4 全部观测。
```

如果数据加载模块暂时没有返回 `r_m/r_c`，可以先用 `m_m/m_c` 作为近似：

```python
if r_m is None:
    r_m = m_m.float()
if r_c is None:
    r_c = m_c.float()
```

但最终建议使用真正的 pooling reliability，而不是二值 mask。

---

## 4. scale_mode 详细设计

### 4.1 配置字段

在 `configs/taxibj.json` 和 `configs/bikenyc.json` 中加入：

```json
{
  "model": {
    "scale_mode": "fine_mid_coarse",
    "use_scale_gate": true,
    "use_reliability_gate": true
  }
}
```

建议默认配置：

#### TaxiBJ 初始建议

```json
{
  "model": {
    "scale_mode": "fine_mid",
    "use_scale_gate": true,
    "use_reliability_gate": true
  }
}
```

原因：TaxiBJ 的 `8×8` coarse 可能过粗，先测试 `fine_mid`。

#### BikeNYC 初始建议

```json
{
  "model": {
    "scale_mode": "fine_mid_coarse",
    "use_scale_gate": true,
    "use_reliability_gate": true
  }
}
```

原因：BikeNYC 上多尺度收益明显。

---

### 4.2 scale_mode 对各模块的影响

#### `scale_mode = fine`

```text
有效尺度：
fine

禁用：
mid
coarse

Shared Branch:
只使用 H_f

Routed Branch:
只使用 Z_f

loss:
关闭 cross-scale loss
关闭 mid/coarse 相关 scale consistency
```

#### `scale_mode = fine_mid`

```text
有效尺度：
fine
mid

禁用：
coarse

Shared Branch:
使用 H_f + H_m

Routed Branch:
使用 Z_f + Z_m

ProgressiveRouteFusion:
只执行 mid → fine
不执行 coarse → mid

loss:
只计算 fine/mid cross-scale loss
不计算 coarse loss
```

#### `scale_mode = fine_mid_coarse`

```text
有效尺度：
fine
mid
coarse

Shared Branch:
使用 H_f + H_m + H_c

Routed Branch:
使用 Z_f + Z_m + Z_c

ProgressiveRouteFusion:
coarse → mid → fine

loss:
计算 fine/mid/coarse cross-scale loss
```

---

### 4.3 scale_mode 工具函数

建议新增 `models/scale_utils.py`：

```python
import torch


def get_active_scales(scale_mode: str):
    if scale_mode == "fine":
        return ["fine"]
    if scale_mode == "fine_mid":
        return ["fine", "mid"]
    if scale_mode == "fine_mid_coarse":
        return ["fine", "mid", "coarse"]
    raise ValueError(f"Unknown scale_mode: {scale_mode}")


def is_scale_active(scale_mode: str, scale: str):
    return scale in get_active_scales(scale_mode)


def build_scale_active_mask(scale_mode: str, batch_size: int, device):
    if scale_mode == "fine":
        mask = torch.tensor([1, 0, 0], device=device, dtype=torch.bool)
    elif scale_mode == "fine_mid":
        mask = torch.tensor([1, 1, 0], device=device, dtype=torch.bool)
    elif scale_mode == "fine_mid_coarse":
        mask = torch.tensor([1, 1, 1], device=device, dtype=torch.bool)
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")

    return mask.view(1, 3).expand(batch_size, 3)
```

后面所有模块都根据这个判断是否使用 mid/coarse。

---

## 5. ReliabilityAwareScaleGate 设计

### 5.1 作用

`ReliabilityAwareScaleGate` 用于 shared branch，在多尺度特征进入 `CrossScaleSharedExpert` 之前，生成每个尺度的权重。

它要解决的问题：

```text
不是所有尺度在所有数据集和所有缺失模式下都可靠。
```

例如：

```text
TaxiBJ:
coarse 8×8 可能过度平滑，应降低权重。

BikeNYC:
mid/coarse 区域趋势有用，应保留较高权重。

高缺失率:
fine 观测不足，mid/coarse 可能更有用。

低可靠 coarse:
r_c 很低，应降低 coarse 权重。
```

---

### 5.2 输入

建议输入：

```text
h_f: [B, D, T, H, W]
h_m: [B, D, T, H/2, W/2]
h_c: [B, D, T, H/4, W/4]

q_f: [B, Q]
q_m: [B, Q]
q_c: [B, Q]

r_m: [B, 1, T, H/2, W/2]
r_c: [B, 1, T, H/4, W/4]
```

其中 `q_s` 是你已有的 observation statistics：

```text
missing_rate
observed_ratio
temporal_missing_score
spatial_missing_score
aggregation_reliability
```

如果当前 `q_s` 里已经包含 `aggregation_reliability`，`r_m/r_c` 仍然建议保留，因为它能更直接反映 pooled cell 的可用观测比例。

---

### 5.3 输出

第一版输出样本级尺度权重：

```text
scale_weight: [B, 3]
```

对应：

```text
w_f, w_m, w_c
```

如果 `scale_mode="fine_mid"`，可以统一输出 `[B, 3]`，但对 inactive scale 做 mask：

```text
fine_mid:
w_c = 0

fine:
w_m = 0, w_c = 0
```

推荐统一输出 `[B, 3]`，这样日志和可视化更方便。

---

### 5.4 结构代码

建议在 `models/fusion.py` 或新文件 `models/scale_gate.py` 中加入：

```python
import torch
import torch.nn as nn


class ReliabilityAwareScaleGate(nn.Module):
    def __init__(self, dim: int, stat_dim: int = 5, hidden_dim: int = 128, num_scales: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_scales = num_scales
        input_dim = dim * 3 + stat_dim * 3 + 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_scales),
        )

    def forward(self, h_f, h_m, h_c, q_f, q_m, q_c, r_m=None, r_c=None, active_mask=None):
        """
        h_f: [B, D, T, H, W]
        h_m: [B, D, T, H/2, W/2]
        h_c: [B, D, T, H/4, W/4]
        q_f/q_m/q_c: [B, Q]
        r_m/r_c: [B,1,T,Hs,Ws]
        active_mask: [B,3], bool, True 表示该尺度启用
        """
        B = h_f.size(0)
        device = h_f.device

        p_f = h_f.mean(dim=(2, 3, 4))
        p_m = h_m.mean(dim=(2, 3, 4))
        p_c = h_c.mean(dim=(2, 3, 4))

        if r_m is None:
            r_m_mean = torch.ones(B, 1, device=device)
        else:
            r_m_mean = r_m.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(-1)

        if r_c is None:
            r_c_mean = torch.ones(B, 1, device=device)
        else:
            r_c_mean = r_c.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(-1)

        gate_input = torch.cat(
            [p_f, p_m, p_c, q_f, q_m, q_c, r_m_mean, r_c_mean],
            dim=-1,
        )

        logits = self.mlp(gate_input)

        if active_mask is not None:
            logits = logits.masked_fill(~active_mask, -1e9)

        weight = torch.softmax(logits, dim=-1)
        return weight
```

---

## 6. 修改 CrossScaleSharedExpert

### 6.1 当前版本

当前逻辑大概是：

```python
h_m_up = F.interpolate(h_m, size=h_f.shape[-3:])
h_c_up = F.interpolate(h_c, size=h_f.shape[-3:])

x = torch.cat([h_f, h_m_up, h_c_up], dim=1)
z_shared = self.net(x)
```

---

### 6.2 新版本思路

改为：

```python
scale_weight = self.scale_gate(
    h_f=h_f,
    h_m=h_m,
    h_c=h_c,
    q_f=q_f,
    q_m=q_m,
    q_c=q_c,
    r_m=r_m,
    r_c=r_c,
    active_mask=active_mask,
)

w_f = scale_weight[:, 0].view(B, 1, 1, 1, 1)
w_m = scale_weight[:, 1].view(B, 1, 1, 1, 1)
w_c = scale_weight[:, 2].view(B, 1, 1, 1, 1)

h_f_w = w_f * h_f
h_m_w = w_m * h_m_up
h_c_w = w_c * h_c_up

x = torch.cat([h_f_w, h_m_w, h_c_w], dim=1)
z_shared = self.net(x)
```

输出中加入：

```python
outputs["gates"]["scale_gate"] = scale_weight
```

形状：

```text
scale_gate: [B, 3]
```

---

### 6.3 建议新增类：GatedCrossScaleSharedExpert

为了不破坏旧代码，建议不要直接改原 `CrossScaleSharedExpert`，而是新增：

```python
class GatedCrossScaleSharedExpert(nn.Module):
    ...
```

它内部包含：

```text
ReliabilityAwareScaleGate
Conv3d(3D → D)
ResidualSTBlock ×2
```

伪代码：

```python
class GatedCrossScaleSharedExpert(nn.Module):
    def __init__(self, dim, stat_dim=5, use_scale_gate=True):
        super().__init__()
        self.use_scale_gate = use_scale_gate
        self.scale_gate = ReliabilityAwareScaleGate(dim=dim, stat_dim=stat_dim)
        self.fuse = nn.Sequential(
            nn.Conv3d(dim * 3, dim, kernel_size=1),
            ResidualSTBlock(dim),
            ResidualSTBlock(dim),
        )

    def forward(self, h_f, h_m, h_c, q_f, q_m, q_c, r_m=None, r_c=None, active_mask=None):
        B = h_f.size(0)
        target_size = h_f.shape[-3:]

        h_m_up = F.interpolate(h_m, size=target_size, mode="trilinear", align_corners=False)
        h_c_up = F.interpolate(h_c, size=target_size, mode="trilinear", align_corners=False)

        if self.use_scale_gate:
            scale_weight = self.scale_gate(h_f, h_m, h_c, q_f, q_m, q_c, r_m, r_c, active_mask)
        else:
            # active scales 均匀分配
            scale_weight = active_mask.float()
            scale_weight = scale_weight / scale_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        w_f = scale_weight[:, 0].view(B, 1, 1, 1, 1)
        w_m = scale_weight[:, 1].view(B, 1, 1, 1, 1)
        w_c = scale_weight[:, 2].view(B, 1, 1, 1, 1)

        x = torch.cat([w_f * h_f, w_m * h_m_up, w_c * h_c_up], dim=1)
        z_shared = self.fuse(x)

        return z_shared, scale_weight
```

---

## 7. 修改 ProgressiveRouteFusion

### 7.1 当前版本

当前 routed branch 是：

```text
Z_c → up to mid
GatedFusion2(Z_m, Z_c_up) → Z_mc
Z_mc → up to fine
GatedFusion2(Z_f, Z_mc_up) → H_route
```

---

### 7.2 新增 scale_mode 分支

#### scale_mode = fine

```python
h_route = self.fine_refine(z_f)
```

不执行 mid/coarse 融合。

#### scale_mode = fine_mid

```python
z_m_to_f = self.up_m_to_f(z_m, target_size=z_f.shape[-3:])
h_route, gate_32_route = self.fuse_f_m(z_f, z_m_to_f)
```

不执行 coarse → mid。

#### scale_mode = fine_mid_coarse

保持当前逻辑：

```python
z_c_to_m = self.up_c_to_m(z_c, target_size=z_m.shape[-3:])
z_mc, gate_16 = self.fuse_m_c(z_m, z_c_to_m)

z_mc_to_f = self.up_mc_to_f(z_mc, target_size=z_f.shape[-3:])
h_route, gate_32_route = self.fuse_f_mc(z_f, z_mc_to_f)
```

---

### 7.3 伪代码

```python
class ProgressiveRouteFusion(nn.Module):
    def forward(self, z_f, z_m=None, z_c=None, scale_mode="fine_mid_coarse"):
        outputs = {}

        if scale_mode == "fine":
            h_route = self.fine_refine(z_f)
            outputs["h_route"] = h_route
            outputs["gate_16"] = None
            outputs["gate_32_route"] = None
            return outputs

        if scale_mode == "fine_mid":
            z_m_to_f = self.up_m_to_f(z_m, target_size=z_f.shape[-3:])
            h_route, gate_32_route = self.fuse_f_m(z_f, z_m_to_f)
            outputs["h_route"] = h_route
            outputs["z_m_to_f"] = z_m_to_f
            outputs["gate_16"] = None
            outputs["gate_32_route"] = gate_32_route
            return outputs

        if scale_mode == "fine_mid_coarse":
            z_c_to_m = self.up_c_to_m(z_c, target_size=z_m.shape[-3:])
            z_mc, gate_16 = self.fuse_m_c(z_m, z_c_to_m)
            z_mc_to_f = self.up_mc_to_f(z_mc, target_size=z_f.shape[-3:])
            h_route, gate_32_route = self.fuse_f_mc(z_f, z_mc_to_f)
            outputs["h_route"] = h_route
            outputs["z_c_to_m"] = z_c_to_m
            outputs["z_mc"] = z_mc
            outputs["z_mc_to_f"] = z_mc_to_f
            outputs["gate_16"] = gate_16
            outputs["gate_32_route"] = gate_32_route
            return outputs

        raise ValueError(scale_mode)
```

---

## 8. 修改 MultiScaleMoEBackbone forward

### 8.1 初始化参数

在 `models/main_branch.py`：

```python
class MultiScaleMoEBackbone(nn.Module):
    def __init__(
        self,
        ...,
        scale_mode="fine_mid_coarse",
        use_scale_gate=True,
        use_reliability_gate=True,
        ...,
    ):
        self.scale_mode = scale_mode
        self.use_scale_gate = use_scale_gate
        self.use_reliability_gate = use_reliability_gate
```

---

### 8.2 forward 输入增加 reliability

当前可能是：

```python
forward(x_f, m_f, x_m=None, m_m=None, x_c=None, m_c=None)
```

建议改为：

```python
forward(
    x_f, m_f,
    x_m=None, m_m=None, r_m=None,
    x_c=None, m_c=None, r_c=None,
)
```

如果 `r_m/r_c` 没传，就内部用 mask 估一个：

```python
if r_m is None and m_m is not None:
    r_m = m_m.float()

if r_c is None and m_c is not None:
    r_c = m_c.float()
```

---

### 8.3 active scale 控制

```python
active_scales = get_active_scales(self.scale_mode)

use_mid = "mid" in active_scales
use_coarse = "coarse" in active_scales
```

第一版为了减少代码改动，可以仍然计算 `h_m/h_c`，但在 `active_mask` 中屏蔽 inactive scale。这样不会影响 shape。

如果你希望节省计算，再进一步跳过不活跃尺度的 embedding 和 routing。

---

### 8.4 shared branch 调用

```python
active_mask = build_scale_active_mask(
    self.scale_mode,
    batch_size=x_f.size(0),
    device=x_f.device,
)

z_shared, scale_gate = self.shared_expert(
    h_f=h_f,
    h_m=h_m,
    h_c=h_c,
    q_f=q_f,
    q_m=q_m,
    q_c=q_c,
    r_m=r_m if self.use_reliability_gate else None,
    r_c=r_c if self.use_reliability_gate else None,
    active_mask=active_mask,
)
```

输出：

```python
outputs["gates"]["scale_gate"] = scale_gate
```

---

### 8.5 routed branch 调用

```python
route_outputs = self.route_fusion(
    z_f=z_f,
    z_m=z_m,
    z_c=z_c,
    scale_mode=self.scale_mode,
)

h_route = route_outputs["h_route"]
```

把 route fusion 中间结果加入输出：

```python
outputs["gates"]["gate_16"] = route_outputs.get("gate_16")
outputs["gates"]["gate_32_route"] = route_outputs.get("gate_32_route")
outputs["features"].update(route_outputs)
```

---

## 9. 修改 loss

### 9.1 cross-scale loss 按 scale_mode 计算

当前 cross loss 可能同时算 mid/coarse：

```python
loss_mid = ...
loss_coarse = ...
loss_cross = loss_mid + loss_coarse
```

改成：

```python
loss_cross = 0.0

if scale_mode in ["fine_mid", "fine_mid_coarse"]:
    loss_cross = loss_cross + loss_mid

if scale_mode == "fine_mid_coarse":
    loss_cross = loss_cross + loss_coarse
```

如果 `scale_mode="fine"`：

```python
loss_cross = 0
```

---

### 9.2 balance loss 按 router 是否启用计算

```python
if use_routed_branch and use_router:
    loss += lambda_importance * loss_importance
    loss += lambda_load * loss_load
else:
    loss_importance = 0
    loss_load = 0
```

建议：

| 配置 | cross loss | balance loss |
|---|---:|---:|
| Fine-Only | 关闭 | 关闭或极小 |
| Shared-Only | 可开小值 | 关闭 |
| Routed-Only | 可开小值 | 开启 |
| Full Model | 开启 | 开启 |
| No Router | 开启 | 关闭 |

---

### 9.3 scale gate 正则

第一版不要加太多正则。先观察 `scale_gate` 是否自然学出合理权重。

后续可选：

```python
L_scale_entropy = -torch.sum(w * torch.log(w + 1e-8), dim=-1).mean()
```

如果希望 gate 更明确，可以最小化 entropy：

```text
loss += lambda_scale_entropy * L_scale_entropy
```

但不建议第一版启用。

---

## 10. 推荐配置

### 10.1 TaxiBJ Full

```json
{
  "model": {
    "scale_mode": "fine_mid",
    "use_scale_gate": true,
    "use_reliability_gate": true,
    "branch_fusion_mode": "shared_plus_routed_residual",
    "route_gamma_init": -4.0
  },
  "loss": {
    "lambda_cross": 0.03,
    "lambda_importance": 0.001,
    "lambda_load": 0.001,
    "lambda_fusion_entropy": 0.0
  }
}
```

### 10.2 BikeNYC Full

```json
{
  "model": {
    "scale_mode": "fine_mid_coarse",
    "use_scale_gate": true,
    "use_reliability_gate": true,
    "branch_fusion_mode": "shared_plus_routed_residual",
    "route_gamma_init": -4.0
  },
  "loss": {
    "lambda_cross": 0.1,
    "lambda_importance": 0.01,
    "lambda_load": 0.01,
    "lambda_fusion_entropy": 0.0
  }
}
```

### 10.3 TaxiBJ scale 消融

```json
{
  "model": {
    "scale_mode": "fine"
  }
}
```

```json
{
  "model": {
    "scale_mode": "fine_mid"
  }
}
```

```json
{
  "model": {
    "scale_mode": "fine_mid_coarse"
  }
}
```

---

## 11. 日志记录

必须新增日志项：

```text
scale_mode
use_scale_gate
use_reliability_gate
scale_gate_f_mean
scale_gate_m_mean
scale_gate_c_mean
scale_gate_f_std
scale_gate_m_std
scale_gate_c_std
route_alpha
route_gamma
effective_route_ratio
lambda_cross
lambda_importance
lambda_load
```

每个 epoch 记录：

```python
scale_gate = outputs["gates"]["scale_gate"]  # [B,3]

log_dict["scale_gate_f"] = scale_gate[:, 0].mean().item()
log_dict["scale_gate_m"] = scale_gate[:, 1].mean().item()
log_dict["scale_gate_c"] = scale_gate[:, 2].mean().item()
log_dict["scale_gate_f_std"] = scale_gate[:, 0].std().item()
log_dict["scale_gate_m_std"] = scale_gate[:, 1].std().item()
log_dict["scale_gate_c_std"] = scale_gate[:, 2].std().item()
```

如果 TaxiBJ 上 `scale_mode=fine_mid_coarse` 时模型自动学到：

```text
scale_gate_c 很小
```

那就说明 scale gate 起作用了。

---

## 12. 推荐实验路线

### Round 1：只加 scale_mode，不加 scale_gate

目的：先确认尺度数量影响。

```text
TaxiBJ:
fine
fine_mid
fine_mid_coarse

BikeNYC:
fine
fine_mid
fine_mid_coarse
```

看哪个尺度组合最优。

### Round 2：加入 scale_gate

对比：

```text
fine_mid_coarse without scale_gate
fine_mid_coarse with scale_gate
```

重点看 TaxiBJ：

```text
scale_gate 是否自动降低 coarse 权重？
```

### Round 3：加入 reliability-aware gate

对比：

```text
scale_gate only
scale_gate + reliability
```

重点看高缺失率或 block missing：

```text
reliability 是否让模型少用低可信 mid/coarse？
```

### Round 4：完整最佳配置

最终候选：

```text
TaxiBJ:
fine
fine_mid + scale_gate + reliability
fine_mid_coarse + scale_gate + reliability

BikeNYC:
fine_mid_coarse + scale_gate + reliability
```

---

## 13. 预期结果

### TaxiBJ

理想结果：

```text
fine_mid + scale_gate + reliability
    接近或超过 Fine-Only

fine_mid_coarse + scale_gate
    不再明显劣于 Fine-Only
```

如果 `fine_mid` 明显优于 `fine_mid_coarse`，说明 coarse 过粗。

如果 `scale_gate_c` 很低，则可以证明模型自动识别 coarse 不可靠。

### BikeNYC

理想结果：

```text
fine_mid_coarse + scale_gate + reliability
    保持或超过当前 Full Model

scale_gate_m / scale_gate_c 不应太低
```

如果 BikeNYC 上 coarse 权重较高，说明区域趋势对 BikeNYC 有帮助。

---

## 14. 文件级修改清单

### 14.1 `models/scale_utils.py`

新增：

```text
get_active_scales
is_scale_active
build_scale_active_mask
```

### 14.2 `models/fusion.py`

新增或修改：

```text
ReliabilityAwareScaleGate
ProgressiveRouteFusion 支持 scale_mode
```

后续可选：

```text
ReliabilityGatedFusion2
```

### 14.3 `models/experts.py`

修改：

```text
CrossScaleSharedExpert 支持 scale_gate / reliability / active_mask
```

或新增：

```text
GatedCrossScaleSharedExpert
```

建议新增新类，避免破坏旧代码。

### 14.4 `models/main_branch.py`

修改：

```text
读取 scale_mode
构造 active_mask
传递 r_m / r_c
调用 gated shared expert
调用 scale_mode-aware route fusion
把 scale_gate 写入 outputs["gates"]
```

### 14.5 `losses.py`

修改：

```text
cross_scale_loss 按 scale_mode 计算
moe_balance_loss 按 use_routed_branch/use_router 启用
```

### 14.6 `configs/*.json`

新增字段：

```json
{
  "model": {
    "scale_mode": "fine_mid_coarse",
    "use_scale_gate": true,
    "use_reliability_gate": true
  }
}
```

### 14.7 `train.py` 或 logger

新增日志：

```text
scale_gate_f/m/c
route_alpha
effective_route_ratio
```

---

## 15. 最终推荐实现顺序

不要一次全改。建议按这个顺序：

```text
1. 加 scale_mode，先不加 scale_gate；
2. 修改 loss 条件启用；
3. 加 ReliabilityAwareScaleGate 到 shared branch；
4. 记录 scale_gate 日志；
5. 再考虑 reliability-aware route fusion；
6. 最后做 gate 可视化。
```

这样每一步都能单独验证是否有效。

---

## 16. 总结

v2.3 的重点不是继续堆更复杂的 MoE，而是让模型学会：

```text
什么时候用多尺度；
用哪些尺度；
这些尺度是否可靠。
```

最终目标结构是：

```text
H_f, H_m, H_c, R_m, R_c
        ↓
ReliabilityAwareScaleGate
        ↓
w_f, w_m, w_c

H_shared = CrossScaleSharedExpert(
    w_f * H_f,
    w_m * Up(H_m),
    w_c * Up(H_c)
)

H_route = ScaleModeAwareProgressiveRouteFusion(Z_f, Z_m, Z_c)

H_main = H_shared + sigmoid(route_gamma) * H_route
```

这个方案直接针对当前实验暴露的两个核心问题：

```text
1. TaxiBJ 上 coarse/multiscale 可能有害；
2. BikeNYC 上多尺度有效但路由分支贡献偏小。
```

通过 `scale_mode + reliability-aware scale gate`，模型可以在不同数据集和缺失模式下自适应调整尺度使用策略。
