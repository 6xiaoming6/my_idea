# Progressive Scale Alignment and Gated Fusion 模块设计文档

## 0. 模块定位

这个模块用于替换当前主分支最后阶段中较简单的：

```text
Upsample(Z_m), Upsample(Z_c)
Concat(Z_f, Z_m_up, Z_c_up, Z_shared)
Conv Fusion
```

新的目标是让多尺度特征融合不再只是“上采样 + 拼接”，而是变成：

> **逐级尺度对齐 + 门控融合。**

也就是：

```text
Z_c: 8×8
   → learnable upsample to 16×16
   → 与 Z_m 融合，得到中粗融合特征 Z_mc

Z_mc: 16×16
   → learnable upsample to 32×32
   → 与 Z_f 和 Z_shared 融合，得到最终 H_main
```

该模块可以作为后续主分支的替换增强模块，主要用于提升多尺度融合的结构性、可解释性和表达能力。

---

# 1. 原始简单融合的问题

当前简单融合方式是：

```text
Z_f:      [B, D, T, 32, 32]
Z_m:      [B, D, T, 16, 16] → upsample → Z_m_up: [B, D, T, 32, 32]
Z_c:      [B, D, T, 8, 8]   → upsample → Z_c_up: [B, D, T, 32, 32]
Z_shared: [B, D, T, 32, 32]

F_fuse = Concat(Z_f, Z_m_up, Z_c_up, Z_shared)
F_fuse: [B, 4D, T, 32, 32]

H_main = FusionNetwork(F_fuse)
```

这个版本可以作为第一版跑通，但作为正式方法会显得偏简单，主要有几个问题：

1. **8×8 粗尺度直接上采样到 32×32 跨度太大**  
   粗尺度特征被直接扩散到大量细粒度位置，容易过度平滑。

2. **没有体现多尺度的层级关系**  
   更自然的多尺度关系应该是：
   ```text
   coarse 8×8 → mid 16×16 → fine 32×32
   ```
   而不是 coarse 直接跳到 fine。

3. **concat + conv 的融合方式不够显式**  
   模型没有明确判断当前位置应该更依赖 fine、mid、coarse 还是 shared 特征。

4. **缺少尺度可靠性建模**  
   由 masked pooling 得到的 mid/coarse 特征可靠性不同，直接上采样拼接会把低可靠和高可靠的尺度信息混在一起。

因此建议将该阶段替换为：

```text
Progressive Scale Alignment and Gated Fusion
```

---

# 2. 新模块总体思路

新的融合模块分成两个阶段。

## Stage 1：coarse-to-mid 融合

```text
Z_c: [B, D, T, 8, 8]
        ↓ LearnableUpsample 8→16
Z_c_16: [B, D, T, 16, 16]

Z_m: [B, D, T, 16, 16]

GatedFuse_16(Z_m, Z_c_16)
        ↓
Z_mc: [B, D, T, 16, 16]
```

含义：

> 先将 coarse 的全局趋势融合进 mid 尺度，得到带有粗尺度上下文的区域级特征。

## Stage 2：mid-to-fine 融合

```text
Z_mc: [B, D, T, 16, 16]
        ↓ LearnableUpsample 16→32
Z_mc_32: [B, D, T, 32, 32]

Z_f:      [B, D, T, 32, 32]
Z_shared: [B, D, T, 32, 32]

GatedFuse_32(Z_f, Z_mc_32, Z_shared)
        ↓
H_main: [B, D, T, 32, 32]
```

含义：

> 再将已经融合 coarse 的 mid 特征与 fine 细节特征、共享跨尺度特征一起融合，得到最终的高分辨率主分支特征。

---

# 3. 输入与输出

## 3.1 输入

来自路由专家分支和共享专家分支：

```text
Z_f:      [B, D, T, H,   W  ]
Z_m:      [B, D, T, H/2, W/2]
Z_c:      [B, D, T, H/4, W/4]
Z_shared: [B, D, T, H,   W  ]
```

以 TaxiBJ 为例：

```text
Z_f:      [B, D, T, 32, 32]
Z_m:      [B, D, T, 16, 16]
Z_c:      [B, D, T, 8, 8]
Z_shared: [B, D, T, 32, 32]
```

其中：

- `Z_f`：fine 尺度路由专家输出；
- `Z_m`：mid 尺度路由专家输出；
- `Z_c`：coarse 尺度路由专家输出；
- `Z_shared`：跨尺度共享专家输出。

## 3.2 可选输入：mask / reliability

如果前面构造多尺度数据时保留了可靠性图，可以额外输入：

```text
M_f: [B, 1, T, H,   W  ]
M_m: [B, 1, T, H/2, W/2]
M_c: [B, 1, T, H/4, W/4]

R_m: [B, 1, T, H/2, W/2]
R_c: [B, 1, T, H/4, W/4]
```

其中：

- `M_s` 表示该尺度是否有观测；
- `R_s` 表示该尺度聚合可靠性，例如 coarse cell 是由多少个 observed fine/mid cells 聚合来的。

第一版可以不接入 reliability，后续增强版本建议接入。

## 3.3 输出

输出最终主分支高分辨率特征：

```text
H_main: [B, D, T, H, W]
```

以 TaxiBJ 为例：

```text
H_main: [B, D, T, 32, 32]
```

然后进入预测头：

```text
x_hat_main = PredictionHead(H_main)
x_hat_main: [B, C, T, H, W]
```

---

# 4. Learnable Scale Alignment

## 4.1 为什么不用普通插值

普通插值：

```text
F.interpolate(Z_c, size=(T, 16, 16))
```

只能完成尺寸变化，不能学习“粗尺度特征如何映射到更细尺度”。

因此建议使用可学习对齐：

```text
interpolate + Conv3D refinement
```

## 4.2 LearnableUpsample 模块

设计：

```text
Input:
Z_low: [B, D, T, H_low, W_low]

Step 1:
Trilinear / Nearest Upsample

Step 2:
Conv3D(D -> D, kernel_size=3, padding=1)
Norm
GELU
ResBlock3D(D)

Output:
Z_high: [B, D, T, H_high, W_high]
```

例如：

```text
Z_c:    [B, D, T, 8, 8]
Z_c_16: [B, D, T, 16, 16]
```

以及：

```text
Z_mc:    [B, D, T, 16, 16]
Z_mc_32: [B, D, T, 32, 32]
```

---

# 5. Gated Fusion

## 5.1 为什么需要门控融合

如果只是：

```text
Concat(A, B) -> Conv
```

模型可以融合，但不够显式。

门控融合可以让模型在每个位置动态判断：

```text
当前位置应该更依赖哪个尺度？
```

例如：

- 随机小缺失：更依赖 fine；
- 空间块缺失：更依赖 mid/coarse；
- 大范围缺失：更依赖 shared cross-scale；
- 低可靠 coarse：降低 coarse 权重。

## 5.2 Stage 1：coarse-to-mid Gated Fusion

输入：

```text
Z_m:    [B, D, T, 16, 16]
Z_c_16: [B, D, T, 16, 16]
```

拼接：

```text
F_16 = Concat(Z_m, Z_c_16)
F_16: [B, 2D, T, 16, 16]
```

生成两个尺度权重：

```text
A_16 = Softmax(Conv3D(F_16), dim=scale)
A_16: [B, 2, T, 16, 16]
```

其中：

```text
A_m: [B, 1, T, 16, 16]
A_c: [B, 1, T, 16, 16]
```

融合：

```text
Z_mc = A_m * Z_m + A_c * Z_c_16
```

输出：

```text
Z_mc: [B, D, T, 16, 16]
```

再经过一个 refinement block：

```text
Z_mc = ResBlock3D(Z_mc)
```

## 5.3 Stage 2：mid-to-fine Gated Fusion

输入：

```text
Z_f:      [B, D, T, 32, 32]
Z_mc_32:  [B, D, T, 32, 32]
Z_shared: [B, D, T, 32, 32]
```

拼接：

```text
F_32 = Concat(Z_f, Z_mc_32, Z_shared)
F_32: [B, 3D, T, 32, 32]
```

生成三个分支权重：

```text
A_32 = Softmax(Conv3D(F_32), dim=scale)
A_32: [B, 3, T, 32, 32]
```

其中：

```text
A_f:      [B, 1, T, 32, 32]
A_mc:     [B, 1, T, 32, 32]
A_shared: [B, 1, T, 32, 32]
```

融合：

```text
H_main = A_f * Z_f + A_mc * Z_mc_32 + A_shared * Z_shared
```

输出：

```text
H_main: [B, D, T, 32, 32]
```

再经过一个 refinement block：

```text
H_main = ResBlock3D(H_main)
```

---

# 6. 完整数据流

以 TaxiBJ、`D=64`、`T=12` 为例：

| 阶段 | 形状 |
|---|---|
| `Z_f` | `[B, 64, 12, 32, 32]` |
| `Z_m` | `[B, 64, 12, 16, 16]` |
| `Z_c` | `[B, 64, 12, 8, 8]` |
| `Z_shared` | `[B, 64, 12, 32, 32]` |
| `Z_c_16 = LearnableUpsample(Z_c)` | `[B, 64, 12, 16, 16]` |
| `F_16 = Concat(Z_m, Z_c_16)` | `[B, 128, 12, 16, 16]` |
| `A_16 = Gate_16(F_16)` | `[B, 2, 12, 16, 16]` |
| `Z_mc = A_m*Z_m + A_c*Z_c_16` | `[B, 64, 12, 16, 16]` |
| `Z_mc_32 = LearnableUpsample(Z_mc)` | `[B, 64, 12, 32, 32]` |
| `F_32 = Concat(Z_f, Z_mc_32, Z_shared)` | `[B, 192, 12, 32, 32]` |
| `A_32 = Gate_32(F_32)` | `[B, 3, 12, 32, 32]` |
| `H_main = A_f*Z_f + A_mc*Z_mc_32 + A_shared*Z_shared` | `[B, 64, 12, 32, 32]` |
| `x_hat_main = PredictionHead(H_main)` | `[B, 2, 12, 32, 32]` |

---

# 7. 推荐 PyTorch 伪代码

## 7.1 ResBlock3D

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
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = F.gelu(x + identity)
        return x
```

## 7.2 LearnableUpsample3D

```python
class LearnableUpsample3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            ResBlock3D(dim)
        )

    def forward(self, x, target_size):
        """
        x: [B, D, T, H_low, W_low]
        target_size: (T, H_high, W_high)
        """
        x = F.interpolate(
            x,
            size=target_size,
            mode="trilinear",
            align_corners=False
        )
        x = self.refine(x)
        return x
```

## 7.3 GatedFusion2

```python
class GatedFusion2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Conv3d(dim * 2, 2, kernel_size=1)
        self.refine = ResBlock3D(dim)

    def forward(self, x1, x2):
        """
        x1: [B, D, T, H, W]
        x2: [B, D, T, H, W]
        """
        gate_logits = self.gate(torch.cat([x1, x2], dim=1))
        gate = torch.softmax(gate_logits, dim=1)

        a1 = gate[:, 0:1]
        a2 = gate[:, 1:2]

        out = a1 * x1 + a2 * x2
        out = self.refine(out)

        return out, gate
```

## 7.4 GatedFusion3

```python
class GatedFusion3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Conv3d(dim * 3, 3, kernel_size=1)
        self.refine = ResBlock3D(dim)

    def forward(self, x1, x2, x3):
        """
        x1: [B, D, T, H, W]
        x2: [B, D, T, H, W]
        x3: [B, D, T, H, W]
        """
        gate_logits = self.gate(torch.cat([x1, x2, x3], dim=1))
        gate = torch.softmax(gate_logits, dim=1)

        a1 = gate[:, 0:1]
        a2 = gate[:, 1:2]
        a3 = gate[:, 2:3]

        out = a1 * x1 + a2 * x2 + a3 * x3
        out = self.refine(out)

        return out, gate
```

## 7.5 ProgressiveScaleGatedFusion

```python
class ProgressiveScaleGatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up_c_to_m = LearnableUpsample3D(dim)
        self.fuse_m_c = GatedFusion2(dim)

        self.up_mc_to_f = LearnableUpsample3D(dim)
        self.fuse_f_mc_shared = GatedFusion3(dim)

    def forward(self, z_f, z_m, z_c, z_shared):
        """
        z_f:      [B, D, T, H, W]
        z_m:      [B, D, T, H/2, W/2]
        z_c:      [B, D, T, H/4, W/4]
        z_shared: [B, D, T, H, W]
        """

        B, D, T, H, W = z_f.shape
        _, _, _, Hm, Wm = z_m.shape

        # Stage 1: coarse -> mid
        z_c_to_m = self.up_c_to_m(
            z_c,
            target_size=(T, Hm, Wm)
        )

        z_mc, gate_16 = self.fuse_m_c(
            z_m,
            z_c_to_m
        )

        # Stage 2: mid -> fine
        z_mc_to_f = self.up_mc_to_f(
            z_mc,
            target_size=(T, H, W)
        )

        h_main, gate_32 = self.fuse_f_mc_shared(
            z_f,
            z_mc_to_f,
            z_shared
        )

        return {
            "h_main": h_main,
            "z_mc": z_mc,
            "z_mc_to_f": z_mc_to_f,
            "gate_16": gate_16,
            "gate_32": gate_32
        }
```

---

# 8. 如何替换到主分支中

原来的简单融合代码大概是：

```python
z_m_up = F.interpolate(z_m, size=z_f.shape[-3:], mode="trilinear", align_corners=False)
z_c_up = F.interpolate(z_c, size=z_f.shape[-3:], mode="trilinear", align_corners=False)

fusion_input = torch.cat([z_f, z_m_up, z_c_up, z_shared], dim=1)
h_main = self.fusion(fusion_input)
```

替换成：

```python
fusion_outputs = self.progressive_fusion(
    z_f=z_f,
    z_m=z_m,
    z_c=z_c,
    z_shared=z_shared
)

h_main = fusion_outputs["h_main"]
```

然后：

```python
x_hat_main = self.pred_head(h_main)
```

并且可以额外保存：

```python
gate_16 = fusion_outputs["gate_16"]
gate_32 = fusion_outputs["gate_32"]
```

用于可视化分析：

```text
gate_16: [B, 2, T, 16, 16]
gate_32: [B, 3, T, 32, 32]
```

---

# 9. 可解释性分析

使用该模块后，你可以分析不同缺失模式下模型对不同尺度的依赖。

## 9.1 gate_16

```text
gate_16[:, 0]：mid 特征权重
gate_16[:, 1]：coarse-to-mid 特征权重
```

可以观察：

- 高缺失率时，coarse 权重是否升高；
- 空间块缺失时，coarse 是否更重要；
- 随机低缺失时，mid 是否占主导。

## 9.2 gate_32

```text
gate_32[:, 0]：fine 特征权重
gate_32[:, 1]：mid-coarse 融合特征权重
gate_32[:, 2]：shared cross-scale 特征权重
```

可以观察：

- 小范围随机缺失：fine 权重是否更高；
- 大范围空间缺失：mid-coarse 权重是否更高；
- 复杂缺失模式：shared 权重是否更高。

这些可视化结果可以作为论文中解释模型有效性的补充。

---

# 10. 消融实验建议

建议后续把这个模块作为一个单独消融点。

| 方法 | 说明 |
|---|---|
| Simple Upsample + Concat | 原始简单版本 |
| Learnable Upsample + Concat | 上采样后加 Conv refine |
| Progressive Fusion | 8→16→32 逐级融合，但不用 gate |
| Progressive Gated Fusion | 逐级融合 + gate，最终主方案 |

如果最终效果满足：

```text
Progressive Gated Fusion > Progressive Fusion > Learnable Upsample + Concat > Simple Upsample + Concat
```

就能证明你的尺度融合模块是有效的。

---

# 11. 建议放进主文档的简短描述

可以在正式设计文档中这样写：

> 为避免简单上采样拼接导致粗尺度信息过度平滑，本文设计 Progressive Scale Alignment and Gated Fusion 模块对多尺度路由专家特征进行逐级融合。该模块首先将 coarse 特征可学习上采样到 mid 尺度，并通过门控机制融合 coarse 与 mid 特征，得到区域级中粗融合表示；随后再将该表示上采样到 fine 尺度，与 fine 尺度路由特征和跨尺度共享专家特征进行门控融合，得到最终高分辨率补全特征。该设计显式建模 coarse-to-mid-to-fine 的层级关系，并允许模型在不同缺失模式下自适应选择不同尺度的信息来源。

---

# 12. 总结

最终推荐替换模块是：

```text
Progressive Scale Alignment and Gated Fusion
```

核心流程：

```text
Z_c 8×8
  → LearnableUpsample 8→16
  → GatedFuse with Z_m
  → Z_mc 16×16

Z_mc 16×16
  → LearnableUpsample 16→32
  → GatedFuse with Z_f and Z_shared
  → H_main 32×32
```

这个模块相比简单上采样拼接有三个优势：

1. **逐级融合**：符合 coarse-to-mid-to-fine 的多尺度层级关系；
2. **可学习对齐**：不是简单插值，而是上采样后再经过卷积细化；
3. **动态门控**：模型可以根据缺失模式自适应选择不同尺度信息。
