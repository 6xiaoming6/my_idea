# v20-single：基于 V14 的缺失几何匹配自验证专家路由 MoE 完整设计修改方案

> **版本号：** `v20-single`  
> **推荐模型名称：** GMSV-MoE  
> **英文全称：** **Geometry-Matched Self-Validated Mixture-of-Experts**  
> **中文名称：** **缺失几何匹配的自验证混合专家模型**  
> **直接基础版本：** `v14-single`  
> **核心创新：** **Probe-before-Impute / 先测后路由**  
> **训练方式：** 从随机初始化开始、单阶段、端到端训练  
> **不使用：** Teacher、知识蒸馏、Teacher Anchor、外部预训练权重、多阶段训练  
> **设计日期：** 2026-09-02

---

# 0. 文档目的

本文档用于指导本地 Agent **直接从 `v14-single` 分支创建 `v20-single` 并完成代码实现**。

V20 不再继续给 V14 叠加普通残差模块，也不重新设计一套复杂的 Hierarchical Router，而是把创新集中到一个非常直观的问题上：

> **传统 MoE 的 Router 是根据输入特征“预测”哪个专家适合当前样本；V20 利用当前样本仍然可见的观测数据构造一组与真实缺失几何相似的伪缺失 Probe，让所有专家先在这些有答案的位置进行一次“现场考试”，通过当前样本上的实际重建误差测量专家能力，再用测得的能力校准传统 Router，最后选择 Top-K 专家处理真正缺失区域。**

因此，V20 的核心区别不是：

```text
多尺度 + MoE
```

而是：

```text
Learned Routing Prior
+
Current-Sample Measured Expert Competence
=
Self-Validated Top-K Routing
```

---

# 1. 当前 V14 代码基础

本方案严格基于仓库 `6xiaoming6/my_idea` 的 `v14-single` 分支。

重点源码：

```text
src/stmoe_imputer/models/main_branch.py
src/stmoe_imputer/models/router.py
src/stmoe_imputer/models/experts.py

src/stmoe_imputer/models/v_single/
├── v14_safe_c2f_moe.py
├── safe_c2f_refiner.py
├── safety_controller.py
└── difficulty_condition.py

src/stmoe_imputer/losses.py
src/stmoe_imputer/engine.py
src/stmoe_imputer/data/transforms.py
src/stmoe_imputer/models/registry.py

configs/v14-single/
├── taxibj.json
├── bikenyc.json
└── chap.json
```

当前 V14 Main Backbone 的专家路径：

```text
Fine / Mid / Coarse observed input
        ↓
ScaleTokenEncoder
        ↓
QualityRouter
        ↓
softmax probability [B, 4]
        ↓
Top-K = 2
        ↓
4 个共享 STExpert 中选择 2 个
        ↓
z_f / z_m / z_c
        ↓
ProgressiveRouteFusion
        ↓
Shared + Routed residual fusion
        ↓
x_base
        ↓
V14 Safe C2F + CorrectionAdapter
        ↓
x_final
```

当前 `QualityRouter` 输入：

```text
pooled scale feature
+
observation statistics q
+
scale embedding
```

并直接：

```text
logits → softmax → Top-K
```

当前 `TopKRoutedExpertPool` 的实现语义上是 Top-K，但内部会先计算全部专家：

```python
expert_outputs = torch.stack(
    [expert(h) for expert in self.experts],
    dim=1,
)
```

然后再根据 Top-K gate 混合。

这使 V20 非常适合做“全部专家现场考试”。

---

# 2. V20 的一句话创新

论文、答辩、导师讨论时统一表述：

> **传统 MoE 根据 Router 预测的概率选择 Top-K 专家，而 V20 在当前样本的已知观测区域构造与真实缺失形态匹配的伪缺失 Probe，让每个专家先完成一次可验证的重建任务，通过实际 Probe 误差测量当前样本上的专家能力，并将该实测能力与 learned routing prior 融合后进行 Top-K 路由。**

口语化：

> **别人是 Router 猜谁靠谱；V20 是让专家先做一道当前样本上的模拟题，按实际成绩选人。**

---

# 3. V20 第一版必须保留的 V14 模块

V20 第一版必须原样保留：

```text
V14 MultiScaleMoEBackbone 的编码器结构
4 个共享 STExpert
top_k = 2
Shared branch
Routed branch
ProgressiveRouteFusion
ReliabilityAwareScaleGate
SharedRoutedResidualFusion
V14 SafeCoarseToFineRefiner
V14 CorrectionAdapter
V14 SafetyController
V14 DifficultyConditionEncoder
V14 所有已有损失
V14 数据尺度设置
V14 训练 budget
V14 checkpoint 逻辑
```

V20 只修改：

```text
专家选择依据
```

即：

```text
V14:
Router probability
→ Top-K

V20:
Router probability
+
Probe measured competence
→ calibrated gate
→ Top-K
```

这是本版本的“单核心创新”原则。

---

# 4. V20 整体架构

```text
                       原始观测输入
                            │
              ┌─────────────┴─────────────┐
              │                           │
        Main Original Path          Probe Exam Path
              │                           │
        原始 Scale Input            Geometry-Matched
              │                      Probe Constructor
              │                           │
        ScaleTokenEncoder            临时隐藏一部分
              │                       已知观测值
              │                           │
        QualityRouter prior          Probe Scale Input
              │                           │
              │                     ScaleTokenEncoder
              │                           │
              │                      ALL 4 Experts
              │                           │
              │                    Shared Probe Decoder
              │                           │
              │                    Expert Probe Error
              │                           │
              │                    Competence Score
              │                           │
              └─────────────┬─────────────┘
                            │
                   Self-Validated Routing
                            │
              prior + measured competence
                            │
                       calibrated gate
                            │
                          Top-K
                            │
                 原 V14 Main Backbone
                            │
                         x_base
                            │
                 原 V14 Safe C2F
                            │
                         x_final
```

核心原则：

```text
Probe 分支只负责“测专家能力”。

Main 分支才负责“正式补全”。

Probe 不替代 Main。
Probe 不产生最终 prediction。
Probe 不读取真实缺失位置 ground truth。
```

---

# 5. 数据定义

对尺度：

```text
s ∈ {fine, mid, coarse}
```

原始输入：

```text
x_s: [B,C,T,Hs,Ws]
m_s: [B,1,T,Hs,Ws]
r_s: [B,1,T,Hs,Ws]
```

Fine：

```text
r_f = m_f
```

Mid / Coarse：

```text
直接复用 r_m / r_c
```

V20 构造：

```text
p_s: [B,1,T,Hs,Ws]
```

其中：

```text
p_s <= m_s
```

即 Probe 只能来自原本已知位置。

考试输入：

```text
m_s_probe = m_s * (1 - p_s)
x_s_probe = x_s * m_s_probe
```

考试答案：

```text
x_s[p_s == 1]
```

这些值原本就是观测值，因此训练、验证、测试时都合法可见。

---

# 6. 核心创新一：Missing-Geometry-Matched Probe

不能简单随机拿几个已知位置考试。

原因：

```text
真实缺失：
可能是连续块、低局部观测率、离已知区域较远。

随机 Probe：
可能只是零散简单点。

这种考试不能代表真实缺失难度。
```

V20 默认使用：

> **Geometry-Matched Probe Constructor**

逻辑：

```text
先分析真正缺失区域的局部观测几何
→
再从已知区域寻找局部条件最相似的位置
→
把这些位置临时隐藏作为 Probe
```

---

# 7. Geometry Descriptor

对每个尺度的 `mask` 和 `reliability` 构造位置级 descriptor。

第一版固定 4 个维度：

```text
D1 = 3×3 spatial neighbor availability
D2 = 7×7 spatial neighbor availability
D3 = 3-step temporal neighbor availability
D4 = 5×5 local aggregation reliability
```

尽量排除位置自身，使 observed candidate 可以与真实 missing 位置公平比较。

位置 descriptor：

```text
D_s(t,h,w) = [D1,D2,D3,D4]
```

真实缺失/低可靠区域的权重：

Fine：

```text
w_target = 1 - m_f
```

Mid / Coarse：

```text
w_target = 1 - r_s
```

这一点必须这样做。

因为 random missing 经过 masked pooling 后：

```text
m_m / m_c
```

往往仍然等于 1，但：

```text
r_m / r_c < 1
```

仍然能反映当前 pooled cell 的 Fine 观测不完整。

Target geometry：

```text
D_target_s =
sum(w_target * D_s)
/
sum(w_target)
```

shape：

```text
[B,4]
```

---

# 8. Probe Candidate Selection

候选：

```text
candidate = m_s == 1
```

候选距离：

```text
distance =
Σ_j weight_j *
|D_candidate_j - D_target_j|
```

默认权重：

```text
spatial3:    1.0
spatial7:    1.0
temporal3:   0.5
reliability: 1.0
```

选 distance 最小的一部分 observed positions。

默认：

```text
probe_ratio = 0.08
probe_min_count = 8
probe_max_count = 128
probe_min_remaining = 16
```

每 sample / scale：

```text
n_probe =
clamp(
    round(observed_count * probe_ratio),
    probe_min_count,
    probe_max_count
)
```

且：

```text
observed_count - n_probe
>= probe_min_remaining
```

否则：

```text
probe_valid = false
final_gate = prior_gate
```

---

# 9. 新文件：`v20_probe_mask.py`

新增：

```text
src/stmoe_imputer/models/v_single/v20_probe_mask.py
```

建议：

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


class GeometryMatchedProbeBuilder:
    def __init__(
        self,
        probe_ratio: float = 0.08,
        min_count: int = 8,
        max_count: int = 128,
        min_remaining: int = 16,
        spatial_kernel_small: int = 3,
        spatial_kernel_large: int = 7,
        temporal_kernel: int = 3,
        reliability_kernel: int = 5,
    ) -> None:
        self.probe_ratio = float(probe_ratio)
        self.min_count = int(min_count)
        self.max_count = int(max_count)
        self.min_remaining = int(min_remaining)
        self.spatial_kernel_small = spatial_kernel_small
        self.spatial_kernel_large = spatial_kernel_large
        self.temporal_kernel = temporal_kernel
        self.reliability_kernel = reliability_kernel

    @staticmethod
    def _spatial_neighbor_mean(
        value: torch.Tensor,
        kernel: int,
        exclude_center: bool = True,
    ) -> torch.Tensor:
        b, _, t, h, w = value.shape

        value_2d = (
            value
            .permute(0, 2, 1, 3, 4)
            .reshape(b * t, 1, h, w)
        )

        total = F.avg_pool2d(
            value_2d,
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        ) * float(kernel * kernel)

        if exclude_center:
            total = total - value_2d
            denom = float(kernel * kernel - 1)
        else:
            denom = float(kernel * kernel)

        result = total / max(denom, 1.0)

        return (
            result
            .reshape(b, t, 1, h, w)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

    @staticmethod
    def _temporal_neighbor_mean(
        value: torch.Tensor,
        kernel: int,
    ) -> torch.Tensor:
        b, _, t, h, w = value.shape

        v = (
            value
            .permute(0, 3, 4, 1, 2)
            .reshape(b * h * w, 1, t)
        )

        total = F.avg_pool1d(
            v,
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        ) * float(kernel)

        total = total - v
        result = total / float(max(kernel - 1, 1))

        return (
            result
            .reshape(b, h, w, 1, t)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )

    def descriptors(
        self,
        mask: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        d_small = self._spatial_neighbor_mean(
            mask.float(),
            self.spatial_kernel_small,
        )
        d_large = self._spatial_neighbor_mean(
            mask.float(),
            self.spatial_kernel_large,
        )
        d_temporal = self._temporal_neighbor_mean(
            mask.float(),
            self.temporal_kernel,
        )
        d_reliability = self._spatial_neighbor_mean(
            reliability.float(),
            self.reliability_kernel,
        )

        return torch.cat(
            (
                d_small,
                d_large,
                d_temporal,
                d_reliability,
            ),
            dim=1,
        )

    def build(
        self,
        mask: torch.Tensor,
        reliability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ...
```

`build()` 必须按以下逻辑实现：

```text
1. descriptors = descriptors(mask,reliability)
2. target_weight = 1-reliability
3. 若 target_weight.sum == 0：
      fallback = 1-mask
4. 若仍为 0：
      valid=false
5. weighted mean 得到 target descriptor
6. candidate = mask==1
7. 对所有 candidate 算 L1 geometry distance
8. 计算 n_probe
9. 选距离最小的 n_probe
10. scatter 成 probe_mask
11. 返回 valid/match_distance/realized_ratio
```

第一版 Probe selection 必须确定性，不引入随机数。


---

# 10. 核心创新二：Expert On-Site Exam

选出 Probe 后，对每个 active scale 执行一次“考试前向”。

尺度 `s`：

```text
x_exam = x_s * (1 - p_s)
m_exam = m_s * (1 - p_s)
```

然后：

```text
h_exam_s = ScaleTokenEncoder(x_exam, m_exam)
```

所有 4 个专家都考试：

```text
z_exam_s_e = Expert_e(h_exam_s)
```

得到：

```text
z_exam_s:
[B,E,D,T,Hs,Ws]
```

必须强调：

```text
Probe 必须先从输入删掉，
再重新编码，
再运行专家。
```

禁止：

```text
完整输入跑完专家
→
再在输出上取 Probe 点
```

因为那样专家已经提前看到答案。

---

# 11. 共享 Probe Decoder

STExpert 输出 latent feature，不能直接和数值真值比较。

新增：

```text
SharedProbeDecoder
```

要求：

```text
所有专家共享；
所有尺度共享。
```

结构：

```text
Conv3d(D,D/2,k3,p1)
→ GELU
→ Conv3d(D/2,C,k1)
```

建议文件：

```text
src/stmoe_imputer/models/v_single/v20_probe_routing.py
```

核心代码：

```python
from __future__ import annotations

import torch
from torch import nn


class SharedProbeDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        c_out: int,
    ) -> None:
        super().__init__()

        hidden = max(16, dim // 2)

        self.in_proj = nn.Conv3d(
            dim,
            hidden,
            kernel_size=3,
            padding=1,
        )

        self.act = nn.GELU()

        self.out_proj = nn.Conv3d(
            hidden,
            c_out,
            kernel_size=1,
        )

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        expert_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        input:
            [B,E,D,T,H,W]

        output:
            [B,E,C,T,H,W]
        """
        b, e, d, t, h, w = expert_features.shape

        value = expert_features.reshape(
            b * e,
            d,
            t,
            h,
            w,
        )

        pred = self.out_proj(
            self.act(
                self.in_proj(value)
            )
        )

        return pred.reshape(
            b,
            e,
            pred.shape[1],
            t,
            h,
            w,
        )
```

---

# 12. 为什么 Probe Decoder 必须零初始化

最后一层零初始化有非常重要的训练稳定作用。

训练初期：

```text
所有 expert 的 Probe prediction 基本相同
```

因此：

```text
所有 expert Probe error 基本相同
→ competence ≈ uniform
→ confidence ≈ 0
→ eta ≈ 0
→ V20 final gate ≈ V14 prior gate
```

即：

> **V20 初始自动退化为 V14 路由，不需要任何分阶段训练。**

随着 Probe Decoder 学会从 expert latent 中读出重建能力：

```text
expert error 开始拉开
→ Probe confidence 上升
→ 现场考试逐渐参与路由
```

这是 Full V20 的关键稳定机制。

---

# 13. Expert Pool 的安全重构

修改：

```text
src/stmoe_imputer/models/experts.py
```

当前 `TopKRoutedExpertPool.forward()` 已先运行全部 Experts。

建议重构成：

```python
def forward_all(
    self,
    h: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [
            expert(h)
            for expert in self.experts
        ],
        dim=1,
    )
```

再把现有 Top-K 混合逻辑抽成：

```python
def mix_from_outputs(
    self,
    expert_outputs: torch.Tensor,
    gate: torch.Tensor,
    routing_mode: str = "topk",
):
    ...
```

最终：

```python
def forward(
    self,
    h,
    gate,
    routing_mode="topk",
):
    expert_outputs = self.forward_all(h)

    return self.mix_from_outputs(
        expert_outputs,
        gate,
        routing_mode,
    )
```

必须保证 V14 数值行为完全不变。

Agent 必须增加回归测试：

```text
修改前
vs
修改后
```

比较：

```text
z
top_indices
top_weights
selected_mask
```

要求：

```text
max abs diff <= 1e-7
```

---

# 14. Probe Path 的梯度隔离

Full V20 中：

```text
Probe 是“测量专家能力”，
不能让专家为了考试成绩专门刷分。
```

因此考试路径建议：

```python
with torch.no_grad():
    h_exam = embed(
        x_exam,
        m_exam,
    )

    expert_features = (
        expert_pool.forward_all(
            h_exam
        )
    )

expert_features = expert_features.detach()

probe_prediction = self.probe_decoder(
    expert_features
)
```

这样：

```text
Probe Loss
→ 只训练 Probe Decoder

Main Loss
→ 正常训练 Experts / Router / V14 主模型
```

同时：

```text
competence / confidence / eta
```

进入正式路由前也必须：

```text
detach
```

防止主任务通过“操纵考试分数”走捷径。

---

# 15. Probe Reconstruction Loss

Probe Decoder 训练目标来自原本已知值。

每尺度：

```text
L_probe_s
=
1/E
Σ_e
SmoothL1(
    pred_s_e,
    x_s
    |
    p_s
)
```

只在：

```text
p_s == 1
```

计算。

总：

```text
L_probe =
mean(valid active scale losses)
```

推荐：

```text
lambda_v20_probe = 0.05
```

由于 expert feature 已 detach，这项损失只影响 Probe Decoder，不会改变 V14 主干优化目标。

---

# 16. Expert Probe Error

训练 Probe Decoder 使用 SmoothL1。

但现场排名统一使用 MAE：

```text
error_s_e
=
MAE(
    probe_pred_s_e,
    x_s
    |
    p_s
)
```

得到：

```text
error_s:
[B,E]
```

为了避免 TaxiBJ / BikeNYC / CHAP 数值尺度不同，现场排名不直接使用 raw MAE。

归一化：

```text
mean_error_s =
mean_e(error_s_e)

normalized_error_s_e
=
error_s_e
/
clamp_min(mean_error_s, eps)
```

因此：

```text
每个 sample / scale
专家平均 normalized error ≈ 1
```

这里关注的是同一现场考试中专家的相对表现。

---

# 17. Competence Score

能力分数：

```text
competence_s
=
softmax(
    -normalized_error_s
    /
    tau_probe
)
```

推荐：

```text
tau_probe = 0.5
```

解释：

```text
考试误差越低
→ competence 越大。
```

---

# 18. Probe Confidence

不能每次考试都强行相信。

例如：

```text
E1=1.01
E2=1.00
E3=0.99
E4=1.00
```

说明专家几乎没有差异。

定义：

```text
best_error =
min_e(normalized_error_s)

probe_confidence =
clamp(
    1 - best_error,
    0,
    1
)
```

由于 normalized error 的专家平均值约为 1：

```text
best ≈ 1
→ confidence≈0

best << 1
→ 有明显赢家
→ confidence↑
```

若：

```text
probe_valid=false
```

强制：

```text
confidence=0
```

---

# 19. V20 最核心路由公式

V14 learned prior：

```text
p_prior_s = QualityRouter(...)
```

现场考试：

```text
p_comp_s = competence_s
```

融合比例：

```text
eta_s =
eta_max
*
probe_confidence_s
```

推荐：

```text
eta_max = 0.85
```

概率几何融合：

```text
log p_final_s
=
(1-eta_s)
*
log(p_prior_s + eps)
+
eta_s
*
log(p_comp_s + eps)
```

最后：

```text
p_final_s =
softmax(
    log p_final_s
)
```

意义：

```text
考试没有区分度：
eta≈0
→ 回退 V14 learned Router

考试有明显赢家：
eta↑
→ current-sample measured competence 主导

eta 最大 0.85：
仍保留 learned prior
```

这比直接删除 Router 更稳健。

---

# 20. 为什么 Full 不是纯 Exam Top-K

纯：

```text
Probe Error 最小的 2 个
→ Top-2
```

虽然最直观，但不建议作为 Full。

原因：

```text
Probe 只是伪缺失；
Probe 与真正 missing 不可能完全等价；
训练早期 Probe Decoder 还没学好；
Probe 数量有限，会有噪声。
```

所以 Full 定义为：

```text
Learned Prior
+
Measured Competence
+
Confidence-Adaptive Fusion
```

正式消融必须保留：

```text
Exam-Only
```

用于证明现场考试本身是否有效。

---

# 21. Main Backbone 最小兼容修改

不要复制完整 `MultiScaleMoEBackbone`。

修改：

```text
src/stmoe_imputer/models/main_branch.py
```

给 `forward()` 增加：

```python
routing_evidence: (
    dict[str, dict[str, torch.Tensor]]
    | None
) = None
```

V14 不传：

```text
routing_evidence=None
```

行为必须完全不变。

当前：

```python
gate_f = self._route(...)
gate_m = self._route(...)
gate_c = self._route(...)
```

改为：

```python
prior_gate_f = self._route(...)
prior_gate_m = self._route(...)
prior_gate_c = self._route(...)

gate_f = self._apply_routing_evidence(
    prior_gate_f,
    None if routing_evidence is None
    else routing_evidence.get("fine"),
)

gate_m = self._apply_routing_evidence(
    prior_gate_m,
    None if routing_evidence is None
    else routing_evidence.get("mid"),
)

gate_c = self._apply_routing_evidence(
    prior_gate_c,
    None if routing_evidence is None
    else routing_evidence.get("coarse"),
)
```

---

# 22. `_apply_routing_evidence()`

在 `MultiScaleMoEBackbone` 中新增：

```python
@staticmethod
def _apply_routing_evidence(
    prior_gate: torch.Tensor,
    evidence: dict | None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if evidence is None:
        return prior_gate

    competence = evidence[
        "competence"
    ].detach()

    eta = evidence[
        "eta"
    ].detach()

    if eta.ndim == 1:
        eta = eta.unsqueeze(-1)

    prior_log = torch.log(
        prior_gate.clamp_min(eps)
    )

    competence_log = torch.log(
        competence.clamp_min(eps)
    )

    mixed_log = (
        (1.0 - eta) * prior_log
        +
        eta * competence_log
    )

    return torch.softmax(
        mixed_log,
        dim=-1,
    )
```

没有新增可学习 Gate。

---

# 23. `prior_gate` 和 `final_gate` 都要保留

V20 Debug / diagnostics 需要：

```text
prior gate
probe competence
final gate
```

Main 实际 Top-K 使用：

```text
final gate
```

现有 MoE balance 也使用：

```text
final gate
```

但论文分析需要比较：

```text
V14 prior 选谁
现场考试认为谁好
V20 最后选谁
```

因此 V20 输出要单独记录 prior gate。

---

# 24. V14 Wrapper 的兼容改动

修改：

```text
src/stmoe_imputer/models/v_single/v14_safe_c2f_moe.py
```

仅给 `forward()` 增加可选：

```python
routing_evidence: dict | None = None
```

然后传给：

```python
self.main_backbone(
    ...,
    routing_evidence=routing_evidence,
)
```

V14 正常训练不传该参数。

必须通过 V14 compatibility test。

---

# 25. V20 顶层模型

新增：

```text
src/stmoe_imputer/models/v_single/
    v20_probe_validated_c2f_moe.py
```

建议：

```python
class V20ProbeValidatedC2FMoE(
    V14SafeC2FMoE
):
    ...
```

即：

```text
完整复用 V14 Safe C2F / Controller / CorrectionAdapter，
V20 只在 Main 路由之前产生 routing evidence。
```

---

# 26. V20 Forward 顺序

严格执行：

```text
Step 1:
Fine/Mid/Coarse 各自构造 Probe。

Step 2:
对 active scales 运行 Expert Exam。

Step 3:
得到 competence/confidence/eta。

Step 4:
调用 V14 forward，
把 routing_evidence 传入 Main Backbone。

Step 5:
Main Backbone 内部：
prior gate + evidence → final gate → Top-K。

Step 6:
后续完全走 V14：
route fusion
shared/routed fusion
x_base
safe C2F
CorrectionAdapter
x_final。
```

推荐框架：

```python
class V20ProbeValidatedC2FMoE(
    V14SafeC2FMoE
):
    def __init__(
        self,
        cfg: dict,
    ) -> None:
        super().__init__(cfg)

        # Build probe modules from cfg["model"]["v20"].
        ...

    def forward(
        self,
        x_f,
        m_f,
        x_m,
        m_m,
        x_c,
        m_c,
        r_m=None,
        r_c=None,
    ):
        if r_m is None:
            r_m = m_m.float()

        if r_c is None:
            r_c = m_c.float()

        probe_result = self.probe_evaluator(
            backbone=self.main_backbone,

            x_f=x_f,
            m_f=m_f,
            r_f=m_f.float(),

            x_m=x_m,
            m_m=m_m,
            r_m=r_m,

            x_c=x_c,
            m_c=m_c,
            r_c=r_c,

            scale_mode=
                self.main_backbone.scale_mode,
        )

        outputs = super().forward(
            x_f=x_f,
            m_f=m_f,
            x_m=x_m,
            m_m=m_m,
            x_c=x_c,
            m_c=m_c,
            r_m=r_m,
            r_c=r_c,
            routing_evidence=
                probe_result[
                    "routing_evidence"
                ],
        )

        outputs[
            "v20_enabled"
        ] = True

        outputs[
            "v20_probe"
        ] = probe_result

        return outputs
```

---

# 27. Scale → Module 映射

Fine：

```text
embed_f
routed_expert_pool
```

Mid：

```text
embed_m
routed_expert_pool_m
```

Coarse：

```text
embed_c
routed_expert_pool_c
```

即使：

```text
share_experts=true
```

也要分别在 Fine/Mid/Coarse 考试。

原因：

```text
同一套共享 Expert 参数，
面对不同尺度 representation 时，
当前样本上的 competence 可能不同。
```

最终：

```text
competence_f [B,E]
competence_m [B,E]
competence_c [B,E]
```

---

# 28. Active Scale

必须尊重 V14：

```text
scale_mode
```

例如：

```text
fine
fine_mid
fine_mid_coarse
```

inactive scale：

```text
不运行 Probe Exam；
routing_evidence=None；
直接 prior routing。
```

绝对不要为了 V20 强制三个尺度全开。

---

# 29. 每个尺度独立考试

Mid / Coarse 的 Probe 不需要重新由 Fine Probe pooling 得到。

直接：

```text
x_m_probe = x_m * (1-p_m)
m_m_probe = m_m * (1-p_m)

x_c_probe = x_c * (1-p_c)
m_c_probe = m_c * (1-p_c)
```

原因：

```text
Probe Path 只负责测该尺度 Expert competence，
不是正式数据路径。

Main Path 仍使用原始、数据预处理产生的
一致 Fine/Mid/Coarse input。
```

这种实现更清晰，也更容易测试。

---

# 30. `ProbeCompetenceEvaluator` 推荐返回值

```python
{
    "probe_loss": scalar,

    "routing_evidence": {
        "fine": {
            "competence": [B,E],
            "eta": [B,1],
        },
        "mid": {...},
        "coarse": {...},
    },

    "fine": {
        "probe_mask": ...,
        "valid": ...,
        "raw_error": ...,
        "normalized_error": ...,
        "competence": ...,
        "confidence": ...,
        "eta": ...,
        "match_distance": ...,
        "realized_ratio": ...,
    },

    "mid": {...},
    "coarse": {...},
}
```

正式日志不要每步保存完整 `probe_mask`，只保存统计量。

---

# 31. No Target Leakage

V20 Forward 允许使用：

```text
已知 observed x_f/x_m/x_c
mask
reliability
```

Probe 答案也是从：

```text
原本 observed 位置
```

产生。

Forward 禁止使用：

```text
真实 missing positions 的 x_f_gt。
```

必须做测试：

```text
固定：
x_obs
mask
r_m/r_c

只修改：
hidden ground truth

比较：
probe mask
probe error
competence
eta
final gate
Top-K
x_hat_main

必须完全不变。
```

hidden ground truth 只能用于：

```text
训练主 Loss
离线 oracle analysis
```

---

# 32. V20 Loss

V14 原 Loss 全部不变。

新增：

```text
L_v20
=
L_v14
+
lambda_v20_probe
*
L_probe
```

推荐：

```text
lambda_v20_probe = 0.05
```

第一版不要加入：

```text
Router-Competence KL
Ranking Loss
Contrastive Loss
Probe Entropy Loss
```

保持核心机制干净。

---

# 33. 为什么不做 Router Alignment Loss

不要第一版加入：

```text
KL(
    competence
    ||
    prior_gate
)
```

因为 V20 需要两种互补信息：

```text
Learned prior:
训练分布上的长期经验。

Probe competence:
当前样本上的现场实测。
```

如果强迫 prior 完全模仿 competence，两者会趋同，反而削弱核心叙事。

---

# 34. Balance Loss

现有：

```text
moe_balance_loss(
    outputs["gates"],
    selected_masks
)
```

V20 的：

```text
outputs["gates"]
```

应保存 actual final calibrated gate。

因此现有 balance loss 自动约束真正的 V20 Top-K，不需要第二套 balance loss。

---

# 35. V20 配置模板

新增：

```text
configs/v20-single/
├── taxibj.json
├── bikenyc.json
├── chap.json
└── smoke.json
```

从各自 V14 JSON 复制，再增加：

```json
{
  "model": {
    "version": "v20-single",
    "architecture": "v20_probe_validated_c2f_moe",

    "v20": {
      "enabled": true,

      "probe_enabled": true,
      "probe_mode": "geometry_matched",

      "probe_ratio": 0.08,
      "probe_min_count": 8,
      "probe_max_count": 128,
      "probe_min_remaining": 16,

      "spatial_kernel_small": 3,
      "spatial_kernel_large": 7,
      "temporal_kernel": 3,
      "reliability_kernel": 5,

      "probe_temperature": 0.5,
      "probe_eta_max": 0.85,
      "probe_eps": 0.000001,

      "probe_decoder_zero_init": true,
      "probe_decoder_dropout": 0.0,

      "detach_probe_features": true,
      "detach_routing_evidence": true,

      "active_scale_only": true,

      "descriptor_weights": {
        "spatial_small": 1.0,
        "spatial_large": 1.0,
        "temporal": 0.5,
        "reliability": 1.0
      }
    }
  },

  "loss": {
    "lambda_v20_probe": 0.05
  }
}
```

同时完整保留：

```text
model.v14
loss.lambda_v14_*
train
```

对应 V14 配置。

---

# 36. 不允许数据集特制 Router

TaxiBJ / BikeNYC / CHAP 必须共用：

```text
Geometry Descriptor
Probe Construction
Competence Formula
Confidence Formula
Prior/Competence Fusion
```

允许保留 V14 本来就存在的数据集差异：

```text
scale_mode
dropout
epochs
batch size
```

禁止：

```text
Taxi exam-only
Bike prior-only
CHAP hybrid
```

否则无法形成统一模型。

---

# 37. Registry

修改：

```text
src/stmoe_imputer/models/v_single/__init__.py
```

导出：

```python
from .v20_probe_validated_c2f_moe import (
    V20ProbeValidatedC2FMoE,
)
```

修改：

```text
src/stmoe_imputer/models/registry.py
```

新增：

```python
"v20_probe_validated_c2f_moe":
    V20ProbeValidatedC2FMoE.from_config,
```

保留 V14 entry，不得覆盖。

---

# 38. `losses.py` 修改

在现有 V14 Loss 完成后追加：

```python
v20_probe = outputs.get(
    "v20_probe"
)

if (
    outputs.get(
        "v20_enabled",
        False,
    )
    and isinstance(
        v20_probe,
        dict,
    )
):
    l_v20_probe = (
        v20_probe.get(
            "probe_loss"
        )
    )

    if not torch.is_tensor(
        l_v20_probe
    ):
        l_v20_probe = (
            _empty_loss_like(
                l_main
            )
        )

    loss = (
        loss
        +
        loss_cfg.get(
            "lambda_v20_probe",
            0.05,
        )
        * l_v20_probe
    )
else:
    l_v20_probe = (
        _empty_loss_like(
            l_main
        )
    )
```

日志：

```text
l_v20_probe
```

必须保存。

---

# 39. Engine

V20 不需要任何阶段切换。

禁止：

```text
freeze stage
warmup model stage
teacher stage
checkpoint preload
第二次训练
```

继续：

```text
一个 optimizer
一个 scheduler
一次完整训练
```

Probe Decoder 自动进入 `model.parameters()`。

第一版与 V14 相同 LR 即可。

---

# 40. 计算成本

V20 在 Main Forward 前额外执行：

```text
active scales
×
ScaleTokenEncoder
+
all 4 experts
+
Probe Decoder
```

Probe Path 不执行：

```text
Shared branch
RouteFusion
BranchFusion
Safe C2F
CorrectionAdapter
SafetyController
```

因此并不是完整模型双倍。

必须实测：

```text
seconds/epoch
peak GPU memory
```

目标：

```text
training time <= V14 × 1.50
peak memory <= V14 × 1.25
```

如果明显超出，再考虑：

```text
Prior Top-M=3
→ 只让 M 个专家考试
→ 再从中选 Top-K=2
```

4 experts 的第一版不要提前做这层优化。


---

# 41. V20 必须新增的诊断

V20 是否成立不能只看 MAE/RMSE。

每个尺度必须记录：

```text
probe_valid_ratio
probe_realized_ratio
probe_match_distance

probe_error_e0
probe_error_e1
probe_error_e2
probe_error_e3

probe_competence_e0
probe_competence_e1
probe_competence_e2
probe_competence_e3

probe_confidence
probe_eta

prior_gate_entropy
competence_entropy
final_gate_entropy

prior_top1
probe_top1
final_top1

prior_probe_top1_agreement
prior_final_topk_overlap
probe_final_topk_overlap
```

继续记录：

```text
expert load / importance
```

用于检查是否出现新的 expert collapse。

---

# 42. 最重要的论文诊断：Probe 是否真的能预测真实 Missing 能力

这是 V20 最关键的科学验证。

正式 Forward：

```text
绝对不能使用 hidden ground truth。
```

但是在测试完成后的**离线分析**中，可以用 hidden ground truth 评估：

```text
“如果每个 Expert 单独去预测真正 missing，
谁实际最好？”
```

得到：

```text
oracle_error_s_e
```

然后比较：

```text
probe_error_s_e
vs
oracle_error_s_e
```

统计：

```text
Spearman rank correlation
Top-1 expert agreement
Top-2 overlap
```

这组实验用于回答：

> **当前样本上的“现场考试成绩”，是否真的能够预测专家在真实缺失区域上的能力？**

如果答案为是，那么 V20 的核心创新就有非常直接的证据。

注意：

```text
oracle_error
只用于 test/offline analysis，
禁止进入 forward routing。
```

---

# 43. Geometry-Matched Probe 必须与 Random Probe 比较

V20 不能只证明：

```text
“遮一些已知值做考试有效”
```

还要证明：

```text
“根据真实缺失几何来选考试题更有效”
```

因此必须比较：

```text
Random Probe
vs
Geometry-Matched Probe
```

指标：

```text
Probe↔Oracle Spearman
Top-1 agreement
Top-2 overlap
Final MAE
Final RMSE
```

理想结论：

```text
Geometry-Matched Probe
具有更高的 expert-rank predictive quality
并带来更好的最终补全结果。
```

---

# 44. 推荐离线分析脚本

新增：

```text
scripts/v20-single/analyze_probe_ranking.py
```

职责：

```text
1. 加载 best.pt。
2. 正常跑 V20 Forward，保存 Probe ranking。
3. 开启 analysis-only all-expert prediction。
4. 用 hidden target 算 oracle ranking。
5. 绝不把 oracle ranking 回写模型。
6. 汇总 Spearman / Top1 / Top2。
7. 分 dataset / pattern / rate 汇总。
```

输出：

```text
outputs/v20-single/.../analysis/
├── probe_oracle_summary.json
├── probe_oracle_per_sample.csv
└── probe_oracle_per_scale.csv
```

---

# 45. 核心单元测试

新增：

```text
tests/
├── test_v20_v14_backward_compatibility.py
├── test_v20_probe_subset_observed.py
├── test_v20_geometry_match.py
├── test_v20_probe_no_leakage.py
├── test_v20_expert_exam.py
├── test_v20_probe_decoder_zero_init.py
├── test_v20_routing_fallback.py
├── test_v20_routing_calibration.py
├── test_v20_inactive_scale.py
├── test_v20_probe_loss_gradient.py
├── test_v20_expert_pool_refactor.py
├── test_v20_checkpoint.py
└── test_v20_three_datasets.py
```

---

# 46. Test：V14 Backward Compatibility

由于 V20 需要修改共享的：

```text
experts.py
main_branch.py
v14_safe_c2f_moe.py
```

所以必须证明：

```text
routing_evidence=None
```

时 V14 没变。

同一参数、同一输入比较：

```text
x_hat_main
x_hat_base
x_hat_ctf
gates
top_indices
top_weights
selected_masks
alpha_final
```

要求：

```text
max abs diff <= 1e-6
```

Expert Pool 重构单独要求：

```text
<=1e-7
```

---

# 47. Test：Probe 只能来自 Observed

必须：

```python
assert torch.all(
    probe_mask <= original_mask
)
```

Probe 点：

```text
原始：
mask=1

考试输入：
mask=0
value=0
```

必须检查：

```python
assert torch.all(
    m_exam[probe_mask.bool()] == 0
)

assert torch.all(
    x_exam.expand_as(x_s)[
        probe_mask.expand_as(x_s).bool()
    ] == 0
)
```

---

# 48. Test：No Target Leakage

构造两份数据：

```text
x_obs 相同
mask 相同
reliability 相同
hidden target 完全不同
```

比较 Forward：

```text
probe_mask
probe_error
competence
confidence
eta
final_gate
top_indices
x_hat_main
```

必须一致。

---

# 49. Test：零初始化自动回退 V14

模型初始化时：

```text
Probe Decoder last layer = 0
```

所以所有 expert prediction 一致。

要求：

```text
competence ≈ uniform
confidence ≈ 0
eta ≈ 0
final_gate ≈ prior_gate
```

建议测试：

```python
assert torch.allclose(
    final_gate,
    prior_gate,
    atol=1e-5,
    rtol=1e-5,
)
```

---

# 50. Test：Probe Loss 不训练 Experts

只保留：

```text
L_probe
```

其他 Loss=0。

Backward 后：

```text
Probe Decoder:
grad != 0

Experts:
grad == 0

ScaleTokenEncoder:
grad == 0

QualityRouter:
grad == 0
```

如果 Expert 有梯度，说明 detach 失败。

---

# 51. Test：Main Loss 正常训练 Router / Expert

关闭 Probe Loss，仅主任务 backward。

必须：

```text
QualityRouter grad != 0
Experts grad != 0
```

说明：

```text
routing evidence detach
没有切断正常 Main training。
```

---

# 52. Test：Fallback

以下情况：

```text
observed candidate 太少
target_weight 无有效位置
probe_valid=false
```

要求：

```text
confidence=0
eta=0
final_gate=prior_gate
```

且：

```text
无 NaN/Inf。
```

---

# 53. Test：Inactive Scale

若：

```text
scale_mode=fine_mid
```

则：

```text
coarse probe 不参与 routing。
```

要求：

```text
coarse evidence=None
或 coarse eta=0
```

不能偷偷改变 V14 coarse behavior。

---

# 54. 三数据集 Shape Test

根据仓库实际数据配置逐一运行：

```text
TaxiBJ
BikeNYC
CHAP
```

必须测试：

```text
Fine Probe
Mid Probe
Coarse Probe
V14 Refiner
Final output
```

全部 shape 正确。

不要在代码中硬编码：

```text
32×32
24×12
```

所有 Probe 操作根据输入 shape 动态处理。

---

# 55. 输出字典

V20 在 V14 outputs 上新增：

```python
outputs["v20_enabled"] = True

outputs["v20_probe"] = {
    "probe_loss": ...,

    "routing_evidence": {
        "fine": ...,
        "mid": ...,
        "coarse": ...,
    },

    "fine": {
        "valid": ...,
        "raw_error": ...,
        "normalized_error": ...,
        "competence": ...,
        "confidence": ...,
        "eta": ...,
        "match_distance": ...,
        "realized_ratio": ...,
    },

    "mid": {...},
    "coarse": {...},
}
```

Debug 模式可以临时带：

```text
probe_mask
```

正式训练默认不要把大 Tensor 写日志。

---

# 56. 推荐模型名和模块名

完整模型：

> **GMSV-MoE**

英文全称：

> **Geometry-Matched Self-Validated Mixture-of-Experts for Spatiotemporal Imputation**

中文：

> **面向时空数据补全的缺失几何匹配自验证混合专家模型**

核心路由：

> **Self-Validated Expert Routing (SVER)**

Probe：

> **Missing-Geometry-Matched Probe (GMP)**

核心组合：

```text
GMP + SVER
```

---

# 57. 论文核心数学形式

传统 Router：

```text
p_prior^(s)
=
softmax(
    R_s(
        h_s,
        q_s
    )
)
```

几何 Probe：

```text
P_s =
GMP(
    m_s,
    r_s
)
```

现场考试：

```text
e_(s,k)
=
MAE(
    D(
        E_k(
            Enc_s(
                x_s ⊙ (m_s-P_s),
                m_s-P_s
            )
        )
    ),
    x_s
    |
    P_s
)
```

归一化：

```text
e_bar_(s,k)
=
e_(s,k)
/
mean_j(
    e_(s,j)
)
```

能力：

```text
p_comp^(s)
=
softmax(
    -e_bar_s / tau
)
```

考试置信度：

```text
c_s
=
clip(
    1-min_k(e_bar_(s,k)),
    0,
    1
)
```

融合：

```text
eta_s =
eta_max * c_s
```

最终：

```text
log p_final^(s)
=
(1-eta_s)
log p_prior^(s)
+
eta_s
log p_comp^(s)
```

再：

```text
TopK(
    p_final^(s)
)
```

---

# 58. 论文三个正式贡献点

## Contribution 1：Self-Validated Expert Routing

传统 MoE：

```text
根据输入表示预测专家适合度。
```

V20：

```text
额外用当前样本的可验证 Probe
直接测量 Expert competence。
```

路由依据从：

```text
predicted suitability
```

升级为：

```text
learned prior
+
measured current-sample competence。
```

这是最核心贡献。

## Contribution 2：Missing-Geometry-Matched Probe

不是 Random Probe。

根据真实 missing 的：

```text
spatial neighborhood support
temporal support
multiscale aggregation reliability
```

在 observed 区域寻找相似考试点。

目标：

```text
让“模拟题”的难度和真实缺失任务尽量一致。
```

## Contribution 3：Confidence-Adaptive Prior/Competence Fusion

如果考试：

```text
分不出专家优劣
```

自动回退：

```text
learned prior。
```

如果考试：

```text
有明显赢家
```

则：

```text
measured competence 主导。
```

避免完全抛弃 learned Router。

---

# 59. 不再把什么当创新

论文不要再写：

```text
“本文提出多尺度 MoE。”
```

因为这不是 V20 最有辨识度的地方。

不要把：

```text
Top-K
Shared Expert
C2F
Residual
```

作为主贡献。

这些属于基础架构。

V20 的主线必须始终围绕：

> **专家选择依据由纯预测式路由升级为“当前样本可验证能力测量 + learned prior”。**

---

# 60. 正式消融设计

V20 消融围绕核心创新，不再铺很多无关模块。

## A0：V14

```text
Learned Router
→ Top-K
```

正式基线。

## A1：Random Probe + Exam-Only

```text
Random observed Probe
+
纯 competence Top-K
```

回答：

```text
现场考试本身是否有效。
```

## A2：Geometry Probe + Exam-Only

```text
Geometry-Matched Probe
+
纯 competence Top-K
```

回答：

```text
Geometry Matching 是否提升考试质量。
```

## A3：Random Probe + Hybrid

```text
Learned Prior
+
Random Probe competence
+
Adaptive Fusion
```

## A4：Geometry Probe + Hybrid

```text
Full V20
```

## A5：Geometry Probe + Prior-Only

构造 Probe，但：

```text
eta=0
```

确认收益确实来自 routing evidence，不是单纯增加 Probe Decoder Loss。

---

# 61. 推荐主消融表

| Variant | Learned Prior | Probe Exam | Geometry Match | Adaptive Fusion |
|---|---|---|---|---|
| V14 | ✓ |  |  |  |
| Exam-R |  | ✓ |  |  |
| Exam-G |  | ✓ | ✓ |  |
| Hybrid-R | ✓ | ✓ |  | ✓ |
| **V20 Full** | ✓ | ✓ | ✓ | ✓ |

要求：

```text
Full V20 在跨数据集平均上最好。
```

---

# 62. Probe Ranking Quality 表

论文单独增加：

| Dataset | Random Probe Spearman | Geometry Probe Spearman | Random Top-2 | Geometry Top-2 |
|---|---:|---:|---:|---:|
| TaxiBJ | | | | |
| BikeNYC | | | | |
| CHAP | | | | |

这张表是 V20 最重要的机制证据之一。

---

# 63. Case Study

推荐可视化一个样本：

```text
Prior Router:

E1 0.41
E2 0.31
E3 0.18
E4 0.10

Probe Error:

E1 1.34
E2 0.61
E3 0.72
E4 1.56

Probe Competence:

E1 0.11
E2 0.44
E3 0.38
E4 0.07

Final Gate:

E1 0.20
E2 0.41
E3 0.31
E4 0.08

V14 Top-2:
E1,E2

V20 Top-2:
E2,E3
```

再对比最终缺失区域误差。

这个例子能够非常直观地展示：

```text
“现场考试纠正了 learned Router 的错误选择。”
```

---

# 64. 实验阶段一：Smoke

点位：

```text
TaxiBJ fixed@0.4
BikeNYC random@0.4
CHAP fixed@0.4
```

每组：

```text
2 epoch
```

检查：

```text
Forward
Backward
Probe valid
Probe loss
No leakage
Router gradient
Expert gradient
No NaN/Inf
Checkpoint
```

---

# 65. 实验阶段二：八点筛选

推荐：

```text
TaxiBJ fixed@0.2
TaxiBJ fixed@0.4
TaxiBJ random@0.4
TaxiBJ random@0.8

BikeNYC fixed@0.6
BikeNYC random@0.4

CHAP fixed@0.4
CHAP random@0.4
```

第一轮：

```text
seed=42
```

比较：

```text
clean V14
V20 Full
```

同时跑：

```text
Probe↔Oracle Ranking Analysis。
```

---

# 66. 八点准入标准

建议同时满足：

```text
1. V20 八点平均 MAE 优于 V14 >=1%。

2. 至少 5/8 点优于 V14。

3. 任意单点退化 <=3%。

4. BikeNYC 两点平均不能退化。

5. Taxi fixed@0.2/@0.4
   不能出现明显大退化。

6. Geometry Probe 的 oracle rank correlation
   高于 Random Probe。

7. probe_valid_rate >=90%。

8. mean eta 不应长期≈0：
   否则 Probe 没有实际参与。

9. eta 也不应全饱和到 eta_max。

10. V14 compatibility 全部通过。
```

---

# 67. 实验阶段三：三随机种子

通过八点后：

```text
seed:
42
2026
3407
```

核心：

```text
Taxi fixed@0.4
Taxi random@0.4
Bike random@0.4
CHAP fixed@0.4
```

共：

```text
4 points × 3 seeds
```

V14/V20 配对。

标准：

```text
平均配对 MAE 改善 >=1%。

至少 3/4 点：
2/3 seeds 获胜。

最差 seed：
单点退化 <=3%。

Probe ranking correlation
跨 seed 趋势一致。
```

---

# 68. 实验阶段四：24 点全量

正式：

```text
3 datasets
×
fixed/random
×
rate 0.2/0.4/0.6/0.8
```

主比较：

```text
V20 vs V14
```

论文 baseline 再按正式实验体系加入。

---

# 69. V20 成为主模型的目标

建议：

```text
MAE：
>=15/24 胜 V14。

RMSE：
>=15/24 胜 V14。

24 点平均相对 MAE：
>=1.5% 改善。

三个数据集八点平均：
均不明显弱于 V14。

至少两个数据集：
平均改善 >=1%。

任一单点退化：
<=3%。

Full：
优于 Random-Probe Hybrid
和 Exam-Only。

Geometry Probe：
ranking quality 明显优于 Random Probe。
```

---

# 70. 参数和资源

新增参数主要来自：

```text
Shared Probe Decoder
```

Probe Builder：

```text
无参数。
```

Prior/Competence fusion：

```text
无参数。
```

所以新增参数应很少。

Agent 完成后必须统计：

```text
V14 total params
V20 total params
新增 params
增长比例
```

目标：

```text
V20 params <= V14 × 1.03
```

---

# 71. 日志

`metrics.jsonl` 增加：

```text
l_v20_probe

v20_fine_probe_confidence
v20_mid_probe_confidence
v20_coarse_probe_confidence

v20_fine_eta
v20_mid_eta
v20_coarse_eta

v20_fine_probe_valid_rate
v20_mid_probe_valid_rate
v20_coarse_probe_valid_rate

v20_fine_prior_probe_agreement
v20_mid_prior_probe_agreement
v20_coarse_prior_probe_agreement

v20_probe_match_distance
```

测试 summary 增加：

```text
probe/oracle ranking statistics
```

---

# 72. 代码文件清单

## 新增

```text
src/stmoe_imputer/models/v_single/
├── v20_probe_mask.py
├── v20_probe_routing.py
└── v20_probe_validated_c2f_moe.py
```

## 修改

```text
src/stmoe_imputer/models/experts.py
src/stmoe_imputer/models/main_branch.py
src/stmoe_imputer/models/v_single/v14_safe_c2f_moe.py
src/stmoe_imputer/models/v_single/__init__.py
src/stmoe_imputer/models/registry.py
src/stmoe_imputer/losses.py
```

## 新增配置

```text
configs/v20-single/
├── taxibj.json
├── bikenyc.json
├── chap.json
├── smoke.json
└── ablations/
```

## 新增脚本

```text
scripts/v20-single/
├── run_smoke.py
├── run_screening.py
├── run_multiseed.py
├── run_full_24.py
├── run_ablation.py
├── analyze_probe_ranking.py
└── summarize_v20.py
```

---

# 73. Git 开发流程

```bash
git switch v14-single
git pull origin v14-single
git status
```

确认：

```text
working tree clean
```

创建：

```bash
git switch -c v20-single
git push -u origin v20-single
```

推荐提交：

```bash
git commit -m "v20-single: refactor expert pool for all-expert outputs"

git commit -m "v20-single: add geometry-matched probe builder"

git commit -m "v20-single: add shared probe decoder and competence evaluator"

git commit -m "v20-single: add external routing evidence support"

git commit -m "v20-single: add self-validated routing wrapper"

git commit -m "v20-single: add probe loss configs and diagnostics"

git commit -m "v20-single: add tests and experiment runners"
```

正式实验：

```text
git_dirty=false
```

---

# 74. Agent 开发顺序

Agent 严格按顺序做。

## Step 1

创建 `v20-single`，先跑现有 V14 tests。

## Step 2

只重构：

```text
TopKRoutedExpertPool
```

加入：

```text
forward_all
mix_from_outputs
```

先证明 V14 数值等价。

## Step 3

实现：

```text
GeometryMatchedProbeBuilder
```

只用 synthetic mask 测。

## Step 4

实现：

```text
SharedProbeDecoder
ProbeCompetenceEvaluator
```

先独立测试。

## Step 5

给：

```text
MultiScaleMoEBackbone
```

加入可选：

```text
routing_evidence
```

确认：

```text
None → V14 完全不变。
```

## Step 6

给：

```text
V14SafeC2FMoE
```

增加 optional passthrough：

```text
routing_evidence=None
```

确认 V14 不变。

## Step 7

实现：

```text
V20ProbeValidatedC2FMoE
```

## Step 8

加入：

```text
L_probe
```

完成 gradient isolation test。

## Step 9

补：

```text
registry
configs
diagnostics
```

## Step 10

跑三数据集 Smoke。

只有全部通过后才能开始正式 screening。

---

# 75. Agent 最终必须生成实现报告

Agent 完成后必须生成：

```text
V20_IMPLEMENTATION_REPORT.md
```

内容：

```text
1. 修改文件清单。

2. 每个文件具体改动。

3. V14 compatibility test。

4. ExpertPool refactor equivalence。

5. Probe subset observed test。

6. No target leakage test。

7. Probe gradient isolation。

8. 三数据集 shape test。

9. 参数量 V14 vs V20。

10. Smoke train/val/test。

11. 一个真实 batch 的：
    prior gate
    probe error
    competence
    eta
    final gate
    Top-K
    示例。

12. 是否存在 TODO。
```

存在未完成 TODO 时不要跑正式实验。

---

# 76. 主要风险与回退

## 风险 1：Probe 与真实 Missing 能力相关性低

看：

```text
Probe↔Oracle Spearman。
```

如果长期：

```text
<0.2
```

优先修改：

```text
GeometryMatchedProbeBuilder
```

不要先：

```text
增加 Expert
增加 Router MLP
增加 Loss
```

## 风险 2：所有专家考试成绩接近

表现：

```text
confidence≈0
eta≈0
```

先看：

```text
Probe Decoder loss 是否下降。
```

然后单因素尝试：

```text
probe_ratio:
0.08 → 0.12
```

不要直接强制 eta。

## 风险 3：Probe 过度主导

表现：

```text
eta≈eta_max
且性能变差。
```

单因素：

```text
eta_max:
0.85 → 0.60
```

## 风险 4：Geometry Probe 只找到 boundary

说明单点 Probe 无法模拟大块中心。

下一版本再考虑：

```text
Patch Probe:
2×2 / 3×3 observed patch
```

V20 第一版不要直接加入。

## 风险 5：计算过慢

若：

```text
time > V14 ×1.5
```

再改：

```text
Prior Top-M=3
→ 3 experts 参加考试
→ Probe Top-K=2
```

第一版不要提前优化。

---

# 77. 为什么 V20 比继续做普通 Router 更有辨识度

继续做：

```text
更复杂 MLP Router
更多缺失统计
更多尺度 Gate
```

很容易仍被归类成：

```text
missing-aware routing
```

V20 的区别是：

```text
传统：
Router 根据特征“预测” Expert competence。

V20：
当前样本给 Expert 出一道有答案的模拟题，
直接“测量” Expert competence。
```

这是一个非常容易被理解、被画出来、被消融验证的创新。

---

# 78. V20 最终结构摘要

```text
V14 Stable Multi-Scale MoE
+
Missing-Geometry-Matched Probe
+
Current-Sample Expert On-Site Exam
+
Measured Expert Competence
+
Confidence-Adaptive Prior/Competence Fusion
+
Top-K
+
V14 Safe C2F
```

核心问题：

```text
“为什么选这两个专家？”
```

V14：

```text
因为 Router 给它们的概率最高。
```

V20：

```text
因为 learned Router 认为它们合适，
并且它们在当前样本、与真实缺失难度匹配的 Probe 上
实际做得好。
```

---

# 79. 论文核心叙事

研究问题：

> **MoE-based spatiotemporal imputation usually relies on a learned router to infer expert suitability from incomplete observations. However, predicted routing confidence does not necessarily represent the actual reconstruction competence of an expert under the current missing pattern.**

方法：

> **V20 constructs geometry-matched pseudo-missing probes from currently observed values and explicitly evaluates each expert on the current sample before routing. The measured expert competence is then confidence-adaptively fused with the learned routing prior for Top-K selection.**

最简一句话：

> **V20 将专家路由从“只靠模型预测谁擅长”，升级为“历史经验 + 当前样本现场实测”。**

---

# 80. 实现后必须回答的三个研究问题

## RQ1

```text
Probe measured competence
是否与真实 missing expert competence 相关？
```

用：

```text
Spearman / Top-K overlap
```

回答。

## RQ2

```text
Geometry-Matched Probe
是否比 Random Probe 更能预测真实 expert ranking？
```

用：

```text
ranking correlation + final MAE
```

回答。

## RQ3

```text
Measured competence
是否真的能纠正 learned Router 的 Top-K？
```

用：

```text
V14
vs
Exam-Only
vs
Random Hybrid
vs
Geometry Hybrid Full
```

回答。

---

# 81. 最终硬性原则

开发过程中始终遵守：

```text
V20 基于 V14。

只把主要创新放在专家路由依据。

V14 Safe C2F 不动。

不使用 Teacher。

不使用蒸馏。

不使用多阶段训练。

Forward 不读取 hidden ground truth。

Probe 只来自 observed data。

Probe 必须先隐藏再考试。

Probe evidence 不允许被 Main Loss 操纵。

Probe Decoder 全专家/全尺度共享。

Probe 无效时必须回退 V14 prior。

Full =
Geometry-Matched
+
Self-Validated
+
Prior/Competence Hybrid Routing。
```

---

# 82. 最终结论

`v20-single` 正式定义为：

> **GMSV-MoE：Geometry-Matched Self-Validated Mixture-of-Experts**

V20 的核心创新不是“又做了一个 MoE Router”，而是：

> **在正式补全之前，从当前样本的已知数据中构造与真实缺失几何相匹配的伪缺失考试，让所有专家先接受可验证的现场测试，用实际 reconstruction error 得到 current-sample expert competence，再与 learned routing prior 融合选择 Top-K 专家。**

相较 V14：

```text
Main Backbone 基本保留；
Expert 保留；
Safe C2F 保留；
训练协议保留；
主要改变的是 Top-K 的证据来源。
```

论文辨识度：

```text
Traditional MoE:
Predict competence.

V20:
Measure competence before routing.
```

这就是 V20 必须坚持的核心创新。
