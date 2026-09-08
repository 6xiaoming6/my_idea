# 研究想法讨论稿（精简版）

> 研究主线：**MoE + 多尺度时空数据补全**

本文只保留两个核心 Idea：

1. **Missingness-Induced Scale Distortion：缺失诱导尺度失真**
2. **Mixture of Coarsening Experts：MoE 自适应不规则粗化**

---

# Idea 1：Missingness-Induced Scale Distortion

## 1. 核心问题

现有多尺度补全通常直接从细粒度数据构造粗粒度数据，例如：

```text
Fine
 ↓ Average / Sum / Pooling
Mid
 ↓
Coarse
```

但当 Fine 数据本身已经存在缺失时，直接做尺度变换可能导致粗粒度表示失真。

这里的问题不是“平均或求和本身错误”，而是：

> **固定聚合区域 + 不完整观测 + 固定聚合权重，可能无法得到真正可靠的粗尺度表示。**

例如：

```text
10  ?
10  ?
```

和

```text
10  10
?   ?
```

两者都可能得到：

```text
mean = 10
observed ratio = 0.5
```

但第一种观测集中在左侧，第二种集中在上侧，它们对后续空间恢复的支持信息并不相同。

再例如：

```text
2   ?
?  18
```

也可能得到：

```text
mean = 10
```

但其内部方差明显更大。

因此，普通的：

```text
mean + observed ratio
```

可能丢失：

```text
局部方差
观测位置
观测方向
局部缺失结构
观测可靠性
```

这就是我们希望研究的：

> **Missingness-Induced Scale Distortion：缺失会参与尺度构造过程，使由不完整 Fine 数据得到的 Mid / Coarse 表示偏离完整数据下真实的尺度表示。**

---

## 2. 基本形式

设完整 Fine 数据为：

\[
X
\]

Mask 为：

\[
M
\]

完整数据真实粗尺度表示为：

\[
Z_s^* = D_s(X)
\]

而从缺失数据构造的尺度表示为：

\[
\hat Z_s = D_s(X\odot M,M)
\]

一般可能存在：

\[
\hat Z_s \neq Z_s^*
\]

我们关注的就是二者之间的尺度失真。

---

## 3. 初步解决方向

构造粗尺度时，不只保留一个平均值，而同时保留更多“观测证据”，例如：

```text
观测和值
观测数量
局部方差
观测位置分布
观测几何
可靠性
```

例如：

\[
S_1=\sum_i M_iX_i
\]

\[
S_2=\sum_i M_iX_i^2
\]

\[
C=\sum_iM_i
\]

从而得到：

\[
\mu=\frac{S_1}{C+\epsilon}
\]

和：

\[
\sigma^2=
\frac{S_2}{C+\epsilon}-\mu^2
\]

这样粗尺度表示不再只是一个 value，而是：

```text
Value
+
Observation Mass
+
Variance
+
Geometry
+
Reliability
```

核心目标是：

> **让缺失条件下构造的粗尺度表示尽量接近完整数据下真正的 coarse state。**

---

# Idea 2：Mixture of Coarsening Experts

## 1. 核心问题

如果固定的：

```text
2×2 Average
4×4 Average
固定 Sum
```

并不一定适合所有数据、所有区域和所有缺失模式，那么可以进一步把：

> **“粗尺度到底应该怎么构造”**

本身变成一个可学习问题。

这里 MoE 的 Expert 不再主要负责直接补全，而是负责：

> **学习不同的 Fine → Coarse 粗化方式。**

因此不是普通：

```text
Mixture of Prediction Experts
```

而是：

```text
Mixture of Coarsening Experts
```

---

## 2. 核心结构

整体可以设计为：

```text
Incomplete Fine Data
        ↓
Fine Encoder
        ↓
Coarsening Router
        ↓
┌────────┬────────┬────────┬────────┐
E1       E2       E3       E4
不同粗化方式
└────────┴────────┴────────┴────────┘
        ↓
Top-K Coarse Views
        ↓
Fine + Coarse
        ↓
后续时空补全模块
        ↓
Final Imputation
```

不同 Expert 可以学习不同的粗化机制，例如：

```text
E1：固定物理 Avg / Sum
E2：局部加权聚合
E3：更大感受野聚合
E4：相似性驱动 / 不规则聚合
```

Router 根据：

```text
Fine feature
Mask
局部观测率
方差
缺失结构
观测可靠性
```

动态决定当前更适合使用哪些粗化方式。

---

## 3. 进一步做成“不规则粗尺度”

更强的版本不是只学习 pooling weight，而是让 Expert 学习：

> **哪些 Fine cells 应该组成同一个 coarse region。**

设第 \(e\) 个 Expert 学习：

\[
A_e\in\mathbb R^{N_f\times N_c}
\]

其中：

\[
A_{e,i,j}
\]

表示 Fine node \(i\) 对 coarse node \(j\) 的贡献程度。

则：

\[
Z_c^{(e)}
=
\frac{
A_e^\top(M\odot H_f)
}{
A_e^\top M+\epsilon
}
\]

这样得到的 coarse region 不再一定是固定：

```text
2×2
4×4
```

规则方块，而可以根据：

```text
数据内容
时空相关性
缺失状态
局部结构
```

形成自适应、不规则的 coarse regions。

也就是说尺度从：

> **固定 Resolution Scale**

变成：

> **Data-Adaptive Functional Scale**

---

## 4. 为什么需要多个 Expert

核心假设是：

> **不存在一种固定粗化方式能够适用于所有缺失场景。**

例如：

```text
局部信息完整
→ 小范围聚合可能更合适

连续大块缺失
→ 更大范围粗化可能更有效

交通热点区域
→ 需要保留局部异质性

平稳区域
→ 可以采用更强的聚合
```

因此使用多个 Coarsening Experts，让不同 Expert 学习互补的尺度构造方式，再根据当前缺失状态动态选择。

---

# 两个 Idea 的关系

两个想法可以组成一条完整逻辑：

```text
固定 Pooling 在缺失条件下可能产生尺度失真
        ↓
Missingness-Induced Scale Distortion
        ↓
说明“粗尺度怎么构造”本身值得研究
        ↓
不再固定 Average / Sum / 规则网格
        ↓
Mixture of Coarsening Experts
        ↓
学习多种缺失感知的粗化方式
        ↓
生成更合理的自适应粗尺度表示
        ↓
Fine + Coarse 联合完成补全
```

因此可以把整个研究方向概括为：

> **研究缺失条件下多尺度数据本身应该如何构造，并使用 MoE 学习多种自适应粗化方式，从不完整 Fine 数据中生成更可靠、更有结构意义的 coarse representations，为后续时空补全提供更好的多尺度表示。**

一句话版本：

> **第一个 Idea 解决“为什么传统尺度构造可能有问题”，第二个 Idea 解决“如何让模型自己学习更合理的尺度构造方式”。**
