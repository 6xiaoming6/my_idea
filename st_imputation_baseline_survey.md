# 时空数据补全 Baseline 纵向对比方案

> 面向当前模型：多尺度时空数据补全 / TaxiBJ、BikeNYC 等网格型 flow 数据。  
> 目标：构建纵向 baseline，而不是只做模型内部消融。  
> 生成时间：2026-06-11

---

## 0. 总体建议

你的模型目前属于：

```text
多尺度输入 + 共享跨尺度主干 + 路由专家残差增强
```

所以 baseline 不要只选深度学习模型，而要覆盖四类：

1. 简单补全方法：证明不是简单均值/插值就能解决；
2. 经典补全方法：对比传统矩阵/张量补全和早期深度补全；
3. 新近补全方法：对比 Transformer、GNN、Diffusion、MoE；
4. 多模态/外部信息方法：为后续辅助分支做铺垫。

建议最终论文主表不要放太多方法，最好控制在 10～14 个左右；其余方法放附录或者补充实验。

---

# 1. 总览表

| 类别 | 方法 | 年份 | 尺度 | 输入类型 | 公开源码 | 推荐程度 | 适配难度 |
|---|---:|---:|---|---|---|---|---|
| 简单方法 | Mean Fill / Global Mean | - | 单尺度 | flow + mask | 自己实现 | 必做 | 低 |
| 简单方法 | Historical Average / Periodic Mean | - | 单尺度 | flow + 时间周期 | 自己实现 | 必做 | 低 |
| 简单方法 | Linear / Temporal Interpolation | - | 单尺度 | flow + mask | 自己实现 | 必做 | 低 |
| 简单方法 | Spatial Neighbor Mean | - | 单尺度 | flow + mask + 网格邻域 | 自己实现 | 推荐 | 低 |
| 简单方法 | KNNImputer | - | 单尺度 | flow 展平特征 | scikit-learn | 推荐 | 低 |
| 经典方法 | SoftImpute / Nuclear Norm MC | 经典 | 单尺度 | matrix/tensor 展平 | R 包 / 可复现 | 推荐 | 中 |
| 经典方法 | LRTC-TNN | 2020 | 单尺度 | traffic tensor | 论文清楚，可复现 | 推荐 | 中 |
| 经典方法 | LATC | 2021 | 单尺度 | sensor × time-of-day × day tensor | 官方代码 | 强烈推荐 | 中 |
| 经典方法 | BRITS | 2018 | 单尺度 | multivariate time series | 官方代码 | 推荐 | 中 |
| 经典方法 | GAIN | 2018 | 单尺度 | tabular/time series 展平 | 官方代码 | 推荐 | 中 |
| 新近方法 | SAITS | 2022 | 单尺度 | multivariate time series | 官方代码 / PyPOTS | 强烈推荐 | 中 |
| 新近方法 | CSDI | 2021 | 单尺度 | probabilistic time series | 官方代码 | 推荐 | 中高 |
| 新近方法 | GRIN | 2022 | 单尺度/图结构 | graph time series | 官方代码 | 强烈推荐 | 中高 |
| 新近方法 | PriSTI | 2023 | 单尺度/图结构 | spatiotemporal + graph prior | 官方代码 | 强烈推荐 | 中高 |
| 新近方法 | ImputeFormer | 2024 | 单尺度/通用 ST | spatiotemporal series | 官方代码 / PyPOTS | 强烈推荐 | 中 |
| 新近方法 | STCPA | 2022 | 单尺度/图结构 | traffic speed graph | 官方代码 | 可选 | 中高 |
| 新近方法 | STAMImputer | 2025 | 单尺度/图结构/MoE | traffic graph | 官方代码 | 强烈推荐 | 高 |
| 新近方法 | PAST | 2025 | 单尺度/图结构 + auxiliary | traffic graph + 时间/节点属性 | GitHub 代码 | 推荐 | 高 |
| 多尺度方法 | UrbanFM / UrbanPy | 2019/2020 | 多尺度 | coarse flow → fine flow + external factors | 非官方/公开实现 | 推荐 | 中 |
| 多尺度方法 | ST-Pyramid Multi-Scale Data Completion | 2026 | 多尺度 | sparse MCS multiscale | 未发现官方代码，论文清楚 | 推荐 | 中高 |
| 多尺度方法 | Non-Aligned Multi-Scale Data Completion | 2025 | 多尺度 | non-aligned multiscale MCS | 未发现官方代码，论文/Slides 清楚 | 推荐 | 高 |
| 多模态方法 | E2-CSTP | 2025 | 单/多尺度可扩展 | flow + text/image/aux modalities | 官方代码 | 作为多模态参考 baseline | 高 |
| 多模态方法 | DSTTN Cross-modal Imputation | 2024 | 单尺度 | time series + spatial modal data | 论文清楚，代码不确定 | 可选 | 高 |
| 多模态方法 | DiffUFlow / DP-TFI | 2023 | 多尺度/生成式 | coarse flow + external/land features | 代码不确定 | 可选参考 | 高 |

---

# 2. 简单补全方法（单尺度）

## 2.1 Mean Fill / Global Mean

对所有缺失位置填充训练集均值：

```text
x_missing = mean(x_observed_train)
```

推荐至少做：

```text
Channel-wise Mean:
inflow 用 inflow 均值填
outflow 用 outflow 均值填
```

这是最低下限 baseline，必做。

## 2.2 Historical Average / Periodic Mean

利用周期性，比如同一时间段、同一星期、同一网格的历史平均值：

```text
x_hat[t, h, w] = mean observed value at same time slot
```

推荐版本：

```text
HA-time
HA-grid-time
HA-channel-grid-time
```

如果你的模型打不过 Historical Average，说明时空建模可能没有发挥作用。

## 2.3 Linear / Temporal Interpolation

对每个网格位置和通道，在时间维上做线性插值：

```text
x_hat[t] = linear interpolation between nearest observed time points
```

如果某个序列前后没有观测，可以 fallback 到 Historical Average 或 Mean Fill。

## 2.4 Spatial Neighbor Mean

对每个缺失网格，用周围邻居网格的观测值平均填充：

```text
x_hat[t,h,w] = mean observed values in 3×3 or 5×5 neighborhood
```

可以作为纯空间局部 baseline，尤其适合网格型 TaxiBJ/BikeNYC。

## 2.5 KNNImputer

KNNImputer 用样本间距离寻找近邻，并用近邻样本均值补全缺失值。

适配方式：

```text
[B, C, T, H, W] -> [B, C*T*H*W]
```

高维情况下 KNN 会比较慢，建议只作为小规模 baseline，或先降维/按时间片处理。

---

# 3. 经典补全方法（单尺度）

## 3.1 SoftImpute / Nuclear Norm Matrix Completion

把数据整理成矩阵，通过低秩矩阵补全恢复缺失值：

```text
min ||P_Ω(X - Y)||² + λ ||X||_*
```

适配方式：

```text
[T_total, C*H*W]
```

优点是实现简单，能证明低秩假设是否有用；缺点是不能显式建模复杂非线性时空关系。

## 3.2 LRTC-TNN

LRTC-TNN 是低秩张量补全方法，用 truncated nuclear norm 建模交通张量低秩结构。

适配方式：

```text
day × time_slot × grid_cell
```

或：

```text
day × time_slot × H × W
```

推荐作为传统张量补全 baseline。

## 3.3 LATC: Low-Rank Autoregressive Tensor Completion

LATC 在低秩张量补全基础上加入 autoregressive temporal regularization，同时建模：

```text
global low-rank consistency
+
local temporal consistency
```

推荐整理成：

```text
grid_cell × time_of_day × day
```

其中：

```text
grid_cell = H × W × C
```

这是强传统 baseline，尤其适合交通数据补全。

## 3.4 BRITS

BRITS 使用双向 RNN 进行时间序列补全，显式利用：

```text
forward temporal dynamics
backward temporal dynamics
missing mask
time gaps
```

适配方式：

```text
[B, T, C*H*W]
```

TaxiBJ 的 feature dimension 较大，可以按 patch 或 channel 拆分。

## 3.5 GAIN

GAIN 是 GAN 风格的缺失补全方法：

```text
Generator 生成缺失值
Discriminator 判断哪些位置是真实观测，哪些是生成值
```

适配方式：

```text
[B, C*T*H*W]
```

或者按时间片：

```text
[B*T, C*H*W]
```

GAIN 可以作为经典生成式补全 baseline，但高维网格上训练可能不稳定。

---

# 4. 比较新的补全方法（单尺度/时空）

## 4.1 SAITS

SAITS 是基于 self-attention 的时间序列补全模型，使用两个 DMSA block 和加权组合机制建模时间依赖、变量相关性和缺失信息。

适配方式：

```text
[B, T, C*H*W]
```

如果维度过高，可以：

```text
1. 每个通道单独跑；
2. 按 patch 展平；
3. 先用 PCA/Conv encoder 降维。
```

推荐程度：强烈推荐。

## 4.2 CSDI

CSDI 是条件扩散式时间序列补全模型，根据观测值条件生成缺失值。

适配方式：

```text
[B, T, N]
N = C*H*W
```

优点是适合高缺失率和不确定性建模；缺点是训练和采样成本较高。

## 4.3 GRIN

GRIN 是图神经网络时序补全方法，将多变量时间序列建模为图上的信号补全问题。

适配方式：

```text
node = grid cell
edge = 4-neighbor / 8-neighbor
feature = inflow/outflow
```

输入可转成：

```text
[B, T, N, C]
N = H × W
```

推荐程度：强烈推荐。

## 4.4 PriSTI

PriSTI 是 spatiotemporal imputation 的条件扩散模型，用 enhanced prior modeling 构造条件信息，再进行扩散补全。

适配方式：

```text
node = grid cell
edge = spatial adjacency
input = [B, T, N, C]
```

推荐程度：强烈推荐。

## 4.5 ImputeFormer

ImputeFormer 将低秩归纳偏置引入 Transformer，用于通用 spatiotemporal imputation。

适配方式：

```text
[B, T, N]
```

或：

```text
[B, T, N, C]
```

推荐程度：强烈推荐。

## 4.6 STCPA

STCPA 用 spatio-temporal attention 和 cycle-perceptual training 做 traffic speed imputation。

适配方式：

```text
grid cell -> graph node
4-neighbor / 8-neighbor adjacency
```

原任务是 traffic speed，和 TaxiBJ/BikeNYC flow grid 有差异，因此作为可选 baseline。

## 4.7 STAMImputer

STAMImputer 是 2025 年的 traffic data imputation 方法，引入 MoE 框架和动态图注意力来处理 block missing 和非平稳交通数据。

适配方式：

```text
node = grid cell
edge = spatial adjacency
```

这是与你当前 MoE 模型最相关的强 baseline 之一。

## 4.8 PAST

PAST 是 Primary-Auxiliary Spatio-Temporal Network，用 primary module 建模内部时空关系，用 auxiliary module 建模时间戳、节点属性等外部特征。

适配方式：

```text
grid cell -> graph node
time-of-day / day-of-week / node position embedding -> auxiliary features
```

和你后续辅助分支思路相似，推荐作为辅助信息增强型 baseline。

---

# 5. 比较新的多尺度补全/推断方法

严格的“多尺度时空补全”方法比单尺度少很多，其中不少是 sparse crowdsensing 或 urban flow inference 方向。它们和你的任务不完全一致，但非常适合作为多尺度纵向对比或论文相关工作。

## 5.1 UrbanFM / UrbanPy

UrbanFM 任务是：

```text
coarse-grained urban flow -> fine-grained urban flow
```

它从粗粒度流量推断细粒度流量，并考虑外部因素。

和你的关系：

```text
你的任务:
sparse fine/mid/coarse observed data -> complete fine data

UrbanFM:
coarse flow -> fine flow
```

可以将你的 `X_m` 或 `X_c` 作为 coarse input，目标是 `X_f_gt`。

## 5.2 Spatio-Temporal Pyramid-Based Multi-Scale Data Completion

该方法面向 Sparse Crowdsensing 的多尺度数据补全，核心包括：

```text
ST-PC: spatio-temporal pyramid construction
ST-PAM: spatio-temporal pyramid attention
cross-scale constraints
```

这是和你当前“多尺度时空补全”最接近的论文之一。未发现官方代码，但结构可以复现。

## 5.3 Non-Aligned Multi-Scale Data Completion

该方法处理 non-aligned compositional relationships 的多尺度数据补全，并强调：

```text
intra-scale correlations
inter-scale correlations
lightweight multi-scale modeling
```

它和你现在“尺度内路由 + 尺度间共享”的思想很接近。未发现官方代码，但论文和 slides 公开，结构可复现。

## 5.4 DiffUFlow / DP-TFI

这类方法面向 fine-grained urban flow inference，常见输入是：

```text
incomplete/coarse urban flow observations
external factors / land features
```

输出：

```text
complete fine-grained urban flow
```

更接近“城市流量推断”而不是标准缺失补全，但对 TaxiBJ 类 flow grid 有参考价值。

---

# 6. 多模态输入方法

严格意义上“多模态输入 + 时空补全”的公开代码方法不多。很多方法是多模态时空预测或城市流量推断，需要适配成补全任务。

## 6.1 E2-CSTP

E2-CSTP 是 causal multi-modal spatio-temporal prediction 框架，使用：

```text
cross-modal attention
gating mechanism
dual-branch causal inference
GCN + Mamba spatio-temporal encoder
```

它不是补全任务，而是 prediction 任务。但它和你“主分支 + 辅助多模态分支”的结构高度相关。

适配方式：

```text
输入:
X_obs, M, F_text/F_img/F_aux

输出:
X_hat

loss:
missing positions reconstruction loss
```

## 6.2 PAST

PAST 把模式分为：

```text
primary patterns: 数据内部时空关系
auxiliary patterns: 时间戳、节点属性等外部因素
```

它属于“辅助信息增强型”时空补全方法，和你的后续辅助分支思路相似。

## 6.3 DSTTN Cross-modal Imputation

DSTTN 用 cross-modal 方法进行 time-series missing data imputation，将 spatial modal data 嵌入到 time-series data 中，通过 dense spatio-temporal transformer 进行补全。

可以作为跨模态补全代表，但未确认公开代码。

## 6.4 UrbanFM / DiffUFlow 类外部因素模型

UrbanFM 和 DiffUFlow 这类 urban flow inference 方法通常会考虑外部因素，例如：

```text
weather
holiday
POI
land features
time metadata
```

如果你后续辅助分支接入图像、文本、POI、天气，这类方法可以作为多模态/多源信息 baseline。

---

# 7. 最推荐最终使用的 baseline 组合

## 7.1 最小但完整版本

| 类别 | 方法 |
|---|---|
| 简单统计 | Mean Fill |
| 简单统计 | Historical Average |
| 简单统计 | Linear Interpolation |
| 经典传统 | SoftImpute |
| 经典传统 | LATC |
| 经典深度 | BRITS |
| 经典深度 | GAIN |
| 新近单尺度 | SAITS |
| 新近单尺度 | CSDI |
| 新近时空 | GRIN |
| 新近时空 | PriSTI |
| 新近时空 | ImputeFormer |
| 多尺度 | UrbanFM / ST-Pyramid 二选一 |
| 多模态参考 | E2-CSTP / PAST 二选一 |

## 7.2 更适合你论文主线的版本

如果你的论文重点是“多尺度 + MoE”，建议主表这样放：

| 层级 | 方法 |
|---|---|
| Simple | HA |
| Simple | Linear Interpolation |
| Classic | LATC |
| Classic | BRITS |
| Deep Single-scale | SAITS |
| Deep Single-scale | CSDI |
| Spatiotemporal | GRIN |
| Spatiotemporal | PriSTI |
| Transformer | ImputeFormer |
| MoE Traffic Imputation | STAMImputer |
| Multi-scale | ST-Pyramid |
| Multi-scale Flow Inference | UrbanFM |
| Ours | Your Full Model |

附录再放：

```text
Mean Fill
KNN
GAIN
STCPA
PAST
Non-Aligned Multi-Scale
E2-CSTP adapted
```

---

# 8. 适配 TaxiBJ/BikeNYC 的统一输入建议

你的数据是：

```text
X: [B, C, T, H, W]
M: [B, 1, T, H, W]
```

## 8.1 传统时间序列方法

例如：

```text
BRITS
SAITS
CSDI
ImputeFormer
```

转换为：

```text
[B, T, N]
N = C × H × W
```

TaxiBJ：

```text
N = 2 × 32 × 32 = 2048
```

BikeNYC：

```text
N = 2 × 24 × 12 = 576
```

TaxiBJ 维度较大，可以按 channel 或 patch 拆分。

## 8.2 图神经网络方法

例如：

```text
GRIN
PriSTI
STAMImputer
PAST
STCPA
```

转换为：

```text
[B, T, N, C]
N = H × W
C = inflow/outflow
```

构造图：

```text
4-neighbor grid graph
8-neighbor grid graph
distance-based graph
learnable adjacency
```

## 8.3 矩阵/张量补全方法

例如：

```text
SoftImpute
LRTC-TNN
LATC
```

构造：

```text
Matrix:
[T_total, C×H×W]

Tensor:
[day, time_slot, C×H×W]
```

或：

```text
[C×H×W, time_of_day, day]
```

## 8.4 多尺度方法

例如：

```text
UrbanFM
ST-Pyramid
Non-Aligned Multi-Scale
```

使用你当前已有的 masked pooling：

```text
X_f, M_f
X_m, M_m
X_c, M_c
```

保持和你的模型输入一致，保证对比公平。

---

# 9. 实验设置建议

## 9.1 缺失模式

至少覆盖：

```text
Random Missing
Spatial Block Missing
Temporal Block Missing
Spatio-temporal Block Missing
```

缺失率：

```text
20%, 40%, 60%, 80%
```

你的模型理论上应该在：

```text
Spatial Block Missing
Spatio-temporal Block Missing
High Missing Rate
```

更有优势。

## 9.2 指标

建议使用：

```text
MAE
RMSE
MAPE
```

如果 MAPE 不稳定，可以只报告：

```text
MAE / RMSE
```

## 9.3 公平性原则

所有深度 baseline 尽量保持：

```text
same train/val/test split
same missing mask
same sequence length T
same optimizer/lr/scheduler if possible
same early stopping strategy
same 3 seeds
```

传统方法不用训练 epoch，但必须使用相同 mask。

---

# 10. 推荐执行顺序

## 第一阶段：快速补齐低成本 baseline

```text
Mean
Historical Average
Linear Interpolation
Spatial Neighbor Mean
KNN
SoftImpute
LATC
```

## 第二阶段：补齐单尺度深度 baseline

```text
BRITS
SAITS
CSDI
ImputeFormer
```

## 第三阶段：补齐时空图 baseline

```text
GRIN
PriSTI
STAMImputer
```

## 第四阶段：补齐多尺度/多模态 reference

```text
UrbanFM
ST-Pyramid
Non-Aligned Multi-Scale
E2-CSTP adapted
PAST
```

---

# 11. 文献与代码链接

## Simple / Classic

- KNNImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
- SoftImpute R package: https://cran.r-project.org/package=softImpute
- GAIN official code: https://github.com/jsyoon0823/GAIN
- BRITS official code: https://github.com/caow13/BRITS
- NIPS-BRITS code: https://github.com/NIPS-BRITS/BRITS
- LATC code: https://github.com/xinychen/autoregressive-tensor

## Recent single-scale / spatiotemporal

- SAITS official code: https://github.com/WenjieDu/SAITS
- PyPOTS: https://github.com/WenjieDu/PyPOTS
- CSDI official code: https://github.com/ermongroup/CSDI
- GRIN official code: https://github.com/Graph-Machine-Learning-Group/grin
- PriSTI official code: https://github.com/LMZZML/PriSTI
- ImputeFormer official code: https://github.com/tongnie/ImputeFormer
- STCPA code: https://github.com/Sam1224/STCPA
- STAMImputer code: https://github.com/RingBDStack/STAMImupter
- PAST code: https://github.com/Hanwen-Hu/PAST

## Multi-scale / urban flow

- UrbanFM PyTorch implementation: https://github.com/yoshall/UrbanFM
- UrbanFM paper: https://arxiv.org/abs/1902.05377
- ST-Pyramid TMC page: https://www.computer.org/csdl/journal/tm/2026/01/11125969/29aJ0j2vDtm
- Non-Aligned Multi-Scale paper: https://cis.temple.edu/~jiewu/research/publications/Publication_files/iwqos2025-final14.pdf
- Non-Aligned Multi-Scale slides: https://cis.temple.edu/~wu/research/publications/Publication_files/25iwqos-slides.pdf

## Multimodal / auxiliary input

- E2-CSTP official code: https://github.com/ZJU-DAILY/E2-CSTP
- E2-CSTP paper: https://arxiv.org/abs/2505.17637
- DSTTN cross-modal imputation article: https://www.aimspress.com/article/doi/10.3934/mbe.2024220?viewType=HTML
- DiffUFlow paper page: https://www.zhangjunbo.org/publication/23-cikm-diffuflow/
- Awesome Multimodal Urban Computing: https://github.com/yoshall/Awesome-Multimodal-Urban-Computing

---

# 12. 最终建议

最应该优先实现的 baseline 是：

```text
1. Historical Average
2. Linear Interpolation
3. SoftImpute
4. LATC
5. SAITS
6. CSDI
7. GRIN
8. PriSTI
9. ImputeFormer
10. STAMImputer
11. UrbanFM 或 ST-Pyramid
12. PAST 或 E2-CSTP-adapted
```

其中最能支撑论文主线的是：

```text
LATC:
传统张量补全强 baseline

SAITS / ImputeFormer:
Transformer 类单尺度补全 baseline

GRIN / PriSTI:
时空图补全 baseline

STAMImputer:
MoE 类时空补全 baseline

UrbanFM / ST-Pyramid:
多尺度补全/推断 baseline

PAST / E2-CSTP:
辅助信息/多模态 baseline
```

如果时间不够，优先完成：

```text
HA + Linear + LATC + SAITS + GRIN + PriSTI + ImputeFormer + STAMImputer + UrbanFM/ST-Pyramid
```

这组已经足够形成比较完整的纵向对比。
