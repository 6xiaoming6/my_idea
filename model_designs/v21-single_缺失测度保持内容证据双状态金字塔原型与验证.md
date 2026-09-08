# V21-single：缺失测度保持的内容–证据双状态金字塔原型与验证

## 1. 本轮目标

本轮只验证“想法一”的最小必要假设，不把未经完整训练验证的部分提前写成结论：

1. 当前 V14 的层级 masked mean 是否存在 Fine→Mid→Coarse 与 Fine→Coarse 不一致；
2. 可加的观测和、平方和、计数及坐标矩能否构造路径一致状态；
3. 内容值与缺失证据能否在不泄漏隐藏真值、不破坏 V14 默认路径的条件下接入 MoE；
4. 该方向是否值得进入完整训练消融。

## 2. 调研后的创新边界

已有工作已经覆盖以下内容，因此不能把它们单独作为创新点：

- Partial Convolution 已使用 mask 对卷积结果重归一化并更新 mask：
  <https://openaccess.thecvf.com/content_ECCV_2018/html/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.html>
- Sparsity Invariant CNN 已研究稀疏输入下按有效观测归一化的卷积：
  <https://arxiv.org/abs/1708.06500>
- Confidence Propagation through CNNs 已把 normalized convolution 与置信度传播结合：
  <https://arxiv.org/abs/1811.01791>
- 多分辨率 Partial Convolution 已用于多尺度图像修复：
  <https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Parallel_Multi-Resolution_Fusion_Network_for_Image_Inpainting_ICCV_2021_paper.html>
- MagiNet 等时空补全工作已经显式采用 mask-aware 表征：
  <https://arxiv.org/abs/2406.03511>
- Deep Sets 及后续理论说明了 sum-decomposable set representation 的表达基础：
  <https://proceedings.mlr.press/v237/tabaghi24a.html>

因此 V21 的可辩护研究点不是“masked average”，而是下面这一整条闭环：

> 缺失诱导尺度失真诊断 → 可加观测测度的层级路径一致性 → 内容/证据双状态 → 证据条件化的多尺度 MoE → 跨数据集消融与证据校准。

目前的检索只能帮助界定近邻工作，不能替代正式投稿前的系统查新，暂时不应使用“首次”或“前所未有”等绝对表述。

## 3. 发现的具体问题

V14 先以 observed mean 生成 Mid，再把所有非空 Mid cell 当成等权样本生成 Coarse。设四个 Mid cell 的观测计数为 \(C_j\)、均值为 \(\mu_j\)，当前 Coarse 实际接近：

\[
\mu_C^{legacy}=\frac{1}{|\{j:C_j>0\}|}\sum_{j:C_j>0}\mu_j
\]

而所有 fine 观测的真实聚合均值是：

\[
\mu_C^{measure}=\frac{\sum_j C_j\mu_j}{\sum_j C_j}
\]

当各 Mid cell 的观测数量不同，两者不相等。原代码虽计算了 `r_m`，但构造 Coarse 内容时没有用它恢复观测质量。

## 4. V21 原型

### 4.1 路径一致的 Content State

每个尺度 cell 从 fine 观测直接计算：

\[
S_1=\sum_i m_ix_i,\qquad S_2=\sum_i m_ix_i^2,\qquad C=\sum_i m_i
\]

并以 \(S_1/(C+\epsilon)\) 作为当前原型的 content。由于 \(S_1,S_2,C\) 都可加，通过任意合法层级路径聚合后再归一化，会得到相同结果。

注意：本轮 content 仍是观测条件均值，还不是已经学成的 mask-invariant 完整尺度内容。是否需要额外的尺度重建监督或跨 mask content consistency，应由本轮消融结果决定。

### 4.2 Evidence State

每个尺度 cell 使用 7 个有界、无量纲证据通道：

1. `coverage`：观测质量 \(C/|cell|\)；
2. `relative_value_variance`：观测值相对方差；
3. `centroid_y`；
4. `centroid_x`；
5. `spread_y`；
6. `spread_x`；
7. `covariance_yx`。

这使“左半侧观测”和“上半侧观测”即使均值、覆盖率相同，也不再被编码成完全相同的尺度状态。

### 4.3 接入方式

- Mid 与 Coarse content 都直接由 fine 观测生成，避免重复 mean pooling 的路径失真；
- Evidence 经独立 `1×1×1 Conv3d` 投影后注入各尺度 token；
- MoE router 对这些 token 做池化，因此路由可以学习 observation regime；
- evidence 投影权重和偏置均零初始化，初始时不引入随机证据扰动；
- V14 原结构、专家数、Top-K、refiner 和训练策略不变；
- 三个 evidence 投影共增加 1,536 个参数，在三个数据集上仅约增加 0.031%。

## 5. 代码范围

- 数据测度与双状态：`src/stmoe_imputer/data/transforms.py`
- NPZ/synthetic 数据接入：`src/stmoe_imputer/data/npz_dataset.py`、`synthetic.py`、`build.py`
- Evidence token 接入：`src/stmoe_imputer/models/embedding.py`
- MoE 与 V14 接口：`main_branch.py`、`imputer.py`、`v_single/v14_safe_c2f_moe.py`
- V21 配置：`configs/v21-single/`
- V21 单任务入口：`scripts/v21-single/train.py`
- 数据诊断：`scripts/v21-single/diagnose_scale_distortion.py`
- 回归测试：`tests/test_v21_observation_moment_pyramid.py`

所有新增行为均由以下配置显式开启：

```json
{
  "data": {"scales": {"pyramid_mode": "observation_moment"}},
  "model": {"main": {"evidence_dim": 7, "evidence_zero_init": true}}
}
```

默认 `pyramid_mode=legacy`、`evidence_dim=0`，旧配置保持原路径。

## 6. 已完成验证

### 6.1 三数据集完整验证集的 24 点无模型诊断

覆盖 TaxiBJ、BikeNYC、CHAP，fixed/random，缺失率 0.2/0.4/0.6/0.8。

- 当前 coarse SDI 均值：`0.166544`
- 路径一致 coarse SDI 均值：`0.165400`
- 相对降低：`0.69%`
- 路径一致方案胜出：`10/24`
- TaxiBJ 平均降低：`6.575%`
- BikeNYC 平均变化：`-2.959%`
- CHAP 平均变化：`-2.354%`
- raw sum/count 层级聚合与 direct 聚合最大相对路径误差约 `3.77e-7`，差异仅来自浮点舍入。

结论：尺度失真与路径不一致确实存在，但“无条件把 coarse mean 换成 count-weighted mean”不能保证所有数据集的最终表示都更接近完整尺度。V21 必须保留 content 与 evidence 两种状态，让模型学习何时信任观测聚合，不能只做固定 pooling 替换。

完整结果位于：

- `outputs/v21-single/diagnostics/scale_distortion/scale_distortion_report.md`
- `outputs/v21-single/diagnostics/scale_distortion/scale_distortion_results.json`

### 6.2 工程验证

- V21 针对性测试：7/7 通过；
- 全项目当前源码回归测试：50/50 通过；
- 覆盖 legacy 等价、空观测数值稳定、隐藏真值不泄漏、观测几何可辨识、证据投影可获得非零有限梯度；
- 覆盖 TaxiBJ `2×12×32×32`、BikeNYC `2×12×24×12`、CHAP `1×7×32×32` 三种输入几何；
- 1 epoch synthetic 完整流程已跑通：train → val → 保存 best → 加载 best → test；
- 三个真实数据集训练命令均通过配置及路径 dry-run。

### 6.3 TaxiBJ fixed@0.6 的真实 1-epoch 对照 smoke

三组实验均使用 seed=42、batch size=32，同一训练/验证/测试划分。可选 evidence 模块使用独立 RNG 域构造，并通过测试确认所有公共参数在初始化时逐元素完全一致。

| 变体 | Val MAE | Test MAE | Test RMSE | Test WAPE |
|---|---:|---:|---:|---:|
| P0 legacy control | 55.5505 | 53.5726 | 85.5924 | 0.54284 |
| P1 measure content，无 7D evidence | 55.4483 | **53.0030** | 86.1382 | **0.53707** |
| P2 measure content + 7D evidence | 55.9347 | 54.2021 | **85.5385** | 0.54922 |

这个 smoke 只能检查真实数据上的优化与评估链路，不能作为最终效果结论。它给出的早期信号是：P1 的 MAE/WAPE 方向较好，P2 的 RMSE 略好，但完整 7D evidence 在一个 epoch 后尚未转化为 MAE 收益。P1 虽不使用新增 7D evidence，仍保留 V14 原有的 `r_m/r_c` 可靠度输入；其变化包括路径一致 content 和由 fine 计数得到的真实 coarse reliability。

## 7. 下一步有效性消融

必须至少比较三个严格控制变量的版本：

| 编号 | Pyramid | Evidence | 目的 |
|---|---|---|---|
| P0 | legacy hierarchical mean | 无 | V14 对照 |
| P1 | path-consistent content | 无新增 7D evidence，保留 V14 reliability | 测测度路径本身 |
| P2 | path-consistent content | 7 维 evidence + V14 reliability | 测完整双状态贡献 |

现有入口分别对应：

```bash
python scripts/v21-single/train.py --variant legacy_control ...
python scripts/v21-single/train.py --variant content_only ...
python scripts/v21-single/train.py --variant dual_state ...
```

判断标准不能只看平均 MAE，还要检查：

- 每数据集、每缺失模式、每缺失率的 MAE/RMSE；
- 高缺失率是否获得稳定收益；
- fixed 与 random 是否方向一致；
- 至少三个 seed 的均值和标准差；
- evidence coverage 与尺度重建误差是否负相关；
- 专家选择是否随 evidence regime 形成可解释差异，而不是少数专家塌缩。

只有 P2 在多数据集上稳定优于 P0/P1，才能把 Content–Evidence Dual-State Pyramid 作为最终模型贡献；当前结果只说明假设值得训练验证，不说明最终补全效果已经优于 V14。
