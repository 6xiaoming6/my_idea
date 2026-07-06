# imputation_benchmark 额外实现的 Baseline

> 以下模型已在 imputation_benchmark 中完成数据适配和训练流程集成，但不在
> `baselines/st_imputation_baseline_survey.md` 的推荐列表内。
> 暂时不加入日常训练矩阵，后续可作为附录或补充实验。

## 模型列表

| 模型 | 全称 | 类型 | 年份 | 核心思路 |
|------|------|------|------|---------|
| **AGCRN** | Adaptive Graph Convolutional Recurrent Network | 图卷积 + RNN | 2020 | 自适应邻接矩阵 + 图卷积 + GRU 时序建模 |
| **ASTGNN** | Attention-based Spatial-Temporal GNN | 时空图注意力 | 2021 | 空间自注意力 + 时间自注意力 + GCN |
| **E2GAN** | End-to-End GAN for Imputation | GAN 补全 | 2021 | 端到端生成对抗网络补全 |
| **GCASTN** | Generative-Contrastive-Attentive STN | 对比生成时空网络 | 2022 | 缺失感知注意力 + 对比学习 + 双编码器 |
| **IGNNK** | Inductive GNN Kriging | 图神经网络克里金 | 2021 | 归纳式图神经网络空间插值 |
| **LAST** | Lattice-based Spatial-Temporal Imputation | 格点时空补全 | 2021 | 非参数格点时空插值算法 |
| **mTAN** | Multi-Time Attention Network | 多时间注意力 | 2019 | 多时间点注意力机制 + 插值 |
| **SSTBAN** | Self-Supervised ST Bottleneck Attentive Network | 自监督时空瓶颈注意力 | 2023 | ISAB 诱导点注意力 + 自监督双通道训练 |

## 在论文主线中的可能位置

这些模型可以作为：

- **附录补充**：与主表 baseline 的完整对比
- **消融参考**：AGCRN（纯 GCN+RNN，无 attention）、mTAN（多时间点注意力）
- **时空图方法扩展**：ASTGNN、GCASTN、SSTBAN

## 训练策略

这些模型已从以下所有默认入口排除：

- `run_all_baseline_train_2gpu.py`
- `run_taxibj_full_baselines.py`
- `run_overview_baseline_1epoch.py`
- `baseline_paper.json`
- `baseline_5epoch_test.json`
- `taxibj_full_no_diffusion.json`

模型实现和正式单模型训练脚本仍保留，但旧的逐模型 smoke 转发脚本已经删除。
如需用于附录实验，需要显式使用允许该模型的自定义 policy；总览表一轮测试入口不会接受这些模型名。
