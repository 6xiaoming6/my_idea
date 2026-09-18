# v24-COE 第一版：单尺度时空专家链

本文描述默认两轮基线。三轮/四轮及扩展专家池见 [深链候选说明](v24_COE_deeper_candidates.md)，原两轮配置仍可独立运行并作为消融基线。

## 可行性评价

结论：可实现，值得作为独立实验版本；尚不能断言优于固定顺序或原模型。

两轮已足以表达 `TT / TS / ST / SS`。非线性方向算子、数值状态更新与观测条件化使顺序可能影响结果。第二轮读取第一轮产生的隐状态、补全值与更新幅度，研究问题明确，能用同参数池的固定顺序和初始状态路由进行检验。

[CoE 原文第 3 节](https://arxiv.org/html/2506.18945v1#S3)支持共享专家池、逐轮重路由和残差通信的实现，但语言建模结果不能直接证明补全收益。[STAMImputer 第 4.1 节](https://arxiv.org/html/2506.08054v1#S4.SS1)已经让空间专家接收时间专家输出，因此本方案的贡献应聚焦于**缺失条件与中间状态驱动的动态操作顺序**。

需要实测解决的风险：路由塌缩为一种固定路径、局部卷积的感受野不足以覆盖长缺口、窗口级决策无法兼顾不同区域，以及两轮计算的收益不足以补偿额外开销。路径分布多样不等于补全效果更好。

## 本版实现

- 注册架构：`model.architecture = "v24_ts_coe"`，默认 `num_steps=2`。
- 时间专家：`(k,1,1)` 深度卷积与非线性点级投影，只沿时间交换信息。
- 空间专家：`(1,k,k)` 二维网格邻域卷积，只在同一时刻交换信息，不将展平索引当作邻接。
- 共享专家：较小的点级前馈网络。归一化只沿通道，避免跨时间或空间引入额外混合。
- 两轮复用同一组时间、空间、共享专家和解码器，每轮有独立 router。
- 原始输入 mask 始终固定。每轮解码得到原始预测 `Z`，再用输入观测回填得到 `V`；`V` 参与下一轮，观测位置的隐表示仍能更新。
- 路由输入包括全部/缺失位置的隐状态和数值状态统计、估计变化量、缺失比例，以及观测支撑的均值、标准差、最大值和缺失位置均值。
- 支撑特征仅从输入 mask 计算：时间/空间覆盖率、连续缺失长度、前后观测距离、无时间观测/无前序/无后序观测标志、空空间邻域及邻域无观测标志。局部覆盖率按窗口内真实邻域大小归一化。
- 本版不含多尺度、修正专家、风险头或分阶段训练；按方案 10.4，第一版不把负载均衡作为训练目标。

数据张量为 `[B,C,T,H,W]`，mask 支持单通道或逐变量。初始位置编码使用窗口内时间位置和二维网格坐标；不声称实现日周期或图传感器建模。

`hard` 训练使用直通 Gumbel-Softmax，同时计算两个候选专家，以获得路由梯度；**训练计算不稀疏**。推理无采样噪声，按整窗口分组，仅执行所选专家。`soft` 在训练与推理都保留两专家加权，应解释为迭代软混合。`fixed` 执行指定顺序。

## 数据、目标与输出

v24 数据入口直接构造细尺度观测，不读取 NPZ 中的中/粗尺度字段，也不调用多尺度池化；奇数空间尺寸可以使用。旧架构仍保留原数据路径。

可选 NPZ 字段 `available_mask` 表示原始可用性 O，`target_mask` 表示人为隐藏目标 Q。输入 mask 会移除 Q，且与 O 和有限值检查相交；没有显式 Q 时，以可用但未观测的位置为监督。非有限值不是标签。使用有限数值占位表示自然缺失时，必须提供 `available_mask`，不能把占位值当真值监督。

第一版仅计算最终解码预测在有效隐藏目标上的 MAE，四份基础配置均设置 `loss.lambda_coe_balance=0.0`。`loss.lambda_coe_mid` 默认 0；启用时加上中间轮次隐藏目标损失的平均值。损失在做差前先索引有效目标，防止 NaN 污染。空监督训练 batch 跳过更新，整个训练/评估集合无有效目标时报错，避免把 MAE=0 当作最佳模型。

`x_hat_main` / `x_hat_final` 是回填前的预测，用于损失和指标；`x_comp` 是最终补全结果，保留输入观测。前向封装只读取输入观测，不使用 `x_f_gt` 回填。`outputs["coe"]` 包含每轮预测、补全状态、变化量、原始 mask、支撑特征和路由 logits/probabilities/weights/paths。

训练与评估日志记录按样本计数的每轮专家使用率和硬路径分布；软版本记录权重，不将 argmax 摘要称作实际执行路径。训练还记录每轮 router 的梯度范数。

## 路由负载均衡辅助损失

**第一版默认关闭。** 按设计 10.4/10.5，四份基础配置、三轮/四轮候选以及主实验使用 `lambda_coe_balance=0.0`、`lambda_coe_mid=0.0`，总目标为有效隐藏位置的 MAE。保留既有 ST Gumbel-Softmax 梯度估计，不新增熵奖励、专家均分约束或探索日程。弱中间监督单独作为可选消融。

先监控每轮专家使用率、路径分布、router/专家梯度，并按窗口缺失率和缺失位置可用的时空观测支撑分组。分组只读取原始观测 mask 和支撑特征，不读取目标真值。使用率偏斜不直接等于塌缩：需要结合不同缺失条件、持续零梯度、替换路径后的误差及独立固定路径基线判断是否失去有效选择能力。具体实验见 [第 13 节实验设计](v24_COE_experiment_design.md)。

辅助损失实现保留为后续开关，不进入第一版训练目标。只有出现上述证据后，再独立测试 `configs/v24/regularization/route_balance_1e3.json`。原 `ablations/no_route_balance.json` 仍可用于显式关闭，它与当前默认设置相同。

[CoE 原论文](https://arxiv.org/html/2506.18945v1)未明确列出辅助损失公式；[作者代码](https://github.com/ZihanWang314/CoE/blob/main/config/models/coe_deepseekv2/modeling_coe.py#L468-L518)采用选择频率与平均路由概率的乘积，[官方配置](https://github.com/ZihanWang314/CoE/blob/main/config/models/coe_deepseekv2/config.json)为 `0.001`。我们保留的可选实现针对窗口路由，按 batch 统计并对轮数平均：

$$
L_{balance}=\frac{1}{K}\sum_{r=1}^{K}E\sum_{e=1}^{E}
\operatorname{stopgrad}(\operatorname{mean}_b w_{b,r,e})
\operatorname{mean}_b p_{b,r,e}.
$$

仅至少含一个有效监督目标的窗口参与统计。硬路由使用实际 Gumbel 选择及无采样噪声的可微概率；软链使用混合权重。固定路径、仅共享专家、单专家、单轮并行基线及空监督 batch 不应用此项。权重为 0 时辅助图不接入训练目标，即便保留未加权诊断。小 batch 的负载统计可能波动，且异构专家的合理分工无需均匀。

日志 `l_coe_balance` 是未加权诊断，`l_coe_balance_weighted` 才是实际加入总损失的贡献，第一版后者为 0。`l_coe_balance_step{r}` 为各轮未加权值。它们不参与最佳模型选择，也不是必须逼近某个值的任务指标。

## 运行

沿用仓库已安装 PyTorch 的 Python 环境；本机为 `/home/students/HuangMingYu/anaconda3/envs/project/bin/python`。以下命令在仓库根目录运行。

```bash
# 合成数据：训练、验证、保存/重载最佳 checkpoint、最终测试
python scripts/train.py -c configs/v24/smoke.json --synthetic --no_plot -n v24_smoke

# BikeNYC：另外两个数据集同理替换配置和 NPZ 路径
python scripts/train.py -c configs/v24/bikenyc.json \
  --train_npz data/BikeNYC/bikenyc_train.npz \
  --val_npz data/BikeNYC/bikenyc_val.npz \
  --test_npz data/BikeNYC/bikenyc_test.npz -n v24_coe

# 固定 T→S 对照：独立从头训练
python scripts/train.py -c configs/v24/bikenyc.json \
  --override_config configs/v24/ablations/fixed_ts.json \
  --train_npz data/BikeNYC/bikenyc_train.npz \
  --val_npz data/BikeNYC/bikenyc_val.npz \
  --test_npz data/BikeNYC/bikenyc_test.npz -n v24_fixed_ts

# 单独评估：CONFIG/CHECKPOINT 为对应训练输出中的实际路径
python scripts/evaluate.py --config CONFIG --checkpoint CHECKPOINT \
  --data_npz data/BikeNYC/bikenyc_test.npz

# CPU 行为测试，无需 pytest
PYTHONPATH=src:tests python -m unittest test_v24_coe
```

真实数据配置另有 `configs/v24/taxibj.json` 和 `configs/v24/chap_beijing.json`。默认读取各数据集 `random_mask/0.4` 的 train/val/test CSV。改变模式或缺失率时，需同时改变对应三个 CSV 路径。真实数据配置现为 80 epoch、每 2 轮验证、全量样本、最佳权重默认保存在 CPU 内存，训练结束恢复后完整测试一次；合成 smoke 仍保留一轮及 checkpoint 检查。需要独立 evaluate 或路径干预时事先设置 `train.save_best_checkpoint=true`。

批量实验使用 `python scripts/v24/run_experiments.py --gpu 0`，流程沿用 v23 后期的统一配置、指纹隔离、完成检查、跳过完整任务、结果汇总。默认 6 组 pilot，后续核心对照用 `--study core`；详细说明见 [实验设计](v24_COE_experiment_design.md)。

## 消融与结论边界

| 覆盖配置（`configs/v24/ablations/`） | 要检验的问题 |
|---|---|
| `fixed_ts.json` / `fixed_st.json` | 学习顺序是否胜过固定时空顺序 |
| `fixed_tt.json` / `fixed_ss.json` | 收益是否只来自增加同类处理深度 |
| `initial_router.json` | 同数量 router 仅读初始状态，验证重路由价值 |
| `soft.json` | 迭代软混合与离散链的差异 |
| `shared_only.json` / `routed_only.json` | 共享专家是否掩盖方向专家的作用 |
| `weak_mid.json` | 弱中间监督是否帮助或妨碍专家分工 |
| `no_route_balance.json` | 显式保持第一版不使用路由正则（等同当前默认） |

这些对照保留相同模块参数布局，实际激活的模块和计算量不同；需分别报告参数、训练耗时和推理耗时，不能仅按参数量宣称完全等成本。现有单轮并行 T/S 对照和冻结专家输入的状态对照见实验设计；当前统一用种子 7、80 epoch 进行机制筛选；仍需实测计算成本，多个随机种子的稳定性验证后续另行安排。

现有 `fixed` / `random` 都选取空间节点并沿整段时间广播；`random` 是每窗口重新选择缺失节点，不是时空点独立随机缺失。该协议可以验证当前任务上的效果，但不足以完整验证所有顺序动机。CSV loader 支持 `T×H×W` 列，后续应加入随时间变化的短/长缺口和区域缺失协议，保持归一化与数据划分不变。

当前交付是可运行实现及正确性验证，不包含完整数据集训练结论。

## 初版已完成的验证（加入路由辅助损失前）

- CPU 全仓库 61 项测试通过，包含 22 项新增 v24 行为测试。
- 合成数据完整一轮训练、验证、最佳 checkpoint 保存/重载和最终测试通过；独立 `scripts/evaluate.py` 重载后复现验证指标。
- TaxiBJ、BikeNYC、CHAP 各取真实训练集的前 2 个样本及其离线 mask，完成一次优化和评估；前向、反向、指标均为有限值，两个 router 有非零任务梯度，观测值保持不变。该检查仅验证数据兼容性，不能作为泛化性能结果。
- CPU bfloat16 autocast 下，hard / soft / fixed 前向与反向通过。本次环境未提供可用 CUDA，尚未验证 GPU 训练速度与显存。

上述结果没有检验动态链是否优于固定顺序；需使用本文的消融配置完成独立训练后再评价。
