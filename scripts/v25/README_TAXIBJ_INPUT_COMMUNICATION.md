# TaxiBJ I01–I05：初始尺度入口、跨轮通信与一致性

原 I01–I03 后追加 I04、I05 两次训练；原始参照为 `outputs/v25/experiments/first_validation/659d3ba9678e1dca` 的 B2，A2/B1只作为辅助冻结参考。各组均从头训练，不是把已训练B2继续微调。本次只新增代码和配置，不启动训练、不执行GPU预检、不改写已冻结三组队列。

|组|初始H0|尺度计算/观测注入|跨轮通信|问题|
|---|---|---|---|---|
|I01|仅F编码；原融合投影的M/C输入槽置零|固定CMFF，各轮注入原生对应尺度可见观测|完整受控差分|改变多尺度观测首次进入状态的时机，是否改变泛化退步|
|I02|原B2的F/M/C融合|每轮每样本硬选一个C/M/F尺度，读取选中尺度观测|完整受控差分|自由尺度与差分是否优于B2固定CMFF|
|I03|原B2的F/M/C融合|固定CMFF|完整受控差分＋X05自适应预测历史池|已有历史读取是否改善多尺度差分的表现|
|I04|原B2的F/M/C融合|固定CMFF|完整受控差分；无历史池|最终F聚合到M/C，对完整粗尺度真值计算辅助损失，作为一致性监督基线|
|I05|原B2的F/M/C融合|固定CMFF|完整受控差分；无历史池|根据可见F/M/C构造独立剩余区域，学习非负一致性校正分配|

## 固定配置与边界

- TaxiBJ，四类混合mask（random_point/node_outage/temporal_gap/spatial_region），缺失率0.4，seed7，batch32，100 epoch，每5轮验证。用户明确选择沿用旧B2的100轮方案，不使用50轮固定学习率。
- AdamW与原B2一致，1e-3余弦到3e-4、AMP、严格确定性。ID val MAE选best；保存best/last，last含优化器/调度器/scaler/RNG。GPU1串行，保留GPU日志，不启用周期线程栈采样。
- F/M/C完整数据预先从F按2×2、4×4空间求和构造，各尺度独立抽观测mask；模型只读可见值，编码时按面积缩放。F/M/C输入形状为[B,2,12,32,32]/[B,2,12,16,16]/[B,2,12,8,8]。
- 所有组 `data.input_scales=FMC`。I01只设 `model.coe.initial_observation_scales=F`，不能把dataset改成F（那会全程抹掉M/C，成为旧A2）。H0使用同一1×1融合投影，M/C特征槽为零；不新增随机层、不改原有参数键与公共初值。
- I01路径：F观测编码→F-only H0→C计算并注入C观测→M计算并注入M观测→F→F。首轮没有可用历史差分；第二、三、四轮均保留原±0.1条件门和RMS匹配差分。
- I02沿用S01的硬尺度、所选分支straight-through估计，无软预热；四轮各执行原生Top2、共8次专家调用。不是每轮同时执行三个尺度，也不是自由专家数。H0已含三尺度。
- I03直接复用 `HistoryReader` 自适应绝对预测读取，保留原生CMFF注入和专家调度，不复用v24旧输入执行路径。池子每个forward初始为空，四轮读取条数为0/1/2/3；写入每轮F网格预测，以真实可见F覆盖后detach。池子不跨batch、不读真值、不含当前或未来轮。
- 历史读取器按位置softmax混合旧预测，打分读取归一化历史、最近预测、两者绝对差、年龄和F可见覆盖。零初始化的注入MLP以tanh、0.1×当前隐藏RMS为界，仅修正F缺失位置的专家输入。历史注入加在受控差分修正之后；当轮router仍读未修正状态。
- I03新增历史参数，公共骨干逐张量核对旧B2初始化；新增参数独立fork RNG，不改变公共参数。零初始化时I03预测等于B2。I01–I03没有乘积差值交互、三平面、FFN、第二视图、一致性或中间监督。
- 六套ID/OOD及mask复测严格复用B2协议；mask复测不是训练种子复验。I01改变输入时机，若改善不能单独证明差分的因果根源，仍缺相同入口的无差分对照；I03也不能单独证明历史池与差分互补。
- 原B2存在历史中断/换卡记录；新实验保持GPU1。公平比较核对数值配置、数据、初始化、协议，不能宣称执行历程完全相同。

## I04：完整粗尺度目标损失

配置为 `model.coe.observation_consistency={"enabled":true,"mode":"full_target_loss"}`，`loss.lambda_observation_consistency=0.1`，`data.return_multiscale_targets=true`。仍按原B2训练，不添加历史池。

1. 最后一轮得到F预测，以原始可见F值覆盖对应位置，形成完整F结果。
2. 使用与数据生成相同的2×2、4×4空间求和得到预测M/C；不聚合时间维，不重新生成目标。
3. 分别对预处理数据中的完整 `x_m_gt`、`x_c_gt` 计算**全部位置**的L1，求和尺度分别除面积4、16，取两个尺度的平均作为一致性损失。
4. 总损失为原F缺失位置L1＋原候选均衡项＋0.1×一致性损失。只监督最终补全；不监督中间轮。

完整M/C真值只返回给训练/验证损失，不进入模型前向，也不作为测试校正信息。这里按用户要求使用完整粗尺度目标，**没有**仅在可见M/C位置计算。I04不增加模型参数；最终输出不强制满足粗尺度等式。均值型数据不除区域面积，但本批固定TaxiBJ求和数据。

## I05：可见观测下的自适应剩余区域校正

配置为 `model.coe.observation_consistency={"enabled":true,"mode":"residual_projection","weight_bound":2.0}`。不打开额外M/C目标返回，不增加粗尺度辅助损失，不添加历史池。

1. 对可见M，用其总量减去内部可见F，得到该M区域尚未观测位置的剩余总量。
2. 对可见C，先扣除已观测M整块总量，再扣除其他区域内可见F，只约束未被上述M约束覆盖的未知F。若全部被更细观测解释，不重复校正。
3. 在互不重叠的剩余未知区域内，以骨干最终F预测为起点，求解带非负条件的加权最小改动投影，使每个区域的未知值之和等于对应剩余总量。
4. 共享轻量权重头读取最后隐藏状态、按观测RMS归一化的F预测、上采样到F的F/M/C可见mask，生成 `exp(2*tanh(logit))`，范围为 `exp(-2)` 至 `exp(2)`；通过原F任务损失学习分配。末层零初始化，初始权重为1，对应固定欧氏投影，**不是**零修正，也不声称初始输出等于B2。

可见F保持原值；没有任何可见粗尺度约束的缺失位置保留骨干预测，因此只有受约束的位置执行非负投影。权重头使用独立fork RNG，保证原B2公共参数初值不变；新增参数为 `main_branch.x_consistency.*`。模型不读取 `x_f_gt/x_m_gt/x_c_gt`。当前适用于精确、对齐、非负、聚合相容的观测，不把它描述为支持任意噪声或非对齐数据的通用求解器。

I04使用全部粗尺度训练目标，I05只使用可见观测；因此两组**不构成等监督的纯软约束/硬约束对照**。虽然M/C完整目标由既有F真值确定性聚合、没有新原始数据，I04仍改变监督目标及权重。若后续有收益，还需可见观测软损失、固定联合投影和等容量普通修正等控制。聚合残差变小不等于F补全MAE变小，更不证明稳定泛化。

## 准备、顺序追加与恢复（以下命令仅供后续启动）

现有三组冻结suite为 `outputs/v25/experiments/taxibj_input_communication/ea1ca776ade869d6`。它的计划、源码和初始化保持不变；不能把新源码塞进旧snapshot，也不能重建五组队列来假定旧run已完成。

只准备新增两组时使用：

```bash
python scripts/v25/run_taxibj_input_communication.py --methods I04 I05 --prepare-only
```

命令会生成独立的两组冻结suite，保留原B2参考。以后只通过该suite的 `source_snapshot/scripts/v25/run_taxibj_input_communication.py --suite <suite>` 启动或恢复；`--suite` 与 `--methods` 不可同时使用。无 `--methods` 的新队列包含I01–I05全部五组，**不是**接续既有三组的入口。

需要严格接在原三组后面时，以下队列脚本可供之后授权启动时使用。`set -e` 确保原队列未成功完成不会提前训练后续组；旧队列已有的完成凭据由其冻结入口检查并跳过。

```bash
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
previous_suite=outputs/v25/experiments/taxibj_input_communication/ea1ca776ade869d6
python -u "$previous_suite/source_snapshot/scripts/v25/run_taxibj_input_communication.py" --suite "$previous_suite"
followup_suite=$(python scripts/v25/run_taxibj_input_communication.py --methods I04 I05 --prepare-only)
mkdir -p "$followup_suite/launcher_logs"
python -u "$followup_suite/source_snapshot/scripts/v25/run_taxibj_input_communication.py" --suite "$followup_suite" 2>&1 | tee -a "$followup_suite/launcher_logs/queue.log"
```

将队列脚本作为专用tmux的主命令运行；训练、六套评估与正式报告完成后主命令退出，会话自动关闭。不要向常驻shell仅发送训练命令，不要关闭其他会话。本次没有执行上述命令。

报告逻辑按实际选定的2/3/5组审计完成状态，逐组列出全部ID/OOD、训练曲线、参数与成本、一致性诊断和失败信息，明确监督差异。CPU验证覆盖目标隔离、残余区域约束、边界mask、梯度与恢复；正式GPU预检仍在以后启动时检查batch32原生与TA/ST路径、AMP梯度和8次稀疏执行，并增加输出约束核验。预检与CPU测试不是实验结果。
