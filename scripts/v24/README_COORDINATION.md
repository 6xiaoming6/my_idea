# V1–V4 三尺度输出协调

## 方法与复现

全部基于U02：四轮共享八专家T/S/TD/SD/TA/ST/TL/SL，原生Top2和原始softmax；固定CMFF，C/M/F=8/16/32（TaxiBJ），时间12。±0.1 RMS匹配状态差分修正专家输入，direct，无completion feedback。

|组|方法|对照|
|---|---|---|
|V1|原U02，无新模块|本批参考|
|V2|第一轮C和第二轮M的原生专家融合输出分别解码区域缺失均值；各0.05辅助L1；不改最终输出|V1，额外监督|
|V3|V2＋最终F输出依次做C、M协调；各修正区域总量差异的10%，缺失位置均匀分配|V2，协调机制|
|V4|V3＋共享16通道1×1分配MLP，缺失位置区域softmax|V3，条件分配|

区域头为最终decoder的独立参数副本，在公共骨干初始化后复制；V4分配头使用隔离RNG、零输出层初始化。V1与U02逐值一致；V2的初始最终输出与V1一致；V3/V4初始一致。区域头接收原生分辨率的专家输出，不从上采样后状态重新降采样。

分配输入：最终细尺度hidden、当前预测、mask、区域缺失均值差（最近邻广播）、尺度factor/4。网络Conv1×1→GELU→Conv1×1，隐藏16，两个尺度共享，输出各数据通道的分配logits。没有额外接受门、候选专家执行或反馈。任务梯度通过校正更新区域头/骨干，V4输入保留梯度；V3→V4包含参数增加，不声称等参数。

对每个时间和流量通道，A_B=sum(observed)+n_missing*mu_B，residual=A_B−sum(observed/current missing predictions)。缺失位置更新0.1*w*residual；观测输出不改，最终x_comp固定真实观测。C后重新计算M残差。软协调不保证两个层级同时完全一致；不是物理流量守恒。0.1不是单点绝对变化上限，不增加裁剪或非负化。

区域辅助目标是缺失位置均值。只监督缺失标签全部有效的区域；部分自然缺失区域不以子集均值冒充全区域目标。目标只进入loss和诊断，模型仅接收可见观测/mask。辅助损失按有效区域等权；主任务按有效缺失位置。原L1＋0.01candidate balance不变。

默认TaxiBJ random0.4、四基础缺失混合、训练每epoch重采样，沿用原时间切分和mask seed。seed7同时设置模型/loader。GPU0单卡串行，batch32，100epoch，val5，AdamW lr1e-3 cosine→3e-4，wd1e-4、clip1、AMP、无早停。best仅ID验证MAE，保存best/last以及优化器/调度器/AMP/RNG。全部原六套固定评估，新方法不改变mask协议。

```bash
python -u scripts/v24/run_coordination_exploration.py --dataset taxibj --gpu 0 --epochs 100 --batch-size 32
# 指定已有冻结队列，代码更新后也可恢复：
python -u scripts/v24/run_coordination_exploration.py --suite /absolute/path/to/suite --dataset taxibj --gpu 0 --epochs 100 --batch-size 32
```

支持variants/dry-run/summary-only，数据集可切换BikeNYC（尺寸须被4整除）。冻结全部源码/配置、数据SHA256与时间戳；已完成跳过，缺评估补跑；中断从last完整恢复，失败停止。

## 诊断与结论边界

精确累计raw（关闭协调）、after_c、after_m的MAE/RMSE及分缺失族误差；两尺度区域均值误差、去均值细节误差、分配熵、修正幅度、观测变动；原路径/专家统计、参数量/时长/显存照常保留。四轮解码诊断保留真实专家轮次输出，最终协调作为独立阶段，不宣称逐轮误差单调。区域误差按完整有效区域等权；细节误差按有效缺失点计数。主指标采用split精确分子分母。

V3/V4同一次forward的raw输出即同一检查点关闭输出协调的推理消融，无需重训/额外专家。V1未设置区域头，相关诊断不适用。专家调用8次、面积代理4.625，不代表所有组严格等FLOPs；新预测头单独计入参数/时长。

日志遵循README_LOGGING：控制台batch tqdm只显示train loss/mae/rmse，文本每epoch精简；详细诊断每5epoch压缩，最终评估保留完整分布。每组完成更新阶段报告；四组全部完成后自动导出experments_report中的正式报告，顺序为方法→结果/曲线/诊断→分析/局限/建议。本批只有seed7，任何优势仍需配对种子复验。
