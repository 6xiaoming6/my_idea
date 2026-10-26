# v25 原生补全 baseline：混合缺失与未见缺失泛化

本批使用独立数据桥接器，不改作者模型源文件，不添加本课题的投影、历史池、差分或 M/C 损失。与 I29 作同为 F-only 的比较；I25 有额外 M/C 观测，仅单列参考。需要日历的原模型可读取真实时间戳，报告明确标注，不声称所有输入字段完全相同。

## 候选与论文证据

|方法|作者源码|与目标的关系|本批状态|
|---|---|---|---|
|T1|[Oppenheimerdinger/T1](https://github.com/Oppenheimerdinger/T1)|[论文](https://arxiv.org/abs/2602.21043)研究未见缺失模式泛化，优先关注|原模型预检后训练|
|ImputeFormer|[tongnie/ImputeFormer](https://github.com/tongnie/ImputeFormer)|[论文](https://arxiv.org/html/2312.01728v3)含跨缺失率、跨序列长度零样本测试；与未见组合并非同一个问题|原模型预检后训练|
|STAMImputer|[RingBDStack/STAMImupter](https://github.com/RingBDStack/STAMImupter)|[论文](https://arxiv.org/html/2506.08054v1)含块缺失叠加随机点缺失；不能等同于同一模型面对未见组合|真实日历已恢复；原模型预检后训练|
|PAST|[Hanwen-Hu/PAST](https://github.com/Hanwen-Hu/PAST)|[论文](https://arxiv.org/html/2511.13414v1)涵盖随机、fiber、block；out-of-sample 不直接等于未见缺失模式|真实日历已恢复；原模型预检后训练|
|GRIN|[Graph-Machine-Learning-Group/grin](https://github.com/Graph-Machine-Learning-Group/grin)|图循环补全强参考；本批额外用统一协议测未见组合|原模型预检后训练|
|SAITS|[WenjieDu/SAITS](https://github.com/WenjieDu/SAITS)|自注意力补全参考；不宣称原论文已验证相同 OOD|原模型预检后训练|
|BRITS、MRNN|[SAITS 仓库中的实现](https://github.com/WenjieDu/SAITS/tree/main/modeling)|循环补全参考；明确使用 SAITS 作者提供的实现，不冒充各论文原始作者源码|原模型预检后训练|
|RITS|[SAITS 仓库](https://github.com/WenjieDu/SAITS)|单向循环参考，属于 BRITS 相关组件，不计为独立新论文|原模型预检后训练|
|MPGRU、BiMPGRU|[GRIN 仓库](https://github.com/Graph-Machine-Learning-Group/grin)|图消息传播 GRU 参考，属于同一仓库变体|原模型预检后训练|
|SPIN|[Graph-Machine-Learning-Group/spin](https://github.com/Graph-Machine-Learning-Group/spin)|[论文](https://arxiv.org/html/2205.13479)采用稀疏时空注意力，适合不规则可见观测|源码已下载，尚未完成原生框架桥接，不列入已启动训练|
|MPIN|[XLI-2020/MPIN](https://github.com/XLI-2020/MPIN)|多属性传感器数据流补全|流式更新与当前独立窗口协议需另行审查|
|LCR|[xinychen/LCR](https://github.com/xinychen/LCR)|低秩与局部时间结构补全|当前仅12步、非连续窗口与长周期设定差异大，暂不强行拼成时间线|

共安排 11 个原生模型/变体，不能将其写成 11 篇独立论文。本批实验的目标是实测统一缺失协议下的适用性，不能把候选论文的缺失设定说成本课题的六套协议。

## 数据与六套评价

- TaxiBJ 原清洗划分：train 2452、val 342、test 707 窗口，每窗口 12 步，两通道、32×32 网格。不拼接窗口、不拆为 8×8 子网格。
- 训练为四族混合 mask 0.4：random_point、node_outage、temporal_gap、spatial_region。沿用 I25/I29 的冻结数据、mask 实现和 seed；每5轮按 ID 验证 MAE 选 best。
- 同一个 best 依次测 `in_distribution`、`unseen_combinations`、`unseen_geometry`、`unseen_triple`、`unseen_combinations_repeat`、`unseen_geometry_repeat`。二元为 node+space、time+space、node+time；未见形态为 moving_region、multi_block；三元为 node+time+space。
- 不依据测试集调参、重新训练或选择检查点。复测只换 mask seed，不是训练种子复验。记录缺失处 MAE/RMSE、各族指标、有效计数和 F mask 哈希。
- OOD 三类别等权：先合并二元两次复测、形态两次复测，再与三元等权。统计量仅由训练集可见 F 计算；输入隐藏槽先清零。评估与模型调用分离，模型推断不接触隐藏真值。

## 日历时间映射

`data/TaxiBJ/TAXIBJ.ext` 含 59006 行、21 个字段，包括 `time`、温度、风速和天气。时间戳覆盖 2013—2016 年；不能按 ext 行号直接与已清洗窗口对齐。

`v24_clean_abc_20260917/manifest.json` 保存了每个清洗窗口的原始 `frame_ids` 和 `raw_timestamps`。v25 来源 manifest 的 SHA256 与该文件一致，且三份 v25 的 F 数组与清洗源逐项相等。3501 个窗口的全部时间均在 ext 中找到；每个窗口仍为连续半小时步长。

ext 有29条重复时间戳记录，涉及验证集；仅提取时间时不产生日历歧义。保留原文件 `Z` 标记的 UTC 时钟，不擅自假定或转换为北京时间。PAST 按作者编码使用日、星期、小时、分钟/15（模型实际嵌入星期/小时/分钟）；STAM 使用星期与每天48个半小时时段。原模型本来就接收日历，不新增模块。温度、风速和天气不使用。

映射另存 `data_bridge/calendar.npz` 和 `calendar_audit.json`，不修改原数据、已有训练或数据划分。

## 原生模型与训练目标

|方法|维度与原生参数|训练监督|
|---|---|---|
|SAITS|全网格2048特征；2组、d_model256、d_inner128、4头|原生 ORT+MIT|
|T1|全网格2048特征；PEMS03配置128头、2+2块、卷积核71/31|缺失位置 MSE|
|BRITS/RITS/MRNN|全网格2048特征、hidden64|原生观测重建；BRITS加原生双向一致性，MRNN原生RMSE|
|GRIN|两通道作为独立1024节点样本；hidden64、4邻接|标准化观测 L1；5%观测隐藏；最终预测与4个中间预测损失相加，各权重1|
|MPGRU/BiMPGRU|同样完整网格与hidden64|原生观测重建和5%观测隐藏|
|ImputeFormer|每通道1024节点、input/output_dim1、32+96嵌入、FF256、4头、3层、投影维8|标准化缺失 L1 +0.01原生 Fourier 项|
|PAST|每通道1024节点、hidden64、3层、order1、alpha0.1、dropout0.1|原生观测 MSE +主预测 detach 后的残差辅助 MSE；使用原生点隐藏路径的10%观测遮蔽|
|STAMImputer|原入口MoE_v2；每通道1024节点、4层、4头、embedding32、dropout0.15；原生邻居采样|物理量空间缺失 MAE，保留作者训练目标排除零值的规则；正式评价仍包含所有缺失位置的零值|

图邻接仅来自四邻接坐标；STAM 按作者最短路近邻方法选取邻居。STAM 的 db1 一层小波只分解遮蔽后的输入，不能对完整真值做分解。STAM 原始前向在 eval 下仍随机采样，保留该行为，预检污染对照和恢复对照使用一致 RNG。

ImputeFormer 按完整源训练数据的可用性设置 `original_mask=1`，混合 mask 是人工隐藏；因此其原生 Fourier 项仅读取预测，不会把隐藏训练值拼入输入。该原生 `fftn` 包含 batch 维，改变 batch 也影响正则化项；不修改此项，在结果中披露。

统一预算为 seed7、100 epoch、每5轮验证、Adam、1e-3余弦至3e-4、梯度裁剪5、FP32。保留原生结构和损失并不意味着完全复现论文训练超参数，不把本批结果写成作者最佳精度。日历、原生监督、通道组织和 batch 差异均需披露。

## 执行、检查点与监控

入口：`imputation_benchmark/scripts/train/run_v25_native_baselines.py`。新建套件冻结源码、作者仓库 commit、模型配置、数据/日历 SHA256；独立依赖目录为 `imputation_benchmark/.native_baseline_deps`，不改现有 conda 环境。

GPU1 串行，等待已有 I29 及其评估队列释放锁。每模型先在完整32×32网格预检 batch32，OOM依次降为16、8、4、2、1。预检覆盖前向、反向、优化器状态、隐藏数据污染与检查点恢复。只调整 batch，不缩小隐藏维度或删除模块；非显存错误保存日志并停止，避免无限重试。batch1仍OOM则明确记为跳过并继续其他候选。

每个 run 保存 best/last、训练曲线、预检日志、每15秒GPU状态；last 包含优化器、学习率、随机数和数据加载器状态。已完成组跳过，中断组从完整检查点恢复。专用 tmux 在训练、六套评价及正式报告结束后自动退出。

队列自动在 `experments_report/` 导出报告；未完成时标注阶段，全部结果、失败、batch回退和成本都披露。没有实测结果前不宣称任何候选优于 I29/I25。
