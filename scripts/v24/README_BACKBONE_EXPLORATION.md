# W01–W30 固定CMFF＋有界差分探索

## 实验方法

共同骨干：U02，CMFF=8/16/32（TaxiBJ，T12），4轮共享8专家，原生Top2、original softmax、direct，无completion feedback。原±0.1 RMS差分门不变。默认TaxiBJ random0.4、四基础缺失、原切分与mask seed，GPU0单卡串行，seed7，batch32，100epoch，val5，AdamW1e-3 cosine至3e-4，wd1e-4、clip1、AMP，无早停。best仅ID验证，保存best/last。

|组|做法|参考|
|---|---|---|
|W01|CMFF＋±0.1差分，严格确定性基准|W01|
|W02|W01同seed完整重放，非独立训练种子|W01|
|W03|零初始化尺度身份向量|W01|
|W04|零初始化轮次身份向量|W03|
|W05|尺度独立零初始化通道仿射|W01|
|W06|轮次独立零初始化通道仿射|W05|
|W07|跨尺度/轮共享rank8残差适配|W01|
|W08|三尺度独立rank8适配|W07|
|W09|四轮独立rank8适配|W08|
|W10|尺度rank8＋局部覆盖率调制，初始倍率1|W08|
|W11|四轮可见重建L1平均×0.02|W01|
|W12|观测残差不扩散，detach，修正专家输入|W01|
|W13|观测残差3×3×3归一化传播，修正专家输入|W12|
|W14|传播残差仅修正含缺失位置|W13|
|W15|W14保留残差内容梯度|W14|
|W16|传播残差只修正Router logits|W01|
|W17|传播残差同时修正专家输入和Router|W14|
|W18|W14＋逐轮可见重建监督|W14|
|W19|前8窗口增加rate0.5嵌套视图，补全L1×0.25|W01|
|W20|增强＋共同缺失预测一致性×0.05|W19|
|W21|一致性教师改为0.99 EMA，评估学生|W20|
|W22|一致性按教师可见误差启发式加权|W20|
|W23|最终F的C/M区域均值监督，各0.025，区域等权|W01|
|W24|区域均值监督改按有效缺失数量加权|W23|
|W25|最终补全的相邻时间差分L1×0.05|W01|
|W26|最终补全的空间邻接差分L1×0.05|W01|
|W27|A+B，仅ID验证冻结候选，从头训练|W01|
|W28|A+C，仅ID验证冻结候选，从头训练|W01|
|W29|B+C，仅ID验证冻结候选，从头训练|W01|
|W30|A+B+C，仅ID验证冻结候选，从头训练|W01|

## 机制定义

尺度适配在投影归一化Z之后：身份Z+e；仿射Z+gamma*Z+beta，均零初始化；rank8 Z+0.1*B*GELU(AZ)，B零初始化。覆盖率版乘2sigmoid(ac+b)，参数零初始化、倍率1。空间1×1投影不额外激活专家。各组总/可训练参数分别报告。

反馈从第2轮起，R=M*(Xobs−上一轮未覆盖观测的解码值)/max(RMSobs(Xobs),1)。归一化尺度detach；除W15外残差内容detach。传播为3×3×3残差求和除观测数、零填充；无观测邻域返回0。输入反馈是0.05 RMS(H)*tanh(Conv1×1→GELU→Conv1×1)，隐藏32、输出零初始化，再乘有效支撑。missing类还乘跨通道平均缺失率。Router读取原路由特征与R/coverage全局/缺失均值，隐藏64零输出头，0.1tanh修正logits。候选均衡使用实际修正logits，公式不变。原差分机制、完成通道不改，历史只在本次forward。

第二视图每batch前8窗，从已有可见位置沿四基础族之一的几何排序追加到rate0.5，不引入散点补丁；独立numpy RNG保存恢复。原已有缺失超过0.5时不揭露数据，保持其原缺失量；此边界不影响默认random0.4。额外学生L1×0.25，不额外计balance或结构loss。一致性×0.05只在两视图共同有效隐藏位置；普通教师是原视图预测detach，EMA系数0.99仅成功更新后同步，评估学生。W22权重exp(-local_observed_error/window_observed_error)，tau下限1e-6，无邻域观测退回窗口误差，共同隐藏位置归一化均值1；下溢时退回均匀权重，不使用真值置信度。

结构监督：观察重建四轮平均L1×0.02；区域监督只聚合最终F的有效隐藏预测与标签，C/M各0.025，区域等权或按有效隐藏数加权，无早期头/输出校正；相邻差分用x_comp，两端标签有效且至少一端有效隐藏，时间或水平/垂直空间邻边合并按边数平均，×0.05。

组合从尺度A、反馈B、训练C各选最低ID val候选；优先比W01改善≥1%，并列较少可训练参数、较小编号。无人达标仍取最小并标记未过初筛，证据哈希冻结，不读测试。顺序原差分+反馈→尺度投影归一化→适配→原生专家，损失按来源相加。从头训练，不加载入选权重。

## 确定性、恢复与检查

严格torch deterministic、cudnn deterministic、benchmark/TF32关闭，CUBLAS_WORKSPACE_CONFIG=:4096:8，SDPA固定math。隐藏空间池化reshape/reduce，上采样固定align_corners=False双线性权重矩阵，FP32累加再恢复dtype。新后端只属于W，不改旧模型默认执行。公共参数由W01冻结到common_initialization.pth，各组先校验同seed构造逐值相同，再复制这份公共模板；额外模块隔离RNG；initialization.json保存公共/完整状态SHA256。

W01/W02全100epoch优先完成，replay_audit.json逐epoch记录主指标、参数/optimizer/scaler/scheduler哈希及原始数据顺序摘要。两个重放必须完全一致，否则停止后续。它们不是两个独立种子。last保存这些审计、numpy额外mask RNG、EMA状态、成功更新数，以及原optimizer/scheduler/AMP/全局loader随机状态。中断按完整epoch恢复，已完成跳过、缺评估补跑。

执行顺序W01/W02→重放验收→W03–10→W11–18→W19–26→冻结A/B/C→W27–30→确认评估。保留六套协议，W19–22及组合的训练可能暴露组合缺失，不能再全部称未见组合。最终对W01/A/B/C/组合使用新mask20261111/12/13＋test偏移确认，不是跨数据集。只有seed7，ID test降≥1%且val改善视作单种子信号；主要OOD退步>5%必须标记。额外前向和参数不是等计算。

## 启动和报告

```bash
python -u scripts/v24/run_backbone_exploration.py --dataset taxibj --gpu 0 --epochs 100 --batch-size 32
# 代码修改后仍使用旧冻结队列继续：
python -u scripts/v24/run_backbone_exploration.py --suite /absolute/path/to/suite --dataset taxibj --gpu 0 --epochs 100 --batch-size 32
```

支持--variants、--dry-run、--summary-only；请求组合自动加入全部26组初筛；任意子集加入W01/W02作为复现参考。配置不匹配拒绝恢复。数据集可切换BikeNYC，按网格1/2、1/4缩放。

控制台只任务切换与train epoch 当前/总数 batch tqdm(train loss/mae/rmse)。日志使用精简epoch文本、每5epoch压缩诊断、最终完整评估。每分块/失败生成阶段报告，全队列及确认完成生成正式报告到experments_report，顺序方法→全部结果/曲线/诊断→分析/局限/建议。不会为报告擅自加跑其他组。
