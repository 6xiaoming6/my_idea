# U01–U30：ID优先的跨轮通信与尺度路由

## 固定实验清单

|编号|做法|主要参考|
|---|---|---|
|U01|原生共享四轮CMFF|新基准|
|U02|RMS状态差分，门±0.1|U01|
|U03|同U02，门±0.25|U02|
|U04|原始状态差分，不RMS匹配，门±0.1|U02|
|U05|操作内变化U(E−Z)，RMS匹配，门±0.1|U02/U01|
|U06|U05消息内容detach，门输入不detach|U05|
|U07|操作变化摘要只修正专家Router，±0.1|U05/U01|
|U08|前两轮自由Top1直通、后两轮F，从头训练|U01|
|U09|前20epoch CMFF，21起四轮自由Top1直通|U01|
|U10|前20epoch CMFF，21起前两轮自由Top1直通、后两轮F|U08/U09|
|U11|U09改为尺度教师，无任务尺度直通|U09|
|U12|U10改为尺度教师，无任务尺度直通|U10|
|U13|前20epoch CMFF，21起三选二尺度softmax融合|U14|
|U14|前20epoch CMFF，21起固定CM/MF/CF/MF等权|U13|
|U15|U11改为尺度分辨率非递减、末轮F|U11|
|U16|U12加3个跨轮共享尺度身份向量，零初始化、21起启用|U12|
|U17|ID第一通信候选＋第一Top1尺度候选|U01及单模块|
|U18|CMFF原生Top2，双响应摘要detach后融合logit差修正±0.1|U01|
|U19/U20|第一通信候选seed17/27|U29/U30|
|U21/U22|第二通信候选seed17/27|U29/U30|
|U23/U24|第一Top1尺度候选seed17/27|U29/U30|
|U25/U26|第二Top1尺度候选seed17/27|U29/U30|
|U27/U28|组合U17 seed17/27|U29/U30|
|U29/U30|U01 seed17/27|新配对基准|

默认TaxiBJ random0.4，四基础mask、原时间切分和mask seed20260917，训练每epoch重采样、val固定。GPU0单卡串行、batch32、100epoch、val5，AdamW1e-3余弦3e-4，wd1e-4、clip1、AMP、无早停。L1+0.01candidate balance；仅教师组加0.1CE。八专家T/S/TD/SD/TA/ST/TL/SL，共享四轮，每激活尺度原生Top2、original组内融合、direct，关闭completion feedback、acceptance、旧partner及其他shared分支。

F/M/C=32/16/8（BikeNYC按输入网格/1,/2,/4），T不变。公共骨干先使用原构造器初始化，所有新增模块在隔离CPU RNG中构建，同seed公共参数和后续全局RNG精确一致；每个组从头独立训练100epoch，不复用20epoch训练预算。尺度头前20epoch被绕过但参数预先注册在优化器中，首次有梯度时才创建Adam状态。

通信：Hexpert=H+αtanh(g)Δ。门读取当前/上一输入状态的全局与缺失池化摘要，输出层零初始化。RMS按每样本每通道THW计算，比例detach、分母下限1e-6。操作消息是在实际选择尺度中，以投影归一化后真实专家输入Z与实际Top2融合输出E构造U(E−Z)，只保存当前forward历史。U06只detach该消息内容，保留门特征梯度。U07以当前H与消息摘要的零初始化MLP产生0.1tanh路由修正，无专家输入修正。

尺度：每轮头输入原Router特征、F/M/C历史计数/4、剩余轮数比例，零输出层及1e-3 CMFF偏置。自由为81路径，后两轮F为9路径，分辨率非递减且末轮F为10路径。约束通过合法候选屏蔽实现，强制非法尺度报错。所有未选尺度不执行；U13/14每轮同一专家对执行于两个尺度。U13选中尺度组内softmax初始化为与U14相同的等权尺度对。面积代理不称FLOPs。

阶段epoch保存为model buffer，重载best后自动保留其阶段，eval不默认强制开放。若best≤20epoch，报告必须声明尚未经历开放阶段。训练恢复在下一完整epoch设置阶段，last包括模型/优化器/调度器/AMP/CPU-CUDA-loader-dataset RNG及教师轮次时钟。20/21切换不能因恢复而重新预热。

教师：第21epoch后每20个全局batch，最多前4个有效标签窗口，学习轮次循环（自由0–3，后两轮F为0–1，单调为0–2）。每窗口只比较合法候选，单候选跳过。实际前缀detach，从干预轮继续正常路由至最终预测，以训练监督mask内MAE给softmax负误差标签，温度max(0.1×合法候选均值,1e-6)。尺度头特征detach，任务无尺度ST，0.1CE只更新当前尺度头。试运行不修改主状态、buffer或全局RNG，推理不试运行。候选次数/耗时/梯度日志不能当总体推理成本。

选择仅用U01–16训练回执与best epoch的ID val日志；至少1%改善优先，其后按val/验证面积/可训练参数/编号排序。通信选择不同机制族（U02–04、U05–06、U07）各至多一种。Top1候选U08–12/U15/U16，U13/14/U18仅单seed探索。不足仍复验排名最高者，标记不通过初筛。冻结证据哈希，恢复仅读取，不重新选择。U17继承各模块原训练时序；U29/30在候选复验之前训练。

所有组原六套test，复验方法及对应seed7方法、U01/U29/U30用20261101/02/03+test偏移30000的新mask确认及rate0.2/0.6/0.8迁移。主要推进条件为三个seed ID测试均降低、平均相对降低≥1%；任何主要OOD退步>5%需突出披露，不用其反向改选择。旧S/M/G/D成绩只背景，不混用其他预算或数据集作公平参考。

```bash
python -u scripts/v24/run_id_priority_exploration.py --dataset taxibj --gpu 0 --epochs 100 --batch-size 32
python scripts/v24/run_id_priority_exploration.py --dry-run
python scripts/v24/run_id_priority_exploration.py --summary-only
```

支持taxibj/bikenyc、单个gpu、epochs、batch-size、variants。正式执行含固定20epoch阶段的组要求epochs>20。动态候选组自动补齐初筛与配对参考依赖。冻结源码/配置/数据size-mtime，失败停止，不隐式降batch或启用第二张卡。已完成训练跳过，缺评估补跑。改配置/代码创建新指纹，原队列不覆盖。

顺序U01–16→U18→冻结→U17→U29/U30→U19–28→确认。每分块阶段报告、最后正式报告自动导出到experments_report，方法→结果/曲线/诊断→分析/代价/建议。控制台仅任务切换和train epoch当前/总数的batch tqdm（train loss/mae/rmse）。本轮无硬截止，估计28–36小时训练评估，开发验收另计。
