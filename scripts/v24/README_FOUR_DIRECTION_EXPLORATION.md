# 四方向60组（M18 / S18 / G12 / P12）

本批为39次首轮探索、20次配对复验、1次组合探索。全部100epoch，GPU0单卡串行、batch32、val5。48小时为阶段节点，不是截止时间；禁止自动缩短epoch、改batch或启用第二张GPU。

## 方法与公共设置

TaxiBJ random0.4，四基础mask近似均衡、每epoch重采样；val固定。模型/loader seed7，复验17、27，mask seed20260917不随模型改变。保留原数据切分。AdamW 1e-3余弦至3e-4（最后epoch到达）、wd1e-4、clip1、AMP、无early stopping。L1+0.01原生candidate balance；仅教师组额外0.1候选交叉熵。八专家T/S/TD/SD/TA/ST/TL/SL、四轮共享、Top2、direct、original fusion；不启用feedback、旧搭档路由、接受门、shared旁路或强制尺度均衡。

F/M/C分别为原始网格/2/4倍池化，TaxiBJ32/16/8，时间12不变。旧文档C16在这里为M。单尺度内只执行被选中的两个专家；三选二为四次/轮，全三尺度为六次/轮。不同尺度及算子不能按激活次数宣称等FLOPs。

|编号|具体做法|对照|
|---|---|---|
|M01|四轮共享direct单F|基准|
|M02|四层独立direct单F|M01|
|M03|h_next=k*h+(1-k)*expert_out，逐轮sigmoid标量k初始0.1|M01|
|M04|专家输入h+tanh(g)*(h0-h)，g末层零初始化|M01|
|M05|专家输入h+tanh(g)*(prev-h)，门初始0.05|M01/M06|
|M06|同M05，门改sigmoid且同样初始0.05|M05|
|M07|专家输入h+tanh(g)*normalized(h-prev)，零门|M01|
|M08|同M07，差分内容detach|M07|
|M09|当前/历史/差分摘要经零初始化头修正Router logits，不改专家输入|M01|
|M10|M07+M09|M07/M09|
|M11|已选双专家响应的全局/缺失均值、身份one-hot、融合权重，64维消息传递到下一轮路由及FiLM|M01|
|M12|64维GRU逐轮读状态和差分摘要，零初始化路由/FiLM输出|M01|
|M13/14|M01，seed17/27|配对参考|
|M15/16|第一信息候选，seed17/27|M13/14|
|M17/18|第二信息候选，seed17/27|M13/14|
|S01|固定CMFF direct|基准|
|S02|自由三选一、selected-only ST|S01|
|S03|前20epoch按0.8p+0.2/3采样，其后argmax；评估argmax|S02|
|S04|S02+最终效果尺度教师|S02|
|S05|S02增加前轮尺度onehot、全局/缺失状态差RMS、前轮专家融合熵，共6项|S02|
|S06|S02+每尺度独立缓存及年龄，零门修正选中尺度输入、零头修正尺度分数|S02|
|S07|每轮三选二，组内尺度softmax|S08|
|S08|固定CM/MF/CF/MF，每对等权|S07|
|S09|全三尺度、学习softmax权重|S10|
|S10|全三尺度等权|S09|
|S11/12|S01，seed17/27|单尺度参考|
|S13/14|S02–06选中候选，seed17/27|S11/12|
|S15/16|S07或S09选中候选，seed17/27|S17/18|
|S17/18|对应S08或S10，seed17/27|匹配参考|
|G01/02|原R5 MMFF/FMMF早期8epoch扰动，seed17/27|旧R2/G03|
|G03|固定MMFF共享，seed27|G02|
|G04/05|R5策略改独立专家，seed7/17|旧R6/G06|
|G06|固定MMFF独立，seed17|G05|
|G07|CMFF的12排列训练epoch1–8，其余固定CMFF，direct|S01|
|G08|同G07但训练全程扰动，评估固定CMFF|G07|
|G09|仅epoch93–100扰动，其余固定CMFF|G07|
|G10/11|G07–09选中候选，seed17/27|S11/12|
|G12|M第一候选+S自由Top1候选，seed7|单模块；仅组合探索|
|P01|原生Top2、等权融合|M01|
|P02|28对分数z_i+z_j+零初始化修正，selected-only ST|M01|
|P03|P02+最终效果教师|P02|
|P04|P03修正头增历史/差分、前轮身份权重|P03|
|P05|原生第一专家+原生分数残差搭档，最终教师|P03|
|P06|P05教师改当轮误差|P05|
|P07|原生Top2，执行后响应摘要detach，融合logit差加0.5*tanh(delta)|M01|
|P08|P03无组合ST任务梯度，修正头只受教师监督|P03|
|P09/10|第一选择候选，seed17/27|M13/14|
|P11/12|第二选择候选，seed17/27|M13/14|

M门输入为当前与参考状态各自全局/缺失均值。M07每样本通道的delta按THW RMS归一化并匹配当前hidden RMS，比例detach、下限1e-6；M08额外detach差分本身。M09输入为当前、历史、差分三份摘要。所有历史因果且不跨forward，第一轮没有最近历史校正。

S01–06尺度头是legacy路由特征+已用F/M/C次数/4+剩余轮数比例，四头独立、末层零初始化、CMFF 1e-3微偏置。S05新增输入头零初始化。S06缓存初始D_s(h0)，只将选中尺度专家输出写入对应缓存，未选不执行；银行摘要+年龄供尺度头使用，专家输入先融合当前状态与缓存再做原state projection。G12先做M细网格输入修正、再做尺度内缓存修正。S07初始硬选择同S08且等权；S09初始等权三尺度。S03及G排列使用独立CPU RNG buffer，检查点保存，不改变mask/loader流。

P02–06/P08保持原生个体分数组内softmax；组合并列优先原生Top2。原生candidate边缘概率计算不改，均衡依然使用实际选择负载。P05当前实现搭档头读取原生路由特征和主专家身份，不额外执行主专家试提议；这是本轮简化分解对照。新模块在骨干构造之后初始化，公共参数保持相同seed下原始初始化。

## 低频教师与评估隔离

每20个全局训练batch选最多4个有效监督窗口，轮次循环1→2→3→4。尺度3候选；专家5个唯一候选，含原生/当前选择，再从原生Top4和独立随机流补齐。保存实际前缀的detach上下文，从干预轮开始试运行剩余轮次；只读取训练隐藏位置的有限真值，自然缺失无标签不计分。P06仅运行/计分当前轮。教师no_grad，不推进主RNG、cache或public forward状态；候选采样推进独立probe_rng并保存。

soft target = softmax(-candidate_mae / max(0.1*mean(candidate_mae),1e-6))。交叉熵系数0.1，输入和原生logits detach，只更新选择头；P08关闭主任务的组合ST梯度，但原生组内融合仍训练个体Router。记录教师耗时/候选数/头梯度。推理无候选试运行。

ID验证best唯一选checkpoint。额外OOD研发使用**val NPZ**，seed20261011/12/13+20000，不能用test NPZ顶替。参考和候选先筛ID验证比值≤1.02，再按J=0.5*IDval比值+0.5*三个OODval MAE比值均值排序；不够两候选时用未过门槛者补足并标明失败。并列按面积代理、总参数、编号排序。S07/S09分别相对S08/S10。冻结来源和证据hash；旧测试不进入selector。

所有模型六套旧测试。复验后对候选/参考与旧N5/R5/R6/R2增加20261021/22/23+30000新mask确认及rate0.2/0.6/0.8迁移，无重新训练。它们共享原测试时间窗口，不能当跨数据集结果。三seed共同改善ID及平均OOD才是推进信号；允许权衡为每seed ID≤1.02、平均OOD≤0.95、各OOD≤1.05。还需检查确认mask和旧强参考成本。G12单seed不替代复验。

## 入口、恢复和报告

```bash
python -u scripts/v24/run_four_direction_exploration.py --dataset taxibj --gpu 0 --epochs 100 --batch-size 32
python scripts/v24/run_four_direction_exploration.py --dry-run
python scripts/v24/run_four_direction_exploration.py --summary-only
```

支持--variants；动态组自动加入39组初筛依赖。静态顺序M01–12/S01–10/G01–09/P01–08，随后冻结选择，执行M13–18/S11–18/G10–11/P09–12，最后G12。每次新队列冻结src/scripts/config与数据文件size/mtime。选择及已解析配置不得随恢复改变；匹配完成回执跳过训练，缺评估补跑。

新训练入口每epoch原子保存last（模型/optimizer/scheduler/scaler/所有RNG/loader/历史日志/最佳权重）。中断中的不完整epoch回滚到上一完整epoch重新执行，不能宣称batch级恢复；SIGKILL也不会覆盖已完成的原子检查点。best仅ID验证选取。完整恢复不能更改epoch总数或学习率计划。首epoch前中断则从头开始该组。

控制台只任务切换及train epoch 当前/总数batch tqdm的train loss/mae/rmse。其他信息写logs。每个初筛分块、48小时后的下一组完成时生成阶段报告；最终统一确认后自动导出正式报告到experments_report。报告方法→结果→分析，不只写队列自动摘要。未完成/失败组保留状态。新模块、loss、probe都有独立回归与显存检查；OOM停止，不自动降batch或用GPU1。
