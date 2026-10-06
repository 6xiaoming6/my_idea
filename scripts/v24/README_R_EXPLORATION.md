# R1–R10：多尺度、未见缺失泛化与跨轮信息传递

## 每组具体做法

| 编号 | 基础模型与具体操作 | seed | 对照问题 |
| --- | --- | --- | --- |
| R1 | N3，四轮共享八专家原生Top-2 | 17 | 共享基准第二种子 |
| R2 | N5，固定CCFF，两粗两细 | 17 | 固定多尺度复现 |
| R3 | N6，配额约束的学习尺度选择，两粗两细 | 17 | N6泛化收益复现 |
| R4 | N6，硬前向始终CCFF，保留尺度概率的直通梯度 | 7 | 额外代理梯度是否有用 |
| R5 | N5，训练前8epoch每窗口等概率CCFF/FCCF，之后CCFF；评估始终CCFF | 7 | 早期路径扰动是否有用 |
| R6 | N2，四层独立专家加固定CCFF | 7 | 尺度收益是否依赖共享专家 |
| R7 | 四轮共享，每轮独立自由尺度Top-1，无配额 | 7 | 自由尺度选择是否有效 |
| R8 | N7，历史注意力改成所有可用历史的均匀平均，门控不变 | 7 | 内容检索是否必要 |
| R9 | N7，只读取最近一份更早历史，门控不变 | 7 | 简单历史校正是否足够 |
| R10 | N3，仅允许跨方向组合或ST搭档，在19对中取个体logits和最高者 | 7 | 简单组合约束是否优于原生Top-2 |

R1/R2/R3同时设置初始化和loader种子为17，mask种子不变。C为空间下采样2倍，F为原分辨率；TaxiBJ为16×16/32×32，时间长度均为12。所有新增模型从头训练。参考N2/N3/N5/N6/N7使用已完成的seed7 best checkpoint，仅补评估。

## 共同设置与复现

默认TaxiBJ、rate0.4、batch32、GPU0单卡串行、100epoch、val_epoch5；AdamW、lr1e-3余弦到3e-4、weight_decay1e-4（沿用bias/norm不衰减）、grad_clip_norm1、AMP。L1+0.01 candidate balance，mid/z为0；没有新增辅助损失。保存best.pth和last.pth，ID验证MAE选best。

八专家T/S/TD/SD/TA/ST/TL/SL，每轮Top-2。原始组内softmax融合，direct更新，关闭completion feedback、搭档路由、接受门、预热、路由噪声、额外shared分支和C3。R6专家按层独立，其余共享。R8/R9仅校正专家输入，Router仍读取未校正当前状态。

```bash
python -u scripts/v24/run_r_exploration.py --dataset taxibj --gpu 0 --epochs 100 --batch-size 32
python scripts/v24/run_r_exploration.py --dataset taxibj --dry-run
# 相同代码和参数重启：跳过已完成训练，补齐缺失评估。
# 可用 --variants R7 单独指定；所选组变化会生成不同队列指纹。
# --dataset bikenyc 可切换数据，当前没有固定的BikeNYC参考checkpoint。
tmux attach -t v24-r-exploration
```

入口位于根目录scripts/v24。配置在configs/v24/r_exploration，继承对应N配置再显式覆盖。旧B/N配置和日志不修改。

## 尺度与记忆实现

粗输入仅由可见观测、覆盖率、当前hidden、原mask支撑和位置生成。无观测粗格使用初始预测占位，coverage保持0。粗专家输出上采样后direct成为下一轮hidden；每轮只执行一个尺度上的两个专家。

R4的硬系数始终CCFF，训练使用hard+p-stop_gradient(p)，概率和预算特征与N6一致。它改变反向传播，不改变尺度前向路径。R5不优化尺度头，用独立随机流选择早期路径，其状态作为buffer保存在checkpoint中；不改变全局、loader或mask随机流。它不是对旧N6实际路径的精确重放。

R7尺度头逐轮独立，读取当前特征、已用粗轮比例、剩余轮比例；无候选屏蔽或次数限制。零输出层和逐轮微小偏置使初始前向为CCFF；之后可选择全部16条路径。只执行选中尺度，通过直通梯度训练，不执行未选尺度生成反事实标签。此梯度是代理梯度，不宣称无偏。

尺度统计按样本精确累计，包括16条路径、每轮选择/概率、各缺失族条件分布。专家网格面积预算为8−1.5×粗轮数；R7可在2–8间变化，固定两粗两细为5。该指标不含完整路由、投影、上采样等代价，不能当作精确FLOPs。

R8/R9保留N7的有符号tanh门及历史梯度。进入第r轮时当前为H(r−1)，历史H0…H(r−2)；均匀模式平均历史，最近模式仅取H(r−2)。query/key保留在state_dict但冻结不用，披露总参数和可训练参数差异。历史仅在当前forward内存在。

R10时间类为T/TD/TA/TL，空间类为S/SD/SL；ST可搭配任意其他专家，允许19对。并列优先合法的原生Top-2，否则固定对顺序。原candidate balance概率计算不改，没有新评分头或额外专家试运行。

## 六套评估协议

训练/验证使用random_point、node_outage、temporal_gap、spatial_region四基础族。训练按epoch重采样，验证固定；各组同一mask协议，不向模型输入族标签。

| 测试集 | 缺失族 | 基础mask seed |
| --- | --- | --- |
| in_distribution | 四基础族 | 20260917 |
| unseen_combinations | 节点+空间、时间+空间、节点+时间 | 20260918 |
| unseen_geometry | 移动区域、多块 | 20260919 |
| unseen_triple | 节点+时间+空间 | 20260930 |
| unseen_combinations_repeat | 原两元组合，换mask实现 | 20261001 |
| unseen_geometry_repeat | 原两种形态，换mask实现 | 20261002 |

实际test种子沿用构建器+30000偏移。所有组使用相同测试窗口与各套mask；最后两套是稳定性复测，不是新缺失族。所有测试只用ID最优checkpoint，不参与训练或选best。

三元mask随机排列三个组件，各贡献约1/3新的缺失位置，沿自己的几何字段排序补足，整数余数按排列顺序分配；总缺失数精确round(T×H×W×0.4)，没有随机散点补丁。边界节点/时间平面可部分缺失，不声称是完整平面的严格组合。旧九模式和旧双组件构造代码保持原行为。

## 队列、输出和恢复

新队列位于outputs/v24-COE/experiments/r_exploration/<dataset>/<fingerprint>/，保存plan.json、protocol.json、源码快照、configs、results、evaluations、reference_evaluations、launcher_logs和summary.json。

先对固定的旧N参考checkpoint评估，再按R1→R10训练与评估。参考配置从checkpoint读取并验证哈希，参考结果只写新队列，不改旧run。训练输出沿用时间戳_方法名_seed/random/rate/...结构。

队列独占锁、逐任务检查GPU空闲，只使用指定单卡。训练失败则停止，不静默跳过。完整训练须有匹配配置/epoch回执、best/last checkpoint和test结果；已完成评估须同时匹配checkpoint/config/protocol。未完成训练目前从头重跑，不将残缺run当作完成，也不自动以last续训。last保存用于明确的后续继续训练或验证。

控制台仅任务切换与train epoch 当前/总数的batch tqdm（train loss/mae/rmse）；评估及路由细节进日志。程序不会在10小时到点时终止训练，已明确允许十组完整跑完。

## 验证要求

旧B/N初始化与前向、旧12族mask位级一致；旧checkpoint可加载。新机制验证初始化、真实稀疏调用、梯度、历史因果隔离、扰动RNG保存恢复、三元组件贡献、按族精确尺度统计和隐藏真值隔离。微型NPZ完成真实训练/保存/六套评估与完成跳过，参考评估不得改旧日志。正式启动前GPU0做batch32 AMP全细自由尺度及两种记忆组显存测试。
