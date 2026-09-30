# N1–N7：B 系列结构上的未见缺失与跨轮尺度实验

本轮直接继承 `configs/v24/b3_c3/B1.json`、`B2.json`、`B3.json`，由 `configs/v24/n_exploration/N*.json` 做明确覆盖。B 系列已有配置、权重和输出不修改，C3 不参与。新的模型子类继承 `TemporalSpatialCoE`，复用其编码器、八专家、原生 Top-2、预测头、损失和稀疏执行。

| 组别 | 基础结构 | 本轮改动 | 研究问题 |
| --- | --- | --- | --- |
| N1 | B1：单层独立八专家 Top-8 | 四基础缺失模式训练 | 并行专家的 ID/OOD 基准 |
| N2 | B2：四层独立八专家 Top-2 | 同上 | 普通多层 MoE 的 ID/OOD 表现 |
| N3 | B3：四轮共享八专家 Top-2 | 同上 | 共享专家链是否改善未见组合泛化 |
| N4 | B3 | 每轮独立 Router 只读初始特征；专家仍读取动态 hidden | 中间状态驱动的重新选择是否必要 |
| N5 | B3 | 固定 coarse/coarse/fine/fine；空间下采样倍数2 | 固定跨轮双尺度是否有效 |
| N6 | B3 | 按当前状态和剩余预算逐轮选尺度；每样本恰好2粗+2细 | 自适应尺度顺序是否优于固定顺序 |
| N7 | B3 | 从更早轮次检索 hidden 记忆，零初始化门控校正本轮专家输入 | 跨轮历史表示是否比仅传上一轮更有用 |

## 共同条件

默认 TaxiBJ、random0.4、seed7、batch_size32、GPU0 单卡串行、100epoch、每5epoch验证、AdamW、lr1e-3 余弦降至最后一轮3e-4、weight_decay1e-4（沿用 bias/norm 不衰减）、grad_clip_norm1、AMP、关闭早停。保存 best.pth/last.pth，best 只由同分布验证集 MAE 选择。

所有组保持 direct 状态传递、completion_feedback=false、原始融合；关闭额外 shared 分支、搭档路由、接受门、路由噪声、预热、C3 局部路由。专家池 T/S/TD/SD/TA/ST/TL/SL。L1、balance0.01、mid0、z0，与 B3 相同。`--dataset bikenyc` 可切换数据；本轮先跑 TaxiBJ。

N1–N7 从头训练，不继承已见过九模式的 B checkpoint。所有组的 train/val mask 完全相同，训练按 epoch 重采样，验证固定。数据按已有时间切分，不因本轮实验重新打乱。

## 缺失协议

完整协议在 `configs/v24/n_exploration/protocol.json`。

- train/val：random_point、node_outage、temporal_gap、spatial_region 四类近似均衡混合；每个窗口一类，缺失率0.4。
- ID test：同四类，独立测试种子。
- unseen_combinations：同窗口内 node_plus_space、time_plus_space、node_plus_time 三类近似均衡混合；构成元素训练已见，组合未见。
- unseen_geometry：moving_region、multi_block 两类近似均衡混合；形态训练未见。

组合构造随机选择组件先后顺序，第一组件占最终缺失预算的40%–60%，第二组件沿自己的几何排序向外扩张，直到贡献足够的新缺失单元。最终缺失数严格为 round(T*H*W*0.4)，不追加随机散点。边界 node/time 平面可能部分缺失，以满足精确体素预算，与原 mask 约定一致；不声称是严格整节点/整帧预算。组件字段不同，但聚合评价不能据此声称真实世界分布已被复现。

三套 test 共用同一测试时间窗口，各自固定 mask；在所有模型之间相同。mask 类别标签只用于日志，不输入模型。自然无效值仍由 available/target mask 排除。训练脚本仅在训练结束后用 ID 最优 checkpoint 测试，额外 OOD 脚本不更新权重，不参与选 best、早停或调参。

旧九模式 `FAMILIES` 默认值与生成结果不变，新增组合名以 `ALL_FAMILIES` 注册用于新协议和日志。

## N5/N6 双尺度的精确定义

- 每轮专家选择仍读取完整细网格上的状态，使用 B3 原生 Top-2。新增尺度决策作用于该轮整窗口的专家执行分辨率，不做 C3 区域裁剪。
- 细尺度输入投影与 B3 完全相同。粗尺度 hidden、原 mask 支撑描述、位置编码在空间2×2平均聚合，时间维不变。
- 粗尺度原始观测值使用 masked mean，另保留分数型观测覆盖率。一个粗格完全没有观测时，使用初始 completion 的池化值作为预测占位，coverage仍为0；预测不成为新观测。
- 粗尺度使用同一批专家权重及同一输入投影；专家输出仅在空间维做双线性上采样，直接成为下一轮细网格 hidden。没有残差累加、另一个预测网络、额外多尺度辅助损失，也不读取隐藏真值。
- 每个样本每轮只执行一个尺度、两个专家。只为实际选择粗尺度的样本构造粗输入，不执行另一尺度的神经专家。N5/N6 每样本都恰好两粗两细，顺序最多6种。
- N6 每轮有独立两输出尺度 Router，输入当前 B3 router features、剩余粗轮预算和剩余轮数。硬选择 argmax；不满足预算的候选先屏蔽；训练使用直通估计，只有实际执行的尺度输出提供代理梯度。
- 尺度头输出层零初始化，固定1e-3粗尺度优先偏置打破并列，使 N6 初始输出与 N5 完全一致。没有尺度预热或额外平衡损失。预算耗尽时强制合法尺度，不宣称强制轮次有尺度选择梯度。
- N5/N6 状态字典包含相同尺度头以匹配初始化和总参数量；N5 的尺度头固定不用、冻结，N6 会训练。因此二者可训练参数量不同，汇总中需披露。
- B3/N3 的 `data.multiscale=false` 保留：粗尺度由新子类在模型内部从可见输入构造，**不是未开启 N5/N6**。实际开关为 `model.coe.spatial_scale.enabled`，不使用旧数据层多尺度分支。

相同8次专家激活不等于相同 FLOPs：N5/N6 的专家网格面积预算为细网格等价5次，N3为8次；每轮完整细网格路由、输入投影、上采样等还有开销。N5/N6尺度面积预算一致，但不同专家类型成本不同。必须同时比较精度、参数量、耗时、显存，不能把面积预算直接写成精确 FLOPs。

日志增加各轮粗尺度比例、可自由选择比例、六种尺度路径比例。N6 若始终采用固定路径，不能据此宣称学到了样本级自适应。

## N7 跨轮状态记忆

N7 仅以 N3 为对照，四轮共享八专家、Top-2、细尺度和其他设置均相同。不把 N5/N6 多尺度合入 N7。开关为 `model.coe.round_memory.enabled`，key_dim默认16。

令 H0 为初始编码表示，H1…H4 为每轮真实执行的专家融合输出。进入第 r 轮时，当前状态为 H(r-1)，可读取的更早记忆为 H0…H(r-2)。第1轮没有更早记忆，完全沿用 N3；第2轮只有 H0；第3轮可读 H0/H1；第4轮可读 H0/H1/H2。记忆不含未来状态，不跨 batch 保留，也不是新增真实观测。

1. 用当前 hidden 的全局均值、缺失位置加权均值构造 query；历史状态的同类统计构造 keys。
2. 点积 attention 在历史轮次间归一化，按窗口计算权重，但检索的 values 保留完整时空网格。query/key 为16维；不执行额外专家。
3. 门控读取当前和检索记忆的统计，输出每窗口、每通道的有符号 tanh 系数 g。专家输入前的状态为 `H_tilde = H_current + g * (H_memory - H_current)`。g∈(-1,1)，它是有符号校正系数，不是置信度或凸融合概率。
4. 原生 Top-2 Router 仍读取上一轮的 H_current，第一版不增加路由特征。仅本轮专家输入中的 hidden 替换为 H_tilde；原始 mask、支撑描述、初始 completion 和位置输入不变。
5. 本轮专家输出继续 direct 更新，completion_feedback 仍关闭。不过跨轮记忆本身确实引入了到更早状态的可学习跳连，应作为 N7 的结构变化明确披露，不能声称 N7 整体没有跳连。
6. 门控最后一层权重、bias 都为0，初始 g=0，逐轮输出、专家选择与 N3 精确一致。第一步任务梯度可训练 gate；gate 打开后 query/key 获得任务梯度。历史值不 detach，主损失可通过记忆路径训练更早表示；不添加辅助损失。
7. 记忆模块 query/key/gate 跨轮共享，只增加轻量参数。它会增加表示存储、计算和跨轮梯度路径，不将收益直接归因于“更聪明的选择”；正式创新归因还需要匹配参数量和固定历史融合等后续消融。

记录各轮 gate绝对值/有符号均值、输入校正量、来自各历史H的平均权重。只有 attention 权重多样、但 gate≈0，不能说明记忆有效。N7 是跨轮信息传递的探索版本，不预先宣称该通用注意力机制具有首创性。

新增 N7 后采用新的冻结队列指纹。此前 N1 仅完成5轮，保留旧日志及 best.pth；本队列从头重跑 N1，不混用旧的部分训练结果。

## N3 路由诊断

N3 完成后，从同分布验证集按模式均衡抽取至多96个窗口。每次仅替换一轮，候选来自原生 Top-4 的5对：(1,2)、(1,3)、(2,3)、(1,4)、(3,4)，后续轮次根据改变的状态正常路由。

记录原生最终 MAE、候选中逐样本最优最终 MAE、按当轮误差选对后的最终 MAE、当轮与最终选择分歧。逐样本/模式/轮次明细保存在 `logs/route_candidate_diagnostic.json`。这是使用隐藏验证标签的离线诊断，不是可部署方法、正式 test 成绩或全路径 oracle。中间解码没有辅助监督，其指标仅用于诊断。原生候选始终包含在参照集合内。

`--route-diagnostic-samples 0` 可禁用；该参数计入队列指纹。诊断不读取 OOD test 标签，不修改后续实验计划。

## 运行、查看与重启

```bash
python -u scripts/v24/run_n_exploration.py --dataset taxibj --gpu 0 --epochs 100 --batch-size 32
# 仅预览，不启动：
python scripts/v24/run_n_exploration.py --dataset taxibj --dry-run
# 查看新会话：
tmux attach -t v24-n-exploration
```

默认 N1→N2→N3→N4→N5→N6→N7 全部运行，不根据测试成绩决定是否启动 N6。可用 `--variants N5 N6` 单独选组，但改变所选组会生成新队列指纹。相同命令、源码与数据下重启跳过已完成训练；额外测试未完成则只补评估。未完成训练从头跑，不自动续训。不同数据集、epoch、batch-size 均可通过参数配置。

计划与冻结源码：`outputs/v24-COE/experiments/n_exploration/<dataset>/<fingerprint>/`；包含 plan.json、configs、source_snapshot、results、evaluations、launcher_logs、summary.json。汇总区分 pending/trained/finished。

每组训练目录遵循 `{年月日_时分秒}_{核心描述}_seed7/random/rate0.4/...`。控制台仅简短任务切换和 batch 级 `train epoch 当前/总数` tqdm，显示 train loss/mae/rmse；验证、OOD和路由详细信息写入日志。新增评估结果写入各组 `logs/n_evaluation.json` 及队列 `evaluations/N*.json`，每个测试集单独统计总体与各缺失类型指标。
