# TS-CoE组队路由与接受反馈：基于最新提交的实验设计 v4

## 0. 版本与工作边界

- 仓库：`6xiaoming6/my_idea`，分支：`v24-COE`。
- 本次两次读取远程分支得到相同HEAD：`6d65c6f5bae7f835e65dd5e9fee0acf3f5c29108`。
- 提交说明：`最新代码更新`；提交时间：2026-09-27 02:39:21 UTC，即2026-09-27 10:39:21 UTC+8。
- 本地 `git ls-remote` 仍因DNS失败；本次通过已连接的GitHub接口成功按不可变提交读取源代码。**这是成功的远程源代码读取，不是完成了本地git clone。**
- 已核对主模型完整forward、外层封装、CoE损失、路由统计、训练主循环与优化器相关代码、计划生成器相关逻辑及关键配置。
- 未运行该仓库的GPU训练、真实数据评估或完整集成测试；未修改或推送远程代码。
- 文中“源码事实”来自该提交；“建议”“拟新增”是后续实施方案。没有将历史报告的训练设置冒充当前提交下的真实运行配置。
- 附带的数学检查只使用独立合成张量，不能替代仓库集成测试。

## 1. 相对上一版方案，必须修正的源码事实

### 1.1 当前Top-2不走Gumbel采样

`TemporalSpatialCoE.forward` 的训练分支对 `top_k>1` 直接执行 `sample_logits.topk(...)`，按选中集合截断clean probabilities，再归一化。Gumbel straight-through及uniform/soft到hard混合位于 `top_k==1` 分支。[S1]

因此，E0必须保留确定性Top-2。不能把原生基线描述成Gumbel模型，也不能给组队模型新增随机探索后把全部改善称作组合交互收益。

**温度日程仍有作用。** `routing_schedule()` 的sampling temperature进入clean probabilities，所以继承3轮warmup＋3轮transition时，会改变Top-2组内权重；但不会使Top-2在前3轮变成六专家Soft输出。代码中的 `hard_fraction` / `effective_mode` 还可能将这段训练标成soft，应修正诊断标签，而不擅自改变实际算法。

### 1.2 当前Top-2推理也计算全部专家

主干进入全候选循环的条件包含：

```python
routing_mode == 'hard' and (self.training or self.top_k > 1)
```

因此三轮六专家Top-2在当前实现下，训练和推理均对每个窗口计算18次路由专家候选；输出中只有6次名义非零专家贡献，另有3次点级共享专家计算。[S1]

本轮先保留相同计算实现，避免同时改dispatch。真正Top-2稀疏推理优化应在结构筛选后统一加到全部相关模型，并检查优化前后输出一致。不得宣传当前Top-2已将推理专家计算降到6次。

### 1.3 `paths`不是完整Top-2路径

当前 `paths = weights.argmax(dim=-1)`，只保存每轮权重最大的一个专家。`routing_metrics.py`还直接用它统计路径；hard模式下名为`usage`的指标实际累计的是融合权重，不是被选择的次数。[S1,S4]

应新增完整选中集合、无序pair id和有序轮次序列。不要把现有路径数量解释成两专家组队路径数量。分组统计还要传递真实`top_k`，避免子统计回落到默认1。

### 1.4 接受门必须进入损失和最终输出

CoE主损失使用 `outputs['x_hat_main']`，外层包装从同一个字段生成 `x_hat_final`、`x_comp`。[S2,S3] 若只保存一个accepted字段但仍返回原始decoder输出，门会被主训练目标和最终推理绕过。

### 1.5 新实验不能直接沿用depth-pool默认队列

`depth_pool_experiments.json`列的是8专家和12专家、20 epoch、默认不保存best checkpoint，数据路径为`data/TaxiBJ/taxibj_*.npz`；`coe_validation_experiments.json`则指向`v24_clean_abc_20260917`目录。[S6,S7]

这只证明配置路径不同，不证明扁平目录里的数据一定有问题。本次没有读取这些NPZ字节。新实验应明确选择并验证数据哈希；不能仅凭目录名称判定数据是否清洁。

`coe_main_base.json`仍为四轮且未显式设置Top-k，按构造默认值是Top-1。[S5] 新基线必须同时覆盖`num_steps=3`、`top_k=2`和长度为3的`fixed_expert_steps`。

### 1.6 原有计划生成器的成本与合并规则要更新

`_candidate_calls()`将所有hard推理记为每轮1次，未识别Top-k全候选实现；三轮Top-2的该字段会错误写成3，而不是当前实际18。[S8]

`build_plan()`只给`coe_validation/coe_dual_mask/coe_mechanism1`绕开旧`experiments/full.json`。新加stage若未加入同类处理，会重新合入旧两专家默认配置。[S8]

这些属于统计/配置可靠性修正，应对所有组一致应用，不当作方法模块。

## 2. 冻结本轮共同主干

| 项目 | 统一规定 |
|---|---|
| 专家链 | 3轮 |
| 路由池 | T/S/TD/SD/TA/ST，共6类、每类1实例 |
| 名义激活 | 每轮2个，跨轮可重复 |
| 专家共享 | 除E10外跨轮共享 |
| Router | 每轮独立，窗口级；legacy输入与现有统计保持 |
| 宽度 | 隐藏64、Router隐藏64；仅E3改Router宽度 |
| 专家组内融合 | 对选中的两个个体logit做softmax，继承相同sampling-temperature日程 |
| 公共点级分支 | 保留当前每轮执行和已有残差scale，不移动到链外 |
| 编码器/投影/解码器 | 不变、原有共享关系不变 |
| 辅助多尺度分支 | 关闭 |
| 门控对象 | 只控制补全估计，不回滚本轮隐藏状态 |
| 结构筛选损失 | 最终缺失位置L1；balance=0.01；mid=0；z=0；不加收益监督 |
| 专家计算 | 第一轮保留现有全候选训练/推理逻辑，统一测量真实成本 |

原始mask始终不变，接受后的预测不会变成真实观测。

## 3. 组队路由：匹配当前确定性基线，而不是新增Gumbel

### 3.1 三种评分方式

原生Top-2：

$$u=g_r(s),\quad \mathcal A_r=\operatorname{Top2}(u).$$

加性对评分：

$$z_{ij}=u_i+u_j,\quad i<j.$$

确定性、无歧义最大值情况下，最大加性对恰好是两个最高个体分数。15维加性分数本身不增加组合交互能力。

带交互项：

$$z_{ij}=u_i+u_j+b_{ij}(s).$$

保留现有Router的隐藏激活，在其上新增`Linear(64,15)`。交互头初始权重、偏置均为0，使初始前向尽量与加性参照一致。保持原个体头控制两专家相对融合比例，不再额外用pair概率缩放实际残差。

### 3.2 为什么必须单独设计选择梯度

如果`b`只用于`argmax(z)`，之后融合权重仍只来自`u`，任务损失不会通过整数选择索引更新`b`。附带独立张量检查确认为`None`梯度。

本轮建议使用**确定性前向＋软代理反向**，不引入Gumbel噪声：

$$q=\operatorname{softmax}(z/(T_{sample}\tau_{pair})),\quad \tau_{pair}=1\text{作为起点},$$

$$q_h=\operatorname{onehot}(\arg\max z),\qquad q_{ST}=q_h+q-\operatorname{stopgrad}(q).$$

前向始终选择一对，反向使用q的代理梯度。各轮温度策略、门控输入和组内权重完全一致。

E1采用加性分数＋这个代理接口；E2采用交互分数＋同一个代理接口。E0不改。**E0/E1因此是反向训练机制对照，不能预设二者从头训练后指标完全相同。**

### 3.3 不重复执行15对专家

对候选对c=(i,j)，组内权重：

$$\rho_i^c,\rho_j^c=\operatorname{softmax}([u_i,u_j]/T_{sample}).$$

其他专家权重为0。将其嵌入6维向量$\rho^c$，定义：

$$\bar w_e=\sum_c q_{ST,c}\rho_e^c,\qquad \Delta H=\sum_e\bar w_eE_e(U).$$

前向只有2个非零有效权重；反向能训练交互评分。每轮仍只计算6个基础专家各一次，而不是15对分别重跑。这个软代理梯度不是对硬离散目标梯度的无偏保证，收益由实验决定。

选中对可能包含两个个体概率很小的专家。直接`gather`其logit再softmax，比先计算全池概率后除一个极小的概率和更稳健；数学上同为选中集合归一化。E0/E1数值等价测试应覆盖正常、极端与并列logit。

### 3.4 正则不悄悄改变

本轮保持当前`route_probs`的六维个体clean概率，以及`route_weights`的六维实际融合权重，继续调用现有逐轮平均balance。

另存`pair_logits/pair_probs/pair_ids`，不要把15维pair概率塞到原六专家统计字段。也不要同时改成“强制15对均匀”。如果后来改用pair分布诱导的专家边际作为balance概率，应作为独立正则对照，不归入纯交互收益。

## 4. 接受机制：只比较真正不同的设计

令$C^r$为本轮观测回填后的原始候选，$V^{r-1}$为上一轮已接受结果：

$$\delta^r=C^r-V^{r-1},$$

$$V^r=M\odot X_{obs}+(1-M)\odot[V^{r-1}+a^r\odot\delta^r].$$

它等价于缺失位置的$(1-a)V^{r-1}+aC^r$。增量门与正确凸组合不应分别跑两组。旧值减新值的增量符号错误；`old+a*new`也不是这种凸组合。

建议点级接受头输入当前H、旧值、候选、差值、原始mask与支撑；使用轻量1×1×1头，跨轮共享，输出[B,C,T,H,W]。初始接受值约0.9是工程起点，不是最优值结论。E5/E6等组使用相同输入处理与初始化。

E6用相同logit-map生成网络，再对时空维度求平均后sigmoid，广播成窗口级系数，从而与点门尽量匹配参数量。接受头不增加卷积空间感受野，避免同时给新方法增加额外时空骨干。

### 4.1 两条数值流必须分开

- `candidate_completion`：C，原始解码加观测回填；
- `accepted_completion`：V，门控结果；
- `candidate_change`：原始候选前后变化；
- `accepted_change`：接受结果前后变化。

主模型E5的Router和专家使用V；Router变化统计应改为`abs(Vr-Vr-1)*(1-M)`。可保留“提议delta”用于门，但不要未经单独实验就再加入新的拒绝历史Router特征。

主损失与最终输出必须使用接受后的缺失位置估计。可以令`x_hat_main`的缺失位置来自V、观测位置保留原decoder值以兼容旧接口；外层`x_comp`仍回填真实观测。端点等价测试比较缺失预测和完整`x_comp`，不要把无监督观测位置的raw decoder值差异误报成机制差异。

`coe['predictions']`应明确是接受后的预测，原始候选另存`candidate_predictions`，防止未来中间监督引用错误张量。

隐藏状态继续演化，不等于其后续轨迹不受门影响：后续专家读取V，仍会改变H的演化。不得宣称所有有害隐藏信息已被消除。

## 5. 最新结构矩阵

以下均是**拟新增/待整合实验**，不是已经跑出的结果。保留旧E编号。

| 编号 | 路由 | 专家参数 | 接受门 | Router数值流 | 专家数值流 | 问题 |
|---|---|---|---|---|---|---|
| E0 | 当前确定性Top-2及原梯度 | 共享6个 | 无 | 当前C | 当前C | 当前提交下基线 |
| E1 | 加性15对＋确定性ST | 共享6个 | 无 | 当前C | 当前C | 选择代理梯度影响 |
| E2 | 交互15对＋确定性ST | 共享6个 | 无 | 当前C | 当前C | 显式组合交互 |
| E3 | 加宽个体头＋加性ST | 共享6个 | 无 | 当前C | 当前C | Router容量因素 |
| E4 | 加性ST | 共享6个 | 位置门 | 当前V | 当前V | 接受闭环 |
| E5 | 交互ST | 共享6个 | 位置门 | 当前V | 当前V | 完整结构 |
| E6 | 交互ST | 共享6个 | 窗口门 | 当前V | 当前V | 局部性是否必要 |
| E7 | 交互ST | 共享6个 | 位置门，仅输出记录 | 当前C | 当前C | 融合或反馈 |
| E8 | 交互ST | 共享6个 | 位置门 | 当前C | 当前V | 显式接受值进Router |
| E9 | 交互ST | 共享6个 | 位置门 | 初始H0/V0/变化0 | 当前V与动态H | 在线重路由 |
| E10 | 加性ST，与E1一致 | 每层6个独立，共18个 | 无 | 当前C | 当前C | 分层MoE强参照 |

除E9外，Router都读取当前H；所有组的mask与支撑信息一致。

### 5.1 首批七组

先执行E0/E1/E2/E3/E4/E5/E10。第一轮固定一个模型seed筛选；不要以一次胜负宣称统计显著。

核心2×2为：

| | 无接受门 | 位置接受门 |
|---|---|---|
| 加性路由 | E1 | E4 |
| 交互路由 | E2 | E5 |

分别比较E2−E1、E5−E4和E4−E1、E5−E2。Full最好不自动意味着两项各自必要。即使单独交互效果小，也完成预先规定的E5，避免漏掉真实交互。

### 5.2 参数量匹配

固定legacy Router、C=2、D=64时，Router输入维度：

$$d_{in}=2D+4C+4(10C)+2=218.$$

| Router | 每轮参数量 |
|---|---:|
| E1，hidden64→6 | 14,842 |
| E2，hidden64→6，加hidden64→15 | 15,817 |
| E3，hidden68→6 | 15,742 |

E2与E3每轮相差75参数，约占E2的0.474%。这是当前选定结构的参数计算；改变输入或采用grouped Router后必须重算。

### 5.3 反馈消融的精确含义

- E7：原候选C继续驱动专家链；门独立维护输出V。不是把V `detach()`后继续反馈。
- E8：Router数值与变化特征都来自C序列，专家使用V；H仍动态，可能间接携带门的信息，所以只能称“去掉显式接受值通道”。
- E9：保留三个独立Router，均读取初始化缓存的特征；专家和门仍处理动态状态。只用已有`router_state='initial'`概念，不设置`expert_state='initial'`。

### 5.4 多层MoE的公平比较

E10以E1为模板，仅解除路由专家跨轮共享。三个池各自包含同样六类专家，其他投影、解码、共享点级分支、每轮Router及反馈规则不变。

仅将全局池改成18个名字不是这个对照。需要按step访问各自池，训练、推理、梯度诊断和保存/恢复全部一致。

每层候选宽度与执行深度相同，但参数总量不同：E10的路由专家参数块为共享池的3倍，不是整网恰好3倍。E10/E1检验绑定与容量的取舍，E10/E5是完整方法对这个强基线的整体比较。胜过E10不等于胜过所有多层MoE，也不单独证明组队和接受两个模块都有效。

## 6. 训练与数据协议

### 6.1 本次建议：新的60轮配对批次

仓库当前主配置和depth-pool配置写的是20轮；本方案建议对七个关键模型统一使用60轮与cosine60，以验证较充分训练的结果。这是新实验预算，不是声称用户最新22.602结果已经用60轮得到。

- 数据路径选择：沿用`coe_validation_experiments.json`明确指向的三个clean NPZ，并在训练服务器核验存在性、哈希与划分；本次未下载数据。
- mask：九族、0.4；训练每epoch重采样，验证/测试固定；模型seed与mask/loader seed分别记录。
- batch16，lr_main=1e-3，weight_decay=1e-4，grad_clip=1，AMP；先不增加独立低Router学习率。
- 主配置3/3温度日程暂时整体继承，使所有组共同经历相同的组内权重温度变化。不要把它描述为Top-2的dense soft预热。若决定取消，整批统一取消并建立新基线，不与旧结果混算。
- 验证间隔2，按验证MAE选best，关闭早停，`save_best_checkpoint=true`。
- 数据归一化、监督mask、丢弃最后batch等不随实验组改变。
- 只用验证集选择配置；测试标签不进入Router、接受头输入或选参过程。

附带`E0_3r6e2k_60ep_proposed.json`由已读主配置字段组成，固定三轮Top-2、60轮和保存best。它没有实现新机制，也未在真实数据上执行。其余组需要先完成源码扩展，不能只填写未知JSON键就运行。

### 6.2 基线命令模板

将E0配置复制到仓库的对应位置，并先检查clean数据哈希后，可按现有训练入口运行：

```bash
python scripts/train.py \
  -c E0_3r6e2k_60ep_proposed.json \
  --train_npz data/TaxiBJ/v24_clean_abc_20260917/taxibj_train.npz \
  --val_npz data/TaxiBJ/v24_clean_abc_20260917/taxibj_val.npz \
  --test_npz data/TaxiBJ/v24_clean_abc_20260917/taxibj_test.npz \
  --no_plot -n pair_accept_v4_E0
```

该命令依据现有计划生成器的调用参数构造；不是本次已运行命令。

### 6.3 多种子和后续验证

先完成E0—E5及E10。主要四格E1/E2/E4/E5优先补两个额外模型seed；若要主张超越普通多层MoE，E10也补同样seed。统一loader与mask协议，报告配对差值、均值和标准差，三seed不是自动的显著性证明。

只有主模型与强对照都仍明显改善时，再统一考虑更长预算。不要只给Full延长训练，也不要将cosine20训练结束后任意续训视为从头cosine60。

## 7. 需要修改的真实文件与接线

| 文件 | 修改内容 |
|---|---|
| `models/temporal_spatial_coe.py` | pair组合/交互头、确定性ST、C/V双流、接受输出、独立池访问、完整pair诊断 |
| `models/coe_router.py`或Router相邻新模块 | 可复用的个体+pair头；保证参数都注册 |
| `models/imputer.py` | 验证x_hat_main/x_hat_final/x_comp都使用接受后的缺失估计；避免字段覆盖误接 |
| `losses.py` | 结构首批保持最终L1与原六专家balance；可选收益损失后加 |
| `routing_metrics.py` | pair路径、选择频率与权重分离、条件组携带top_k、修正soft标记 |
| `engine.py` | 检查新头优化器归组、任务梯度诊断、各缺失族误差累计 |
| `scripts/v24/build_experiment_plan.py` | 注册新stage/变体、绕过旧full、配对断言、实际候选调用统计 |
| `scripts/train.py`/checkpoint工具 | 保存best；需要精确恢复时再补last及scheduler/scaler/RNG状态 |

拟新增配置名只是接口设计，例如`pair_mode`、`expert_sharing`、`acceptance`等。当前`from_config`没有消费这些字段，当前构造器也没有实现组队/接受/分层池。新增时必须显式接入，最好拒绝未知方法配置键。

优化器中`lr_router`仅显式匹配`main_branch.routers.*`，`gate_lr_mult`仅匹配含scale_gate或branch_gate的参数名。[S9] 建议pair head放在对应Router模块内部；新接受头的学习率不能靠名称想当然。先打印每个参数的归属、确保无遗漏和重复。

当前checkpoint保存model、optimizer、epoch、config与metrics，但未保存完整scheduler/scaler/RNG状态。[S11] 保存best足够做固定checkpoint评估；精确中断恢复需要另行实现，不能声称已有完整恢复能力。

## 8. 必须记录的结果

### 8.1 精度与成本

记录Val MAE、Test MAE/RMSE、best epoch、每缺失族累计绝对误差/平方误差/有效标签数；不能只平均各batch MAE。MAPE不作为主选择依据。

成本区分：有效非零专家数、实际处理窗口数、共享专家调用、总参数、Router/Gate参数、训练时间、推理时间、峰值显存、实际优化更新与AMP跳步。

当前三轮Top-2成本标签应为：路由专家名义6次贡献，实际18次训练候选＋18次推理候选；共享专家3次。不要沿用生成器的hard每轮一次元数据。

### 8.2 组合记录

新增`selected_experts[B,R,2]`、`pair_ids[B,R]`、`pair_logits/probs[B,R,15]`，保留个体`route_probs/weights[B,R,6]`。同轮内专家对按固定索引排序，轮与轮之间有序。三轮最多15³种pair路径，不把组内两个专家排名交换算作新路径。

每层选择频率之和为2；组内融合权重均值之和为1；pair选择频率之和为1。三种量不能混名。

### 8.3 接受质量

同checkpoint、同一前向轨迹保存旧值、原始候选与接受值，计算：

$$H_c=\operatorname{mean}[\max(e_c-e_{old},0)],\quad
H_a=\operatorname{mean}[\max(e_a-e_{old},0)].$$

同时记录相反方向的有益改善量。接受门应减少有害修改且保留有益修改；全部拒绝不是成功。检查峰值/区域边缘的分析分组需要预先定义，不根据测试误差挑漂亮样本。

### 8.4 候选选择质量

在验证子集固定前缀状态，比较模型选择与15个最后一轮候选的最终代价。含门模型必须对每个候选重新计算自己的门。

$$Regret=J(c_{chosen})-\min_{c\in\mathcal C}J(c).$$

使用标签选出的最优候选是诊断参照，不是可部署成绩，更不是全局最优模型。不同模型的候选与状态不同，regret不能取代误差主指标。

## 9. 可选的收益监督：结构确定后再做

| 组别 | 结构 | 后续收益路由监督 | 接受辅助监督 |
|---|---|---|---|
| S0 | E5复用 | 无 | 无 |
| S1 | E5 | 有 | 无 |
| S2 | E5 | 无 | 有 |
| S3 | E5 | 有 | 有 |

组合收益必须评价`当前候选→本轮接受→剩余专家链与接受→最终监督误差`，而不是只看未经门处理的候选。训练候选采样中包含当前选择并抽样其他组合；试运行仍有前向成本，推理时不使用标签、不枚举候选。

接受辅助目标可以使用停止梯度的新旧候选，在训练隐藏点直接优化插值误差。最终损失主导，不要求每点每轮严格单调。

## 10. 验收测试

### 10.1 本次已完成的独立数学测试

仅用合成张量与PyTorch CPU，不是仓库集成测试：

- 无并列随机logit下，原Top-2与最大加性pair前向权重一致，误差约1.11e-16。
- 零交互的确定性ST前向与原Top-2一致，仍只有两个非零权重、权重和为1。
- 交互项只经argmax时无选择梯度；加入确定性ST后得到有限非零梯度。
- 正确增量门与凸组合等价；门0/1端点与观测保护通过。
- 上述三类Router参数计数已用独立模块核对。

详见`math_checks.json`、`verify_pair_router_math.py`。这些测试不能证明新方法精度更好，也不能证明源码集成已经正确。

### 10.2 批量训练前必须完成的仓库测试（本次未执行）

1. 新功能关闭时，原模型缺失预测与x_comp逐项一致。
2. E0/E1无歧义输入前向一致，明确允许梯度不同。
3. E2交互全零时退回E1；交互头从纯任务损失获得非零梯度，而不只是balance梯度。
4. gate强制1时还原候选，强制0时保留旧补全，原始mask与可见值不变。
5. 改变接受头后，主损失与最终输出确实变化，不被外层包装覆盖。
6. E7后续输入确为C，E8所有显式数值统计来自C，E9三个Router保持独立且只读初始特征。
7. E10各层专家参数存储独立；第一层更新不会自动改动第二层；独立池进入优化器和诊断。
8. B=1、全观测、全缺失、边界网格、极端logit、AMP和空监督都无非有限结果。
9. 矩阵生成后差分配置只出现白名单字段；新stage不被full.json改回两专家。
10. 统计和实际执行相符；pair路径与选择频率不能沿用argmax路径含义。

## 11. 结果决策

| 结果模式 | 后续选择 |
|---|---|
| E2稳定优于E1与E3 | 保留组合交互 |
| 仅E1优于E0，而E2≈E1 | 主要可能是训练代理改变，暂不主张组队交互收益 |
| E4优于E1且E5优于E2 | 接受机制存在跨路由版本增量 |
| E5与E6接近 | 优先更简单的窗口门，不强写局部保护 |
| E5与E7接近 | 更像跨轮输出融合，弱化闭环必要性主张 |
| E5与E9接近 | 初始条件可能足以安排组合 |
| E10弱于E1而E5≈E1 | 主要是权重绑定/容量取舍，新模块未立住 |
| E5稳定优于各单项和E10 | 完整方案值得进一步跨数据集与强外部基线验证 |
| S系列增量小且成本高 | 不保留额外收益监督 |

没有预设Full必须胜出。目标是保留确有增量的最简单方案。

## 12. 来源索引（全部固定到本次提交）

下列链接是源码依据，不是论文性能结果。本设计不依赖旧缓存分支或README中的过时层数说明。

- [S1] [src/stmoe_imputer/models/temporal_spatial_coe.py](https://github.com/6xiaoming6/my_idea/blob/6d65c6f5bae7f835e65dd5e9fee0acf3f5c29108/src/stmoe_imputer/models/temporal_spatial_coe.py) — 主模型、路由分支、完整forward与返回值。
- [S2] [src/stmoe_imputer/models/imputer.py](https://github.com/6xiaoming6/my_idea/blob/6d65c6f5bae7f835e65dd5e9fee0acf3f5c29108/src/stmoe_imputer/models/imputer.py) — 外层预测与观测回填。
- [S3] [src/stmoe_imputer/losses.py](https://github.com/6xiaoming6/my_idea/blob/6d65c6f5bae7f835e65dd5e9fee0acf3f5c29108/src/stmoe_imputer/losses.py) — 监督掩码、CoE损失与均衡项。
- [S4] [src/stmoe_imputer/routing_metrics.py](https://github.com/6xiaoming6/my_idea/blob/6d65c6f5bae7f835e65dd5e9fee0acf3f5c29108/src/stmoe_imputer/routing_metrics.py) — 路径与使用统计。
- [S5] [configs/v24/coe_main_base.json](https://github.com/6xiaoming6/my_idea/blob/6d65c6f5bae7f835e65dd5e9fee0acf3f5c29108/configs/v24/coe_main_base.json) — 当前主配置。
- [S6] [configs/v24/depth_pool_experiments.json](https://github.com/6xiaoming6/my_idea/blob/6d65c6f5bae7f835e65dd5e9fee0acf3f5c29108/configs/v24/depth_pool_experiments.json) — 当前8/12专家配置。
- [S7] [configs/v24/coe_validation_experiments.json](https://github.com/6xiaoming6/my_idea/blob/6d65c6f5bae7f835e65dd5e9fee0acf3f5c29108/configs/v24/coe_validation_experiments.json) — 明确的clean数据路径。
- [S8] [scripts/v24/build_experiment_plan.py](https://github.com/6xiaoming6/my_idea/blob/6d65c6f5bae7f835e65dd5e9fee0acf3f5c29108/scripts/v24/build_experiment_plan.py) — 配置合并、阶段注册、调用统计。
- [S9] [src/stmoe_imputer/engine.py](https://github.com/6xiaoming6/my_idea/blob/6d65c6f5bae7f835e65dd5e9fee0acf3f5c29108/src/stmoe_imputer/engine.py) — 优化器归组与scheduler。
- [S10] [scripts/train.py](https://github.com/6xiaoming6/my_idea/blob/6d65c6f5bae7f835e65dd5e9fee0acf3f5c29108/scripts/train.py) — 实际训练、验证、best选择与输出元数据。
- [S11] [src/stmoe_imputer/utils/checkpoint.py](https://github.com/6xiaoming6/my_idea/blob/6d65c6f5bae7f835e65dd5e9fee0acf3f5c29108/src/stmoe_imputer/utils/checkpoint.py) — 保存与恢复字段。

附带文件说明：`experiment_design_manifest.json`是设计描述，不是当前runner可执行的配置列表；除E0配置外，其余机制必须先实现。包内没有仓库完整副本或训练权重。
