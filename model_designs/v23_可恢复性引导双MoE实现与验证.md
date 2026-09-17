# V23：目标相关可恢复性引导双 MoE——实现与验证

日期：2026-09-15。开发分支：v23。基础提交：398e550。

本文描述新增代码与预注册验证方案，不是实验结果报告。现有 ST_DILATED 仍是稳定主力，
新机制显式 opt-in；不删除旧文件、不改变旧预设、不导入历史分数作为本轮基线。

## 1. 要验证的问题与边界

相同观测数量下，空间成线、时间集中、时空分散的观测布局，可能约束不同的局部变化。
当前区域加权特征、有效样本数和方差，不显式表达这些方向性约束。
新机制让每个组织专家的局部重建同时依赖原始可见观测，以及对未充分约束模式的学习先验。

重要边界：

- 以固定局部表示为条件研究可恢复性，不保证任意真实数据可恢复，不是因果可辨识性。
- 加权最小二乘、Gram 矩阵、岭回归、数据一致性/零空间思想均已有研究，不能宣称首次提出。
- 第一版使用固定归一化 `[1,x,y,time]` 四个基，先验证机制，不引入可学习基及其缩放自由度。
- 这是离线窗口补全，时间轴可双向使用窗口内可见值，不是因果预测。
- `Q` 是软衰减算子，不是严格零空间投影；先验可学习，最终神经主干仍可能补偿这种衰减。
- 约束作用在新增局部重建支路，不对 ST_DILATED 最终输出施加硬约束，也不声称最终输出有精确恢复保证。
- 一个学习到的支持分数不是校准不确定性。日志中称其 `weakness`，不称预测概率。

基础参考：[归一化卷积与置信度传播](https://bmva-archive.org.uk/bmvc/2018/contents/papers/0591.pdf)、
[图信号采样理论](https://arxiv.org/abs/1503.05432)、
[Deep Null Space Learning](https://arxiv.org/abs/1806.06137)、
[ImputeFormer](https://arxiv.org/abs/2312.01728)。新的潜在贡献应落在组织、目标相关约束传递、
局部先验补充及其机制证据，而不是给经典公式换名字。

## 2. 实际实现

源文件：`src/stmoe_imputer/models/recoverability.py`。
开关：`model.dual_moe.recoverability.mode`，默认 `off`。
仅支持 `target_readout_v1 + routed_shared`，不默默改变历史 anchored/grid 设计。

### 2.1 保留的结构

- 前端中、粗两尺度，各8个学习区域组织专家、目标读出Top4，节点数32/8。
- 每个组织专家都先从可见观测构造区域，不在源观测阶段Top-K丢弃信息。
- 细粒度ST块、区域时空交互与原细/中/粗预测头保留。
- 后端8个ST_DILATED路由专家＋1个逐点共享专家，Top3；空间/时间dilation均为2。
- 稀疏的是混合权重，所有专家仍密集计算，没有新增稀疏dispatch或提速承诺。

### 2.2 原始可见值构造局部约束

输入值先沿用可见数据统计得到的归一化，缺失位置在运算前清零。
不从已经经过ST卷积的隐藏特征计算观测方程，避免把推断出来的上下文当成原始观测证据。

对专家e、区域k，使用其学习映射 `A[B,E,T,N,K]`。
将同一槽位在窗口时间维上的观测汇总，构造权重：

`w_i = A_i * m_i / max(sum_j A_j, eps)`。

分母是在完整参考窗口求和，不是在观测子集求和；不含前端统一源门控的1/E因子，
避免改变专家数任意改变岭正则强度。
节点始终按各自专家的映射读回，不混合不同专家的粗节点编号。

`phi=[1,x,y,time]` 的三个坐标轴在完整参考网格上标准化为单位均方。
长度为1的轴置零，既不除零，也不伪造该方向信息。

对每个区域计算：

```text
G = sum_i w_i phi_i phi_i^T       # 4×4
b = sum_i w_i phi_i z_i^T         # 4×C
c_obs = solve(G + lambda I, b)
Q = lambda * solve(G + lambda I, I)
```

使用FP32求解；AMP外围仍按原训练配置运行。不显式求逆，不构造N×N协方差矩阵。
`lambda=0.05` 是初始共同超参数，不能解释为已估计的数据噪声方差。
无观测时 `c_obs=0, Q=I`，完全依靠局部先验；退化方向由正岭项稳定处理。

### 2.3 可学习先验与目标读出

区域时空交互后的隐藏特征在时间上平均，通过小线性层预测每区域 `4×C` 先验系数。
完整候选D的系数为：

`c_hat = c_obs + Q @ c_prior`。

其满足局部岭方程 `(G+lambda I)c_hat = b + lambda c_prior`。
强约束方向的先验被衰减，弱约束方向允许更多先验参与；这不是对最后输出的硬一致性保证。

对目标q，使用自己的 `phi_q` 与每个专家的映射恢复局部预测，计算：

`weakness(q) = sum_k A_qk * phi_q^T Q_k phi_q / ||phi_q||^2`。

weakness及coverage作为条件信息时detach，不对其直接施加“置信度越高越好”的损失。
拟合、先验与映射本身仍有重建梯度；只读取可见值，隐藏真值只在损失/评估中使用。

将局部预测与目标weakness通过小线性投影，以固定 `strength=0.1` 加到各专家已对齐的
区域隐藏表示，然后进行原目标Top4读出。后续scale condition和ST_DILATED不改变结构。
因此新机制同时影响候选表示和后续补全输入，不是只给最终router附加一个标量。

### 2.4 损失

原有missing SmoothL1、三尺度辅助项、partition项、前后负载均衡全部保留原权重。
启用组额外对前端Top4融合后的中/粗局部重建预测使用归一化missing SmoothL1，权重0.01。
只有训练集损失参与优化；验证/测试同样记录该损失但不反向传播。
不强迫每个后端残差专家独立重建完整目标。

## 3. 五组对照

| 标识 | 观测重建 | 学习先验 | 用途 |
|---|---|---|---|
| A_ST_DILATED | 原模型 | 原模型 | 不启用新模块的本轮基线 |
| B_SCALAR | 标量coverage下的常数局部拟合 | `lambda/(coverage+lambda) * c_prior` | 普通支持度/新增容量控制 |
| C_MATRIX | 完整G的局部拟合 | 直接`c_prior` | 有矩阵表示、无模式级先验限制 |
| D_RECOVERY | 完整G的局部拟合 | `Q @ c_prior` | 完整研究候选 |
| E_FIT | 完整G的局部拟合 | 0 | 检查经典局部拟合能解释多少收益 |

B/C/D/E的新增参数形状、初始化、注入方式和辅助权重相同。
E保留匹配先验头但输出乘零，其参数对任务无作用，不能用名义参数量掩盖这一差异。
E仍使用学习区域和神经主干，不是独立的经典插值baseline。
B虽然共享用于实现/诊断的计算工具，但非恒定观测矩与G方向信息不进入其预测。

C与D是最干净的单变量对照：仅改变局部先验是否经过Q。
D与A是完整新增机制（含辅助损失和容量）的比较；需联合B/C/E解释，不能单独归因。
关闭模块不构造新参数，旧state_dict键、初始化随机序列及相同输入输出保持一致。

## 4. 默认验证矩阵和训练协议

统一配置：`configs/presets/dual_moe_recoverability_experiments.json`。

| 数据集 | TRAIN样本 | epoch | batch | val_epoch |
|---|---:|---:|---:|---:|
| TaxiBJ | 2491 | 140 | 16 | 2 |
| BikeNYC | 511 | 100 | 16 | 2 |
| CHAP | 2626 | 150 | 16 | 2 |

全部原始训练、验证、测试样本；fixed/random × 0.4/0.8；seed42；每个条件五组，共60组。
fixed/random按相同缺失率交叉，避免上一轮fixed@0.4/random@0.8的混杂。
这是两个代表缺失率的验证，不是四个缺失率全覆盖；单seed不代表统计显著性。
可在JSON统一扩充rates与对应points、seeds，不能只为有利组增加预算。

每2轮及最后一轮验证。按验证MAE选最好的一轮，完整state_dict在CPU内存覆盖保存；
最后恢复最佳权重，完整TEST只运行一次。不写模型checkpoint，不启用早停，不按截止时间削减epoch。
模型/候选选择使用验证结果，测试结果用于最终描述。

## 5. 执行顺序

### 今晚22点前的短预算初筛（后补）

当前时间约2026-09-15 10:32，至22:00约11.5小时。
原60组长预算预计48～60小时，保留原配置，新增同文件`profiles.tonight`，不修改模型、损失或训练样本数。

```bash
conda activate difftdi
python scripts/run_recoverability_experiments.py --profile tonight --dry-run
python scripts/run_recoverability_experiments.py --profile tonight --gpu 0
```

| 顺序 | 数据集 | epoch | 实验数 |
|---|---|---:|---:|
| 1 | BikeNYC | 60 | 10 |
| 2 | CHAP | 50 | 10 |
| 3 | TaxiBJ | 40 | 10 |

五组A/B/C/D/E全部保留；fixed/random都取0.8，seed42；全部训练、验证、测试样本，batch16，val_epoch=2。
减少的是训练轮数与缺失率覆盖，不是为某个候选选择更少的数据或不同预算。
优先高缺失条件基于上一轮ST_DILATED实验观察，是预先声明的初筛，不代表其他缺失率。
短预算下的结果也包含收敛速度差异，不能当作最终最优性能或新颖性已经成立的证据。

用前轮实测ST_DILATED时间折算，无新增开销约6.3小时；预留新模块、读盘和测速后暂按8～10小时安排。
后续用户本机测速为约12.37小时，原8～10小时预估偏乐观。用户明确22点不是强制要求，
现已取消所有时间准入限制：测得ETA加30分钟余量超过22点时，只保存timing.json、schedule.json并提醒，
继续全部正式训练，不改变任何组的预算，也不会为赶时间强制终止训练。
目标时间已经过去也允许启动/续跑。schedule.json记录deadline_mode=advisory、within_target及accepted=true。
只有显式`--calibrate`会在测速后正常返回。其他任务、温度和频率变化仍可能影响实际完成时间。

输出为`outputs/v23/target_dual_moe/recoverability/tonight/<指纹>/`。
中断重跑、测速、汇总都加相同的`--profile tonight`；不加该参数默认仍是原60组。
以后继续这个短预算即使原日期已过也不必修改deadline；如需更新显示的参考时间，可以使用
`--deadline "新的日期 22:00"`。它只改时间提示，不改变该profile的模型、数据或科学指纹。

### 原60组长预算

在项目内层根目录执行：

```bash
conda activate difftdi

# 只检查60组计划，不训练、不生成正式结果
python scripts/run_recoverability_experiments.py --dry-run

# 可选：只做CPU上的等观测数几何机制检查
python scripts/run_recoverability_experiments.py --mechanism-only

# 单卡完整验证（会先做短时测速，再逐组正式训练）
python scripts/run_recoverability_experiments.py --gpu 0

# 中断后仍执行上面同一条命令，只跳过经审计完整结束的任务

# 重新汇总，不训练
python scripts/run_recoverability_experiments.py --summary-only
```

如只想估算时长，使用 `--calibrate --gpu 0`；测速会对一次性模型做短训练/验证，丢弃权重，
不把测速结果记作正式实验。完整命令也会测速，并输出30%余量的预计完成时间，不保证实际时长。
单卡顺序运行，不启用DDP，不启动另一张卡；脚本不能保证服务器电源/散热稳定。

单独训练新候选：

```bash
python scripts/train_scale_completion.py --preset dual_moe_recoverability \
  --dataset TaxiBJ --mask random --rate 0.8 --gpu 0
```

这使用单跑预设 `dual_moe_recoverability.json`，不属于配对60组队列。
配对队列从原ST_DILATED预设构建各组，以实验JSON为准；不要误以为修改单跑预设会修改对照实验。

## 6. 日志、恢复与判读

输出：`outputs/v23/target_dual_moe/recoverability/comparison/<代码与配置指纹>/`。

- `protocol.json / plan.json / data_manifest.json`：代码指纹、逐组计划、数据规模。
- `configs/`：各组完整解析配置。
- `logs/`：逐组控制台.log与审计status.json。
- `runs/`：原规范层级下的train.log、val.log、test.log、metrics.jsonl。
- `geometry.json / geometry.log`：固定区域的合成几何检查。
- `timing.json`：本卡短时测速与参数规模。
- `summary.csv / summary.json / comparison.json`：结果、机制诊断、同条件同seed配对差异。

代码/配置/数据变化会创建新的指纹目录。运行期间禁止修改这些输入；开始/结束均校验，
中途变化不会标记为有效完成。即使产生test.log，也必须满足完整epoch、验证日程、
最佳权重恢复、有限MAE/RMSE、机制日志齐全及verified审计，才跳过该组。
没有磁盘checkpoint，中断单组只能从第1轮重跑；已完成的其他组不会覆盖。

新增`recovery_{mid/coarse}_{all/low/medium/high}_*`记录：

- count：缺失通道元素数量，空分箱只报count=0，不写NaN指标；
- weakness / coverage：目标条件软未约束程度/可见覆盖度；
- mae/rmse：最终输出的原始量纲误差；
- branch_mae/branch_rmse：新增局部重建支路原始量纲误差；
- weakness_error_corr：与最终绝对误差的描述性相关；无方差时省略并标记corr_defined=0。

weakness分箱预先固定为[0,1/3)、[1/3,2/3)、[2/3,1]。各组学习映射不同，分箱目标集合可能不同，
不可直接拿两组分箱均值当严格配对改善；相关性也没有自动控制密度/距离，不是因果证据。

合成检查使用64个已知仿射时空场，三种布局各12个观测，在同一批共同缺失目标上评估，
不使用任何学习先验。这只能验证数学实现及现象，不证明真实数据也符合仿射模型。
真实数据机制结论仍需进一步的固定观测数量、固定目标集合和密度/距离控制实验。

建议顺序：D对A确认整体收益；D对B确认不仅是标量支持；D对C确认软限制有作用；
D对E确认学习先验有额外作用。再看各数据集/模式/缺失率的一致性和多seed。
如果D与C/B/E相当，应降低创新性判断，不强行包装有效。两个MoE各自必要性仍需独立消融。

## 7. 本地工程验证

```bash
PYTHONPATH=src:tests python -m unittest discover -s tests -p 'test_recoverability*.py' -q
```

覆盖基归一化、退化轴、同数量不同布局、无观测极限、岭方程、区域置换、隐藏真值/NaN隔离、
历史初始化与输出兼容、各模式梯度、CPU混合精度、训练/验证/最佳恢复/测试和日志、队列配对及恢复审计。
另用三个真实数据集各一个TRAIN/VAL/TEST窗口、正式模型尺寸进行了CPU流程smoke；
这些都不是正式60组训练结果，也不构成CUDA大batch显存与训练性能的保证。
