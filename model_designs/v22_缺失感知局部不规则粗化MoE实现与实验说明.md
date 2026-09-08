# V22：缺失感知局部不规则粗化 MoE

日期：2026-09-06。开发分支：`v22`。

当前优先使用第 11 节的 `--profile quick`：8 组、每组 120 epoch。原有完整实验保留，不加 profile 仍会启动 30 组长预算实验。

## 1. 版本边界

基于《研究想法讨论稿_缺失诱导尺度失真与MoE自适应不规则粗化.md》的 Idea 2，新增独立的 `v22_coarsening_moe` 架构。保留所有 V14/V21 代码和已有未提交修改，不切换分支，不覆盖旧配置、脚本或输出。

V22 不继承 V14 安全残差包装、不叠加 V21 风险校准、不保留旧的预测专家池。只复用基础时空残差块、数据接口、训练/日志/检查点设施。V22 与 V14 同时存在骨干和归一化差异，所以 V14 不能作为“仅改了粗化”的单变量对照；机制结论来自 V22 内部控制组。

本阶段检验的假设：局部区域的观测支撑与内部结构保真存在权衡；多种受约束粗化方式是否比固定区域或单个学习粗化更适合缺失恢复。尚不能宣称证明了新颖性、改进了效果，或恢复了不可观测信息。

## 2. 代码布局

- `src/stmoe_imputer/models/v_single/v22_coarsening_moe.py`：独立模型、稀疏分配/聚合/回投。
- `configs/v22/experiments.json`：模型、损失、训练预算、候选和 Core-6 点位。
- `scripts/v22/train.py`：单点训练，不继承旧版本模型配置。
- `scripts/v22/run_experiments.py`：单 GPU 顺序调度、已完成检查、只读汇总。
- `scripts/v22/train_ddp.py`：同一实验的多进程 DDP 训练与分布式评估。
- `tests/test_v22_*.py`：数学对照、梯度/边界/协议与可选真实小样本检查。
- `outputs/v22/`：本版本实验。`outputs/v22/smoke/` 单独存合成流程检查。

共享代码只增加架构注册、V22 专用损失分支和诊断指标收集，不改变旧架构的计算路径。

## 3. 数据流和结构

输入 `x_f_obs, m_f` 为 `[B,C,T,H,W]`、`[B,1,T,H,W]`。只支持所有通道共享二值缺失掩码。

1. 用 `where` 清除缺失位置占位值，只用观测值计算每样本每通道的均值、标准差。标准差下限为 1。全缺失时均值为 0、尺度为 1。最终预测逆变换回输入的原始范围。
2. 归一化值、掩码、归一化空间坐标进入共享 1×1×1 stem，得到观测特征。共享 3D 时空残差块生成用于路由和细网格恢复的上下文特征。
3. 分别直接从 Fine 构造 stride=2 和 stride=4 两级粗节点（不是先 Mid 再 Coarse）。时间维不下采样，粗节点数为 `ceil(H/s)*ceil(W/s)`。
4. 每一级有三个粗化方式：固定所属块；允许分配至所属粗节点周围 3×3 锚点；允许分配至周围 5×5 锚点。后两者由内容相似性减去空间距离先验，经过 masked softmax 得到行归一化的软分配。
5. 所有 Fine 位置，包括缺失位置，都有分配；仅观测位置向粗节点贡献证据。计算观测特征均值、覆盖比例、有效标记、观测特征方差、观测坐标质心。方差不是完整区域不确定性的保证，覆盖比例不是概率校准结果。
6. 无证据粗节点使用可学习 empty token 与有效标记，而不是把除零后的 0 当可靠观测。共享粗节点时空块处理每个专家的粗特征，同一级专家共享这部分权重。
7. 每个专家使用自己的分配矩阵回投到 Fine 网格，再通过等权或逐位置 softmax MoE 融合。绝不按粗节点编号直接相加。两个尺度与 Fine 特征拼接，经共享融合/解码块输出。

软分配是局部受约束的不规则表示，不承诺硬连通区域或任意全局功能聚类。粗节点依附固定空间锚点，粗节点卷积采用锚点邻接，而不是精确学习后的区域邻接图。

### 稀疏实现

不构建 `[B,T,N_f,N_c]` 稠密分配矩阵。只保存 `[B*T,N_f,K]` 权重及 `[N_f,K]` 索引，其中 K=1/9/25。聚合使用 scatter_add，回投使用 gather 与加权。其运算量仍随专家数量增长，不能因为用了 MoE 就宣称节省 FLOPs。

聚合和归一化累积使用 FP32。初版为 dense soft routing，不使用 Top-K，避免初期专家饿死。默认 FP32 训练；AMP 可由配置开启，但 GPU 混合精度需另外实测。

## 4. 损失、归一化和训练

主损失为缺失位置上、观测统计归一化空间中的 Smooth L1。

`L = L_missing + 0.001 * L_assignment_mass + 0.001 * L_route_balance`。

- assignment mass：限制几何区域容量相对局部均匀分配参考的偏离，不强迫观测数量均匀，不允许删除缺失位置来伪造高覆盖率。
- route balance：轻量约束平均专家使用率，不能把均衡使用本身当成有效专门化。
- 不对不规则 latent coarse token 使用固定 2×2/4×4 物理真值损失；不保留 V14 regret、多级监督或固定跨尺度约束。

默认 `dim=48, batch_size=8, lr=1e-3, weight_decay=1e-4, dropout=0.1, grad_clip=1`，cosine 调度。训练原始全量数据，不裁剪窗口，关闭提前停止。

配置中的 `model.main.use_router=false` 仅停用旧预测专家的统计接口，不会关闭 V22 粗化路由；实际路由由 `model.v22.mode` 控制。两通道默认模型的总参数为 523,257；fixed / single_local / uniform / moe 的可训练参数分别为 511,443 / 514,515 / 517,587 / 523,257（包含通用包装器的参数）。相同初始化不等于不同分支模式下的 dropout 随机轨迹完全相同。

| 数据集 | epoch | val_epoch |
|---|---:|---:|
| TaxiBJ | 160 | 5 |
| BikeNYC | 140 | 2 |
| CHAP | 150 | 5 |

每隔 val_epoch 验证，最后一轮也验证。仅当验证 MAE 更低时覆盖 `checkpoints/best.pt`。完整训练后重载 best，测试一次。train/val/test 的 MAE、RMSE 使用逆变换后的原始输入范围；训练损失不与旧版本原始尺度 loss 直接比较。

当前 NPZ 已提供完整训练目标，主损失允许使用训练掩码位置真值；模型输入和路由不读取这些真值。并不声称此版本已经适配只有不完整训练真值的纯自监督设定。

## 5. 消融矩阵

| variant | 粗化和融合 | 目的 |
|---|---|---|
| fixed | 固定 masked mean；不显式提供 5 维统计 | 固定粗化参照 |
| fixed_stats | 固定粗化＋5 维统计 | 统计增强是否已经足够 |
| single_local | 单个局部自适应粗化 | 多专家是否必要 |
| single_wide | 单个宽范围自适应粗化 | 更大范围是否已能解释收益 |
| uniform | 三个专家等权融合 | 排除简单集成解释 |
| moe | 三个专家动态融合 | 完整候选 |
| moe_no_stats | 动态融合但路由显式统计置零 | 显式统计的增益 |

`moe_no_stats` 不是完全 mask-blind：共享特征仍编码了掩码。fixed 的 stem 仍包含位置和掩码；它只去掉粗节点处的额外统计。

所有模式按同一顺序构造模块，保证相同 seed 下共享参数初始值相同。未使用分支冻结，不计为可训练参数。单专家与多专家实际 FLOPs 不同，日志里的总/可训练参数量都应报告；这里没有伪称严格计算量匹配。后续需要参数量/耗时匹配基线、三个 seed，以及同一完整样本下等观测率不同几何的机制对照。

## 6. 执行顺序

在项目内层根目录 `my_idea` 执行。下面保留单卡命令；双卡 DDP 使用第 10 节命令。不同实验仍顺序运行。

### A. 先验证命令和数据路径（不训练）

```bash
python scripts/v22/run_experiments.py --gpu 0 --dry-run
```

### B. 可选合成流程检查（不能当成真实数据适配证据）

```bash
python scripts/v22/train.py --dataset BikeNYC --mask fixed --rate 0.4 --variant moe --smoke --cpu
```

### C. 正式 Core-6：默认五候选 × 六点，共 30 组

```bash
python scripts/v22/run_experiments.py --gpu 0
```

点位是 TaxiBJ fixed/random@0.6、BikeNYC fixed/random@0.4、CHAP fixed/random@0.4，seed=42。此处 fixed 表示项目已有固定掩码，不自动等价于连续大块缺失。

如果先只在 BikeNYC 预筛：

```bash
python scripts/v22/run_experiments.py --gpu 0 --datasets BikeNYC
```

随后执行默认命令，会跳过已完成且协议匹配的 BikeNYC 实验。`--epochs 1` 是真实全量数据上的一轮训练，不是限制一个 batch；与正式预算的 fingerprint 不同，不能冒充完成。

### D. 加强单专家/统计消融

```bash
python scripts/v22/run_experiments.py --gpu 0 --variants single_wide moe_no_stats
```

### E. 根据验证集决定是否升级，而非反复根据测试集筛选

```bash
python scripts/v22/run_experiments.py --summary
python scripts/v22/run_experiments.py --gpu 0 --variants fixed_stats single_local uniform moe --seeds 42 2026 3407
```

确认机制对照有效后，再冻结方案运行 24 点：

```bash
python scripts/v22/run_experiments.py --gpu 0 --points full24 --variants fixed_stats moe
```

调度按 fixed 再 random 的点位顺序运行，每个点位依次跑候选。失败即退出，不自动反复重试；修复后重跑同一命令。中断实验从头重训，并非加载 last checkpoint；只有已完成的实验被跳过。

脚本默认通过 conda 的 difftdi 环境调用通用训练器；已在合适 Python 环境中时可传 `--conda-env current`。路径以脚本位置解析，不依赖外层/内层目录习惯。

## 7. 输出与续跑判据

`outputs/v22/{dataset}/ablation/v22_{variant}/{fixed|random}/rate{rate}/{timestamp_seed_bs}/`

包含 config.json、checkpoints/best.pt，以及 logs/train.log、val.log、test.log、metrics.jsonl。常规日志保持已有紧凑格式；完整专家使用率、分配熵、位移、无证据比例、覆盖率诊断保存在 metrics.jsonl，测试诊断也进入 test.log。

只有配置/源码 fingerprint 相同、达到指定训练轮数、存在最佳验证记录与 best.pt、测试记录恰好一次且 MAE/RMSE 有限、测试 best_epoch 匹配且训练正常结束，才跳过。fingerprint 包含数据文件大小/mtime，属于文件身份检查而非完整内容哈希；修改数据应保留可追踪记录。

`--summary` 只输出匹配当前协议的 JSON 汇总，不挑测试值最好的 run，不写报告，也不启动训练。

## 8. 初次验证状态与限制

开发验证在 CPU 上执行，不占用 GPU 训练，也不修改其他运行进程：

- 合成单元检查：masked mean 数学一致性、稀疏/稠密参考一致性、梯度、共享初始化、全缺失/全观测/常数、隐藏真值隔离、不同尺寸、CPU bfloat16。
- 实际通用训练入口已完成 BikeNYC 尺寸合成数据 1 epoch → val → best.pt → reload → test，检查点只保留 best。
- 三个真实数据集 × fixed/random × 七个模式，共 42 个 CPU 小样本流程通过：分别从真实 train/val/test 读取首个窗口，各一次训练/验证/测试，保存与加载内存检查点，指标有限。流式读取 NPZ 首窗口，不加载完整训练数组。
- 旧 V21 系列回归检查通过，V14 回归也单独执行。
- V22 与选定 V14/V21 回归合计 47 个单元/协议检查通过。最新 TaxiBJ 尺寸合成入口也跑通；续跑识别函数确认完成，目录仅有一个 `best.pt`，metrics.jsonl 只有一次最终 test。

这些仅是可运行性检查。没有完成正式 30 组训练，没有证明优于 V14，没有验证真实 GPU 整轮显存峰值和长时间稳定性，也没有提供发表或新颖性保证。

可重跑真实流程检查：

```bash
V22_REAL_SMOKE=1 PYTHONPATH=src:tests python -m unittest test_v22_real_smoke
```

## 9. 应当停止或修改路线的情况

如果 fixed_stats 已达到相同效果，说明额外粗化机制可能没有必要；如果 single_local/single_wide 不逊于 MoE，则多专家必要性不足；如果 uniform 与 moe 相同，则路由没有独立增益。不能只看到最终模型比最弱 fixed 好，就宣称 MoE 不规则粗化成立。

## 10. 双 GPU DDP 基础验证（2026-09-06 新增）

此次不修改 V22 模型结构。两张 GPU 共同训练同一模型，同步梯度；实验之间顺序执行。不是 GPU0/GPU1 各跑一个实验，也不是两个独立窗口同时启动同一命令。

### 先做一次实际双卡启动检查

```bash
python scripts/v22/train.py --dataset BikeNYC --mask fixed --rate 0.4 --variant moe --gpus 0 1 --smoke
```

这是合成数据 1 epoch 的 train/val/test 流程检查，输出独立放在 smoke 子目录，不说明真实任务效果。然后可做真实全量数据的一轮检查：

```bash
python scripts/v22/train.py --dataset BikeNYC --mask fixed --rate 0.4 --variant moe --gpus 0 1 --epochs 1
```

### 基础效果实验

```bash
python scripts/v22/run_experiments.py --gpus 0 1
```

默认 5 个方法 × Core-6 × seed42，共 30 组，每组两卡共同训练。完整数据、160/140/150 epoch、原来的验证间隔；不是五轮测试。可先加 `--datasets BikeNYC` 跑 10 组。检查配置而不运行用 `--dry-run`。

基础实验依据：fixed→fixed_stats 检查统计增益；fixed_stats→single_local 检查自适应粗化；single_local→uniform 检查多分支互补；uniform→moe 检查路由。主要以相同点位验证 MAE 判断，最终 test 只报告冻结检查点。若初步有效，补 single_wide 并进行三 seed 重复；本轮不足以证明完整不规则粗化的创新成立。

### batch 与执行语义

`configs/v22/experiments.json` 中：

```json
"distributed": {
  "per_rank_batch_size": 4,
  "timeout_seconds": 1800
}
```

默认每卡 4，全局 batch=8，与单卡基线 batch=8 对齐；学习率不自动翻倍。DDP 仍有采样、dropout 和正则项聚合差异，不能声称与单卡逐步数值一致。若减小 per_rank_batch_size，所有对照应使用相同设置，不要混用不同预算的结果。

- 启动器设置 CUDA_VISIBLE_DEVICES=0,1，再通过 `python -m torch.distributed.run --standalone --nproc_per_node=2` 启动；内部设备是 local_rank。
- 训练使用 DistributedSampler，每轮 set_epoch；为保持两进程训练步数一致，样本数不能整除 rank 数时补齐少量重复样本，重复数量写入 train.log。
- 验证/测试使用无补齐分片，不复制样本。各进程运行未包装的模型进行评估，允许不同 batch 数乃至空分片，避免 DDP forward 同步死锁。
- MAE/RMSE/MAPE/WAPE 汇总误差分子和分母，绝不直接平均各卡 RMSE。训练任务损失梯度按当前全局 batch 的缺失元素数加权；辅助正则仍按 rank 平均。
- 只有 rank0 写 config、train.log、val.log、test.log、metrics.jsonl 和 best.pt。best 临时写入后原子替换，不保留多轮检查点；所有进程重载同一 best 后分片完成一次 test。
- epoch 时间/显存记录各 rank 最大值。完整训练记录、分布式配置和 world size 纳入完成检查；单卡、双卡、1 epoch、smoke 不相互冒充。
- 禁止自动重启失败 rank。任一 rank 出错，由 torchrun 结束同组进程；修复后重跑命令，跳过已完成且配置匹配的组，中断组从头训练。

双卡结果汇总也需传相同配置：

```bash
python scripts/v22/run_experiments.py --gpus 0 1 --summary
```

### 安全边界与已验证范围

DDP 并不限制显卡功率，也不能保证解决之前的整机宕机。两卡可能同时高负载；每个进程还会各自加载 NPZ 和掩码，主机内存开销高于单进程。先确认没有其他任务占用目标显卡，再自行进行双卡试跑。本次没有设置硬件功率上限，没有关闭其他任务，也没有主动启动 GPU 训练。

已通过两个 CPU/Gloo 进程的真实分布式检查：7 个候选分别执行连续多步训练，参数同步；3 样本不等长评估分片和 1 样本/空分片均通过，汇总指标与单进程精确统计一致。另已用实际启动入口跑通双进程合成 train→val→best→reload→test。GPU/NCCL 与长时间双卡稳定性仍需本地验证。

```bash
PYTHONPATH=src:tests python -m unittest test_v22_ddp test_v22_protocol
```

## 11. 精简对照：8 组 × 120 epoch

为先观察当前原型是否出现有意义的差异，新增 JSON 命名策略 `profiles.quick`，不修改原来的完整预算、不改变模型结构，不增加其他目录。

```bash
python scripts/v22/run_experiments.py --profile quick --gpus 0 1
```

- 只跑 BikeNYC 的 fixed@0.4、random@0.4；保留两种缺失模式，不对三个数据集做全面结论。
- 四个候选：fixed_stats、single_local、uniform、moe。依次检查局部自适应粗化、多分支互补、动态路由。删去较弱 fixed 对照，使用带统计信息的固定粗化作为基线。
- seed=42，每组完整读取训练集训练 120 epoch，每 2 epoch 验证一次，共 60 次验证；只保存验证 MAE 最好的 checkpoint，最后测试一次。
- DDP 每卡 batch=4、全局 batch=8；四个候选使用相同预算、归一化、数据掩码和学习率设置。cosine 调度按 120 epoch 执行，不截取长预算训练的前 120 轮冒充同协议。
- 共 8 组，累计 epoch 数由原来的 4500 降到 960；这是预算数量比较，不是承诺运行时间按相同比例下降。
- 输出独立放在 `outputs/v22/quick/`，支持中断后重跑跳过已经完成且协议匹配的组。

如果之前的长预算命令还在运行，请在启动它的终端按 Ctrl+C，确认 torchrun 子进程退出后再启动 quick。修改配置不会缩短已加载旧配置的训练。本次不主动结束用户进程。

检查命令和后续汇总：

```bash
python scripts/v22/run_experiments.py --profile quick --gpus 0 1 --dry-run
python scripts/v22/run_experiments.py --profile quick --gpus 0 1 --summary
```

中断后继续运行同一命令即可，也可以显式写出默认的跳过策略：

```bash
python scripts/v22/run_experiments.py --profile quick --gpus 0 1 --skip-completed
```

启动时显示 `completed / skip / pending / pending_epoch_budget`，已完成组打印 `SKIP`。
只有配置、代码与数据指纹匹配，达到完整 epoch 预算、存在最佳检查点，且验证及最终测试成功的组才会跳过。
中断组会新建运行目录从第 1 轮重跑，不从中断 epoch 或 best.pt 续训，也不删除旧日志和检查点。
`--dry-run` 只显示跳过项和待执行命令，不启动任何训练子进程。
不要加 `--rerun-completed`，除非确实需要重跑已完成组；该选项与 `--skip-completed` 互斥。
本功能只是队列级继续执行，不修复 SIGILL 或保证双卡硬件稳定。

### 判读边界

这是短预算、单种子的探索，不保证一定出现差异，也不能凭此证明显著性或最终收敛效果。优先对比同点位的验证 MAE，并查看最后数次验证趋势；如果最优点都集中在最后几轮且曲线仍明显下降，不宜因短预算排名否定某个候选。若 uniform≈moe，不能归功于动态路由；若 single_local≈moe，尚无证据支持多专家必要性。

只有差异在两种掩码下较一致且不是单次验证波动时，再考虑扩展 CHAP 或多 seed。若需要延长，应给比较中的候选统一预算重跑，不能只给表现较差或较好的候选额外训练。120 epoch 的结果不与旧 30 epoch 或完整预算混作公平对照。已经启动的任务不会自动变成 120 轮；旧 30 轮完成记录也不会被识别为新预算已完成。

## 12. Quick 后续验证：多种子、路由干预与固定几何对照

入口统一为 `scripts/v22/verification.py`，策略为 `configs/v22/verification.json`。
不修改历史 V22 模型文件、训练入口或其源代码指纹，所以原来 8 组 quick 结果仍然有效。
正式数据训练由用户手动启动；以下命令均在项目内层根目录执行。

### 12.1 推荐执行顺序

第一步：补充三种子稳定性实验。

```bash
python scripts/v22/verification.py --stage multiseed --gpus 0 1
```

- BikeNYC fixed/random@0.4，候选 fixed_stats、uniform、moe，seed=42/2026/3407。
- 每组 120 epoch、val_epoch=2、每卡 batch=4，全局 batch=8，仍按验证 MAE 选 best，最后 test 一次。
- 共 18 组；当前 6 组 seed=42 结果可直接复用，只新增 12 组。
- 各 seed 仍使用同一组离线掩码：检验训练随机性，而不是新掩码泛化。
- 自动跳过完整且匹配的结果；中断组新建目录从头训练，不覆盖旧记录。

第二步：对最佳 MoE 检查点执行验证集路由诊断，不进行训练。

```bash
python scripts/v22/verification.py --stage diagnose --gpus 0 1 --diagnostic-device cuda:0
```

`--gpus 0 1` 在该阶段只用于匹配原 DDP 实验协议；推理仅使用 `--diagnostic-device` 指定的一张卡。
也可指定 `--diagnostic-device cpu`。不要让它和训练队列争用同一张 GPU。
当前只有 seed=42 时也可运行；缺失检查点会明确打印 MISSING，完成多种子后再运行会诊断全部 6 组。

每个验证 batch 在同一 checkpoint 上做 9 次前向：

1. 原始 learned 路由。
2. 两尺度同时替换为 uniform；保持专家和解码器参数不变。
3. 两尺度路由权重分别在每个样本内置乱时空位置，保留专家总体使用量。
4. 每次只干预一个尺度，强制选择专家 0/1/2；两个尺度共 6 个条件，另一尺度保持原路由。

所有条件均以原始数据单位、仅在验证集缺失位置计算 MAE/RMSE 等指标。
模型前向不接收 `*_gt`；预测完成后才用验证真值计算指标，不训练、不读取测试集、不用 oracle 产生部署预测。
在缺失位置额外统计路由熵、专家使用量、路由首选与最小干预误差的一致率，以及首选/软权重/等权的干预误差 regret。
误差一致率以跨通道平均绝对误差判定，并处理并列最优。

判读边界：专家输出是潜特征，不是独立预测头。强制选择改变整层空间特征，随后还经过非线性解码器，因此上述误差是**干预代理量**，不是严格的独立专家风险。
事后 oracle 是各位置的最小干预误差，仅作诊断，不是可实现的模型指标。
已训练 MoE 改为平均后变差，可以说明模型依赖路由与专家的共同适配，但不能证明它优于从头训练的 uniform；后者仍需三种子实验检验。
置乱只使用一个固定随机种子，是探索性检查，不作为独立显著性检验。

原路由重算 MAE 必须在配置容差内复现 checkpoint 记录，否则保留结果并报错，不当作有效诊断。
默认相对容差 1e-4、绝对容差 1e-5，实际偏差、设备、PyTorch 版本和容差均记入结果。
这是允许 CPU/GPU 和分批归约的小幅数值误差，并非允许更换模型、数据或权重。

第三步：固定几何对照，已扩展为 12 组（两候选 × 两模式 × seed=42/2026/3407）。已有 4 组 seed=42 结果可复用，本次只新增 8 组，每组仍训练 120 epoch。

```bash
python scripts/v22/verification.py --stage controls --gpus 0 1
```

| 候选 | 几何分配 | 粗尺度处理器 | 用途 |
|---|---|---|---|
| uniform_fixed_shared | 三路相同固定分区 | 共享 | 重复计算和训练时多次 dropout 的负对照 |
| uniform_fixed_independent | 三路相同固定分区 | 三套独立参数 | 同几何多分支容量/集成对照 |
| 原 uniform | 固定/局部/更宽自适应三种分区 | 共享 | 几何多样性候选，复用旧结果 |

共享对照在 eval 下三路相同，不应称为真正的独立集成；无 dropout 时同初始化输出应与 fixed_stats 一致，已有测试验证。
独立对照在潜特征层等权融合，共享最终解码器，不是三个独立端到端模型的预测集成；其参数量增加，**不是参数量严格匹配实验**。
独立处理器各自初始化；其余已初始化的主干保留。所有固定对照禁用自适应分配投影，冻结不参与计算的参数以兼容 DDP。
对照 builder 只在专用 worker 进程注册，复用既有 DDP 训练/验证/最佳保存/测试流程，不影响普通 V22 启动器。
对照使用独立 variant 目录，并将实现文件的 SHA256 写入有效配置，防止与普通 uniform 或修改后的对照相互冒充。
JSON 中 `control_seeds` 已设为 `[42, 2026, 3407]`，执行同一命令会自动补齐缺少的种子。本次只调整调度种子列表，不改变模型、单组训练配置或已有 seed=42 的实验指纹。

第四步：汇总。

```bash
python scripts/v22/verification.py --stage summary --gpus 0 1
```

按候选/缺失模式统计原尺度验证和测试指标的均值、样本标准差，列明实际完整 seed 数。
同时计算相同 seed 配对的 uniform−fixed_stats、moe−uniform、uniform−两个固定几何对照差值（负数更好）。
不会把不同缺失模式混成同一组，也不会把单个 seed 的标准差写成 0；缺失结果明确列出。
纳入当前最佳 epoch 匹配且基准 MAE 复核通过的最新路由诊断。
三个 seed 仅提供初步稳定性证据，不自动宣称统计显著或证明创新性。

### 12.2 输出与安全

- 训练：沿用 `outputs/v22/quick/BikeNYC/ablation/v22_<variant>/...`，train/val/test.log 和 best.pt 规则不变。
- 路由：对应运行的 `logs/route_validation_<时间>.json`、同名 `.log`；不覆盖原始指标或旧诊断。
- 汇总：`outputs/v22/verification_summary_<时间>.json`、同名简洁 `.log`；不新增杂乱目录。
- 所有阶段都支持 `--dry-run`，不执行训练/推理或写汇总文件。
- DDP 沿用双卡协同训练，同一时刻只跑一组；遇到进程失败立即停止，不自动重试，不承诺解决此前 SIGILL。

```bash
python scripts/v22/verification.py --stage multiseed --gpus 0 1 --dry-run
PYTHONPATH=src:tests python -m unittest test_v22_protocol test_v22_verification
```

新增测试覆盖协议和历史指纹保持、共享对照 eval 等价性、独立分支参数不共享及 checkpoint 重载，以及两个新对照真实 CPU/Gloo 双进程 train→val→best→reload→test（包括连续两步反向传播和不等长验证分片）。这不是长时间 GPU/NCCL 稳定性保证。
