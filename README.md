# ST-MoE Imputer — V23

## 2026-09-15 当前主力：ST_DILATED 后端

主力预设为 `configs/presets/dual_moe_st_dilated.json`：前端保持 E8/Top4，
后端8个扩张时空卷积路由专家（hidden32、dilation2）＋1个逐点MLP共享专家，Top3。
不修改模型类的历史默认行为；旧 `dual_moe_shared_topk` 仍为MLP，31组对照配置保持原定义。

```bash
python scripts/train_scale_completion.py --preset dual_moe_st_dilated \
  --dataset TaxiBJ --mask random --rate 0.8 --gpu 0
```

输出独立放在 `outputs/v23/target_dual_moe/st_dilated/`。继承原主力完整预算
TaxiBJ140/BikeNYC100/CHAP150轮（不是上一轮探索的80/100/80轮）、batch16、每2轮验证，
最佳权重只保存在CPU内存，最后恢复测试一次，不保存checkpoint。不自动启动训练。
可加 `--dry-run` 核对实际配置。入口脚本更新会改变旧实验队列的代码指纹；
已完成31组结果保留在原目录，无需重新运行历史队列。

## 2026-09-14 新旧后端验证＋时空专家结构探索（31组）

```bash
conda activate difftdi
python scripts/run_backend_architecture_experiments.py --gpu 0
```

单GPU顺序执行两阶段，不启动多卡任务。统一配置：`configs/presets/dual_moe_backend_architecture.json`。
覆盖TaxiBJ/BikeNYC/CHAP的fixed@0.4与random@0.8、seed42，保留全部原始TRAIN/VAL/TEST。
TaxiBJ80、BikeNYC100、CHAP80轮，batch16，每2轮及最后一轮验证；以验证MAE选择最佳权重，
CPU内存覆盖保留，最后恢复最佳权重测试一次，不保存checkpoint文件。80轮是匹配预算探索，不宣称充分收敛。

| 阶段 | 对照 | 要回答的问题 |
|---|---|---|
| 新旧验证（13组） | OLD_D vs NEW_MLP；TaxiBJ random@0.8另加OLD_L | 新后端整体修改是否优于旧动态融合及该点历史强对照？ |
| 结构探索（18组新增） | MLP_WIDE、ST_LOCAL、ST_DILATED；共用第一阶段NEW_MLP | 专家额外时空交互是否优于逐点MLP及近参数量控制？ |

OLD_D为原3尺度动态融合，OLD_L为原全局可学习均匀/动态混合。
NEW_MLP为8个路由专家＋1共享专家、Top3、隐藏宽32。MLP_WIDE仅把路由专家隐藏宽改为46，
每专家参数量与两个时空候选相差不足1%。ST_LOCAL/ST_DILATED为宽32的逐点输入投影后，
增加`hidden + 0.1 × PWConv(GELU(DWTemporalConv(GELU(DWSpatialConv(hidden)))))`再输出。
空间卷积核1×3×3、时间核3×1×1；local dilation=1，dilated dilation=2（时间/空间有效范围5）。
二者参数数量完全相同，时间双向交互适用于窗口内离线补全，不能用于宣称因果预测。
共享专家保持原逐点MLP，前端E8/Top4、路由器、支持度输入、损失、数据、优化策略均固定。
空间/时间交互仅使用观测构成的隐藏特征，不读取缺失真值。旧默认结构和历史配置保持不变。

第一阶段是专家数量/共享分支/输出组织/均衡项的**整体升级对比**，不能独立归因于某一项；
第二阶段仅探索路由专家结构。候选由验证集判断，测试集只做最终描述；不同数据集不混合原始量纲误差。
单seed和6个条件只用于确定后续方向，非全条件多种子结论，也不是双MoE完整2×2证明。
卷积专家是本项目的探索设计，参考的是局部时空建模原则，不声称直接复现某篇论文的专家。

脚本首先对各数据集/候选短时测速，丢弃测速权重，打印含30%余量的总时长和北京时间ETA。
目标为2026-09-15 10:00，**仅提示超时，不自动砍样本、epoch或跳过候选**。
`--dry-run`只检查31组计划；`--calibrate`只测速；`--summary-only`重新汇总。
`--stage upgrade`只跑阶段一；`--stage structure`跑阶段二并自动补齐/复用本套NEW_MLP锚点。
同一命令重跑只跳过完整训练＋定期验证＋最佳模型最终测试的任务，中断单组从第1轮重新训练。
运行期间不要修改代码/配置/数据；指纹变化会产生独立队列，不导入历史分数。

输出统一在`outputs/v23/target_dual_moe/backend_architecture/<指纹>/`：
`configs/`为逐组解析配置，`logs/`为控制台原始日志，`runs/`内保存各模型的train.log/val.log/test.log及metrics.jsonl，
根目录有plan.json、data_manifest.json、timing.json、summary.csv/json、comparison.json。
配对变化按相同数据集、模式、缺失率、seed计算，负数表示改善；不完整任务不计入有效配对。

## 2026-09-14 后端8路由专家＋1共享专家（新增模型入口）

前端仍为中/粗两个E8、Top4观测组织读出。新后端是**8个独立路由专家＋1个始终启用的共享专家**，每个位置在8个路由专家中选3个；共享专家不占Top3、不参加路由负载均衡。
全部专家读取细/中/粗隐藏表示及mask/support，独立小型MLP输出；共享专家给基础预测，路由专家给修正，最终为`shared + sum(top3_weight * routed_residual)`。
新布局不使用L的等权回退，避免未选中的专家也参与混合。现阶段是稀疏混合、密集计算，不能宣称只计算3个专家或一定提速。

```bash
conda activate difftdi
python scripts/train_scale_completion.py \
  --preset dual_moe_shared_topk \
  --dataset TaxiBJ --mask random --rate 0.8 --gpu 0
```

配置集中在`configs/presets/dual_moe_shared_topk.json`，可加`--dry-run`只检查、`--epochs`统一覆盖单次预算。
默认全量TRAIN/VAL/TEST、TaxiBJ140/BikeNYC100/CHAP150轮、batch16、每2轮验证、最佳CPU内存权重测试、不落checkpoint。
前后端均衡权重均0.001；原来的三个尺度头保留0.01辅助监督，不把8个修正专家当作三个尺度头监督。
输出`outputs/v23/target_dual_moe/shared_topk/`，日志分别记录e0–e7权重/负载、共享预测、共享加单专家预测误差。
旧三尺度D/H/L配置和历史结果均保留；新模型容量与机制均有变化，不能把与旧模型的比较称作只改变专家数的纯消融。
此入口是单次训练，不自动跳过已完成任务；旧36组稳定性队列仍是原U/D/H/L，并未被替换为新模型。

## 2026-09-13 后端稳定性验证（历史四组入口）

根据18组结果，保留前端E8/Top4，比较U等权、D完全动态、H固定半动态、L全局可学习动态强度四组。
BikeNYC random@0.4为100轮，TaxiBJ random@0.4/0.8为140轮；各跑42/2026/3407三个种子，共36组。
使用全部TRAIN/VAL/TEST，batch16、每2轮验证、单GPU顺序运行，无默认截止、不自动降低预算。

```bash
conda activate difftdi
python scripts/run_scale_completion_experiments.py \
  --config configs/presets/dual_moe_backend_stability.json --gpu 0
```

先加`--dry-run`检查36组计划；`--calibrate`只做短时测速并估时；`--summary-only`只汇总。
再次执行同一命令会跳过本套已经完整训练/验证/最佳测试的组；中断的单组从头重跑，不从中间epoch恢复。
最优权重仍在CPU内存覆盖保留，最后恢复测试一次，不保存checkpoint。

新增训练/验证/测试诊断：动态强度alpha、局部3×3观测比例分组误差、尺度专家误差相关性与胜率、权重集中比例、训练分支的裁剪前梯度范数。
原尺度权重均值/标准差继续记录，全部写入各run的`logs/train.log`、`val.log`、`test.log`和`metrics.jsonl`。
输出集中在`outputs/v23/target_dual_moe/backend_stability/<指纹>/`，包括`summary.csv/json`、分缺失率的`comparison.json`、`diagnostics.json`和`mechanism_summary.json`。

**本次科学源码和诊断改变，36组均新训练，不复用旧分数。旧配置、旧模型默认行为和已有结果保留；不要在队列运行期间修改代码/配置/数据，否则身份保护会停止队列。**
该实验检验受约束动态融合，不是完整双MoE的2×2证明，也不是新增创新点成立的保证。具体假设及日志字段说明见[全条件验证方案第9节](model_designs/20260911_V23双MoE全条件验证方案.md)。

## 2026-09-13 后端长预算复核（历史入口）

以下5组复用/13组新增描述对应当时源码状态；当前已增加稳定性功能，旧复用校验会因科学源码变化而拒绝复用。这不会删除或修改任何历史结果。

冻结模型设计和前端配置，仍端到端训练，比较Q10后端等权、SB全局可学习权重、Q11目标条件密集权重。完整数据：TaxiBJ random@0.4/0.8各140轮，BikeNYC random@0.4为100轮；三个条件×三组×seed42/2026共18组。batch16、每2轮完整验证、最佳权重保存在CPU内存，训练后恢复并完整测试一次，不保存checkpoint。

```bash
conda activate difftdi
python scripts/run_scale_completion_experiments.py \
  --config configs/presets/dual_moe_backend_confirmation.json --gpu 0
```

已核实历史coverage实验可复用BikeNYC的Q10/Q11两seed和SB seed42共5组，所以当前只需新增13组：BikeNYC SB seed2026，随后TaxiBJ两缺失率的12组。**复用不是新增独立实验，也不复制旧权重或伪造新训练日志。** 若科学源码/预设、实际配置、完整数据/掩码或完成记录不匹配，则不复用，改为正常训练。

输出统一放在`outputs/v23/target_dual_moe/backend_confirmation/<指纹>/`，含`plan.json`、`reuse_manifest.json`、新训练日志、带`result_origin/reference_suite/run_dir`的`summary.csv/json`、按缺失率隔离的`comparison.json`和收敛诊断。旧日志保留原位置，从汇总中的run_dir可定位。

`--dry-run`检查18组RUN/REUSE列表；`--calibrate`仅短测并估算剩余任务；`--summary-only`汇总。默认无截止时间，不沿用已过期的上一轮期限。同一命令可跳过已完成任务；不完整的任务从头训练。`--no-reuse`强制18组全部重跑，会产生独立指纹队列。不要并行运行两个队列或在训练中修改源码/配置/数据。详见[验证方案新增的后端复核部分](model_designs/20260911_V23双MoE全条件验证方案.md#8-2026-09-13后端长预算复核18组)。

## 2026-09-11 全条件双 MoE 验证（最新实验入口）

使用**全部 TRAIN/VAL/TEST 样本**，单卡串行117组：三数据集 × fixed/random × 四缺失率 × 四个2×2对照共96组；三个 random@0.4 点追加独立种子12组、全局可学习权重/密集前端机制对照9组。后端改用无均衡密集 Softmax；不覆盖下方历史配置。

```bash
conda activate difftdi
python scripts/run_scale_completion_experiments.py \
  --config configs/presets/dual_moe_coverage.json --gpu 0
```

默认全数据预算：TaxiBJ70/BikeNYC100/CHAP80轮、batch16、每2轮完整验证、最佳权重只存CPU内存并恢复后完整测试。严格预测目标为**2026-09-13 09:00（北京时间）**，当前短测估算117组约31.42小时（已含20%余量），不是硬截止保证。若预计超时会在正式启动前拒绝，不临时缩减个别组预算。全条件只有seed42，额外seed2026仅在预先指定的random@0.4；不宣称全部条件多种子或保证收敛。

支持 `--dry-run` / `--calibrate` / `--summary-only`；重新执行同一命令跳过完整任务。`--deadline "2026-09-13 18:00"` 仅明确放宽准入时间，不改实验指纹。`--budget extended` 使用全数据140/100/150轮的新队列，取消默认截止，预计明显更长；不要同时启动两个队列。

所有产物放在 `outputs/v23/target_dual_moe/coverage/<指纹>/`，包括逐组 `.log`、`summary.csv/json`、按缺失率隔离的 `comparison.json` 和 `diagnostics.json`。代码/数据/配置改动会进入不同指纹目录；运行中不要修改。方案和可证伪判据见 [全条件验证方案](model_designs/20260911_V23双MoE全条件验证方案.md)。

## 2026-09-11 最新候选：保留观测的目标条件双 MoE

基于已完成的 E3–E8 容量实验和两数据集路由实验，新增可选预设 `configs/presets/dual_moe_target.json`。**不是已经证明提升的最终模型**：前端中/粗尺度各8个组织专家，所有专家先聚合可见观测，各自恢复到细网格后按目标位置 Top-4；后端 fine/mid/coarse 三专家 Top-2。前/后均衡系数0.001/0.01，维度32、120轮、每2轮验证、batch4，最佳验证权重仅保留CPU内存，最终恢复后完整测试。不覆盖以下历史模型或实验。

```bash
conda activate difftdi
python scripts/train_scale_completion.py --preset dual_moe_target \
  --dataset TaxiBJ --mask random --rate 0.4 --gpu 0 --seed 42 --train-windows 2048
```

此例与上一轮 TaxiBJ 的2048训练窗口口径一致；BikeNYC改为 `--dataset BikeNYC --train-windows 511`。省略窗口参数即完整训练集；任何训练窗口选择都不截断验证/测试。输出 `outputs/v23/target_dual_moe/`。每次单任务启动会新建运行记录，不自动跳过已完成任务。

同一个入口支持 `--front-mode uniform/topk --back-mode uniform/topk` 四组对照，名称Q00/Q10/Q01/Q11分别表示关闭/开启前后路由，默认Q11。uniform保留全部可学习专家、只等权融合并关闭该侧均衡项；不是固定空间平均池化。推荐先将Q01与Q11同预算、同种子比较，按验证结果选候选；不能把更换训练子集后的分数与旧结果直接归因比较。未启用真正稀疏计算加速。

改动依据、论文来源、公式和完整四组命令见 [本轮设计说明](model_designs/20260911_观测保留与目标条件双MoE改进依据.md)。

### 三数据集对照队列（2026-09-11 16:30 预算）

```bash
python scripts/run_scale_completion_experiments.py \
  --config configs/presets/dual_moe_target_comparison.json --gpu 0
```

单卡串行12组：TaxiBJ/BikeNYC/CHAP各Q01、Q11、Q10、Q00，random@0.4、seed42。**TaxiBJ 140轮/2048训练窗口，BikeNYC 100轮/全量511，CHAP 150轮/2048**；两个2048子集均等间隔取原训练集，并按相同行号同步抽取缺失掩码；所有组完整验证/测试、val每2轮、batch4。不同数据集轮数写在JSON的`dataset_epochs`，同一数据集四组预算必须相同。

启动前仅用短训练更新测时，丢弃测时模型，估算整个剩余队列并增加20%余量；若预计超过**北京时间2026-09-11 16:30**，不启动正式任务。时限是预测准入，不是保证或强制中断；不会临时减少某组epoch。不要同时开第二个GPU训练任务。晚启动或机器负载变化可能导致拒绝开跑。

`--dry-run`检查任务和样本数；`--calibrate`只测时；`--summary-only`只汇总。重复原命令会跳过配置匹配且训练/验证/最佳权重测试均完整的组，未完整的组从头重跑。参数不落盘，最佳模型保留CPU内存并在最后恢复测试；输出`outputs/v23/target_dual_moe/comparison/<指纹>/`，包含分组配置、train/val/test.log、原始控制台日志、summary.csv和comparison.json。运行中不要修改模型、脚本、数据或配置，否则触发源码指纹保护；完成组在截止时间过去后仍可跳过和汇总。

本轮只检验一个缺失率和一种模式，不是全缺失率或多种子论文终评。等权组保留可学习空间组织，不等于固定平均池化。重点看Q11对Q01、Q10是否同时改善，配对变化与2×2交互项自动汇总；不跨数据集直接平均原始MAE，也不把单种子差异当作显著性证据。

### 双 MoE 收益独立种子复现（2026-09-11 晚间）

```bash
python scripts/run_scale_completion_experiments.py \
  --config configs/presets/dual_moe_target_replication.json --gpu 0
```

仅新增seed2026：TaxiBJ、BikeNYC各Q01/Q11/Q10/Q00，共8组；训练源码和超参数不改，TaxiBJ140轮/2048窗口、BikeNYC100轮/全511，random@0.4、batch4、val每2轮、完整val/test、内存最佳模型恢复测试。已有seed42不重跑、不覆盖；新结果在`outputs/v23/target_dual_moe/replication/<指纹>/`，其中summary仅包含本轮2026结果，分析时再与配置中`seed42_reference_suite`记录的旧目录配对，而不是假装本轮有两个种子。

已有同预算8组耗时约3小时7分钟，20%余量后约3小时44分钟。JSON的`deadline_mode=advisory`将当天20:00设为软目标：预计超时只提醒，仍按全部epoch跑完；没有早停、预算缩短或20:00强制停止。旧配置未指定该字段时仍是strict准入保护。原命令重跑继续跳过本轮已完成组；运行中不要编辑训练代码/数据/配置。

这次检验收益是否跨种子重复，不检验新的结构。CHAP上一轮的负面结果仍须保留，暂不扩展该数据集；两个种子的结果也不能当作统计显著性或全数据集普遍有效的证明。详细判定规则见[设计说明第8节](model_designs/20260911_观测保留与目标条件双MoE改进依据.md)。

### 固定前端、检验后端融合与均衡（2026-09-11 23:00预算）

```bash
python scripts/run_scale_completion_experiments.py \
  --config configs/presets/dual_moe_backend_study.json --gpu 0
```

前端结构和超参固定为各尺度E8/目标Top4、均衡0.001，权重仍端到端训练，并非冻结已有模型。后端只改变融合方式和均衡：BU等权、BD密集softmax、BK0无均衡Top2、BK1有0.01均衡Top2。BD用Top3-of-3确保与BK0/BK1路由头初始化匹配；关闭无效的硬计数均衡。三个补全专家及其辅助监督保持不变。

先完成BikeNYC四后端×seed42/2026共8组（全511训练窗口、100轮），再跑TaxiBJ四后端×seed42共4组（统一1280训练窗口、140轮）。共12组，均random@0.4、batch4、val每2轮、完整验证/测试、内存最佳模型恢复后测试一次。TaxiBJ数据子集与此前2048窗口不同，只比较本轮匹配四组，不能拿旧绝对分数作同预算对照。

GPU0短测约3.31小时（含20%余量）；JSON设置当天23:00、strict预测准入，预计超过时限不启动正式队列，不缩减epoch，不保证硬截止。原命令重跑跳过已完整组。输出`outputs/v23/target_dual_moe/backend_study/<指纹>/`，保留各组train/val/test.log、配置、原始控制台日志和配对汇总。重点比较BD对BU、BK0对BD、BK1对BK0；不是新的前后端2×2实验，不计算其交互项。具体判定见[设计说明第9节](model_designs/20260911_观测保留与目标条件双MoE改进依据.md)。

## 历史候选与对照协议（保留原行为）

当前新增 **B01基础双Top-K候选**：前端每尺度3聚合专家Top-2，后端3尺度专家Top-2，前/后端各加入0.01负载均衡。配置集中在 `configs/presets/dual_moe_topk.json`，不覆盖B01及已完成的实验。保留独立尺度预测，不使用硬残差限幅。

```bash
conda activate difftdi
python scripts/train_scale_completion.py --preset dual_moe_topk --dataset TaxiBJ --mask random --rate 0.4 --gpu 0 --train-windows 512
```

默认120轮、val_epoch=2，验证/测试完整，省略窗口参数为完整训练集。输出 `outputs/v23/topk_dual_moe/`；旧命令不加preset仍是原行为。Top-K指路由权重稀疏，当前不跳过所有未选专家的计算。实现、负载损失和日志含义见[模型说明](model_designs/v23_双MoE聚合与尺度补全模型.md)。已完成一组 TaxiBJ random@0.4 全量训练（test MAE 13.1246、RMSE 20.4789），但旧 B01 使用512样本，不能据此归因于结构改进。

### 全量2491样本的双Top-K公平对照

```bash
conda activate difftdi
python scripts/run_scale_completion_experiments.py \
  --config configs/presets/dual_moe_topk_comparison.json --gpu 0
```

沿用同一个统一脚本，不新建训练目录/入口。默认 TaxiBJ random@0.4、seed42、120 epoch、val每2轮、batch4；五组都读取原始2491训练样本、356验证样本、712测试样本，训练集数量不符直接报错，不截断也不填充。

| 组别 | 前端专家融合 | 后端专家融合 | 前/后均衡系数 |
|---|---|---|---|
| T00 | 等权 | 等权 | 0 / 0 |
| T10 | Top-2 | 等权 | 0.01 / 0 |
| T01 | 等权 | Top-2 | 0 / 0.01 |
| T11 | Top-2 | Top-2 | 0.01 / 0.01 |
| T11_NB | Top-2 | Top-2 | 0 / 0 |

保留相同的3聚合专家及fine/mid/coarse补全路径，分支辅助损失0.01、分区正则0.001一致；**等权融合仍保留可学习空间聚合器，不等于固定平均池化**。2×2检验“Top-K路由及对应均衡”的组合作用，T11_NB额外检验联合均衡约束；不是精确等训练参数量对照，关闭路由会冻结路由头。

单GPU串行，先短测时再正式训练，不会根据截止时间缩短训练。只保存最佳验证参数，训练结束重载它测试一次。输出为 `outputs/v23/topk_dual_moe/comparison/<协议指纹>/`：`protocol.json`、`data_manifest.json`、`configs/`、`runs/`、`logs/`、`summary.csv`、`comparison.json`。协议包含代码哈希及数据文件路径/大小/mtime；每个任务前检查是否变化，避免混跑。此前独立Top-K结果保留作参考，不自动导入，本轮T11重新训练。

重复同一命令自动跳过本轮已完整完成且配置匹配的实验；不完整实验从头重跑。`--dry-run`只检查数据量及任务列表，`--summary-only`只汇总，`--calibrate`只短测时。汇总包含六个成对比较和交互项 `T11-T10-T01+T00`；负交互项表示超加性误差下降，但单种子不能证明统计显著性。应看 T11 是否同时优于 T10/T01，不以测试集调整候选。

### 前端专家数量探索（3 / 4 / 6 / 8）

```bash
python scripts/run_scale_completion_experiments.py \
  --config configs/presets/dual_moe_expert_count.json --gpu 0
```

本轮只增加**前端每个尺度**的聚合专家数，不增加后端尺度数；后端始终是fine/mid/coarse三个专家Top-2、均衡系数0.01。前端E=3/4/6/8，每个E都安排等权融合U和Top-2路由K，默认共8组。前端K组均衡系数0.01，U组为0；分区正则0.001、分支监督0.01、每专家粗节点数32/8保持不变。四个E共享相同训练预算：TaxiBJ random@0.4、seed42、2491训练样本、完整356验证/712测试、120轮、val每2轮、batch4。

本阶段不落盘模型权重：此队列JSON设置 `"save_best_checkpoint": false`，统一下发到每组 `train.save_best_checkpoint`。仅在CPU内存中保留一份独立的最佳验证模型参数/缓冲区，遇到更优验证结果就替换；训练后恢复该内存快照测试，而不是用最后一轮替代最佳轮。不会创建checkpoint目录或保存优化器状态，配置、train/val/test.log、metrics.jsonl和汇总仍照常保存。完成判断依据完整训练/验证记录、最佳epoch及明确的内存最佳权重测试记录，不因缺少best.pt而重跑成功组。若进程中断，内存权重丢失，未完成组从头训练；已完成组仍可跳过。恢复保存只需将该JSON字段改为true；未配置此开关的历史训练仍默认保存，不删除旧参数。

已启动的进程不会热更新配置；源码/配置改变也会触发队列指纹保护。因此改开关后需要在自己的终端停止旧队列，再执行同一命令。新协议指纹会保留旧目录并使用新目录，不能把旧保存策略的任务自动认作新策略任务。

代码支持任意正整数 `aggregation_experts`；此探索协议限制3..16且包含E3参照，并要求同一K满足1<K<最小E。配置内 `expert_counts` / `aggregation_top_k` 与每组patch必须一致，拒绝遗漏同容量的等权对照或单独改变训练预算。默认E3两组会在新源码指纹下重跑，不把历史结果混入当前队列。原有预设仍是3专家，旧结果不删除。

输出在 `outputs/v23/topk_dual_moe/expert_count/<指纹>/`，沿用自动测时、单GPU串行、最佳验证参数重载测试、完整任务跳过和.log记录。`summary.csv`增加专家数量/前端模式；`comparison.json`同时给出同E的K对U、各E对E3的容量比较，以及 `front_routing_gain`：`uniform_minus_topk`为正表示路由有益，`gain_change_vs_E3`为正表示路由优势扩大。未完成的配对不会计算收益。

解释限制：增加E也增加参数量、总粗节点容量及实际计算/显存；K固定为2时选中比例从2/3降至2/8。这不是纯等容量跨E实验，更不是稀疏计算加速实验。同E的U/K专家初始权重保持一致（路由头除外），但跨E只固定种子，不保证共享层初始权重逐元素相同。只有同E的K稳定优于U，才支持更大专家池中前端路由有额外价值；如果两者一起变好，主要证据是容量收益。不要为扩大消融差距而选择绝对精度更差的E；单种子仅用于探索，候选应按验证集确定后再做多种子确认。

### 两数据集前端截断机制对照（明早9点预算）

```bash
conda activate difftdi
python scripts/run_scale_completion_experiments.py \
  --config configs/presets/dual_moe_routing_study.json --gpu 0
```

统一脚本和JSON，不另建训练入口。`deadline`默认北京时间2026-09-11 09:00，启动时先短测时再做截止准入；预算含15%测时余量及每组固定开销，不能保证硬截止。预计超过截止时间则不启动正式队列，不能通过临时缩减某一组epoch来凑时间。`--deadline "YYYY-MM-DD HH:MM"`覆盖截止时间；`--dry-run`只显示任务/数据量，`--calibrate`只测时，`--summary-only`只汇总。用户本地执行，未代为启动正式实验。

数据：TaxiBJ训练窗口从2491等间隔取2048，所有候选和种子复用同一选择及对应mask行；BikeNYC仅511个训练窗口，保留全量。TaxiBJ验证/测试356/712、BikeNYC验证/测试73/147均完整，不参与筛选训练窗口。两数据集random@0.4、每个模型每尺度8聚合专家、120epoch、val每2轮、batch4、同优化器/调度器/辅助项、单GPU串行。CPU内存保留最佳验证权重后测试，不生成参数文件。

| 组别 | 前端 | 后端 | 前/后均衡系数 | 每个数据集的种子 |
|---|---|---|---|---|
| U8 | 8专家等权 | 3尺度Top-2 | 0 / 0.01 | 42、2026 |
| K2 | 8专家Top-2 | 3尺度Top-2 | 0.01 / 0.01 | 42、2026 |
| K4 | 8专家Top-4 | 3尺度Top-2 | 0.01 / 0.01 | 42、2026 |
| D8 | 8专家稠密可学习softmax | 3尺度Top-2 | 0 / 0.01 | 42、2026 |
| K4_NB | 8专家Top-4 | 3尺度Top-2 | 0 / 0.01 | 42 |
| D8_BU | 8专家稠密可学习softmax | 3尺度等权 | 0 / 0 | 42 |

合计20组，16组主对照+4组机制对照，预计约6.5–7.5小时，实际取决于当次测时、起跑时刻和机器负载。D8通过现有Top-8-of-8实现，与完整softmax严格等价；保留与K2/K4一致的路由头初始化，不修改模型结构。选中全部专家时Switch-style硬计数均衡为常数，因此关闭前端该项；D8的硬分派负载天然均匀，不能据此判定权重未塌缩，应看原有门控权重均值/方差/熵及有效支持度。所有组仍实际计算全部专家，K改变不代表计算量按K下降。

预先声明判断标准：

- K4 vs K2：放宽硬截断是否改善验证精度及观测支持？
- D8 vs U8：保留全部观测分派时，自适应权重是否优于等权融合？
- D8 vs K2/K4：稠密路由是否更合适？此处同时有均衡项差异，不能完全归因为截断；K4_NB提供部分拆解。
- K4 vs K4_NB：Top-4是否仍需要前端均衡？后端均衡保持不变。
- D8 vs D8_BU：稠密前端基础上，后端路由及其均衡是否有作用？不是纯后端权重单变量。
- 候选先看各数据集配对验证MAE、双种子方向一致性、误差幅度和原始量纲RMSE；测试只作最终描述，不根据测试值自动改参数。两个种子仍不能支持正式显著性主张。若需要证明双MoE缺一不可，必须确认D8优于U8且D8优于D8_BU，并跨种子/数据集复现；不能只挑一个有利点。

输出 `outputs/v23/topk_dual_moe/routing_study/<协议指纹>/`，自动汇总 `summary.csv` 和 `comparison.json`；新增 `paired_groups`记录每项比较的预期/完成种子、配对百分比均值/样本标准差和获胜种子数。不合并不同数据集的原始MAE，不把未完成组当作结果，不把单种子机制组伪装成双种子结论。样本数变化后旧2491样本实验仅作背景，不混入本轮公平对照。运行时不要修改源码/预设，以免触发指纹保护。

## 历史候选：有界尺度修正

此前候选为 **共享细预测的有界尺度修正 MoE**（`design=anchored_scale_moe`）：取消前端动态门控，默认均匀组合三个可学习聚合器；中/粗路径产生有界修正，后端 MoE 决定使用程度，配合0.01分支监督。另提供规则、观测归一化金字塔。

```bash
conda activate difftdi
python scripts/train_scale_completion.py --dataset TaxiBJ --mask random --rate 0.4 --gpu 0 --train-windows 512
```

`--aggregation regular_grid` 切换规则聚合；省略 `--train-windows` 使用全训练集，验证/测试始终完整。支持 `--dry-run`。配置在 `configs/presets/scale_completion.json` 和 `scale_completion_grid.json`；新输出在 `outputs/v23/scale_completion/`。完整结构与限制见 [模型说明](model_designs/v23_双MoE聚合与尺度补全模型.md)。只完成流程与约束检查，效果是否提升尚待实验。

### 新模型的统一对照验证

```bash
python scripts/run_scale_completion_experiments.py --gpu 0 --deadline "2026-09-10 09:00"
```

三数据集 × fixed/random@0.4：三主方案各三种子 + 四消融各 seed42，共78组；统一120轮、每2轮验证、最多512训练窗口、完整验证/测试。单GPU顺序执行，先测时检查预算，自动跳过同协议完整任务。实测加余量约8.5小时，不保证硬截止。JSON配置：`configs/presets/scale_completion_experiments.json`；结果：`outputs/v23/scale_completion/comparison/`。支持 `--calibrate` / `--dry-run` / `--summary-only`。完整实验矩阵、限制及解读方法见 [验证方案](model_designs/v23_尺度修正MoE对照验证方案.md)。

## 前一阶段：可学习区域双 MoE（保留）

此前 `v23` 分支的模型为 **观测聚合 MoE + 多尺度补全 MoE**，实现位于
`src/stmoe_imputer/models/dual_moe.py`，使用现有训练入口，不新增版本模型文件夹。

前端根据细粒度观测和掩码，学习细节点到潜在粗节点的软分配；只固定中/粗节点数（32/8），不固定区域边界、邻域半径或成员。每个聚合专家独立组织区域，经节点注意力和时间处理后，按自身分配映射回到细网格，再组合专家。后端结合细上下文，在细、中、粗三个补全路径之间逐位置选择。一次端到端训练，不使用 V14 教师或安全残差控制器。这里是软区域，不保证连通或硬分区。

- 默认入口配置：`configs/presets/default.json`，架构 `dual_moe`。
- 合并已有数据集配置：`configs/presets/dual_moe.json`，保留各数据集通道数及 CSV 掩码设置。
- 输出：`outputs/v23/learned_regions/`，只保存一个最佳参数，最终重载后测试；与早期固定半径 V23 输出区分。
- 结构、数据流、统计含义与验证边界：[V23 说明](model_designs/v23_双MoE聚合与尺度补全模型.md)。

CPU 两轮训练—验证—测试流程检查：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/train.py \
  -c configs/presets/default.json \
  --override_config configs/presets/dual_moe_smoke.json \
  --synthetic --no_plot --name smoke_dual_moe
```

真实 V23 训练需要三份 NPZ 和 JSON 中的 train/val/test CSV；不再隐式回退到合成数据。
旧数据集配置与 V14 launcher 保留原用途，使用它们时需显式选择新架构，不能把旧入口误认为已经在跑 V23。

V23 四组路由对比（A00/A10/A01/A11，三数据集 × fixed/random @0.4，共 24 次）：

```bash
conda activate difftdi
python scripts/run_dual_moe_comparison.py --gpu 0 --deadline "2026-09-09 20:00"
```

统一策略在 `configs/presets/dual_moe_comparison.json`：80 epoch、每 2 轮验证、seed=42、batch=4；训练集均匀选最多 512 个窗口，验证/测试完整。前端 uniform 仍然学习区域分配，只关闭动态专家门控。单卡顺序执行，不启用 DDP。
`--calibrate` 仅测时，`--dry-run` 仅打印任务，`--summary-only` 汇总已有结果。原命令重跑自动跳过**同协议已完整训练并测试**的实验；失败实验重新训练，不从最佳参数假冒断点续训。
日志/唯一 best.pt/summary.json 位于 `outputs/v23/learned_regions/comparison/<协议指纹>/`。代码、配置或数据文件状态改变会产生新指纹，避免混用实验；运行中不要修改源码。截止时间是估时准入与停止新增任务的检查，不是强制杀掉正在运行的实验，也不保证服务器负载变化时准时完成。

### V23 三阶段机制诊断

```bash
python scripts/run_dual_moe_diagnostics.py --gpu 0
```

默认按 1→2→3 顺序：已有 A01/A11 检查点的验证集粗信息干预；单聚合器与三聚合器对比；A01/A11 的轻量分支监督对比。配置在 `configs/presets/dual_moe_diagnostics.json`，原数据/预算/seed 从已完成对照批次继承，不改动原模型源码或历史结果。默认新增 30 次训练，输出放在 `outputs/v23/learned_regions/diagnostics/<协议指纹>/`。
`--dry-run` 查看计划；`--stages 1` 只做验证干预；`--summary-only` 更新汇总。原命令重跑会跳过完整任务。参考实验缺失或源码/数据改变时拒绝混用旧成绩，不会偷偷重跑或替换对照。

## 历史 main / V6 说明（以下不是 V23 结构）

面向时空网格数据补全任务的 PyTorch 项目（第 6 版）。核心模型 `DualBranchSTImputer` 以 `MultiScaleMoEBackbone` 为骨干，在 TaxiBJ（出租车流量）、BikeNYC（共享单车）、CHAP（PM2.5 浓度）三个数据集上，评估 fixed / random 两种离线缺失掩码策略下的补全性能。

关键设计：多尺度表示（fine/mid/coarse）、质量感知稀疏路由（QualityRouter + TopKRoutedExpertPool）、可靠性感知跨尺度共享专家（GatedCrossScaleSharedExpert + ReliabilityAwareScaleGate）、共享-路由双分支残差融合（SharedRoutedResidualFusion）。模型仅以观测值作为输入，缺失位置真值不参与前向计算。

> **分支说明**：`main` 分支是项目的核心基线，包含经过完整实验验证的模型结构和训练流程。其他分支均在 `main` 的基础上尝试修改或优化模型结构，属于实验性探索，不代表最终方案。
>
> **分支命名规则**：
> - `single-v{i}` — 单分支架构第 i 个版本，不包含多模态辅助分支（`aux.enabled=false`）
> - `dual-v{j}` — 双分支架构第 j 个版本，在单分支基础上增加多模态辅助分支（`aux.enabled=true`）

---

## 项目结构

```
my_idea/
├── src/stmoe_imputer/          # 核心模型与训练源码
├── configs/                    # 训练配置
│   ├── datasets/               #   TaxiBJ / BikeNYC / CHAP 数据集配置
│   ├── presets/                #   合成数据默认 + smoke test 配置
│   └── policies/               #   训练策略（epochs、batch 等）
├── scripts/                    # 训练、评估、数据处理脚本
├── data/                       # 数据集与离线 mask（不提交 Git）
├── outputs/                    # 训练输出与实验索引（不提交 Git）
├── experments_report/          # 实验分析报告
├── model_designs/              # 模型设计演进文档
├── changes/                    # 代码结构改动记录
└── README.md
```

---

## 模型架构

### 总体数据流

```
NPZ 数据 → x_f_gt [B,C,T,H,W], m_f [B,1,T,H,W]

数据预处理 (transforms.py):
  x_f_obs = x_f_gt * m_f                              ← 仅观测值可见
  x_m_obs, m_m, r_m = masked_pool2d(x_f_obs, m_f, 2)  → [B,C,T,H/2,W/2]
  x_c_obs, m_c, r_c = masked_pool2d(x_m_obs, m_m, 2)  → [B,C,T,H/4,W/4]
  q_f, q_m, q_c = compute_observation_stats(m)         → [B,5] each

模型前向 (imputer.py → main_branch.py):

  x_f_obs,m_f        x_m_obs,m_m        x_c_obs,m_c
      │                   │                   │
  ┌───▼────┐         ┌───▼────┐         ┌───▼────┐
  │Embed_F │         │Embed_M │         │Embed_C │    ScaleTokenEncoder
  └───┬────┘         └───┬────┘         └───┬────┘      value+mask+scale+time+space
  h_f [B,64,T,H,W]  h_m [B,64,T,H/2,W/2] h_c [B,64,T,H/4,W/4]
      │                   │                   │
      ├───────────────────┼───────────────────┤
      │                   │                   │
  ┌───▼────┐         ┌───▼────┐         ┌───▼────┐
  │Router_F│         │Router_M│         │Router_C│    QualityRouter
  └───┬────┘         └───┬────┘         └───┬────┘     MLP(h_pool|q|scale_embed)
 gate_f[B,4]       gate_m[B,4]       gate_c[B,4]
      │                   │                   │
      └────────┬──────────┴──────────┬────────┘
               │                     │
        ┌──────▼──────┐              │
        │ ExpertPool  │ (4 experts,  │              TopKRoutedExpertPool
        │ top_k=2     │  3尺度共享)   │              STExpert = Conv3d+GELU+ResBlock
        └──────┬──────┘              │
      z_f,z_m,z_c                    │
               │                     │
        ┌──────▼──────┐              │
        │ Progressive │              │              c→m→f 渐进上采样+门控融合
        │ RouteFusion │              │
        └──────┬──────┘              │
          h_route                    │
               │                     │
               ├─────────────────────┤
               │                     │
        ┌──────▼──────────────────────▼──────┐
        │  GatedCrossScaleSharedExpert      │       可靠性感知尺度门控
        │  ├─ ReliabilityAwareScaleGate      │       MLP(209→128→3)→softmax
        │  └─ Conv1x1+2×ResBlock(concat)    │       加权融合 h_f,h_m_up,h_c_up
        └──────┬────────────────────────────┘
               │
          z_shared
               │
        ┌──────▼──────────────────────┐
        │  SharedRoutedResidualFusion │             双分支残差融合
        │  z_shared → 2×ResBlock → h_shared       │
        │  h_route → Conv+ResBlock → h_route_proj │  (+Dropout3d 0.1)
        │  h_main = h_shared + γ·h_route_proj     │  γ = sigmoid(trainable)
        └──────┬──────────────────────┘
               │
          h_main [B,64,T,H,W]
               │
        ┌──────┼──────────┬──────────┐
        ▼      ▼          ▼          ▼
    pred_head  shared_aux_head  route_aux_head      Conv3d×2
        │          │              │
  x_hat_main  x_hat_shared  x_hat_route
  [B,C,T,H,W]
```

### 关键模块

**ScaleTokenEncoder** — 多尺度时空嵌入
- `value_embed(x)` + `mask_embed(m)` + `scale_embed` + `time_embed` + `space_embed`
- 每个尺度独立参数，将 [B,C,T,H,W] 映射到 [B,64,T,H,W]

**QualityRouter** — 质量感知路由
- 输入：token 空间池化 [B,64] + 观测统计 q [B,5] + 尺度嵌入 [B,64]
- 输出：softmax(gate) [B,num_experts]
- `compute_observation_stats(m)` 返回 5 维统计量（缺失率、观测率、时间缺失分数、空间缺失分数、聚合可靠性）

**TopKRoutedExpertPool** — Top-K 稀疏专家池
- 4 个 STExpert（Conv3d→GroupNorm→GELU→ResidualSTBlock），3 尺度共享
- `top_k=2`：每样本激活 2/4 专家，加权组合输出

**ProgressiveRouteFusion** — 渐进路由融合
- Coarse(8×8) → Mid(16×16) → Fine(32×32) 逐级上采样
- 每级用 GatedFusion2 学习逐位置门控权重

**GatedCrossScaleSharedExpert** — 跨尺度共享专家
- `ReliabilityAwareScaleGate`：MLP(209→128→3) → softmax，综合 3 尺度特征+观测统计+可靠性评分，动态输出 [w_f, w_m, w_c]
- 加权拼接后经 Conv1x1+2×ResidualSTBlock → z_shared
- 默认 `shared_input_mode="pre"`：接收原始嵌入（非专家输出），与 Routed 分支互补

**SharedRoutedResidualFusion** — 双分支残差融合
- Shared：z_shared → 2×ResidualSTBlock → h_shared
- Routed：h_route → Conv3d(k1)+ResidualSTBlock+Dropout3d(0.1) → h_route_proj
- Fusion：`h_main = h_shared + sigmoid(γ) · h_route_proj`（γ 初始 sigmoid(-3)≈0.047，可学习）

### 损失函数

```python
L = SmoothL1(x_hat_main, x_gt)           # 主损失（仅 hidden 位置）
  + 0.10 × SmoothL1(x_hat_pooled, x_obs) # 跨尺度观测损失（mid+coarse）
  + 0.01 × Σ(gate_mean - 1/E)²          # 专家重要性均衡
  + 0.01 × Σ(load_mean - avg_load)²      # 专家负载均衡
  + 0.05 × SmoothL1(x_hat_shared, x_gt)  # 共享分支辅助
  + 0.10 × SmoothL1(x_hat_route, x_gt)   # 路由分支辅助
  + 0.003 × cos²(h_shared, h_route_proj) # 特征互补约束
```

### 默认超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| dim | 64 | 隐藏维度 |
| num_experts | 4 | 专家数（3 尺度共享） |
| top_k | 2 | 每 token 激活专家数 |
| c_in | 2 / 1 | TaxiBJ=2(in/out), BikeNYC/CHAP=1 |
| routing_mode | topk | 稀疏路由 |
| shared_input_mode | pre | 共享分支接收原始嵌入 |
| branch_fusion_mode | residual | h_shared + γ·h_route_proj |
| scale_mode | fine_mid_coarse | 三尺度全开 |
| route_gamma_init | -3.0 | γ 初始≈0.047 |
| route_dropout | 0.1 | 路由分支 Dropout3d |
| aux_branch | 关闭 | NullResidualBranch |

---

## 缺失掩码

支持 `fixed` 和 `random` 两种离线掩码，由 CSV 文件提供。

**fixed**：同一缺失率下所有样本共享同一个空间 mask，train/val/test 使用相同掩码。
**random**：每个样本有独立空间 mask，train/val/test 使用不同 seed 偏移生成。

```text
data/{dataset}/{fixed,random}_mask/{rate}/
├── train.csv    # fixed: 1×N | random: N_train×N
├── val.csv      # fixed: 1×N | random: N_val×N
└── test.csv     # fixed: 1×N | random: N_test×N
```

---

## 快速开始

```bash
# 安装
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .

# Smoke test（合成数据，快速验证前向+loss）
python scripts/train.py -c configs/presets/smoke.json --synthetic
```

---

## 真实数据训练

### 统一调度器（推荐）

```bash
# 单数据集、单模式
python scripts/run_experiments.py --dataset TaxiBJ --gpu 0 --mask-pattern fixed --mask-rate 0.4

# 全部数据集、全部模式和缺失率
python scripts/run_experiments.py --dataset all --gpu 0 --mask-pattern all --mask-rate all

# 使用训练策略（控制 epochs、batch、早停等）
python scripts/run_experiments.py \
  --dataset all --gpu 0 --mask-pattern all --mask-rate all \
  --experiments full \
  --training-policy configs/policies/full_model_paper.json
```

参数 `all` 展开：`--dataset all` → TaxiBJ, BikeNYC, CHAP；`--mask-pattern all` → fixed, random；`--mask-rate all` → 0.2, 0.4, 0.6, 0.8。

### 单次训练

```bash
python scripts/train.py \
  -c configs/datasets/taxibj.json \
  --train_npz data/TaxiBJ/taxibj_train.npz \
  --val_npz data/TaxiBJ/taxibj_val.npz \
  --test_npz data/TaxiBJ/taxibj_test.npz \
  -n my_experiment
```

配置中需包含离线 mask 路径（调度器自动生成）。合成数据不需 CSV：

```bash
python scripts/train.py -c configs/presets/default.json --synthetic
```

---

## 输出结构

```
outputs/{dataset}/{experiment_type}/{variant}/{mask}/rate{rate}/{timestamp}_seed{seed}_bs{bs}/
├── config.json
├── checkpoints/
│   └── best.pt
├── logs/
│   ├── train.log
│   ├── val.log
│   ├── test.log
│   └── metrics.jsonl
└── training_curves.png
```

`--name` 自动归类：`full` → `full/model`，`ablation_*` → `ablation/*`，`smoke_*` → `debug/*`。

汇总索引：`outputs/summary/experiment_index.csv` 记录每次训练的 run_dir、数据集、mask、缺失率、best epoch、best val MAE、耗时、显存等。

---

## 数据格式

NPZ 文件需包含：`x_f_gt` 或 `x_f` [N,C,T,H,W] 或 [N,T,H,W,C]。可选：`m_f`, `x_m_obs/m_m`, `x_c_obs/m_c`, `r_m/r_c`。

中粗尺度若未预存，`ensure_multiscale()` 自动从 fine 观测值通过 masked pooling 构造。

---

## 源码结构

```
src/stmoe_imputer/
├── data/
│   ├── npz_dataset.py     # NPZ 数据集加载 + 离线 mask CSV
│   ├── transforms.py      # masked_pool2d_spatial, ensure_multiscale, ensure_observed
│   ├── masks.py           # mask 生成与转换
│   ├── synthetic.py       # 合成数据集
│   └── build.py           # Dataset/DataLoader 构建
├── models/
│   ├── imputer.py         # DualBranchSTImputer（顶层封装）
│   ├── main_branch.py     # MultiScaleMoEBackbone（核心骨干，forward 编排）
│   ├── embedding.py       # ScaleTokenEncoder
│   ├── router.py          # QualityRouter
│   ├── experts.py         # STExpert, TopKRoutedExpertPool
│   ├── fusion.py          # ProgressiveRouteFusion, GatedCrossScaleSharedExpert,
│   │                        SharedRoutedResidualFusion, ReliabilityAwareScaleGate,
│   │                        AdaptiveBranchGate, ExpertEnhancedSharedInput
│   ├── blocks.py          # ResidualSTBlock
│   ├── stats.py           # compute_observation_stats
│   ├── scale_utils.py     # build_scale_active_mask
│   └── aux_branch.py      # NullResidualBranch
├── engine.py              # train_one_epoch, evaluate, build_optimizer/scheduler
├── losses.py              # compute_main_stage_loss, masked_loss, cross_scale_loss
├── metrics.py             # masked_metrics (MAE/RMSE/MAPE)
├── config.py              # 配置加载与深度合并
└── utils/
    ├── checkpoint.py      # save/load checkpoint
    ├── device.py          # get_device
    ├── seed.py            # set_seed
    └── train_logger.py    # TrainLogger（epoch/test 日志 + metrics.jsonl）
```

---

## 常用类名速查

| 类名 | 职责 |
|------|------|
| `DualBranchSTImputer` | 顶层模型，组合主分支+辅助分支 |
| `MultiScaleMoEBackbone` | 多尺度 MoE 骨干，编排完整前向 |
| `ScaleTokenEncoder` | 单尺度时空嵌入 |
| `QualityRouter` | 质量感知专家路由 |
| `TopKRoutedExpertPool` | Top-K 稀疏专家池 |
| `STExpert` | 单个专家（Conv3d+ResBlock） |
| `GatedCrossScaleSharedExpert` | 跨尺度共享专家+可靠性门控 |
| `ProgressiveRouteFusion` | 渐进路由融合 |
| `SharedRoutedResidualFusion` | 共享-路由残差融合 |
| `ReliabilityAwareScaleGate` | 可靠性感知尺度门控 |
| `ExpertEnhancedSharedInput` | 专家增强共享输入适配器 |
| `ResidualSTBlock` | 时空残差块（Conv3d×2+GroupNorm+GELU） |
