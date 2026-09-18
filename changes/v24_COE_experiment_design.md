# v24-COE 单尺度实验方案

本方案按[详细方案第 10.4、13.1–13.10 节](../model_designs/v24_Temporal-Spatial_Chain-of-Experts_详细方案.md)组织实验。第一版 `pilot/core` 检验两轮 T/S 的顺序差异、动态选择和中间状态作用，以及单轮并行对照。完成核心实验后，新增 `chain4` 阶段，直接运行四轮六专家并加入路由辅助损失；轮数 × 专家池网格仍暂缓。所有正式对照从头独立训练，不加入多尺度。

当前执行入口已按历史 v23 实验框架对齐为 [`scripts/v24/run_experiments.py`](../scripts/v24/run_experiments.py)，策略集中在 [`configs/v24/experiments.json`](../configs/v24/experiments.json)。默认运行 `chain4`，显式 `--study pilot/core` 可选择原无正则对照。

训练使用全量数据、单种子 7、80 epoch、每 2 轮验证（最后一轮必验证），关闭早停。每次验证改善时独立复制参数和缓冲区到 CPU 内存；训练后恢复最佳状态，再完整测试一次。默认不写权重文件；`training.save_best_checkpoint=true` 时改为原子覆盖唯一 `best.pt`。不保存权重的运行结束后无法再单独调用离线路径干预工具，需事先为该实验打开保存选项。

队列以源码、输入 NPZ、CSV、实验配置及 Python/依赖版本的指纹隔离结果。每项任务启动前和结束后检查文件内容；变化会停止队列，不能混入已完成对照。相同队列有文件锁，防止重复启动。只有轮数完整、验证周期正确、恢复验证最优权重并完成一次有限值测试、日志和配置一致的任务才会跳过。中断或失败任务从第一轮重跑，原日志和每次 attempt 保留。

终端只显示训练进度条；每个任务保留 `train.log / val.log / test.log / metrics.jsonl`，队列另存原始进度输出。`summary.csv` 记录完成情况和指标，`comparison.json` 记录同协议同种子的 Full 对照差值，四个固定路径都完成后仅按验证 MAE 选路径；`diagnostics.json` 汇总测试时专家使用率、路径及条件诊断。详细训练梯度仍在每轮 `metrics.jsonl` 中。`--summary-only` 只更新当前指纹的汇总，`--dry-run` 只显示计划，不训练、不创建实验目录。

当前统一使用单种子 `7`、每组 `80 epoch`，暂不进行三种子重复；短程 smoke 仅用于运行检查。原 `pilot/core` 仅用隐藏目标 MAE，`lambda_coe_mid=0`、`lambda_coe_balance=0`。新增 `chain4` 使用 `lambda_coe_balance=0.01`、`lambda_coe_mid=0`；具体定义与限制见[深链候选说明](v24_COE_deeper_candidates.md#四轮六专家路由辅助损失后续实验)。异构专家的合理使用率可以不同，仍需结合使用率、路径、梯度和任务指标判断。

## 1. 分阶段推进，而非先扩大模型

| 阶段 | 实验 | 每个数据集/缺失协议的默认运行数 | 决策用途 |
|---|---|---:|---|
| 运行检查 | 少量样本的前后向、观测保护、mask 和 checkpoint 检查 | 按需 | 只验证实现与数据兼容性 |
| `pilot` | Full、TT、TS、ST、SS、Parallel，单种子 7、80 epoch | 6 | 先确认固定顺序间的差异与动态选择的潜在空间 |
| `core` | 下表 11 个结构对照，单种子 7、80 epoch | 11 | 检验动态链、路由输入、专家输入与共享部分 |
| `chain4`（当前默认） | K=4、E=6、均衡损失权重 0.01，单种子 7、80 epoch | 1 | 在五种缺失结构上检验扩展模型及路由正则后的表现，共 5 组 |
| `depth`（暂缓） | K∈{2,3,4}、E∈{2,4,6} 九组完整网格，仅保留配置 | 当前不运行 | 后续另行决定是否研究链长和扩展专家池 |
| `optional` | Full 加 `lambda_coe_mid=0.1`，单种子 7 | 1 | 与相同 Full 结果配对检验中间监督 |

先选一个已有数据集及合法、可管理的数据子集做机制检查。子集应固定样本索引与时间范围，并对所有模型共享；训练、验证、测试保持原有时间划分，不能打散重叠窗口制造信息泄漏。确认主要机制后再扩展到 TaxiBJ、BikeNYC、CHAP 以及更多缺失率。CHAP 的目标区域、裁剪范围和上下文边界始终保持一致。

`pilot` 的固定路径比较和离线诊断用于判断是否存在可利用的差异；增加专家数不能替代这个检验。当前四轮候选作为独立后续阶段运行，三轮与九组深度网格继续保留配置。

## 2. 核心结构对照

配置均在 [`configs/v24/experiments/`](../configs/v24/experiments/) 中。每个文件是可检查的 JSON patch；计划生成器将其合并为完整配置。

| 名称 | 配置区别 | 主要检验 |
|---|---|---|
| `full` | K=2、T/S 专家池，当前状态硬路由 | 完整动态链 |
| `fixed_ts` / `fixed_st` | 分别固定 T→S、S→T | 动态选择是否胜过两种固定顺序 |
| `fixed_tt` / `fixed_ss` | 保留两轮，只执行 T→T、S→S | 收益是否仅来自同类处理深度 |
| `initial_router` | 各轮独立 router 均读初始状态，专家仍读当前输入 | 中间状态是否帮助后续路由 |
| `no_expert_state_update` | router 仍读当前状态，共享和路由专家始终读初始统一输入 | 专家间传递更新状态的价值 |
| `parallel` | K=1，T/S 对同一初始输入计算，由学习到的 softmax 权重融合 | 一次并行融合与串行处理的区别 |
| `shared_only` | 两轮仅保留共享专家执行 | 点级共享部分是否已经主导任务 |
| `routed_only` | 两轮去掉共享专家执行 | 共享部分是否提供独立收益 |
| `soft` | 两轮均使用全部专家的软混合 | 离散路径与迭代软混合的差异 |

`no_expert_state_update` 保留残差状态的逐轮累积、解码及观测回填，只冻结每轮专家读取的输入；它与“router 仅读初始状态”是两项不同的控制。`parallel` 训练和推理都执行两个候选专家；两轮 `soft` 仍存在逐轮状态传递，不能代替一次并行对照。

同一专家池、相同 K 的 Full、Initial Router 和 No Expert-State Update 都使用相同的全部候选专家训练方式，适合做机制比较。固定路径和 Shared-/Routed-Only 的实际激活参数与计算不同；Parallel 也减少了一次状态更新和解码。这些对照需要同时报告资源开销，不能仅因总参数相近而称作严格等容量、等计算实验。如结论需要更强的容量匹配，再独立增加经过测量的并行容量对照，并明确其参数调整。

来源 mask 始终固定、真实观测保持不变属于当前实现的信息边界检查。这里没有提供“移除来源信息”的受控结构消融，因此不能声称已独立证明第 13.10 节中的来源保留贡献；该项机制主张需要额外对照。

## 3. 缺失协议与数据边界

| 协议名 | 结构 | 主要问题 |
|---|---|---|
| `random_point` | 时间和空间上的随机点 | 简单缺失下是否有额外开销或性能退化 |
| `node_contiguous` | 节点上的连续时间缺口 | 时间支撑不足时的空间补偿 |
| `spatial_region` | 空间区域同步缺失 | 邻域受损时的时间信息与区域边界 |
| `spatiotemporal_block` | 连续时间段与连续空间区域同时隐藏 | 多维支撑受损时的状态传递 |
| `mixed` | 在窗口集合中混合上述不同结构，具体构成以 metadata 为准 | 不同窗口是否需要不同处理流程 |

首先使用相同的目标缺失率 `0.4` 比较五种结构，再对保留的关键对照扩展 `0.2/0.6`。结构完整性与离散网格取整可能造成实际率偏差；必须报告每个 split 的实际率均值、最小值、最大值，不能只写配置中的目标率。主实验让训练、验证、测试覆盖匹配的协议，跨缺失模式泛化另列，不能把未覆盖模式的结果作为默认保证。

当前旧 `random_mask` CSV 常为每个窗口选择空间节点并沿整段时间广播，属于节点缺失；不能把它命名为随机时空点缺失。计划生成器未传 `--mask-root` 时沿用现有 mask 并将协议标记为 `legacy_csv`，用于先完成可运行机制实验。结构化 CSV 的训练字段仍设置 `data.mask.pattern="random"`，这是 loader 的“逐窗口一行”格式标签；真实结构名保存在计划与 mask metadata 中。

生成结构化 mask 的示例：

```bash
python scripts/v24/generate_structured_masks.py \
  --train-npz data/BikeNYC/bikenyc_train.npz \
  --val-npz data/BikeNYC/bikenyc_val.npz \
  --test-npz data/BikeNYC/bikenyc_test.npz \
  --output-dir outputs/v24_masks/bikenyc_seed2026 \
  --patterns random_point node_contiguous spatial_region spatiotemporal_block mixed \
  --rates 0.4 --seed 2026
```

每个 `<pattern>/<rate>/` 保存 `train.csv`、`val.csv`、`test.csv` 和 `metadata.json`。CSV 每行对应一个原始窗口，列数为 `T*H*W`，0 表示隐藏、1 表示观测，各变量共享该空间时间 mask。相同 CSV 对所有结构、所有模型训练种子复用；mask seed 与模型 seed 分开管理。metadata 记录 NPZ 路径、实际 N/C/T/H/W、率和 SHA-256，计划生成器检查来源、维度和文件哈希。

NPZ 内的 `m_f` 优先于外部 CSV，`target_mask` 也会改变监督集合，因此结构化 mask 工具与计划生成器会拒绝这两类嵌入字段；计划生成器对沿用旧 CSV 的真实数据实验也执行该检查，避免声明的 mask 与实际监督不符。若数据含这些字段，先明确它们的含义，再单独准备可审计的实验数据副本；不能默默覆盖原协议。`available_mask` 可以保留，它与有限值检查共同决定原始可用位置，最终有效 Q 的数量及比例仍需从实际 loader 统计。隐藏率不是自动等于有效监督率。未知自然缺失不能当作真实标签。

计划生成器只读 NPZ 头部验证形状，不从配置猜测错误的数据尺寸。若小尺寸 channels-last 数组的布局与当前 loader 的推断冲突，必须先明确转换数据布局；不能通过改窗口设置来迁就实验结果。

## 4. 生成可执行计划

[`scripts/v24/build_experiment_plan.py`](../scripts/v24/build_experiment_plan.py) 只生成完整 JSON 配置、`manifest.json` 和 `commands.sh`，不启动训练，不覆盖已有计划文件。默认 `core`、单种子 7、单个已有 mask 协议，共 11 项；真实数据基础配置的训练预算为 80 epoch；`--variants` 可选择子集，`--stage` 可选 `pilot/core/chain4/depth/optional`。默认最多 120 项，超过时须显式缩小范围或修改 `--max-plans`。

先检查计划生成和少量合成数据流程：

```bash
python scripts/v24/build_experiment_plan.py \
  --base-config configs/v24/smoke.json \
  --output-dir outputs/v24_plans/smoke_core \
  --synthetic --variants full initial_router no_expert_state_update parallel \
  --seeds 7
```

合成数据随模型 seed 改变，仅用于运行检查；它不读取结构化 CSV，脚本会拒绝 `--synthetic` 与 `--mask-root` 的组合。单种子结果也不属于正式统计结论。

例如用当前 BikeNYC 数据及原有 CSV 生成 6 项基础顺序实验：

```bash
python scripts/v24/build_experiment_plan.py \
  --base-config configs/v24/bikenyc.json \
  --output-dir outputs/v24_plans/bikenyc_pilot_legacy \
  --train-npz data/BikeNYC/bikenyc_train.npz \
  --val-npz data/BikeNYC/bikenyc_val.npz \
  --test-npz data/BikeNYC/bikenyc_test.npz \
  --stage pilot --seeds 7 --epochs 80
```

已有结构化 mask 后，例如生成一个协议下的完整 core：

```bash
python scripts/v24/build_experiment_plan.py \
  --base-config configs/v24/bikenyc.json \
  --output-dir outputs/v24_plans/bikenyc_core_node_contiguous \
  --train-npz data/BikeNYC/bikenyc_train.npz \
  --val-npz data/BikeNYC/bikenyc_val.npz \
  --test-npz data/BikeNYC/bikenyc_test.npz \
  --mask-root outputs/v24_masks/bikenyc_seed2026 \
  --patterns node_contiguous --rates 0.4 \
  --stage core --seeds 7 --epochs 80
```

确认计划中的输入、预算与输出路径后，可执行新生成的 `commands.sh`。该文件调用统一队列的 `--plan manifest.json` 入口，输出保存在计划目录的 `managed/<指纹>/` 下，支持检查后跳过已完成任务。计划里的完整配置是固定的，后续不会被基础配置静默覆盖。每个 NPZ split 必须显式给出，脚本不会在缺少真实路径时默默退回合成数据。

`--stage depth` 仍保留生成九组 K/E 网格的能力，但当前不生成或启动该阶段。若后续重新开展，应固定 E 比较 K、固定 K 比较 E；E 增加时也引入不同算子和更多参数，因此结果属于这组专家池的增益，不能单独归因为“专家数量”。三轮四专家、四轮六专家是其中两格，与原两轮两专家直接比较会混合这两个因素。

## 5. 两种训练预算分别报告

**相同更新次数口径（默认 `--budget-regime updates`）。** 所有变体使用相同 epoch、batch size、数据、mask、优化器和验证规则。计划统一 `drop_last=false`，每个 epoch 完整暴露训练样本，并关闭 early stopping 以避免提前结束造成预算偏差。manifest 记录的是 `expected_update_slots`；空 Q batch、AMP 溢出会减少实际更新，因此最终使用训练日志中的 `train_optimizer_steps`、`train_seen_samples`、`train_skipped_empty_batches`、`train_skipped_amp_steps` 核对累计预算。相同种子用于配对，不能据此假设不同结构消耗 RNG 后每个 batch 的顺序完全相同。

**近似相同训练计算口径（`--budget-regime approx_compute`）。** 先在相同硬件、数据、batch size、精度和计时边界下测量各变体每 epoch 的训练秒数，GPU 测量需要预热和同步。成本配置必须含 `reference_variant`、`seconds_per_epoch` 和测量 `context`，例如结构如下；这里的数字仅展示格式，正式实验必须替换为实测值：

```json
{
  "reference_variant": "full",
  "seconds_per_epoch": {"full": 10.0, "fixed_ts": 6.5},
  "context": {"device": "填写实际硬件", "batch_size": 16, "dataset": "BikeNYC", "amp": true}
}
```

将该文件通过 `--cost-profile PATH` 传给计划生成器，使用 `--variants full fixed_ts --budget-regime approx_compute`。`--epochs` 或基础配置的 epoch 数定义参考模型总预算；每个模型用 `floor(参考总秒数/该模型每epoch秒数)` 换算 epoch，manifest 记录目标、估计总秒数和取整差额。预算不足一个完整 epoch 时脚本报错。实际训练时间仍需重新测量，特别是 GPU 调度和数据读取不同的情况下；这不是精确 FLOPs 匹配，也不是运行时严格限时器。

这一口径允许各模型的样本暴露、优化器更新数和验证机会不同，应完整披露；不可与同更新次数口径混成一张未标注预算的主表。scheduler 使用该次计划的完整训练长度，记录其配置。若实测成本配置不覆盖某个变体，脚本不推测其成本。

硬路由训练每轮计算 E 个候选，两轮两专家为 4 次候选计算，而固定 TS 为 2 次；四轮六专家为 24 次。推理硬路由每轮只选一个可路由专家，soft/parallel 执行全部候选。manifest 列出这些调用次数以及共享专家次数，但 T、S、TA、ST 等成本不同，次数不能替代 FLOPs 或耗时。最终同时报告总参数、实际参与/激活参数的统计口径、训练累计时间与吞吐、推理延迟、推理峰值显存，必要时另列训练峰值显存。推理应在相同 batch、输入尺寸和设备上测量，并报告样本内不同路径分组的调度开销。

## 6. 四路径离线诊断与干预

对同一个训练好的 K=2、E=2 checkpoint，固定输入和 mask，强制执行 TT、TS、ST、SS。每个窗口仅在有效隐藏目标 Q 上计算误差，排除空 Q 窗口。记窗口误差为 `ell_b(path)`，比较：

\[
\Delta_{\mathrm{oracle}}=
\min_{\pi\in\{TT,TS,ST,SS\}}\frac{1}{B}\sum_b\ell_b(\pi)
-\frac{1}{B}\sum_b\min_{\pi\in\{TT,TS,ST,SS\}}\ell_b(\pi).
\]

这里按有效窗口等权；它与按有效目标点数加权的总体 MAE 是两个统计量，窗口监督数不等时不能混用。oracle 用到了目标真值，只能描述事后可选择空间，不能作为部署路径或主表中的模型指标。若差距小，动态选择空间有限；若差距大但学习路由收益小，再检查路由输入和训练，而不是直接归因于专家不足。

先在验证集上选择一个固定路径，再将该路径带到测试集：

```bash
python scripts/v24/analyze_paths.py \
  --checkpoint CHECKPOINT \
  --data-npz data/BikeNYC/bikenyc_val.npz \
  --mask-csv MASK_ROOT/node_contiguous/0.4/val.csv \
  --output-dir outputs/v24_diagnostics/val \
  --max-samples 128 --device cpu --split val

# TS 仅为参数格式示例；正式执行使用上一步验证集选出的路径
python scripts/v24/analyze_paths.py \
  --checkpoint CHECKPOINT \
  --data-npz data/BikeNYC/bikenyc_test.npz \
  --mask-csv MASK_ROOT/node_contiguous/0.4/test.csv \
  --output-dir outputs/v24_diagnostics/test \
  --max-samples 128 --device cpu --split test --selected-fixed-path TS
```

工具输出 `summary.json` 与 `per_window.csv`，包括 `oracle_gap_posthoc`、学习策略相对窗口 oracle 的 regret、逐点指标和逐轮误差。默认取前 128 个窗口，是有限诊断范围，须记录实际样本索引，不能把这项诊断当作完整测试集结果；误差单位沿用 NPZ 的存储数值尺度。固定路径的正式性能基线还需要 TT、TS、ST、SS 独立训练并在验证集选择；同一动态 checkpoint 下的罕见路径可能缺乏训练，不能把退化完全归因于操作顺序。

解释路由时，在相同输入下替换或交换路径，观察误差变化，并结合该路径的训练使用率、每轮误差及更新幅度。按缺失结构、时间/空间观测支撑、连续缺口长度分析路径占比与干预效果；路径分布图本身不证明决策正确。只评估真实执行的整窗口路径，不能从不同空间块各自最优的结果拼出不存在的全局 oracle。

## 7. 汇总与判断标准

当前每项只运行种子 7、80 epoch，报告该次运行的 MAE/RMSE 和相对同种子基线的差值，不报告跨种子的均值或标准差。若后续需要验证稳定性，再另行增加种子；所有模型共享验证、测试 mask。固定顺序与 checkpoint 选择只使用验证集；不得在测试集挑出最好路径作为事先确定的基线。

主指标只覆盖实际有效 Q。当前训练入口的 MAE/RMSE 处于输入数据的数值尺度；只有掌握训练集拟合的归一化参数及逐变量变换后，才能逆变换预测和真值并报告原始量纲。缺失缩放信息时明确标注“归一化空间”，不能把这些数值直接称作流量或浓度误差。若使用 MAPE，单独说明零附近分母保护和目标定义。

时间窗口相关性强时，按独立时间块做配对统计或区块重采样，不能把每个时空点当成独立样本。当前单种子结果用于机制筛选，不能据此推断跨种子的稳定性。记录数据/CSV 哈希、代码版本、完整配置、实际训练次数和预算口径，保留失败或无收益的对照结果。

支持当前核心思路的证据应同时包括：主要缺失结构和多个种子下，Full 稳定优于并行融合与验证集选定的固定顺序；当前状态路由优于初始状态路由；实际专家状态传递有可重复收益；成本与收益相称。单张主表最好、路径更加多样或四轮优于两轮，都不足以独立支持这些机制。当前交付是实验方案、配置生成与诊断工具，不包含完整数据集性能结论。

## 8. 旧版材料与当前执行方式

- BikeNYC 五种结构、目标缺失率 0.4、mask seed 2026 的共享 train/val/test CSV 与每协议 metadata 已生成在 [`outputs/v24-COE/section13/bikenyc_masks_seed2026/`](../outputs/v24-COE/section13/bikenyc_masks_seed2026/)。
- 旧版 55 份完整配置及清单保存在 `outputs/v24-COE/section13/bikenyc_core_s7_e80/`，旧 pilot 及其日志也保留。这些配置仍是每轮验证、磁盘保存最佳权重；不修改旧清单、不自动导入新框架结果。新入口采用上面的集中配置及独立指纹目录。
- 计划生成器的 12 项针对性测试通过，包含机制配置、九组网格、实测成本换算、mask 来源/哈希验证、嵌入 mask 冲突、布局歧义和不执行训练/不覆盖已有计划的检查。
- 本次全仓库 135 项 CPU 测试通过；Full、Parallel、No Expert-State Update、三轮和四轮分别完成一轮合成数据训练、验证、最佳 checkpoint 重载与最终测试。各次训练日志均满足 `loss=l_main`、`l_coe_balance_weighted=0`，并记录实际优化器更新数。
- 五种共享缺失协议 × 11 个核心变体共 55 组真实小批次检查通过，每组使用 BikeNYC 前两个训练窗口和配置中的 64 维隐状态：前后向有限、观测精确保留、加权路由损失为 0，条件日志正常。结果记录在 [real_batch_validation.json](../outputs/v24-COE/section13/real_batch_validation.json)。这些是实现验证，不是完整训练的性能对比，也不提供 GPU 效率结论。

55 项清单用于检查覆盖范围，不要求一次全部执行。第一阶段运行已经生成的节点连续缺失 pilot：6 个模型 × 单种子 7 × 80 epoch，再根据固定顺序差异和四路径诊断决定 core 对照的优先级；depth 网格继续暂缓。

```bash
python scripts/v24/run_experiments.py --study pilot --gpu 0

# 后续需要完整 core 时使用此入口；不同 study 不自动导入结果
python scripts/v24/run_experiments.py --study core --gpu 0

# 查看计划 / 只重新汇总，不启动训练
python scripts/v24/run_experiments.py --study pilot --dry-run
python scripts/v24/run_experiments.py --study pilot --summary-only
```

前面曾提供的 `bikenyc_pilot_node_contiguous/commands.sh` 与 `bikenyc_core_no_balance/commands.sh` 仍转发到旧的单种子/80-epoch 清单，不会自动切换为本次新流程。请使用上面的统一入口。此次框架迁移只执行小样本验证和正式计划检查，不代替用户启动正式训练。
