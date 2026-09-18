# v24-COE：三轮、四轮专家链候选

两轮并不一定不足，但它最多串联两次方向处理，在只含时间、空间两个专家时仅有四种可选路径。三轮、四轮能表达更长的交替处理与重复细化，也允许局部、较远邻域、完整时间窗口和联合时空处理参与组合。是否更好需要独立训练和消融验证，不能由路径数量直接推出。

本次在原 v24 架构中增加可配置专家池与链深度，保留原有两轮基线。三轮、四轮候选都是单尺度：所有隐状态、专家输入输出与补全结果保持原始 `T×H×W` 分辨率，没有池化、下采样或粗尺度分支。空洞卷积改变采样间隔，不构建额外尺度的表示。

## 候选结构

每一轮独立 router 从同一个专家池选择一个专家，同时执行点级共享专家；两个增量加入隐状态，再解码并回填原始观测。下一轮读取更新后的隐状态、补全值和预测变化量重新选择。专家和解码器跨轮复用，router 与残差系数按轮独立。

| 配置 | 轮数 K | 可路由专家数 E | 专家池 | 理论可选路径数 |
|---|---:|---:|---|---:|
| 原基线 / `ablation_grid/k2_e2.json` | 2 | 2 | T、S | 4 |
| `candidates/chain3.json` | 3 | 4 | T、S、TD、SD | 64 |
| `candidates/chain4.json` | 4 | 6 | T、S、TD、SD、TA、ST | 1296 |

这里的专家数只计可路由专家，另有一个每轮始终执行的共享专家。例如四轮候选总共含六个可路由专家和一个共享专家，并不是每轮新建六个专家。

| 专家 | 信息交换方式 | 期望捕获的模式 |
|---|---|---|
| T | 时间深度卷积，默认核 `(3,1,1)` | 相邻时刻的局部变化 |
| S | 空间深度卷积，默认核 `(1,3,3)` | 同一时刻的局部邻域关系 |
| TD | 时间空洞卷积，默认核 `(3,1,1)`、dilation=2 | 间隔更远的时间上下文 |
| SD | 空间空洞卷积，默认核 `(1,3,3)`、dilation=2 | 间隔更远的空间邻域关系 |
| TA | 每个网格位置在完整输入时间窗口内进行多头注意力 | 窗口内较长距离、内容相关的时间关系 |
| ST | 联合时空深度卷积，默认核 `(3,3,3)` | 同时依赖相邻时间与相邻位置的变化 |
| 共享专家 | 点级前馈网络 | 每个位置的通用特征变换 |

“期望模式”来自算子的结构偏好，并不保证训练后形成清晰分工。TA 仅访问模型输入窗口，不能据此声称获得窗口之外的周期信息。当前路由仍是每个样本、每轮对整个窗口选择一次，不是每个缺失位置分别选择。

三轮、四轮表示计算次数，不预设专家顺序，允许重复选择。例如三轮可以选择 `T→S→T` 或 `TD→S→SD`，四轮可以选择 `TA→S→ST→T`。全部 E 个专家在每轮均可选，因此理论路径数是 `E**K`；它只表示结构允许的组合数，不代表实际学到的有效模式数。

## 配置与运行

`configs/v24/candidates/` 中的文件都是 JSON 覆盖配置，需要通过 `--override_config` 合并到现有 `configs/v24/smoke.json`、`bikenyc.json`、`taxibj.json` 或 `chap_beijing.json`；不能直接作为独立的 `-c` 配置运行。覆盖文件显式设置 `fixed_path: null`，避免继承旧基线中长度为二的固定路径。

两个候选与九份消融配置统一使用动态硬路由、点级共享专家和 `residual_init=0.1`，不随深度调整残差初值。新增配置项为 `model.coe.expert_pool`、`temporal_dilation=2`、`spatial_dilation=2`、`attention_heads=4`。注意力头数必须整除隐通道数；现有真实数据的 64 通道与 smoke 的 16 通道均满足。其余数据、mask、优化器和训练轮数继承基础配置。

原始 `candidates/chain3.json`、`candidates/chain4.json` 继承第一版的 `loss.lambda_coe_balance=0.0` 与 `lambda_coe_mid=0.0`，仅以隐藏目标 MAE 训练。按方案 10.4，先观察缺失条件下的选择、专家使用率、路径和梯度，不强制异构专家平均分工。按方案第 13 节补齐的同深度结构对照、mask 协议与预算规则见 [实验设计](v24_COE_experiment_design.md)。当前启用正则的新阶段见下节。

## 四轮六专家路由辅助损失后续实验

完成第一版核心实验后，当前默认队列改为 `chain4`：使用 [`experiments/chain4_balance.json`](../configs/v24/experiments/chain4_balance.json)，4 轮共享 6 个路由专家（T、S、TD、SD、TA、ST），每轮独立 router，保留原有点级共享专家。训练使用硬 Gumbel 路由，推理按窗口选择一个专家，分辨率保持不变。

总损失为 `L = MAE + 0.01 × L_balance`。对每轮 k，在含有效隐藏目标的 batch 窗口上计算实际采样的专家使用率 `f[k,e]` 和干净 softmax 概率的均值 `p[k,e]`，再计算：

`L_balance = mean_k(6 × sum_e(stop_gradient(f[k,e]) × p[k,e]))`。

这里按四轮取平均，避免链路加深使正则权重自动扩大。该项对 batch 层面的使用失衡施加软惩罚，不要求每个窗口概率均匀，但确实引入了偏好均衡的先验；`0.01` 是本轮待验证的初始权重，不能保证消除塌缩，也不能预设均衡必然改善任务。仍记录逐轮使用率、路径、梯度、不同缺失条件下的选择和各轮辅助损失；中间监督权重保持 0。

使用 TaxiBJ 全量数据（2491/356/712 个训练/验证/测试窗口，形状 2×12×32×32）、五种结构化 mask（缺失率 0.4）、seed 7、80 epoch、每 2 轮验证、最后恢复验证最优状态测试一次，共 5 组。RTX 3090 上 batch 32 的真实批次检查出现显存不足，本轮统一设置 batch 16，每轮 156 个训练 batch，末批保留。TaxiBJ 掩码独立生成于 `outputs/v24-COE/section13/taxibj_masks_seed2026`，mask seed 为 2026。默认入口为：

```bash
python scripts/v24/run_experiments.py --study chain4 --gpu 0
```

同时修复 AMP GradScaler 每轮重建的问题：整个训练复用同一个 scaler，日志增加 `train_amp_scale_start/end`，保留实际更新数和 AMP 跳步数。新结果与旧两轮无正则结果同时存在链长、专家池、正则和 AMP 实现差异，只能视为组合方案比较，不能单独归因于路由损失；后续因果消融应固定其他条件。

在仓库根目录使用已安装 PyTorch 的 `project` 环境运行：

```bash
# 三轮、四轮候选：先检查完整训练和 checkpoint 重载流程
python scripts/train.py -c configs/v24/smoke.json \
  --override_config configs/v24/candidates/chain3.json \
  --synthetic --no_plot -n v24_chain3_smoke

python scripts/train.py -c configs/v24/smoke.json \
  --override_config configs/v24/candidates/chain4.json \
  --synthetic --no_plot -n v24_chain4_smoke

# BikeNYC：使用对应的 NPZ 与基础配置中的离线 mask CSV
python scripts/train.py -c configs/v24/bikenyc.json \
  --override_config configs/v24/candidates/chain3.json \
  --train_npz data/BikeNYC/bikenyc_train.npz \
  --val_npz data/BikeNYC/bikenyc_val.npz \
  --test_npz data/BikeNYC/bikenyc_test.npz -n v24_chain3

python scripts/train.py -c configs/v24/bikenyc.json \
  --override_config configs/v24/candidates/chain4.json \
  --train_npz data/BikeNYC/bikenyc_train.npz \
  --val_npz data/BikeNYC/bikenyc_val.npz \
  --test_npz data/BikeNYC/bikenyc_test.npz -n v24_chain4
```

不同结构应分别从头训练。旧两轮 checkpoint 不能直接作为三轮、四轮候选的等价训练结果。

重载时使用 checkpoint 保存的配置，尤其保持 `expert_pool` 的顺序一致，因为该顺序决定 router 每一列对应哪位专家。项目的 `load_checkpoint` 会拒绝专家顺序不一致的加载。默认两轮仍保留原参数名，已有两轮 checkpoint 可以继续使用。自定义专家池移除 T/S 时，也会移除对应未使用模块的参数。

## 分开检验链深度和专家池大小

直接比较两轮两专家、三轮四专家和四轮六专家，会同时改变深度和专家池，无法判断收益来自哪一项。因此提供完整的 `K∈{2,3,4}`、`E∈{2,4,6}` 九组覆盖配置：

| 轮数 | E=2：T/S | E=4：T/S/TD/SD | E=6：T/S/TD/SD/TA/ST |
|---|---|---|---|
| K=2 | `k2_e2.json` | `k2_e4.json` | `k2_e6.json` |
| K=3 | `k3_e2.json` | `k3_e4.json`（三轮候选） | `k3_e6.json` |
| K=4 | `k4_e2.json` | `k4_e4.json` | `k4_e6.json`（四轮候选） |

以上文件位于 `configs/v24/candidates/ablation_grid/`。同列比较深度，同一行比较专家池。专家数增加时也改变了算子类型和参数量，因此同一行结果检验的是这组专家池的增益，不能单独归因为“专家数更多”。如需进一步区分新增专家的贡献，可从选定配置继续做单专家移除对照。

```bash
# 一次完整的九组 smoke 检查；真实数据实验替换基础配置及数据路径
for steps in 2 3 4; do
  for experts in 2 4 6; do
    python scripts/train.py -c configs/v24/smoke.json \
      --override_config "configs/v24/candidates/ablation_grid/k${steps}_e${experts}.json" \
      --synthetic --no_plot -n "v24_k${steps}_e${experts}_smoke"
  done
done
```

正式消融应采用一致的数据划分、归一化、缺失协议、优化器设置和训练预算，并在各组使用相同的一组随机种子独立训练，报告均值与标准差。当前 CLI 没有 `--seed` 参数；多种子实验可在独立保存的基础配置副本中修改顶层 `seed`，再合并上述覆盖配置。当前离线 mask 的节点缺失协议不会随顶层种子自动变化，如需改变缺失样本，应同步管理对应 train/val/test CSV。

固定顺序对照必须显式提供与 `num_steps` 等长、且成员属于当前专家池的 `fixed_path`，例如三轮 `["T", "S", "T"]`。原 `configs/v24/ablations/fixed_ts.json` 等文件仅适用于两轮。`--override_config` 一次接受一份覆盖文件；需要把候选与旧路由消融组合时，先保存合并后的 JSON 再运行。

第一版不把路由正则作为必做对照。若后续出现持续且有干预证据的塌缩，再将 `regularization/route_balance_1e3.json` 合并到选定候选，作为独立后续实验；该结果不能混入无正则的第一版主比较。原 `no_route_balance.json` 仅显式保持当前默认 0。

## 日志与计算成本

日志按实际专家名统计每一轮的使用率和完整路径。旧两轮 T/S 基线仍使用 `TT`、`TS`、`ST`、`SS`；包含多字符专家名的路径使用 `__` 分隔，例如 `T__ST__TA`，避免把单个联合专家 ST 与两步 S→T 混淆。路由覆盖率与补全 MAE 应一起分析；更高路径多样性本身不是性能提升。

日志还记录每轮 router 梯度范数、每位专家的梯度范数、逐轮预测变化幅度，以及硬路径种类数、熵和最高路径占比，用于检查深链梯度、专家训练情况及路径塌缩。专家没有被选中时，其任务梯度可能为零，应结合多个 batch 的使用率分析。

辅助损失日志为 `l_coe_balance`（各轮平均的未加权值）、`l_coe_balance_weighted`（实际加入总损失的值）及 `l_coe_balance_step{r}`（逐轮未加权值）。第一版加权贡献恒为 0；未加权值仅作记录，不要求专家使用分布接近均匀。

硬路由训练沿用 ST Gumbel-Softmax：每轮计算全部 E 个候选专家，以获得 router 梯度，所以增加专家池也会增加训练计算和显存。推理时每个样本每轮只执行选中的一个可路由专家，加上共享专家；一个 batch 中不同样本可能选中不同专家，因此会按选择分组执行。

对于同分辨率输入，链深度增加会增加顺序计算轮数；跨轮共享专家参数并不能消除重复执行的耗时。TA 的注意力长度为时间窗口 T，各空间位置独立计算，其注意力计算量随 T 的平方增长。ST 的 `(3,3,3)` 邻域也比 T 或 S 更大。训练候选专家的执行次数可粗略对比为 `K×E`，但不同算子的单位成本不同，不能把这个乘积当作准确 FLOPs 或耗时倍率。

正式实验除 MAE、RMSE 外，应记录参数量、训练吞吐、推理耗时和峰值显存，并在相同硬件、batch size、时间窗口下比较。若显存不足而调整 batch size，应记录变更并重新核对训练公平性。所有残差系数保持相同初值有助于控制变量，但深链可能需要进一步检查梯度与中间预测稳定性。

本次提供可运行候选和消融配置，不包含完整数据集性能结论。通过 smoke 或前后向检查只能证明运行与基本行为，不能证明三轮、四轮优于两轮，也不能证明新增专家已经学到不同模式。

## 配置验证

11 份新增 JSON 已分别与现有四份 v24 基础配置合并检查，共 44 组配置均通过 JSON 解析、专家池与轮数一致性、固定路径清空、注意力头数可整除及单尺度设置检查。三轮、四轮候选内容分别与网格中的 `k3_e4.json`、`k4_e6.json` 完全一致。这是配置检查，不是完整训练的性能验证。

## 候选初版已完成的运行验证（加入路由辅助损失前）

以下为最初候选的检查记录；当前实验使用无路由正则的默认设置。新结构对照和协议工具的验证另见 [实验设计](v24_COE_experiment_design.md)。

- 全仓库 77 项 CPU 测试通过，含 16 项新增深链测试。覆盖全部 router 的任务梯度、专家实际传播范围、稀疏推理分派、逐轮状态更新、隐藏值隔离、观测保护、专家子集、配置组合及 checkpoint 重载。
- chain3 和 chain4 均完成合成数据一轮训练、验证、最佳 checkpoint 保存/重载与最终测试。
- 两个候选分别在 TaxiBJ、BikeNYC、CHAP 的前 2 个真实训练样本及对应离线 mask 上完成一次优化与评估，共 6 组检查：各轮 router 梯度非零，输出、梯度及指标有限，各轮观测值精确保留。该小批次检查不代表泛化效果。
- 新专家和完整链在 CPU bfloat16 autocast 下的训练、推理检查通过；本次没有 GPU 速度或显存结论。
- 原两轮 smoke checkpoint 使用当前代码重载后，复现原验证 MAE/RMSE，确认默认模型兼容。

当前 `dim=64`、含外层封装的参数量如下；共享专家已计入，不同深度复用同一专家池。

| 版本 | C=2（TaxiBJ / BikeNYC） | C=1（CHAP） |
|---|---:|---:|
| 两轮 / 两专家 | 91,571 | 84,194 |
| 三轮 / 四专家 | 142,257 | 131,976 |
| 四轮 / 六专家 | 211,507 | 198,322 |

这些是实现与数值检查；尚未完成候选之间的完整数据集性能对比。
