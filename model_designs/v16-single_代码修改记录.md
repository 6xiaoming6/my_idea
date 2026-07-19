# v16-single 代码修改记录

## 1. 版本信息

- 版本：`v16-single`
- 模型名称：TA-CRC-MoE
- 中文名称：教师锚定连续残差校准混合专家模型
- 开发分支：`v16-single`
- 开发基础：`v15.1-single`
- 修改前提交：`3db79f8d20c186c63bf639d35e43a0a5414e9138`
- 修改后提交：尚未提交
- 输出根目录：`outputs/v16-single`

V16 依据 `v16-single_教师锚定连续残差校准MoE详细修改说明.md` 实现。开始修改前，`v16-single` 已安全 fast-forward 到本地完整的 `v15.1-single` 提交，未改写、删除或覆盖 V15.1 分支内容。

## 2. 实际模型结构

```text
对应实验点 V14 best.pt（仅训练期）
  ├─ 同名同 shape 主干权重 -> 初始化 V16 student backbone
  └─ frozen teacher prediction -> Base Teacher Anchor Loss

V16 student MultiScaleMoEBackbone
  -> x_base, z_f, z_m, z_c, h_main, shared/route predictions, scale gate

复用 V15.1 ScaleGuidedResidualAdapter（24维）
  -> delta_raw
  -> delta_candidate = 0.05 * scale_ref * tanh(delta_raw)

12维 observable condition
  -> ContinuousResidualCalibrator(12 -> 32 -> 1)
  -> alpha = sigmoid(logit), 0 <= alpha <= 1

x_final = x_base + alpha * delta_candidate
```

没有改动原 main Backbone 的结构与参数定义，没有增加专家、路由分支或更深残差金字塔。V16 额外参数只有 residual proposer 和小型 calibrator。

| 数据集 | Main 参数 | V16 总参数 | Residual Proposer | Calibrator | V16 新增 |
|---|---:|---:|---:|---:|---:|
| TaxiBJ | 4,682,754 | 4,734,421 | 51,154 | 512 | 51,667 |
| BikeNYC | 4,620,930 | 4,672,597 | 51,154 | 512 | 51,667 |
| CHAP | 4,677,471 | 4,729,121 | 51,137 | 512 | 51,650 |

总数比两项模块之和多 1，是历史 `DualBranchSTImputer.alpha` 标量，不是 V16 新结构。

## 3. 连续校准条件

原 V15.1 的 9 维条件保留：

```text
missing rate / temporal missing / spatial missing
mid reliability / coarse reliability
fine / mid / coarse active scale weight
candidate relative RMS
```

新增三个只依赖推理可见信息的样本级条件：

```text
shared-route hidden disagreement RMS
observed base MAE
observed candidate relative gain
```

Forward 不读取 `x_f_gt`。隐藏 target 只允许在 Loss 中生成 `alpha_star`，修改隐藏 target 不会改变 `alpha_pred`、condition 或预测结果。

校准器使用 `12 -> 32 -> 1`，最后一层零初始化，固定 bias 为 -2，初始 `alpha≈0.1192`。Residual Head 同样零初始化，因此初始 `delta_candidate=0`，`x_final` 与 `x_base` 逐值严格相等。

## 4. Oracle Alpha 与损失

对每个样本在隐藏位置使用与训练主目标一致的 Smooth-L1，搜索：

```text
[0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
```

联合阶段总损失为原主损失加：

```text
0.30 * (L_base_gt + 0.50 * L_base_teacher)
+ 0.05 * L_candidate
+ 0.10 * L_calibration
+ 0.10 * L_safe
```

V16 Full 不再使用 V15.1 的正/负/不确定二值 BCE 标签。`binary_acceptance` 消融才恢复等价的 BCEWithLogits 监督。

前 12 epoch 的 warm-up 只优化 `L_candidate`：学生主干和 calibrator 冻结，residual proposer 独立学习方向，固定 `alpha=1`。第 13 epoch 起自动解冻并切换联合损失；optimizer 在冻结前已经包含全部参数，因此解冻后无需重建 optimizer，也不会漏掉主干参数。

## 5. 教师解析与复现约束

新增 `src/stmoe_imputer/teacher_utils.py`：

- `AUTO_RESOLVE` 只在 `outputs/v14-single/<dataset>/full/model/<mask>/rate<rate>/` 查找 `best.pt`；
- 严格核对 V14 architecture、dataset、mask、rate 和 teacher seed；
- 明确指定 checkpoint 时执行同样核对，禁止静默加载错误实验点；
- V14 `main_backbone` 的 324 个 state tensor 全部按名称后缀和 shape 拷贝到 V16 `student_backbone`；
- 教师永久冻结，训练 Forward 使用 `no_grad`；
- warm-up 不计算未参与损失的教师 Forward，降低显存与功耗；
- 保存教师 branch、commit、checkpoint path、SHA256、best epoch、seed 和初始化 tensor 数量。

教师不注册为学生子模块，所以学生 checkpoint 不含任何 teacher state key。验证/测试时可以额外计算 teacher hidden MAE 作为诊断，但教师结果不参与学生 Forward，最终预测仍完全来自已重新加载的 V16 `best.pt`。

## 6. 连续诊断指标

新增日志包括：

```text
teacher/base/candidate/final/oracle hidden MAE
base vs teacher gap
candidate/final vs base gain
final vs teacher gain

alpha pred/star mean/std/min/max
alpha absolute error / RMSE / Pearson / tie-aware Spearman
alpha zero/full/middle target rate
calibration regret
final non-regression violation rate

alpha 与 missing rate、branch disagreement、observed gain、
candidate relative RMS、fine/mid/coarse weight 的相关性
```

常量或单样本 batch 的相关性稳定记为 0，不产生 NaN。Spearman 对 Oracle Grid 的重复值使用平均秩，避免普通 ordinal rank 对大量 ties 产生虚假相关。

## 7. 配置、脚本与消融

正式配置：

```text
configs/v16-single/taxibj.json
configs/v16-single/bikenyc.json
configs/v16-single/chap.json
configs/v16-single/smoke.json
```

核心消融：

```text
no_teacher_anchor
fixed_alpha
original_9d_condition
binary_acceptance
```

训练入口：

```text
scripts/v16-single/train.py
scripts/v16-single/run_validation_matrix.py
scripts/v16-single/run_full_experiments.py
```

正式 epoch 与验证周期：

| 数据集 | Epoch | val_epoch | Batch size（继承数据配置） |
|---|---:|---:|---:|
| TaxiBJ | 160 | 5 | 32 |
| BikeNYC | 120 | 2 | 16 |
| CHAP | 150 | 5 | 32 |

BikeNYC 使用设计文档建议的 100–120 范围上限 120，而不是沿用 V15.1 的 140；数据量较小且每 2 epoch 验证一次，能够更快观察过拟合，但正式配置仍关闭 early stopping，保证公平跑满。

六点三 seed 验证矩阵为：

```text
TaxiBJ random@0.4 / random@0.8
BikeNYC fixed@0.6 / fixed@0.8
CHAP fixed@0.4 / random@0.8
seed = 42 / 2026 / 3407
```

Student seed、mask seed、teacher seed 分离。默认 teacher seed 固定为 42，以复用已经完成的 V14 对应实验点；不会把 student seed2026/3407 误当作教师 checkpoint seed。

## 8. 训练、验证、测试与保存策略

通用训练器保持项目既有策略：

```text
多轮 train
-> 每 val_epoch 验证（最后一轮强制验证）
-> val MAE 改善时覆盖唯一 checkpoints/best.pt
-> 训练结束重新加载 best.pt
-> 完整 test 一次
```

`best.pt` 包含学生模型、optimizer、epoch、指标、完整配置和教师来源 metadata，只存在一个最佳文件，不会逐 epoch 堆积 checkpoint。训练、验证、测试信息继续分别写入 `.log` 和 `metrics.jsonl`。

## 9. 验证结果

标准库测试：

```bash
PYTHONPATH=src:tests conda run --no-capture-output -n difftdi \
  python -m unittest discover -s tests -p 'test_*.py' -v
```

结果：V16 新增 11 项测试及历史回归共 42 项全部通过，覆盖：

- 三个数据集实际 shape；
- zero-init 等价与 residual bound；
- Oracle 选择 0/0.5/1；
- Forward 无隐藏 target 泄漏；
- warm-up 冻结与联合解冻；
- proposer/calibrator 有限梯度；
- teacher anchor loss；
- 教师冻结和 324/324 tensor 初始化；
- 无教师 checkpoint 加载与推理；
- 四个消融可构建并 Forward；
- 历史 V14/V15/V15.1 Registry 和行为不回归。

synthetic 端到端闭环已经完成一次 train、validation、覆盖保存唯一 `best.pt`、重新加载最佳权重和 test，checkpoint 中无 teacher state key。

真实数据单 batch 验证：

| 数据集 | 实际输入 shape | V16 Loss | 初始 Alpha | 对应教师 best epoch | 结果 |
|---|---:|---:|---:|---:|---|
| TaxiBJ | `(1,2,12,32,32)` | 4.782253 | 0.119203 | 160 | finite |
| BikeNYC | `(1,2,12,24,12)` | 0.684530 | 0.119203 | 108 | finite |
| CHAP | `(1,1,7,32,32)` | 3.326745 | 0.119203 | 150 | finite |

以上只证明工程链路正确，不代表正式性能。

## 10. 建议运行顺序

先运行设计文档规定的六点三 seed 验证矩阵：

```bash
python scripts/v16-single/run_validation_matrix.py --gpu 0
```

若只想先验证命令和链路：

```bash
python scripts/v16-single/run_validation_matrix.py \
  --gpu 0 --epochs 5 --seeds 42
```

满足设计文档的性能、Base Anchor 和 Calibration 进入标准后，再运行 24 点正式实验：

```bash
python scripts/v16-single/run_full_experiments.py --gpu 0
```

完整输出全部进入 `outputs/v16-single/`，不会覆盖 V15.1 或更早版本结果。
