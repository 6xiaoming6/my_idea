# V14 可微 Top-K 负载感知路由完整探索方案

> 基础版本：`v14-single`  
> 路线代号：`RLB`（Routing Load Balancing）  
> 路线性质：单变量、单阶段、端到端训练  
> 核心目标：让训练目标真正感知并纠正 hard Top-K 的实际专家选择失衡  
> 结构约束：不增加网络层、不改变专家结构、不改变 `top_k`、不增加模型参数  
> 评价原则：原始范围 MAE 为主，RMSE 为安全指标，路由统计只用于解释机制  

## 1. 为什么选择这条新路线

E02 已经通过 Core-6、多种子、24 点全量和低缺失率诊断被完整证伪：

- E02 全量只有 10/24 个明确 MAE 胜点；
- TaxiBJ `random@0.2` 出现 MAE `+10.332%`、RMSE `+16.581%`；
- 把权重从 `1e-4` 降低至 `5e-5` 或 `2.5e-5` 仍不能解决 TaxiBJ 和
  CHAP 的退化；
- 因此不应继续沿 residual 正则权重微调，也不应把多个 residual 技巧
  组合起来。

本路线转向 V14 内部另一个与 E02 正交的问题：MoE 路由训练信号。

V14 当前同时记录：

```text
l_importance：基于 soft gate 概率的专家重要性均衡
l_load：基于 hard Top-K selected_mask 的实际负载均衡
```

但代码中的 hard mask 来自：

```python
top_values, top_indices = torch.topk(gate, k=top_k, dim=-1)
selected_mask = torch.zeros_like(gate)
selected_mask.scatter_(1, top_indices, 1.0)
```

随后当前 `l_load` 使用：

```python
load = selected_mask.mean(dim=0)
l_load = ((load - load.mean().detach()) ** 2).sum()
```

`top_indices` 是离散索引，`selected_mask` 不具有梯度。自动求导检查表明：

```text
selected_mask.requires_grad = False
l_load.requires_grad = False
总 balance 梯度与仅使用 l_importance 时完全相同
```

这意味着当前日志中的 `l_load` 可以反映专家选择不均衡，却不能通过反向
传播直接改变路由器。当前 `lambda_load_balance` 实际只给总 loss 增加了
一个无梯度数值。

这是一个明确、局部、可以被实验否定的优化问题，适合作为下一条探索路线。

## 2. 现有结果中的机制证据

对 V14 的 24 个正式实验、3 个尺度进行汇总后，hard Top-K 选择存在明显
集中现象。以下“槽位”指一个实验中的一个尺度—专家组合。

| 尺度 | 总槽位数 | 接近从不选择 | 接近始终选择 |
|---|---:|---:|---:|
| Fine | 96 | 41 | 41 |
| Mid | 96 | 37 | 36 |
| Coarse | 96 | 36 | 38 |

部分代表点包括：

| 实验点 | 代表性 hard load 现象 | `l_load` |
|---|---|---:|
| BikeNYC `random@0.2` | 某些尺度接近 `[0, 1, 0, 1]` | 0.8659 |
| BikeNYC `random@0.8` | fine/coarse 接近固定选择同两个专家 | 0.7810 |
| TaxiBJ `fixed@0.2` | fine、mid 均出现固定专家组合 | 0.5000 |
| TaxiBJ `fixed@0.4` | mid/coarse 接近固定专家组合 | 0.4940 |
| TaxiBJ `random@0.2` | mid/coarse 选择明显集中 | 0.4110 |
| CHAP `random@0.2` | 多个专家接近始终选择或从不选择 | 0.3220 |

更重要的是，部分实验的 soft gate 熵并不低，但 hard Top-K 仍长期选择相同
专家。这说明：

```text
soft importance 相对均衡
≠
hard Top-K 实际选择均衡
```

因此，单纯提高现有 `l_importance` 权重不能准确检验这一问题；必须让
辅助损失同时感知 soft 概率和 hard 选择。

## 3. 核心假设

### 3.1 待验证假设

V14 的部分性能瓶颈来自：

```text
soft gate 只受到 importance 均衡约束
        ↓
相邻概率的排序仍可长期固定
        ↓
hard Top-K 长期只选择少数专家
        ↓
共享专家池中的部分专家训练不足
        ↓
专家多样性和跨缺失模式泛化能力受限
```

如果使用一个可微的、由实际 Top-K 负载引导的辅助损失，那么在不改变模型
结构和推理路径的情况下，可能改善专家利用率和最终插补精度。

### 3.2 可证伪条件

这条路线不是默认“路由越均匀越好”。以下任一结果都应视为假设失败：

1. hard load 明显变均衡，但验证集 MAE 没有改善；
2. 路由均衡后 RMSE 或低缺失率结果稳定退化；
3. 专家集中实际上是有益的任务分工，强制均衡破坏了专门化；
4. 改动只改变 loss 数值，没有稳定改变 hard selection；
5. 改善只出现在单种子或少数测试点，无法通过多种子和全量实验。

失败时应停止这条路线，不继续细化大量权重。

## 4. 唯一模型改动

### 4.1 可微负载感知损失

对所有启用尺度的 gate 和 hard mask 进行聚合：

```text
p_i = mean(soft_gate_i)
f_i = mean(hard_selected_mask_i)
q_i = stop_gradient(f_i / sum_j(f_j))
```

其中：

- `p_i` 是可微的平均路由概率，且和为 1；
- `f_i` 是专家被 Top-K 实际选中的频率，和为 `top_k`；
- `q_i` 是归一化且停止梯度的实际负载分布。

新负载感知项定义为：

```text
L_RLB = E × sum_i(p_i × q_i)
```

`E` 为专家数。其作用是：

- hard load 过高的专家具有更大的 `q_i`；
- 反向传播会降低这些过载专家的 soft probability；
- 当 hard load 已均匀时，`q_i` 为常数，经过 softmax 后梯度接近 0；
- 不需要对离散 `topk` 本身求导。

完整 balance loss 为：

```text
L_balance
= lambda_importance × L_importance
+  lambda_load × L_RLB
```

其中现有 `L_importance` 保持不变。

### 4.2 为什么不直接使用 hard load 方差

hard load 方差适合作为诊断指标，但不能直接反向传播到 gate。保留原方差
用于日志：

```text
hard_load_variance
hard_load_cv
selection_entropy
dead_expert_rate
always_selected_rate
soft_hard_load_gap
```

训练项和诊断项必须分开命名，避免再次出现“日志有值但训练无梯度”的误解。

### 4.3 配置兼容方式

建议只增加一个配置字段：

```json
{
  "loss": {
    "load_balance_mode": "legacy_hard",
    "lambda_importance_balance": 0.001,
    "lambda_load_balance": 0.001
  }
}
```

可选值：

```text
legacy_hard：保持原 V14 行为，用于精确复现
switch_topk：启用新的可微负载感知项
```

默认值必须为 `legacy_hard`，从而保证历史 V14 配置、checkpoint 和结果不被
静默改变。

### 4.4 明确禁止同时改变的内容

本路线实验期间不得同时改变：

- 专家数量；
- `top_k`；
- router 网络结构；
- gate temperature；
- expert block；
- scale gate；
- residual 参数化；
- correction gate；
- 主损失、优化器、学习率和训练轮数；
- 数据划分、离线 mask 和指标计算方式。

否则无法判断结果来自负载损失修复，还是来自其他改动。

## 5. 实现前单元验证

在运行正式训练前必须通过以下测试：

1. `legacy_hard` 的 loss 输出与原 V14 完全一致；
2. 证明 legacy hard load 项没有 router 梯度；
3. 构造失衡路由时，`switch_topk` 对 router 产生有限且非零的梯度；
4. 构造均匀 hard load 时，额外 router 梯度接近 0；
5. dense/no-router 模式不会误加该损失；
6. 只聚合数据集实际启用的尺度；
7. TaxiBJ 未启用 coarse 时，coarse 不进入 loss 和统计；
8. loss、梯度和日志中无 NaN/Inf；
9. 新模式不改变模型参数量和推理输出路径；
10. 不读取真实缺失值生成路由目标，不产生标签泄漏。

还应在一个真实 batch 上记录：

```text
主损失梯度范数
importance 项梯度范数
RLB 项梯度范数
加权后 RLB/主损失梯度范数比
```

如果辅助梯度远大于主损失梯度，应先缩小候选权重，不能直接进入完整训练。

## 6. 完整探索流程

### 阶段 0：无训练路由审计

目的：固定问题证据，并建立之后可比较的路由指标基线。

范围：V14 的 24 个 seed=42 正式 checkpoint。

输出：

```text
routing_audit_per_run.csv
routing_audit_per_dataset.csv
routing_gradient_check.json
routing_audit_summary.md
```

每个实验至少记录：

```text
soft importance
hard load
hard load CV
hard selection normalized entropy
dead expert rate（load < 0.01）
always-selected rate（load > 0.99）
soft-hard load L1 gap
Top-K 边界 margin
```

阶段 0 只诊断，不改变任何实验结论。

### 阶段 1：三点权重预筛选

选择三个具有明显路由集中、同时覆盖三个数据集的点：

| 数据集 | 模式与缺失率 | 选择理由 |
|---|---|---|
| TaxiBJ | `fixed@0.4` | 路由集中明显，且不是已知最脆弱的 random 低缺失点 |
| BikeNYC | `random@0.2` | 现有 24 点中 `l_load` 最大 |
| CHAP | `random@0.2` | 存在集中，同时能检查低缺失率安全性 |

候选仅改变 `lambda_load_balance`：

| 实验编号 | 模式 | 权重 |
|---|---|---:|
| R00 | `legacy_hard`，V14 对照 | 原配置 |
| R01 | `switch_topk` | `1e-4` |
| R02 | `switch_topk` | `1e-3` |
| R03 | `switch_topk` | `1e-2` |

新增训练共：

```text
3 个权重 × 3 个实验点 = 9 组
```

权重必须只根据验证集选择。即使训练程序按统一流程自动运行了测试集，也不能
查看测试集排序后再选择权重。

#### 阶段 1 通过标准

至少有一个权重同时满足：

- 3 点中至少 2 点的验证 hard load CV 相对 V14 降低不低于 25%；
- dead/always-selected rate 有实质下降；
- 三点平均验证 MAE 改善不低于 0.3%，或至少不退化超过 0.3%；
- 任一点验证 MAE 退化不超过 1%；
- 没有 NaN/Inf、均匀随机路由或单专家极端塌缩；
- 参数量不变，单 epoch 时间增长不超过 5%。

若多个权重通过，优先选择：

```text
验证 MAE 更好
→ RMSE 更安全
→ 权重更小
```

若所有权重都未通过，立即终止 RLB，不增加更多权重。

### 阶段 2：Core-6 单种子因果验证

固定阶段 1 选出的唯一权重，在以下 Core-6 上运行 seed=42：

| 数据集 | Fixed | Random |
|---|---|---|
| TaxiBJ | 0.4 | 0.4 |
| BikeNYC | 0.6 | 0.8 |
| CHAP | 0.2 | 0.4 |

新增训练：

```text
6 组
```

#### 阶段 2 通过标准

- MAE 明确改善（超过 0.5%）至少 4/6 点；
- 六点宏平均 MAE 改善不低于 0.5%；
- 六点宏平均 RMSE 不退化超过 0.5%；
- 任一点 MAE 退化不超过 2%；
- 任一点 RMSE 退化不超过 3%；
- hard load CV 平均下降不低于 25%；
- 每个数据集至少有一个点改善；
- 6/6 均完整训练、验证、加载 best checkpoint 和测试。

若只改善路由统计而不改善 MAE，应判定“路由集中主要是有益专门化”，停止
扩展，不能因为机制图更好看而进入多种子。

### 阶段 3：Core-6 三种子稳定性验证

仅阶段 2 通过后执行。

种子：

```text
42, 2026, 3407
```

保持每个实验点的 offline mask 不变，只改变模型训练种子。已有 seed=42 可
跳过，因此通常新增：

```text
6 点 × 2 个新种子 = 12 组
```

与相同种子、相同 mask 的 V14 进行 paired comparison。

#### 阶段 3 通过标准

- 至少 4/6 点的三种子平均 MAE 优于 V14；
- 六点三种子宏平均 MAE 改善不低于 0.5%；
- 每个数据集的平均 MAE 均不得退化超过 0.5%；
- 总体 RMSE 不退化；
- 任一配对种子的 MAE 退化不超过 3%；
- 路由负载改善在三个种子中方向一致；
- 报告 mean、std、paired delta 和层次 bootstrap 置信区间。

若平均改善完全由单个种子贡献，则不进入全量实验。

### 阶段 4：24 点全量验证

仅阶段 3 通过后执行：

```text
3 数据集 × 2 缺失模式 × 4 缺失率 = 24 点
seed = 42
```

已完成的 Core-6 可跳过，因此预计新增 18 组。

#### 阶段 4 通过标准

- 24 点全部完整结束；
- 超过 0.5% 的明确 MAE 胜点至少 14/24；
- 三个数据集的平均 MAE 均不得退化超过 0.3%；
- 至少一个数据集的平均 MAE 改善超过 1%；
- 三个数据集的平均 RMSE 均不得退化；
- 任一点 MAE 退化不超过 3%；
- 任一点 RMSE 退化不超过 5%；
- 参数量增长为 0；
- 平均训练时间增长不超过 5%；
- 路由负载改善与 MAE 改善具有合理但不过度宣称的对应关系。

若 24 点未通过，不再用按数据集或按缺失率手工开关来补救。这样的规则会把
路线变成测试集驱动的特例集合。

### 阶段 5：论文级补充验证

只有阶段 4 通过，才考虑：

1. 在代表点补齐三种子；
2. 单独改变 random mask seed，验证缺失位置随机性；
3. 汇报参数量、FLOPs、峰值显存和训练/推理耗时；
4. 做 `legacy_hard`、`importance-only`、`switch_topk` 消融；
5. 绘制训练过程中的 hard load CV、专家利用率与验证 MAE 曲线；
6. 分析改善是否集中在高缺失率、特定数据集或特定尺度。

阶段 5 仍不加入第二个结构改动。

## 7. 结果表格要求

### 7.1 精度结果

每个点必须记录：

| 字段 | 说明 |
|---|---|
| Dataset / Pattern / Rate / Seed | 唯一实验条件 |
| Best epoch | 最佳验证轮次 |
| Val MAE / RMSE | 选择 checkpoint 的依据和安全指标 |
| Test MAE / RMSE / WAPE | 原始范围最终指标 |
| V14 paired delta | 同条件相对变化 |
| Complete | 是否训练、验证、best 加载、测试均完成 |

### 7.2 路由结果

每个尺度必须记录：

| 字段 | 说明 |
|---|---|
| Soft importance | 平均 gate 概率 |
| Hard load | 实际 Top-K 选择率 |
| Hard load CV | 实际负载离散程度 |
| Selection entropy | 归一化选择熵 |
| Dead expert rate | 选择率低于 1% 的专家比例 |
| Always-selected rate | 选择率高于 99% 的专家比例 |
| Soft-hard gap | 概率分布与实际选择分布差距 |

不能只报告辅助 loss，因为辅助 loss 下降不等于模型效果改善。

## 8. 预计实验成本

按现有 V14 的大致单实验耗时估算：

| 阶段 | 新增实验 | 预计单卡 GPU 时间 |
|---|---:|---:|
| 阶段 0 | 无训练审计 | 数十分钟至 2 小时 |
| 阶段 1 | 9 | 约 7 小时 |
| 阶段 2 | 6 | 约 4.5～5 小时 |
| 阶段 3 | 12 | 约 9～10 小时 |
| 阶段 4 | 18 | 约 14 小时 |
| 合计 | 45 个新增训练 | 约 35 小时 |

由于每阶段都有停止规则，失败路线通常只消耗前 7～12 小时，不会直接浪费
完整 24 点和三种子预算。

## 9. 预期贡献与风险

### 9.1 如果成功

可形成一条简洁的论文贡献：

> V14 的 soft importance 均衡无法约束离散 Top-K 的实际专家负载。通过
> 一个不增加参数的可微负载感知辅助项，使路由概率获得来自实际选择频率的
> 反馈，从而改善专家利用率和插补性能。

它具有以下优点：

- 不改变 V14 推理结构；
- 不增加模型参数；
- 不引入教师模型或多阶段训练；
- 不需要按数据集手工设计结构；
- 可以通过路由统计和精度结果同时验证。

### 9.2 主要风险

- 当前专家集中可能是合理专门化，而不是有害塌缩；
- 共享专家池并不要求每个 batch 完全均匀；
- 过大的负载权重会牺牲任务损失；
- 路由改善可能只改善训练稳定性，不改善最终 MAE；
- 小数据集或低缺失率可能不需要充分专家多样性。

因此目标不是追求绝对均匀，而是验证“适度减少长期固定选择”是否带来稳定的
泛化收益。

## 10. 最终决策规则

```text
阶段 0：确认旧 hard load 无梯度且实际路由集中
  ├─ 否 → 停止，不实现 RLB
  └─ 是 → 阶段 1

阶段 1：负载改善且验证性能安全
  ├─ 否 → 停止，不继续搜权重
  └─ 是 → 固定唯一权重，进入 Core-6

阶段 2：Core-6 至少 4/6 胜且 RMSE 安全
  ├─ 否 → 停止
  └─ 是 → 三种子

阶段 3：跨种子稳定
  ├─ 否 → 保留为机制实验，不升正式版本
  └─ 是 → 24 点全量

阶段 4：全量达到预注册阈值
  ├─ 否 → 路线关闭
  └─ 是 → 作为 V14 后续正式版本候选
```

## 11. 当前建议

这条路线值得作为 E02 之后的第一优先级探索，但目前只能称为
“有明确代码证据的候选假设”，不能提前称为改进。

建议严格按以下顺序执行：

```text
路由审计和梯度单测
→ 三点三权重预筛选
→ Core-6
→ Core-6 三种子
→ 24 点全量
→ 论文级补充实验
```

在阶段 1 完成前，不组合 E02、E03、S01/S02、residual 放大、通道校正或
其他结构模块。若 RLB 失败，则保留 V14，转向数据条件化的缺失模式表征，
而不是继续对负载权重做密集搜索。
