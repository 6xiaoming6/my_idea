# V19-single：基于 V14 的单阶段通道残差增益校准 MoE

> 版本：V19-single  
> 基础版本：V14-single  
> 建议名称：Channel-Calibrated V14 MoE，简称 CC-V14-MoE  
> 训练方式：从头开始、单阶段、端到端训练  
> 设计原则：只增加一个小模块，不改变 V14 主体结构

## 1. 为什么重新简化 V19

V14 与 V18 的全量实验说明：

- V14 在 24 个配对点中赢了 14 个，整体性能更稳定；
- V18 的严格有界残差在 CHAP 上有效，但在 BikeNYC 上整体退化；
- V18 在 TaxiBJ fixed@0.2/0.4 上出现大幅退化；
- V14 的完整 correction 路径虽然存在尺度解释问题，但实际拟合能力很强；
- V14 的 24 个最终测试结果全部优于各自训练中的 `x_base`。

因此 V19 不应该：

- 再增加一套残差金字塔；
- 再增加新的方向头；
- 冻结、解冻多个阶段；
- 先训练 V14 再训练 V19；
- 依赖已有 V14 checkpoint；
- 引入 teacher、distillation 或多阶段调度；
- 大幅修改现有 loss。

V19 应该只解决一个最明确的问题：

> V14 使用一个样本级标量控制所有通道的 correction，不同通道无法独立调整修正强度。

TaxiBJ 和 BikeNYC 都有两个通道。两个通道的流量范围、缺失难度和时空规律并不完全相同，但 V14 最终使用同一个 `alpha_final`。这给 V19 留下了一个清晰、低风险的改进空间。

## 2. V19 的唯一结构改动

完整保留 V14：

```text
MultiScaleMoEBackbone
DifficultyConditionEncoder
SafeCoarseToFineRefiner
ObservedConsistencyEvaluator
SafetyController
CorrectionAdapter
```

V14 原始输出为：

```text
x_v14 = x_base + alpha_final × delta_ctf
```

定义 V14 已经生效的最终残差：

```text
r_v14 = x_v14 - x_base
```

V19 只增加一个通道残差增益：

```text
gamma ∈ [0.5, 1.5]^(B×C×1×1×1)
```

最终输出：

```text
x_v19 = x_base + gamma × r_v14
```

其中：

- `gamma=1`：严格等于 V14；
- `gamma<1`：减弱该通道的 V14 correction；
- `gamma>1`：增强该通道的 V14 correction。

V19 不预测新方向，不改变 V14 correction 的方向，只调整已经被证明总体有效的 correction 强度。

## 3. 单阶段总体流程

```text
输入 fine/mid/coarse 数据和 mask
                ↓
         完整 V14 前向传播
                ↓
       x_base、x_v14、r_v14
                ↓
   观测位置通道级相对统计量
                ↓
    ChannelResidualGain MLP
                ↓
       gamma，初始严格为 1
                ↓
 x_v19 = x_base + gamma × r_v14
                ↓
         原有损失 + 两个小约束
                ↓
       单阶段端到端反向传播
```

整个训练只有一个 optimizer、一个 scheduler、一套 epoch，不存在阶段切换。

## 4. 通道增益控制器

### 4.1 输入特征

控制器只使用观测位置和模型中间输出，不读取缺失位置 target。

对每个样本、每个通道计算：

```text
observed_scale
base_observed_error_relative
v14_observed_error_relative
v14_observed_gain_relative
effective_residual_mean_relative
effective_residual_q95_relative
observed_zero_ratio
```

共 7 个特征。

### 4.2 具体定义

令：

```text
O = observed mask
N = sum(O)
```

观测尺度：

```text
s = mean(|x_obs| × O) / N
```

并使用：

```text
s = clamp_min(s, 1e-3)
```

相对误差：

```text
e_base = MAE(x_base, x_obs | observed) / s
e_v14  = MAE(x_v14,  x_obs | observed) / s
gain   = (e_base - e_v14) / clamp_min(e_base, 1e-6)
```

有效残差统计：

```text
r_mean = mean(|r_v14| | observed) / s
r_q95  = q95(|r_v14| | observed) / s
```

零值比例：

```text
zero_ratio = mean((|x_obs| < 1e-6) | observed)
```

所有特征均按 `[B,C]` 计算，随后输入同一个共享 MLP。共享参数可以避免为不同数据集硬编码通道逻辑。

### 4.3 MLP 结构

建议：

```text
Linear(7,32)
→ LayerNorm(32)
→ GELU
→ Dropout(0.1)
→ Linear(32,16)
→ GELU
→ Linear(16,1)
```

输出：

```text
gamma = 1 + gamma_range × tanh(raw_gamma)
```

默认：

```text
gamma_range = 0.5
```

因此：

```text
gamma ∈ [0.5,1.5]
```

最后一层权重和偏置全部零初始化，训练开始时：

```text
raw_gamma = 0
gamma = 1
x_v19 = x_v14
```

## 5. 为什么只做通道增益

### 5.1 保留 V14 已验证的方向

V14 的最终 correction 在 24 个实验中都使整体 MAE优于自己的 `x_base`。当前没有充分证据证明需要重新学习 correction 方向。

### 5.2 比 V18 更少限制

V18 将新残差限制在观测 RMS 的固定比例内，可能无法补偿较差的 `x_base`。V19 不限制 V14 原有 correction，只在 `[0.5,1.5]` 范围内重新调整其强度。

### 5.3 解决 V14 最明显的粒度不足

V14 的 `alpha_final` 是：

```text
[B,1,1,1,1]
```

V19 的 `gamma` 是：

```text
[B,C,1,1,1]
```

新增的只是通道独立性，不引入空间 gate、时间 gate或第二套多尺度网络。

### 5.4 对单通道 CHAP 仍然有效

CHAP 的 `C=1`，此时 V19 退化为样本级 V14 correction 强度校准。模型仍可以根据相对观测收益判断 V14 correction 应略微增强还是减弱。

## 6. 输出和回退关系

V19 应同时保留：

```text
x_hat_base
x_hat_v14
x_hat_main = x_hat_v19
x_hat_final = x_hat_v19
```

用于日志记录：

```text
base_mae
v14_anchor_mae
v19_final_mae
v19_vs_v14_gain
v19_vs_base_gain
```

当 `gamma=1` 时：

```text
x_hat_v19 == x_hat_v14
```

该性质必须由单元测试保证。

## 7. 损失函数

V19 继续使用 V14 的全部原始损失：

```text
L_main
L_cross
L_balance
L_shared_aux
L_route_aux
L_complementary
0.25 L_v14_base
0.05 L_v14_mid
0.03 L_v14_coarse
0.10 L_v14_regret
0.0001 L_v14_gate
```

只增加两个小约束。

### 7.1 V14 锚点非退化约束

按样本计算缺失位置 MAE：

```text
e_v19 = MAE(x_v19, target | missing)
e_v14 = MAE(stopgrad(x_v14), target | missing)
```

定义：

```text
L_v19_anchor_regret =
    mean(ReLU(e_v19 - e_v14))
```

建议权重：

```text
lambda_v19_anchor_regret = 0.05
```

该约束只抑制 V19 增益校准比当前 V14 锚点更差，不要求多阶段训练。

### 7.2 增益正则

```text
L_v19_gain =
    mean((gamma - 1)^2)
```

建议权重：

```text
lambda_v19_gain = 0.001
```

该项使模型只有在主损失确实受益时才偏离 `gamma=1`。

### 7.3 总损失

```text
L_v19 =
    L_v14_original
  + 0.05 L_v19_anchor_regret
  + 0.001 L_v19_gain
```

不增加方向损失、probe 损失、teacher 损失或多个阶段的独立目标。

## 8. 单阶段训练策略

V19 从随机初始化开始整体训练，不加载 V14 checkpoint。

```text
初始化 V14 主体
初始化 gamma=1
→ 训练一个 epoch
→ 按配置定期验证
→ 验证 MAE更优时覆盖 best.pt
→ 继续训练到配置上限
→ 加载唯一 best.pt
→ 测试一次
```

训练配置保持和 V14 一致：

| 数据集 | Epoch | Batch size | Val interval |
|---|---:|---:|---:|
| TaxiBJ | 160 | 32 | 5 |
| BikeNYC | 140 | 16 | 2 |
| CHAP_Beijing | 150 | 32 | 5 |

优化器：

```text
AdamW
lr_main = 1e-3
lr_v14 = 1e-3
lr_v19_gain = 5e-4
weight_decay = 1e-4
cosine scheduler
eta_min = 1e-6
grad_clip_norm = 1.0
AMP = true
```

不使用：

- V14 预训练；
- 参数冻结；
- warm-start；
- 第二阶段微调；
- teacher checkpoint；
- 单独校准集。

## 9. 配置建议

```json
{
  "model": {
    "version": "v19-single",
    "architecture": "v19_channel_calibrated_v14_moe",
    "v19": {
      "enabled": true,
      "base_architecture": "v14_safe_c2f_moe",
      "gain_hidden": 32,
      "gain_dropout": 0.1,
      "gain_range": 0.5,
      "gain_zero_init": true,
      "scale_eps": 0.001
    }
  },
  "loss": {
    "lambda_v19_anchor_regret": 0.05,
    "lambda_v19_gain": 0.001
  },
  "train": {
    "lr_v19_gain": 0.0005
  }
}
```

三个数据集只修改 epoch、batch size、scale mode 和数据路径，不在模型代码中写 dataset 分支。

## 10. 代码结构

V19 只建议增加两个模型文件：

```text
src/stmoe_imputer/models/v_single/
  channel_residual_gain.py
  v19_channel_calibrated_v14_moe.py
```

以及版本独立配置和脚本：

```text
configs/v19-single/
scripts/v19-single/
outputs/v19-single/
```

最小修改共享位置：

```text
models/v_single/__init__.py
models/registry.py
losses.py
engine.py
optimizer 参数分组
```

不得修改 V14 类的现有行为。V19 wrapper 内部调用 V14，并在其输出后执行通道增益校准。

## 11. 必须通过的测试

### 11.1 初始化严格等于 V14

```text
max_abs(x_v19 - x_v14) <= 1e-6
```

### 11.2 无缺失 target 泄漏

固定观测输入和 mask，只改变缺失位置 target：

```text
V19 forward output 不得变化
```

### 11.3 增益范围

```text
0.5 <= gamma <= 1.5
```

极端输入下不得出现 NaN/Inf。

### 11.4 三数据集 shape

必须通过：

```text
TaxiBJ       [B,2,12,32,32]
BikeNYC      [B,2,12,24,12]
CHAP_Beijing [B,1,7,32,32]
```

### 11.5 梯度

单次反向传播后：

- V14 主体梯度有限；
- gain controller 梯度有限且非零；
- 不存在 unused parameters。

### 11.6 checkpoint

- 只保留一个 `best.pt`；
- 更优验证指标原子覆盖；
- 训练结束加载 best；
- 测试只执行一次。

## 12. 实验流程

训练流程本身始终是单阶段。实验验证只分为正常的 smoke 和 full：

### 12.1 一轮 smoke

三个数据集各运行：

```text
1 epoch train
1 次 val
1 次 test
```

确认：

- forward/backward 正常；
- gamma 初始为 1；
- 日志和 checkpoint 正常；
- 没有 target 泄漏。

### 12.2 全量实验

smoke 通过后直接运行：

```text
3 datasets
× fixed/random
× rates 0.2/0.4/0.6/0.8
= 24 组
```

先运行 seed=42。只有 V19 确认优于 V14 后，再决定是否运行多随机种子。

## 13. 结果判断标准

V19 的目标应保持现实：

1. 24 点至少 14 点不差于 V14；
2. 三个数据集八点平均 MAE均不能明显退化；
3. BikeNYC 平均 MAE至少不差于 V14；
4. TaxiBJ fixed@0.2/0.4 不能出现 V18 式大幅退化；
5. 至少一个数据集平均 MAE改善超过 1%；
6. 新增参数量控制在 20K 以内；
7. 训练时间增加不超过 2%；
8. 单点退化超过 3% 时应判定该设计不稳定。

## 14. 预期效果

V19 只调整 correction 强度，预期提升不会特别大，但风险和解释成本都更低。

合理预期：

| 数据集 | 预期 |
|---|---|
| TaxiBJ | 利用 inflow/outflow 通道差异，争取改善 0.5%–2% |
| BikeNYC | 避免统一 gate 对零值密集通道校准不足，争取改善 0.5%–2% |
| CHAP_Beijing | 单通道样本级增益校准，争取持平或小幅改善 |

如果该简单模块无法超过 V14，说明 V14 的主要瓶颈不在最终 correction 强度，届时再根据 gamma 分布和误差诊断决定下一步，而不是提前堆叠复杂模块。

## 15. 最终设计结论

V19 的结构只比 V14 多一个共享小型 MLP：

```text
V14:
x_v14 = x_base + r_v14

V19:
x_v19 = x_base + gamma_channel × r_v14
```

它具有以下特点：

- 基于 V14，而不是 V18；
- 从头开始训练；
- 单阶段端到端；
- 没有冻结和解冻；
- 没有 teacher；
- 没有第二套残差网络；
- 没有多尺度增量金字塔；
- 没有复杂 probe；
- 初始严格等于 V14；
- 只增加通道级 correction 强度校准；
- 训练、验证、保存和测试流程与 V14 完全一致。

这是目前结合 V14/V18 实验结果后，最简洁、风险最低、最容易写入论文并验证是否有效的 V19 改进方向。

