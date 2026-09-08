# V21 缺失测度保持双状态金字塔验证协议

## 1. 验证对象

| 方案 | 尺度构造 | 新增7维Evidence | 作用 |
|---|---|---|---|
| P0 | V14层级masked mean | 否 | 严格基线 |
| P1 | 路径一致的sum/count measure pyramid | 否 | 验证尺度测度本身 |
| P2 | 与P1相同 | 是 | 验证Content–Evidence双状态 |

三者保持数据划分、mask CSV、模型主干、专家数、Top-K、损失、优化器、epoch与seed一致。新增 Evidence 模块使用独立 RNG 域构造，公共参数初始化逐元素一致。

## 2. 为什么不直接跑全量

验证分三阶段进行。前一阶段未通过的方案不得进入后一阶段，避免在明显无效候选上浪费完整 24 点和三种子算力。

### Stage 1：Core-6 单种子完整训练

共 `3 variants × 6 points = 18` 组，使用 seed 42 和各数据集正式 epoch。

Core-6 同时包含诊断中有利和不利的尺度点：

- TaxiBJ fixed@0.6、random@0.6；
- BikeNYC fixed@0.4、random@0.4；
- CHAP fixed@0.4、random@0.4。

执行：

```bash
python scripts/v21-single/run_validation.py \
  --phase core6 \
  --gpu 0
```

中断后重复同一命令会检查 `best.pt`、完整 test.log、有限 MAE/RMSE、seed、epoch 和实验策略，并自动跳过真正完成的任务。

汇总：

```bash
python scripts/v21-single/summarize_validation.py --phase core6
```

### Stage 2：晋级方案的三种子 Core-6

只运行 Stage 1 晋级者及 P0。假设 P1 晋级：

```bash
python scripts/v21-single/run_validation.py \
  --phase multiseed \
  --variants P0 P1 \
  --seeds 42 2026 3407 \
  --gpu 0
```

已完成的 seed42 会自动跳过，因此实际只补缺失任务。

汇总：

```bash
python scripts/v21-single/summarize_validation.py \
  --phase multiseed \
  --variants P0 P1 \
  --seeds 42 2026 3407
```

### Stage 3：最终候选全量 24 点

仅对最终候选和 P0 执行三个数据集、fixed/random、0.2/0.4/0.6/0.8。假设 P1 最终晋级：

```bash
python scripts/v21-single/run_validation.py \
  --phase all24 \
  --variants P0 P1 \
  --gpu 0
```

汇总：

```bash
python scripts/v21-single/summarize_validation.py \
  --phase all24 \
  --variants P0 P1
```

## 3. 预注册晋级标准

候选相对 P0 必须同时满足：

1. 至少 2/3 匹配点的 Val MAE 改善达到 0.5%；
2. Val MAE 宏平均改善至少 0.5%；
3. Val RMSE 宏平均退化不超过 0.5%；
4. 最大单点 Val MAE 退化不超过 2%；
5. 任一数据集的 Val MAE 宏平均退化不超过 1%。

要宣称 7 维 Evidence 有独立贡献，P2 还必须相对 P1 满足：

1. Val MAE 宏平均改善至少 0.3%；
2. 至少 2/3 匹配点获胜；
3. 最大单点退化不超过 2%。

模型选择和晋级只看验证集。测试集在每次训练结束后由最佳验证 checkpoint 运行一次，但不用于选择候选或调整阈值。

三种子阶段还要求候选至少在 2/3 个 seed 的 Core-6 宏平均上优于 P0，且最差 seed 的宏平均退化不超过 1%。汇总器同时报告 seed 间样本标准差。三个 seed 只用于检查稳定性，不以小样本显著性检验包装成统计证明。

## 4. 机制验证

若 P2 进入三种子阶段，除最终误差外，还要对每个数据集至少选择一个 random-mask 的完整 P2 run，分析：

- coverage、观测方差与缺失位置 MAE 是否形成稳定相关关系；
- 专家门控概率是否随 coverage、方差和几何矩变化；
- Top-1 专家是否形成不同 evidence regime，而不是所有样本落到同一专家。

命令中的 `RUN_DIR` 指向某次训练时间戳目录：

```bash
python scripts/v21-single/analyze_mechanism.py \
  --run-dir RUN_DIR \
  --split val \
  --gpu 0
```

机制分析默认只读取验证集。测试集机制图只在最终方案冻结后生成，避免观察测试结果后继续修改模型。

## 5. 快速工程测试

正式 Stage 1 前可先验证 1 epoch 任务矩阵：

```bash
python scripts/v21-single/run_validation.py \
  --phase core6 \
  --gpu 0 \
  --epochs 1
```

该结果只能证明训练链路和日志汇总可用，不能用于方案晋级。

若只想检查命令而不训练：

```bash
python scripts/v21-single/run_validation.py \
  --phase core6 \
  --gpu 0 \
  --dry-run \
  --rerun-completed
```

## 6. 单卡分缺失模式运行

运行器支持筛选，可先 fixed 后 random：

```bash
python scripts/v21-single/run_validation.py --phase core6 --patterns fixed --gpu 0
python scripts/v21-single/run_validation.py --phase core6 --patterns random --gpu 0
```

不建议同时用两张 GPU 跑两个高负载任务；此前服务器已有双卡并发不稳定记录。协议默认单 GPU 串行。
