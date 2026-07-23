# V17.1 探索性消融运行说明

该目录用于执行 `V17.1-single_探索性消融实验详细执行方案.md` 中的 E0–E7。正式结果与 V17 全量实验隔离，统一写入：

```text
outputs/v17.1-single/{dataset}/ablation/{variant}/{pattern}/rate{rate}/{run_id}/
```

## 消融版本

| 编号 | variant | 唯一变化 |
|---|---|---|
| E0 | `full` | 完整 V17 复现 |
| E1 | `no_adapter` | 关闭三尺度 Adapter |
| E2 | `decoupled_expert_router` | 只把专家头改为三个独立 QualityRouter，仍保留 Hierarchical Scale Router |
| E3 | `progressive_fusion` | 只把 Routed 并行融合改为使用相同外部尺度权重的渐进融合 |
| E4 | `no_fine_floor` | 不使用 Fine Floor |
| E5 | `hard_fine_floor` | 仅在 Fine 权重低于 0.25 时执行硬下限投影 |
| E6 | `global_route_gamma` | 样本 Gate 改为可学习全局 Gamma |
| E7 | `independent_shared_scale` | Shared 使用独立可靠性尺度门，Routed 保留 V17 尺度权重 |

## 先做最小 smoke

只检查一个代表点和一个版本：

```bash
python scripts/v17.1-single/run_exploratory_ablation.py \
  --gpu 0 \
  --points P1 \
  --variants full \
  --smoke
```

`--smoke` 会执行 1 个训练 epoch、1 次验证和 1 次测试，并写入 debug 目录，不会被正式结果汇总脚本计入。

检查全部 8 个结构能否完成一轮：

```bash
python scripts/v17.1-single/run_exploratory_ablation.py \
  --gpu 0 \
  --points core4 \
  --variants all \
  --smoke
```

## 第一阶段正式实验

完整执行 8 个版本 × 4 个点，共 32 次：

```bash
python scripts/v17.1-single/run_exploratory_ablation.py \
  --gpu 0 \
  --points core4 \
  --variants all \
  --seeds 42
```

脚本按文档推荐顺序单卡串行执行，并自动跳过已经完成且 MAE/RMSE/Loss 有限的任务。E0 四点完成后，脚本会将 seed=42 的 Test MAE 与 `outputs/v17-single/` 中原 V17 结果逐点配对；默认任一点相对差异超过 0.5% 就停止，不再执行 E1–E7。中断后再次运行同一命令即可续跑。

如果原 V17 结果不在本机，脚本也会停止。只有在已经人工确认基线可比时，才可显式使用 `--skip-reproduction-check`；不建议把跳过校验后的结果直接用于机制结论。

计算资源有限时，先运行优先级最高的 16 次：

```bash
python scripts/v17.1-single/run_exploratory_ablation.py \
  --gpu 0 \
  --points core4 \
  --variants full hard_fine_floor global_route_gamma independent_shared_scale \
  --seeds 42
```

只打印命令、不启动训练：

```bash
python scripts/v17.1-single/run_exploratory_ablation.py \
  --gpu 0 \
  --points core4 \
  --variants all \
  --dry-run
```

## Stage 1 结果汇总与组件选择

32 组完成后先汇总，不要直接开始组合：

```bash
python scripts/v17.1-single/summarize_exploratory_ablation.py \
  --seeds 42 \
  --require-complete
```

先根据汇总判断 E1–E7 中哪个单因素版本最好，并检查 E7、E5、E6 是否分别提供了正向证据。若 E7 明显有害，不应机械执行后续以 E7 为基础的组合。

## Stage 2 逐步组合实验

组合严格按以下顺序定义：

| 组合 | 内容 | 是否新增训练 |
|---|---|---:|
| C1 | E7：Independent Shared Scale | 否，直接复用 Stage 1 E7 |
| C2 | C1 + Hard Fine Floor | 4 组 |
| C3 | C2 + Global Route Gamma | 4 组 |

因此 Stage 2 只新增 8 组，不重复训练 C1：

```bash
python scripts/v17.1-single/run_stage2_combinations.py \
  --gpu 0 \
  --seeds 42 \
  --combinations all
```

脚本要求 Stage 1 的 32 组结果全部存在，否则会拒绝启动。完成后汇总：

```bash
python scripts/v17.1-single/summarize_stage2_combinations.py \
  --seeds 42 \
  --require-complete
```

根据该汇总在 C1/C2/C3 中选择最佳组合。应重点检查：成功点 P1/P2 是否保留、失败点 P3/P4 是否改善，以及路由塌缩是否减少。

## Stage 3 三随机种子确认

只验证三个模型：

1. E0 Full；
2. Stage 1 最佳单因素版本；
3. Stage 2 最佳组合版本。

下面用 `hard_fine_floor` 和 C3 作为命令示例，实际执行时应替换为前两份汇总选出的版本：

```bash
python scripts/v17.1-single/run_stage3_multiseed.py \
  --gpu 0 \
  --stage1-variant hard_fine_floor \
  --combination-variant c3_independent_shared_hard_floor_global_gamma \
  --seeds 42 2026 3407
```

Stage 3 总表包含 3 模型 × 4 点 × 3 seeds = 36 条记录，但 seed=42 的 12 条会复用 Stage 1/2 结果，所以正常只新增 24 次训练。

完成后使用完全相同的两个候选名称汇总：

```bash
python scripts/v17.1-single/summarize_stage3_multiseed.py \
  --stage1-variant hard_fine_floor \
  --combination-variant c3_independent_shared_hard_floor_global_gamma \
  --seeds 42 2026 3407 \
  --require-complete
```

稳定有效的最低条件是：平均 MAE 改善超过 1%，且三个 seeds 中至少两个改善。完成 Stage 3 后即可撰写探索报告；V17.1 本轮不要求继续跑三数据集 24 点全量实验。只有当最佳组合准备升级为下一正式版本时，才为新版本安排完整 24 点验证。

## 推荐执行顺序与新增数量

```text
Stage 1：E0-E7 核心四点              32 条总记录
Stage 1 汇总：选择最佳单因素          不训练
Stage 2：C2/C3 核心四点               8 次新训练（C1复用E7）
Stage 2 汇总：选择最佳组合            不训练
Stage 3：三模型×四点×三seed           24 次新训练（12条seed42复用）
Stage 3 汇总与实验报告                不训练
```

因此在 Stage 1 的 32 组全部完成后，标准流程还需要新增 **32 次训练**，不是 48 次。

所有汇总写入 `outputs/v17.1-single/summary/`，包含 Markdown、CSV 和 JSON。`--require-complete` 会在结果缺失时返回非零状态，防止误用不完整结果。

## 每次运行保存的文件

```text
config.json
git_metadata.json
checkpoints/best.pt
logs/train.log
logs/val.log
logs/test.log
logs/metrics.jsonl
router_diagnostics.json
```

`router_diagnostics.json` 记录尺度、专家、Top-2、跨尺度一致性、分支 Gate、Adapter 和自动塌缩标志。最终测试指标来自验证集 MAE 最优 checkpoint。

正式实验前应先提交并固定当前代码，确保日志中的 `git_dirty=false`。
