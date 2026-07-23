# V17.2 Adapter-Free HSA-MoE

V17.2 是 V17.1 `E1 No Adapter` 的正式化版本。唯一结构变化是删除
Fine/Mid/Coarse 三个 Scale-Specific Adapter。V17/V17.1 的原模型、配置、
脚本和输出目录均保持不变。

## 强制结构约束

- 架构名：`v17_2_no_adapter_hierarchical_scale_moe`
- 输出根目录：`outputs/v17.2-single/`
- 即使配置误写 `adapter_enabled=true`，包装类仍强制关闭 Adapter。
- 正式训练启动前自动执行协议审计。
- 正式训练默认要求 Git working tree clean；`--allow-dirty` 仅供调试。
- 每次运行额外保存 `resolved_config.json`、`config_sha256.txt`、
  `mask_metadata.json`、`parameter_report.json` 和 `protocol_audit.json`。

## 0. 测试与协议审计

```bash
cd /home/students/HuangMingYu/code/py/my_idea/my_idea
conda activate difftdi

PYTHONPATH=src python -m unittest discover \
  -s tests \
  -p 'test_v17_2*.py' \
  -v

python scripts/v17.2-single/audit_protocol.py \
  --dataset TaxiBJ \
  --mask random \
  --rate 0.4 \
  --seed 42
```

已经固定的基线清单位于：

```text
configs/v17.2-single/baseline_manifest.json
```

它包含 seed 42 的完整 24 点，以及核心四点 seeds 2026/3407，共 32 条
显式 Full 基线。代码或基线改变后重新生成：

```bash
python scripts/v17.2-single/build_baseline_manifest.py
```

## 1. 三数据集真实数据 Smoke

每个数据集执行 1 epoch 训练、1 次验证和 1 次测试：

```bash
python scripts/v17.2-single/run_smoke.py \
  --gpu 0
```

Smoke 使用 debug 输出，不作为论文结果。

## 2. Stage 1：核心四点三随机种子 clean 复现

先提交当前 V17.2 代码，确认 `git status` 为空，然后运行：

```bash
python scripts/v17.2-single/run_multiseed.py \
  --gpu 0 \
  --points all \
  --seeds 42 2026 3407
```

中断后重新执行同一命令即可；完整结果自动跳过。不要使用
`--force-rerun`，除非明确需要重跑。

汇总：

```bash
python scripts/v17.2-single/summarize_v17_2.py \
  --points core4 \
  --seeds 42 2026 3407 \
  --name stage1_core4_multiseed \
  --require-complete
```

进入 24 点实验前，至少确认：

- P3 平均改善约 9%，至少 2/3 seeds 获胜；
- P4 平均改善约 4%，至少 2/3 seeds 获胜；
- P1/P2 没有出现平均超过 5% 的新退化。

## 3. Stage 2：seed 42 完整 24 点

执行顺序固定为全部 fixed，再全部 random：

```bash
python scripts/v17.2-single/run_full_24.py \
  --gpu 0 \
  --seed 42 \
  --datasets all \
  --patterns fixed random \
  --rates 0.2 0.4 0.6 0.8
```

先检查任务但不训练：

```bash
python scripts/v17.2-single/run_full_24.py \
  --gpu 0 \
  --seed 42 \
  --datasets all \
  --patterns fixed random \
  --rates 0.2 0.4 0.6 0.8 \
  --dry-run
```

汇总：

```bash
python scripts/v17.2-single/summarize_v17_2.py \
  --seeds 42 \
  --name stage2_full24_seed42 \
  --require-complete
```

## 4. Stage 3：针对性多随机种子

核心四点可以继续使用 `--points`。24 点中新发现的最佳/最差点通过可重复的
`--custom-point DATASET PATTERN RATE` 加入：

```bash
python scripts/v17.2-single/run_multiseed.py \
  --gpu 0 \
  --points P1 P2 P3 P4 \
  --custom-point TaxiBJ random 0.8 \
  --custom-point CHAP fixed 0.8 \
  --seeds 42 2026 3407
```

非核心点的 seeds 2026/3407 必须先准备同协议 Full 基线并重新生成 manifest；
如果 manifest 缺失，脚本会拒绝启动，防止错误配对。

## 单实验命令

```bash
python scripts/v17.2-single/train.py \
  --dataset TaxiBJ \
  --mask fixed \
  --rate 0.4 \
  --seed 42 \
  --gpu 0
```

V17.2 正式实验不应使用 `--allow-dirty`。该参数仅用于功能调试。

## 输出完整性

一个可被断点续跑逻辑识别为完成的正式运行必须同时包含：

```text
checkpoints/best.pt
logs/train.log
logs/val.log
logs/test.log
logs/metrics.jsonl
router_diagnostics.json
parameter_report.json
protocol_audit.json
git_metadata.json
```

同时要求：

- Test MAE/RMSE/Loss 均为有限值；
- `protocol_audit.passed=true`；
- `adapter_parameter_count=0`；
- 模型版本和架构严格为 V17.2。
