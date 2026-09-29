# v25 RAS-CoE 十组夜间实验

`run_ras_night.py` 按 E0→E9 顺序运行，TaxiBJ clean split、九种混合缺失、缺失率 0.4、seed 7、全局 batch 32、**100 epoch / cosine100**、E0 每 2 epoch 验证、E1–E9 每 5 epoch 验证。每组仅在最优验证 MAE 的 checkpoint 上测试一次；每组保存完整 `best.pt`。多尺度和 monotonic loss 均关闭。

```bash
python -u scripts/v25/run_ras_night.py --dry-run
python -u scripts/v25/run_ras_night.py --gpus 0
```

Runner 独占队列锁，检测到 GPU 上还有计算任务时等待，组间串行，仅使用 GPU 0。batch 32 已通过 E5/E9 单卡真实尺寸 AMP 前向、反向和优化器更新的显存测试；原 batch 16 中断日志保留在 `outputs/v25-RAS/night_20260928/`。结果写在 `outputs/v25-RAS/night_20260928_bs32/`，包含 `night_status.json`、`summary.csv`、`comparison.md` 和 E0–E9 各组的配置、回执、日志、最佳 checkpoint 与分析指标。某组异常会记录 `failure_receipt.json` 并继续下一组；重新执行同一命令会跳过完整组，并为未完成组创建新 attempt。

```bash
python -u scripts/v25/run_ras_night.py --start E5 --gpus 0
python -u scripts/v25/run_ras_night.py --only E3 E4 E5 --gpus 0
python -u scripts/v25/run_ras_night.py --only E3 --force --gpus 0
python -u scripts/v25/run_ras_night.py --summary-only
```

`--force` 保留旧 attempt，另起一次训练；不用于正常断点续跑。单 seed 结果只用于机制筛选，不代表统计显著性。
