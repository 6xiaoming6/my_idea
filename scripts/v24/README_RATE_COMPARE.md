# A0/A2 TaxiBJ 缺失率对照

本轮共 8 组，按单卡顺序执行：

- 方案：A0（软预热后硬路由）、A2（全程 Conditional Soft）
- mask：`random`
- 缺失率：0.2、0.4、0.6、0.8
- 数据：`data/TaxiBJ/taxibj_{train,val,test}.npz`
- seed=7、batch=16、100 epoch、每 5 epoch 验证、cosine scheduler 总周期 100
- 每个方案在同一 pattern/rate 下使用完全相同的 train/val/test CSV
- 保存每组 `best.pt`，按验证 MAE 选最佳权重后测试

这里使用原始 legacy `random` 协议：每个样本一行 mask；本轮不是九类 mixed mask。

## Dry-run

```bash
cd /home/students/HuangMingYu/code/py/my_idea/my_idea
conda activate difftdi
python scripts/v24/run_rate_compare.py --dry-run
```

## 单卡顺序运行

```bash
tmux new-session -s v24-rate-compare \
  'python -u scripts/v24/run_rate_compare.py --gpu 0'
```

入口会检查 GPU 是否已有计算进程；已存在结果回执的任务会跳过。每个任务的完整训练输出写入 `outputs/v24-COE/experiments/coe_rate_compare/launcher_logs/`，控制台只打印任务开始和完成信息。

## 时间估计

上一轮 TaxiBJ mixed9 实测：A0 60 epoch 约 94.7 分钟，A2 60 epoch 约 97.5 分钟。当前旧 TaxiBJ train 样本略多（2491 vs 2452），按 100 epoch 估算：

- A0：每组约 2.6–2.8 小时；4 组约 10.5–11.2 小时
- A2：每组约 2.7–2.9 小时；4 组约 10.8–11.6 小时
- 总计约 21.5–23 小时，建议按 25 小时预留

这是单卡串行时间，不包含服务器负载波动；fixed 和 random 的单组时间通常接近。`rate_compare_experiments.json` 和运行脚本会保留完整配置及顺序。
