# v24 早期四轮八专家主骨干

2026-09-29 后续基础基线改为开启反馈、direct 更新的 B3，见 [B3/C3 说明](README_B3_C3.md)。本文入口保留原 no-feedback 结构，不能替代新的 B3。

后续 v24 主骨干是四轮共享专家池，每轮从 `T/S/TD/SD/TA/ST/TL/SL` 中按样本选出两个专家，只运行选中的专家，并将其权重归一化融合。四轮都使用当前隐藏状态重新路由；训练采用 3 epoch 软预热、3 epoch 过渡，随后为硬 Top-2。默认关闭显式逐轮补全反馈，保留逐轮隐藏状态更新。当前配置保持候选重要性均衡项 `lambda_coe_balance=0.01`；这是沿用最近五组对照的训练设置，结构效果仍需后续实验验证。

TaxiBJ 和 BikeNYC 均采用缺失率 0.4 的九类混合 mask：训练每轮重采样，验证和测试各自固定采样，三者使用相同的缺失类型分布。默认 seed 7、全局 batch 32、100 epoch，按验证 MAE 选择最优轮次并测试；初始学习率 `1e-3`，余弦衰减至最后一轮 `3e-4`，每 5 epoch 验证一次。按验证 MAE 保存 `best.pth`，结束时另存最后一轮的完整状态 `last.pth`。此入口不运行只适用于三轮的 Oracle 或搭档探针。

```bash
python -u scripts/v24/run_coe_main_s4_e8.py --dataset taxibj --gpu 0 --dry-run
tmux new-session -s v24-coe-main \
  'python -u scripts/v24/run_coe_main_s4_e8.py --dataset taxibj --gpu 0 --epochs 100'
```

`--dataset bikenyc` 可换数据集；`--gpus 0,1` 可运行一个双卡 DDP 任务；`--epochs N` 同时更新训练轮数与余弦调度周期。每次启动前脚本会检查 GPU 是否空闲，同一输出队列使用文件锁。历史三轮六专家和四轮六专家配置与实验计划均保留，不受新默认入口影响。
