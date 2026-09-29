# v24 直接传递 MoE 基准

入口：`scripts/v24/run_direct_baselines.py`。默认 TaxiBJ，也可用 `--dataset bikenyc`。三组按表中顺序串行训练；双卡 DDP 只用于当前一组。

| 顺序 | 实验 | 层数 | 专家池 | 每层激活 |
|---|---|---:|---|---:|
| 1 | `direct_moe_s1_top8` | 1 | 本层独立八专家 | 8 |
| 2 | `direct_moe_s4_top2` | 4 | 每层独立八专家 | 2 |
| 3 | `direct_shared_s4_top2` | 4 | 四层共享八专家 | 2 |

每层以路由选择的专家输出加权融合结果作为下一层隐藏状态；不加跨层残差，也不使用公共专家或显式补全反馈。所有组使用原生 Router Top-K、选中组内归一化权重、相同八种异构专家类型、dim 64 和九类混合 random 0.4 mask。训练 mask 每轮重采样；验证与测试 mask 固定且同分布。默认 seed 7、100 epoch、每 5 epoch 验证、双卡全局 batch 32（每卡 16），按验证 MAE 保存 `checkpoints/best.pth` 并在测试前加载。可以用 `--epochs` 和 `--batch-size` 覆盖默认值。

```bash
python -u scripts/v24/run_direct_baselines.py --dataset taxibj --gpus 0,1
```

输出在 `outputs/v24-COE/experiments/coe_direct_baselines/`；已完成作业经训练记录和检查点校验后跳过。使用 `--dry-run` 检查队列。
