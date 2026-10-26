# v25 实验导航

当前实验基线为 **I37**，见 [基线说明](README_I37_BASELINE.md)。真实F/M/C输入，前三轮自由尺度、末轮F；受控差分、历史池、学习分配投影；固定前后F损失0.5/0.5。

|入口|用途|
|---|---|
|[I37基线](README_I37_BASELINE.md)|当前固定参考与独立发布入口|
|[实验档案](../../experiment_records/v25/README.md)|已完成I42–I75配置、指标、曲线与报告|
|[观测驱动迭代](README_OBSERVATION_ITERATION.md)|I42–I53历史研究批次|
|[输入整理](README_INPUT_REFINEMENT.md)|I54–I63历史研究批次|
|[最后一轮输入控制](README_FINAL_INPUT.md)|I64–I75，已完成，未替换I37|

下面保留早期单尺度RAS方向说明，不代表当前研究决定或默认训练入口。

<details><summary>早期RAS探索</summary>

# v25 RAS-CoE（单尺度第一版）

v25 使用独立的 `v25_ras_coe` 架构、`src/v25_ras_coe/` 模型/损失/指标和 `configs/v25/` 配置。v24 骨干与 v24 实验配置保持原样。沿用 C2 的四轮共享八异构专家、稀疏 Top-2、残差搭档路由及其排名监督；新机制依次执行候选解码、位置级接受、选择性提交 hidden、重新解码 accepted completion，再把接受后的状态和上一轮拒绝统计交给下一轮 Router。目标值只参与训练损失和评价诊断，不进入模型前向。

| 组别 | 反馈 | 接受机制 | Oracle BCE | 拒绝反馈给 Router |
|---|---|---|---|---|
| A0 | 关闭 | 无 | 关闭 | 关闭 |
| A1 | 开启 | 无 | 关闭 | 关闭 |
| A2 | 开启 | v24 输出端 point gate | 关闭 | 关闭 |
| A3 | 开启 | latent selective commit | 关闭 | 关闭 |
| A4 | 开启 | latent selective commit | 0.1 | 关闭 |
| A5 | 开启 | latent selective commit | 0.1 | 开启 |

默认 TaxiBJ、seed 7、100 epoch、每 5 epoch 验证、双卡全局 batch 32。六组串行运行；学习率从 `1e-3` 余弦衰减至最后一轮 `3e-4`。`best.pth` 按验证 MAE 更新，最终测试前恢复；恢复前另存最后一轮的完整训练状态 `last.pth`。正式长训前先按方案运行 20 epoch pilot，确认机制指标和训练稳定性；BikeNYC 使用 `--dataset bikenyc`。不包括多尺度。

```bash
python -u scripts/v25/run_ablations.py --dataset taxibj --gpus 0,1 --dry-run
python -u scripts/v25/run_ablations.py --dataset taxibj --gpus 0,1 --epochs 20
python -u scripts/v25/run_ablations.py --dataset taxibj --gpus 0,1
```

结果写入 `outputs/v25-RAS-CoE/experiments/ras_coe_ablation/`。运行器校验完成回执、训练/验证/测试记录与 `best.pth` 后跳过已完成组；组间共享数据和 mask 分布。输出还包括逐轮 MAE、有害候选与接受后伤害率、Oracle 选择下界、Gate 分类指标、单样本逐轮不退步率和训练开销。

</details>
