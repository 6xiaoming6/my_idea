# route20_next：路由探索修正实验

本阶段只在单卡上顺序运行三组 20 epoch 实验，训练、验证和测试继续使用相同的九类混合缺失 mask。三组共享 TaxiBJ 数据、seed 7、batch 16、学习率和训练调度。

| 组 | 设置 | 目的 |
|---|---|---|
| N1 `route20_next_noise` | 四层六专家、balance=0.01，前两层训练期路由输入加入相对尺度 0.1 高斯噪声 | 修正原 R4 配置错误，单独检验噪声 |
| N2 `route20_next_warmup` | R1：3 epoch 软预热、3 epoch 过渡，之后硬路由 | 复现当前最佳 R1 并继续观察第二层 |
| N3 `route20_next_warmup_fixed2` | N2 + 第二层固定选择 TA，其余层仍动态路由 | 判断第二层 TA 集中是否损害性能 |

N1 和 N2 都是四层六专家；N3 只固定第二层，不能与整条固定链混淆。验证/测试不加噪声；N3 的第二层固定设置在训练和评估都生效。当前阶段不保存 `best.pt`：训练仍按验证 MAE 在内存中保留最佳模型，并用该模型执行最终测试。

## 启动

```bash
cd /home/students/HuangMingYu/code/py/my_idea/my_idea
conda activate difftdi
python scripts/v24/run_followup_next.py --dry-run
tmux new-session -s v24-route20-next 'python -u scripts/v24/run_followup_next.py --gpu 0'
```

入口会拒绝已有 GPU 计算进程，也会阻止第二个同阶段队列。预计单卡约 1.5 小时，实际以 N1 速度为准。
