# v24 五组精简对照与新组合路由

这组实验落实 [设计文档](../../model_designs/v24_TSCoE_精简实验与新组合路由_2fb0ab1.md) 的 M0/M1/C0/C1/P1 矩阵。它回答两个问题：跨轮共享专家池的多轮 CoE 相比每轮独立专家池有什么精度、参数和耗时取舍；根据主专家的实际提议选择搭档，能否优于个体分数最高的 Top-2。旧六组 focus 结果只作为历史证据，新五组统一使用修订后的 balance importance，不能与旧组当作严格配对实验。

| 编号 | 配置名 | 专家池 | 补全反馈 | 选择方法 | 主要对照 |
| --- | --- | --- | --- | --- | --- |
| M0 | `focus_moe` | 每轮独立 | 关 | 原生 Top-2 | M1：反馈；C0：共享 |
| M1 | `focus_coe_independent` | 每轮独立 | 开 | 原生 Top-2 | C1：共享 |
| C0 | `focus_shared_no_feedback` | 跨轮共享 | 关 | 原生 Top-2 | C1：反馈 |
| C1 | `focus_coe_shared` | 跨轮共享 | 开 | 原生 Top-2 | P1：伙伴选择 |
| P1 | `focus_partner` | 跨轮共享 | 开 | 主专家条件化搭档 | C1：新选择策略 |

五组按表中顺序串行；默认每组在两张卡上 DDP，也可用 `--gpu 0` 单卡运行。公共设置：三轮、六种专家 T/S/TD/SD/TA/ST、每轮实际激活两个、dim 64、Router 64、无接受门、九类混合 random 0.4 mask、全局 batch 16（双卡每卡 8，单卡每卡 16）、seed 7、80 epoch、cosine80、每 5 epoch 验证。训练和验证/测试都使用九类混合，验证/测试 mask 固定。硬路由只执行所选专家。路由均衡项使用与各组实际候选对策略对应的六维 importance，双卡时负载在每卡 8 个样本内计算，再经 DDP 梯度同步；单卡时在 16 个样本内计算。因此单卡与双卡的 balance 目标并非严格相同。

P1 先按原 Router 选主专家并执行，用临时解码和可见状态给共享搭档评分头提供输入，再从其余五位选一位；两个专家读取相同的本轮输入，按原个体 logits 的组内 softmax 融合。推理只执行这两位专家。训练时每卡每 16 个 batch 对一个轮次试运行至多三个候选搭档：原生第二名、当前选中者、随机其他候选，重复时补足；试运行无梯度，最终隐藏训练标签只用于构造排名目标，不输入评分头。日志记录额外完整前向次数、训练耗时和评分头梯度。P1 与 C1 的差别也包含这项额外监督与计算，若 P1 有收益，下一轮再做去掉主专家输出特征的消融和同训练时长对照。

最佳轮次按验证 MAE 选择并保存 `checkpoints/best.pt`。随后在同一最佳权重的验证集子集上，对末轮 15 个专家对分别重跑前向，得到 `J_top2`、`J_anchor_best`、`J_all_best`。Oracle 使用验证标签，只作选择空间诊断，绝不用于部署、训练或测试排名；候选包含关系天然使 `J_all_best ≤ J_anchor_best ≤ J_top2`，真正要看的是差距大小。每个候选重跑完整前向，因此这项诊断另有计算成本。当前默认抽取 48 个验证样本。前两轮的 Oracle 留待末轮诊断后按需要扩展。

启动前先核对计划：

```bash
python -u scripts/v24/run_coe_partner.py --dataset taxibj --gpus 0,1 --dry-run
```

在本地启动：

```bash
tmux new-session -s v24-coe-partner \
  'python -u scripts/v24/run_coe_partner.py --dataset taxibj --gpus 0,1 --epochs 80'
```

单卡启动时把 `--gpus 0,1` 改为 `--gpu 0`；由于每卡 batch 和 balance 聚合范围会变化，应将单卡与双卡结果视为不同训练协议。前面用户确定后续默认 TaxiBJ，因此入口默认 TaxiBJ；设计文档写的第一轮 BikeNYC 可用 `--dataset bikenyc` 切换。两套数据不能直接混排精度数值。`--epochs N` 同时覆盖五组和余弦周期；`--summary-only` 重建已有结果摘要。已完成且回执、日志、最佳权重、Oracle 文件完整的组会跳过；未完成的组从头重跑。不要同时启动其他 GPU 队列。

实验结果位于 `outputs/v24-COE/experiments/coe_partner_<dataset>/`，其中 `summary.csv` 与 `comparison.json` 汇总配对差值。各运行目录按现有 v24 数据集/实验名/mask/rate 结构保存 `config.json`、`logs/metrics.jsonl`、`logs/oracle_last_step.json` 和 `checkpoints/best.pt`。队列保存生成配置及实际源码快照/哈希；checkpoint 附带模型、优化器、调度器、AMP scaler 与各 rank RNG 状态。测试日志包含端到端 `test_time_sec`、显存和纯模型前向 `forward_ms_per_batch_per_rank`；后者是各卡每 batch 耗时的平均值，不代表整批端到端延迟。单 seed 结果只作机制筛选，最终关键比较还需至少三 seed 和第二数据集复核。
