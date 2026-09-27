# v24 TS-CoE 组队路由与接受反馈：首批对比实验

本批次对应 [实验设计 v4](../../model_designs/v24_TSCoE_实验设计_v4_已核对最新提交_6d65c6f.md) 的 §5.1。顺序固定为 E0、E1、E2、E3、E4、E5、E10，每组独立从头训练 50 epoch，不自动开始训练。E6–E9 是首批结果出来后的反馈机制消融，当前队列不包含它们。

| 组别 | 相对基准的唯一结构变化 | 主要检验 |
| --- | --- | --- |
| E0 | 原生确定性 Top2 | 三层、六专家、跨层共享的基准成绩 |
| E1 | 15 个专家对的加性评分及确定性 ST 选择代理 | 与 E0 比较代理梯度的影响；前向选择规则应等价 |
| E2 | E1 + 专家对交互评分头 | 与 E1 比较显式“专家搭档”是否有收益 |
| E3 | E1 + 个体 Router 隐层 64→68 | 与 E2 比较，排除仅增加 Router 参数量的解释 |
| E4 | E1 + 点级接受门 | 与 E1 比较，验证拒绝有害候选、跨轮反馈是否有效 |
| E5 | E2 + 点级接受门 | 与 E2、E4 比较两项机制的增量与组合效果 |
| E10 | E1 + 每层独立六专家池 | 与 E1 比较权重共享和容量；与 E5 比较更强的分层 MoE 参照 |

E1/E2/E4/E5 构成“加性/交互 × 无门/点门”的 2×2 设计。重点看验证集配对差值，再报告 best checkpoint 的测试集 MAE/RMSE。一次 seed 7 只用于筛选，不能据此声称统计显著。

所有组采用 TaxiBJ clean train/val/test NPZ；九种缺失模式、缺失率 0.4；训练每轮重新采样 mask，验证/测试使用各自固定的同分布 mask。统一三轮链路、专家池 T/S/TD/SD/TA/ST、Top2、batch 16、L1、路由 balance 0.01、3+3 温度日程、cosine 50、每 2 epoch 验证、关闭早停并保存 best checkpoint。Top2 仍是确定性选择，温度日程只改变组内权重；当前实现每轮会计算六个候选专家，三轮合计 18 次候选计算，最终只有六次非零贡献。数据文件与 clean manifest 的哈希会在队列启动前核验。

先检查配置与顺序：

```bash
python -u scripts/v24/run_experiments.py --config configs/v24/team_accept_v4_experiments.json --study coe_team_accept_v4 --gpu 0 --dry-run
```

历史 TaxiBJ 策略在本机单卡顺序运行：

```bash
tmux new-session -s v24-team-v4 \
  'python -u scripts/v24/run_experiments.py --config configs/v24/team_accept_v4_experiments.json --study coe_team_accept_v4 --gpu 0 --epochs 50'
```

队列会拒绝已有 GPU 计算进程或重复启动。重启相同命令会跳过已验证完成的组；未完成的组从头开始。各组训练日志和 checkpoint 按 `outputs/v24-COE/TaxiBJ/` 下的实验名组织，队列汇总在 `outputs/v24-COE/experiments/coe_team_accept_v4/coe_team_accept_v4/<fingerprint>/summary.csv` 和 `comparison.json`。训练控制台保持 batch 级 tqdm，仅显示 train loss、MAE、RMSE；详细诊断写入训练日志。日志另含每缺失族累计误差/有效标签数、每轮专家选择频率与专家对路径、接受门有害及有益修改量、梯度和 AMP 更新信息。

这批实验不会枚举全部 15 个末轮候选来计算 oracle regret；该诊断需在选定 checkpoint 后对固定验证子集单独运行，不能用测试标签为模型选参。


## 后续 BikeNYC 实验

当前队列已精简为六组 CoE 与分层 MoE、个体 Top-2 与组合评分的对照，使用双卡 DDP 和稀疏 hard Top-2 执行。启动命令与设计见 [README_COE_FOCUS.md](README_COE_FOCUS.md)。本页上方的七组 E0–E10 是历史首批设计；原 BikeNYC 七组策略仍保存在 `configs/v24/team_accept_v4_bikenyc_experiments.json`，需显式传给 `run_experiments.py` 才会执行。
