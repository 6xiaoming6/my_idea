# TaxiBJ：20 epoch 同分布混合缺失路由诊断

当前入口为项目根目录 `scripts/v24/run_followup.py`，配置为 `configs/v24/followup_experiments.json`，统一基配置为 `configs/v24/route20_base.json`。旧 A–E 和 followup 结果保留；本轮重新训练，不能与旧 50/70 epoch 数字直接当作配对结果。

## 共同协议

- TaxiBJ 清洁数据，seed 7，batch 16，20 epoch，cosine 周期也为 20；每 2 epoch 验证，关闭早停。
- train/val/test 均使用九类近似等比例混合：random_point、node_outage、temporal_gap、spatial_region、spatiotemporal_block、stripe、moving_region、multi_block、composite；全部缺失率 0.4。
- 每个窗口分配一种模式，composite 本身允许组合模式。不是对每个窗口同时叠加九类 mask。
- 训练每 epoch 重采样，所有组共享相同 mask seed、epoch/index 规则和样本顺序。训练 2452 个窗口，每类每 epoch 272–273 个。
- 验证 342 个窗口，每类 38 个；测试 707 个窗口，每类 78–79 个。验证/测试使用独立 seed（基础 mask seed +20000/+30000），固定 mask，跨组相同。
- 各组按混合验证 MAE 选择最佳 checkpoint，结束后仅测试一次。路由诊断同时查看最后 5 epoch 与最佳 checkpoint，不能只看早期最好的一次验证。

## 运行顺序与对照

| 顺序 | 配置名称 | 相对 R0 的改变 | 检验问题 |
|---|---|---|---|
| R0 | route20_base | 四层六专家、从头硬路由，balance=0.01 | 同分布基准，优先运行 |
| R1 | route20_warmup | 3 epoch 软预热 + 3 epoch 过渡，uniform_mix_start=0.5、sampling_temperature_start=2 | 早期探索能否避免饱和；第 7–20 epoch 硬路由是否保持分化 |
| R2 | route20_grouped | 分组归一化路由输入 + 观测差分特征 | 路由输入表达是否影响坍缩；不叠加 R1 |
| R3 | route20_previous | 传递上一层实际选择的专家身份 | 专家选择历史是否有帮助；不叠加 R1/R2 |
| R4 | route20_noise | 训练阶段仅给前两层 router 输入加相对尺度 0.1 的高斯噪声，验证/测试关闭 | 随机扰动能否打破前两层固定选择 |
| R5 | route20_fixed | 固定 TA→ST→S→TA；无可学习路由，balance=0 | 学习路由相对一条预先指定链路是否有收益 |
| R6 | route20_small | 两层三专家 T/S/ST，balance=0.01 | 小链路候选是否更稳定、更高效 |

R2 同时改变特征表达与归一化，是一个路由输入方案对照，不能拆解各组成部分的贡献。R6 同时改变层数和专家集合（也移除了 TA），不是严格的深度单变量消融。固定路径的路径来自旧 A 的验证行为，只是一条预先指定参照，不能视为最优固定路径。D-static 暂不进入本轮队列，集中检验路由设置。

## 判断方式

每层分别看专家使用率、最大占比、路由熵、logit margin、router 梯度；整条链看路径数、最大路径占比、路径熵；按九种缺失模式分别看路由分布。验证采用确定性选择，避免把训练 Gumbel 随机性误认为学到的分工。固定路径组的集中使用是实验定义，不是训练失败。

20 epoch 用于早期筛选，不证明最终精度或长期不坍缩。改变验证/测试分布本身不提供训练梯度；同分布协议用于正确评估及选模。所有组都在完全相同的混合测试 mask 上比较，汇总以 R0 为参照。

## 启动

```bash
cd /home/students/HuangMingYu/code/py/my_idea/my_idea
conda activate difftdi
python scripts/v24/run_followup.py --dry-run
tmux new-session -s v24-route20 'python -u scripts/v24/run_followup.py --gpu 0'
```

单卡顺序执行，不使用第二张卡。已有 GPU 任务时入口拒绝启动；如果旧队列还在运行，需要先在原 tmux 里 Ctrl-C 退出旧队列再启动。修改磁盘配置不会把已启动进程自动改成 20 epoch。旧队列检测到代码/配置变化时也可能在任务边界退出。

新结果目录：`outputs/v24-COE/experiments/route20/followup/<fingerprint>/`。任务名和协议明确标注 `mixed9_rate0.4`，不再沿用误导性的 random_point 名称。终端仍只显示训练进度条，详细统计写入日志。

## 时间

五个动态四层组按此前约 94 秒/epoch，各约 31–33 分钟；固定路径和两层候选预计更快。七组总计暂估 **3–3.5 小时**，预留到 4 小时；以本轮实际速度为准。未启动新的训练。
