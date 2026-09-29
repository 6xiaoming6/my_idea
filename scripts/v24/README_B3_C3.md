# B3 基础基线与 C3 局部支撑链

2026-09-29 起后续 CoE 改进统一对照 **B3**：四轮、共享 T/S/TD/SD/TA/ST/TL/SL、每轮原生 Top-2、组内 softmax、direct hidden 更新、关闭 completion_feedback。没有额外 shared 分支、搭档路由、接受门、多尺度、残差累加或路由软预热。前次开启反馈的队列已中断；本轮从头训练，两组均不继承旧权重。

B1/B2/B3 的精确覆盖配置在 `configs/v24/b3_c3/`：B1 单层独立 Top-8，B2 四层独立 Top-2 且反馈关闭，B3 四轮共享 Top-2 且反馈关闭。本轮只依次启动 B3、C3；B1/B2 需显式选择。

共同协议：TaxiBJ random 0.4、九类混合 mask，训练按 epoch 重采样，验证/测试固定采样但类型分布与训练相同；seed 7，**全局 batch 32、单卡、100 epoch、初始 1e-3、余弦下限 3e-4、val 每 5 epoch**。优化器 AdamW、weight_decay=1e-4（沿用 bias/norm 不衰减分组）、grad_clip_norm=1.0、AMP 开启、早停关闭。L1、lambda_coe_mid=0、lambda_coe_balance=0.01、lambda_coe_z=0、balance_importance=candidate，不额外加 C3 损失。最后保存 best.pth 和 last.pth，test 使用验证最优的 best。`--dataset bikenyc` 可切换数据集，`--epochs`、`--batch-size` 可覆盖默认值。

`completion_feedback=false` 沿用现有模型语义：每轮专家输入投影读取当前 hidden，但 completion 固定为初始补全；路由中的 completion 同样固定，change 输入为零。逐轮解码仍用于最终输出及评价，不回送下一轮；C3 的路径支撑状态继续演化。

## C3 的精确定义

- 固定单尺度 8×8 空间区域，保留全时间窗口 T=12；TaxiBJ 32×32 分成16区。每轮每区独立原生 Top-2。四轮路由参数独立，同轮各区域共用路由；八个专家跨区域及跨轮共享。
- 原生路由输入统计按区域计算，原始 mask 支撑特征和坐标先在完整网格计算再分区。隐藏状态和初始完成值只在本区域池化。
- 初始 R0=M、D0=M：R 是原始观测沿条件计算路径的可达性，D 是观测支撑密度的代理量。D 不是置信度或独立原始观测数；路由权重本身携带的信息不计入 R，所以不声称它刻画整个网络的全部依赖。
- 每个候选仅进行低维 mask 运算：按专家真实卷积核及 dilation，对 R 做邻域 OR、D 做有效邻域均值。TA 使用全时间范围。没有试执行所有神经专家。
- 路由额外读取区域 R/D 的全体和缺失位置均值，以及各候选在缺失位置的 R/D 增量。独立小 MLP 产生路由 logit 修正，其输出层零初始化；原生区域路由和专家初始化与 B3 对齐。
- 真正执行的两位专家分别读取上一轮全局 hidden 的对应区域及所需 halo，执行后只提交中心区。全图边界使用裁剪输入，避免填充假位置经过带 bias 的点级层。所有区域同步读取旧状态，不依赖区域遍历顺序。
- 下一轮 R 仅对选中专家的候选支撑取并集，D 仅按这两个专家实际融合权重加权；原始观测位置始终锚定为1。未选专家的支撑不会进入状态。预测值从不变为新增观测。
- direct 指 `hidden_next = weighted_expert_output`；输入投影仍读取当前 hidden、初始 completion、原始 mask 和原始 mask 特征，与 B3 一致。

C3 的路由统计单位是区域，B3 是窗口；日志 `coe_regions_per_window`、`coe_routing_sample_count` 区分两者。监督、MAE/RMSE 和逐轮补全指标仍按原始窗口/缺失点计算。均衡损失只统计有监督目标的区域。日志同时记录每轮可达性、D、每区域执行专家数、含 halo 的专家面积开销和区域组合差异。C3 固定执行2个专家/区域，但 halo 会增加计算，不能宣称与 B3 FLOPs 严格相等。

这次只检验整套 C3 是否超过 B3，无法单独证明支撑演化优于纯局部路由。后续归因需要相同局部结构下禁用支撑或冻结支撑的对照；配置已支持 `support_enabled` 和 `support_evolution`，本轮不额外启动。

## 运行

```bash
python scripts/v24/run_b3_c3.py --dataset taxibj --gpu 0 --dry-run
tmux new-session -s v24-b3-c3 \
  'python -u scripts/v24/run_b3_c3.py --dataset taxibj --gpu 0'
# 观察：tmux attach -t v24-b3-c3
```

程序冻结源码后串行运行，完整回执与 best/last 均存在才跳过该组；配置或源码变化使用新的指纹目录。训练输出保持 batch 级 `train epoch 当前/总数` tqdm，仅显示 train loss/mae/rmse。结果目录遵循 `日期时间_核心名_seed7/random/0.4/...`；计划、启动日志、汇总位于 `outputs/v24-COE/experiments/b3_c3/<dataset>/<fingerprint>/`。中断未完成任务重跑，不自动断点续训。
