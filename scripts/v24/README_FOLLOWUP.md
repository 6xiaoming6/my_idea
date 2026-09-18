# 后续诊断实验：评估分布、固定路径、mask 拆分与 F

本轮在已完成 ABCDE 结果基础上继续诊断，不重复训练 A/D/E。代码和配置已经写入根目录；本轮没有启动 GPU。

## 现有 checkpoint 的九类固定 mask 面板

目的：直接回答 D/E 是否只是损害 random_point，还是在结构化缺失上有收益。使用 ABCDE 已按共同 random_point 验证集选出的 A、C、D、E 最佳 checkpoint，在同一批测试窗口上分别生成九类固定 mask：random_point、node_outage、temporal_gap、spatial_region、spatiotemporal_block、stripe、moving_region、multi_block、composite。这个面板只做诊断，不重新选择 checkpoint，也不改变原正式测试结果。

```bash
cd /home/students/HuangMingYu/code/py/my_idea/my_idea
CUDA_VISIBLE_DEVICES=0 python scripts/v24/run_mask_panel.py --device cuda --output outputs/v24-COE/experiments/abcde/abcde/38027eefadf2a586/mask_panel_20260918.json
```

脚本逐个加载模型、释放显存，单卡运行。每种 mask 的 MAE/RMSE 和路由统计写入 JSON；不要同时在另一张卡启动训练。

## 四个新训练对照

新队列有四个任务，各 70 epoch、batch 16、seed 7，按下面顺序在同一张卡上运行：

| 任务 | 设置 | 用途 |
|---|---|---|
| `abc_d_eval_mixed` | D 的九类训练 mask；验证和测试也各自使用九类固定混合 mask | 修正训练/评估分布不一致的问题，作为本轮第一个任务 |
| `fixed4_ta_st_s_ta` | 四轮六专家固定 `TA→ST→S→TA` | 与 A 的动态路由比较；路径来自 A 的验证集行为，不使用测试选择 |
| `abc_d_static` | D 的九类 mask，但每个训练窗口只生成一次，70 轮不重采样 | 与已有动态 D 比较，拆开“形态多样性”和“逐轮重采样” |
| `abc_f` | 两轮、三专家 `[T,S,ST]`，硬路由，balance=0.01 | F：较小链路候选，与 A 的四轮六专家比较 |

D-eval-mixed 的训练/验证/测试都使用相同的九类和 40% 缺失率分布。验证和测试 mask 按各自数据集固定生成，避免验证指标随 epoch 改变；它们与训练 mask 的家庭分布一致，但不是同一批样本的 mask。该组的 test MAE 不能直接和 random_point 测试协议下的其他组做数值比较，应在同一评估协议内比较。

D-static 的训练 mask 与动态 D 使用同一 seed、同一九类、同一 40% 缺失预算；区别只有 `resample_each_epoch=false`。它的验证和测试仍使用原来的 random_point 协议。固定路径 `TA→ST→S→TA` 是 A 在验证集形成的路径，不能作为所有可能固定路径的最优证明；该实验只回答“这条固定链是否已足够解释 A 的成绩”。

F 不是 A 的严格单变量消融，因为它同时减少轮数和专家池，并改变候选算子集合；它是容量/链路预算候选。若 F 优于 A，只能说明较小候选在本任务上有竞争力，不能直接归因于某一个专家。

## 启动

先确认服务器所有 GPU 都没有计算任务，然后只运行一张卡：

```bash
cd /home/students/HuangMingYu/code/py/my_idea/my_idea
conda activate difftdi
tmux new-session -s v24-followup 'python -u scripts/v24/run_followup.py --gpu 0'
```

只查看计划，不训练：

```bash
python scripts/v24/run_followup.py --dry-run
```

单卡启动入口会检查任意 GPU 上的计算进程，发现已有任务就拒绝启动；不会停止其他任务，也不会自动使用第二张卡。四项任务按 D-eval-mixed→固定路径→D-static→F 顺序执行。调度器不会把不同评估 mask 协议的结果混成一个 paired delta。

## 预计时间

按之前 TaxiBJ 四轮六专家约 91–94 秒/epoch 估计，固定路径会更快，F 会更快，四项合计预计约 **6–7 小时**，但仍以实际日志为准。正式结果应与已有 A/D 动态结果一起看：

- D-eval-mixed vs D：先比较同一九类评估协议下的各家庭结果；
- A vs fixed path：动态路由是否超过一条验证选出的固定链；
- D-static vs D dynamic：逐轮重采样是否是主要代价/收益来源；
- F vs A：两轮三专家是否在误差和路由稳定性之间更平衡；
- 九类 mask panel：D/E 的多路径是否真的对结构化缺失有帮助。

输出根目录为 `outputs/v24-COE/experiments/followup/followup/<fingerprint>/`。所有任务仍保存最佳 checkpoint，按验证 MAE 选择并最后只测试一次。
