# V24 TaxiBJ A → B → C → D → E 单卡顺序实验

代码与配置已同步到项目根目录下的 src/、configs/v24/ 和 scripts/v24/；outputs 下旧工作副本只保留历史记录，不作为启动入口。未启动正式训练。用户自行在本地 tmux 中执行。禁止两张 GPU 同时训练；启动脚本检测到任意 GPU 上已有计算进程时会直接拒绝启动，不会停止其他任务。

## 五组对照

| 组别 | 设置 | 主要比较 |
|---|---|---|
| A | 清洁数据 + 原四轮六专家路由 + balance=0.01 | 共同参照 |
| B | A + 路由数值缩放、分组输入、观测时空差分 | B−A |
| C | B + FP32 路由、路由小学习率、软到硬预热、z-loss | C−B |
| D | A + 九类缺失模式、逐轮重采样训练 mask | D−A |
| E | D + 将上一轮实际选中专家的信息传给下一轮路由 | E−D |

D/E 使用完全相同的 mask 随机源和 epoch/窗口索引，同一窗口同一轮的 mask 一致；A/B/C 共享固定训练 mask。五组验证/测试仍共用相同固定 random_point 40% mask，按验证 MAE 选择最佳 checkpoint，结束后做一次最终测试。多样模式评估工具 evaluate_mask_panel.py 可在主实验完成后单独调用，不自动增加本夜队列负担，也不用于挑选 checkpoint。

## E 如何传递专家选择

四轮仍各自使用独立路由器，专家池仍跨轮共享 T/S/TD/SD/TA/ST。原模型的动态状态已经隐含前面专家的处理结果；E 再补充显式的上一轮专家身份。

第 1 轮保持 D 的原路由。第 k=2,3,4 轮将上一轮实际选择的专家转为 6 维 one-hot，再映射到路由器隐藏层：

```text
state_k ── LayerNorm ── Linear ── (+) ── GELU ── Linear ── logits_k
                                 ↑
one_hot(expert_{k-1}) ── Embedding_k
```

每轮新增一个独立的 6×64 可学习嵌入，三轮共增加 1152 个参数。嵌入全部零初始化，不消耗额外初始化随机数，E 的公共参数、初始预测和初始路由与 D 一致；随后通过原任务损失和 balance 辅助项学习如何利用专家身份，不新增损失项。

训练阶段传递的是 Gumbel 采样实际选中的专家，不是无噪声 logits 的 argmax；验证/测试传递确定性 argmax 选择。one-hot 作为离散条件 detach，不通过这条新增通道对之前的选择反传；动态状态的正常跨轮梯度仍保留。每个 batch 的第 1 轮重新开始，不保存上个 batch 的专家信息；不使用未来选择、标签或缺失类型标签。

不禁止重复专家，也不强制轮换。E 可能学到有用的专家衔接，也可能学到固定转移，因此需要同时观察 E−D 的误差、硬路由路径分布、逐类 mask 的专家选择和梯度。

配置开关：model.coe.previous_expert_context=true。本次仅支持无软预热的 legacy hard 路由；混用 grouped/soft 会显式报错，避免把软权重的 argmax 错当实际执行专家。A/B/C/D 不启用该开关。

## 训练预算与预计时间

所有组统一 **70 epoch**，单种子 7，batch size 16，四轮六专家，无多尺度。清洁 TaxiBJ 训练/验证/测试为 2452/342/707 个窗口，每轮 154 batches。每 2 轮验证，共 35 次；不开 early stopping，保存最佳 checkpoint。余弦学习率完整覆盖 70 轮（不是之前 20 轮筛选时的 80 轮前缀）。

依据已有真实日志：旧 TaxiBJ 最近 10 轮平均约 91.76 秒/epoch；清洁数据 A 首轮约 91.71 秒。五组共 350 轮，基础估算约 8.9 小时。考虑 D/E 在线 mask 生成、B/C 路由计算、验证/保存和运行波动，建议按 **9–10.5 小时**规划。2026-09-17 晚 22:15 左右开始，约在 2026-09-18 07:15–08:45 完成；实际完成时间随启动时刻和服务器负载顺延，不设置到点强制终止。

这个 epoch 数为五组共同预算，不会针对某一组看效果后单独增加轮数。70 轮是本夜比较预算，不保证所有方案都已充分收敛。

## 一条命令在 tmux 运行

```bash
cd /home/students/HuangMingYu/code/py/my_idea/my_idea
tmux new-session -s v24-abcde '/home/students/HuangMingYu/anaconda3/envs/difftdi/bin/python -u scripts/v24/run_abcde.py --gpu 0'
```

Ctrl+B 后按 D 可退出 tmux 界面、保留运行；重新查看：

```bash
tmux attach -t v24-abcde
```

已经在 tmux 内时，直接在上述目录执行：

```bash
/home/students/HuangMingYu/anaconda3/envs/difftdi/bin/python -u scripts/v24/run_abcde.py --gpu 0
```

只查看五组计划，不运行训练：

```bash
/home/students/HuangMingYu/anaconda3/envs/difftdi/bin/python scripts/v24/run_abcde.py --dry-run
```

不要同时启动旧 chain4/ABC/AD 队列。新入口会先检查任意 GPU 上是否已有计算任务，再启动一个串行队列；运行中不要再手动启动第二个 GPU 训练。脚本不会修改 GPU 功率等服务器设置。

## 结果与重启

配置：configs/v24/abcde_experiments.json；E 覆盖配置：configs/v24/experiments/abc_e.json。

正式输出位于主项目 outputs/v24-COE/experiments/abcde/abcde/<内容指纹>/。终端仅保留简洁 train epoch 进度；详细记录写到 runs/**/logs/。summary.csv 汇总五组，comparison.json 另外明确输出 C−B、E−D，并保留各组对 A 的差值，负数表示误差更低。

相同入口重复运行时，仅跳过已有完整训练、验证、最终测试和最佳模型回执且校验通过的任务。被中断的任务从 epoch 1 重跑，不支持从中间 epoch 恢复。运行时不要修改项目中的源码、配置或数据，否则内容指纹检查会阻止队列继续。所有实验设置和输入文件哈希写入 protocol.json。

## 验证范围

175 项 CPU 回归测试（174 通过、1 项 CUDA 用例跳过），包括：D/E 初始参数、RNG 和输出一致；传递实际采样专家；不跨 batch 泄漏历史；上下文可影响 logits 并获得有效梯度；CPU bfloat16 前向/反向有限；五组顺序、70 轮预算、E−D 汇总与单卡启动检查。另用小型虚构 NPZ 在 CPU 验证五组完整生命周期及 checkpoint 重载；未启动正式 TaxiBJ 训练，未进行新的 GPU 训练验证。
