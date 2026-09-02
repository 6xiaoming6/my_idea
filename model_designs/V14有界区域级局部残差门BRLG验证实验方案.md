# V14 有界区域级局部残差门（BRLG）验证实验方案

## 1. 实验目标

本实验验证 V14 当前“每个样本一个最终残差门”是否是 MAE改善但 RMSE退化的结构瓶颈。
只修改最终残差接纳粒度，不改变主干、多尺度特征、MoE专家池、Top-K路由、损失函数、
优化器和单阶段训练流程。

原始 V14：

\[
\hat X=\hat X_{base}+\alpha_{global}\Delta_{ctf},
\qquad \alpha_{global}\in\mathbb R^{B\times1\times1\times1\times1}.
\]

BRLG：

\[
M=1+\delta\tanh(G_{local}),
\]

\[
\alpha_{local}=\operatorname{clip}
(\alpha_{global}M,0,\alpha_{max}),
\]

\[
\hat X=\hat X_{base}+\alpha_{local}\Delta_{ctf}.
\]

最后一层卷积严格零初始化，因此训练开始时 \(M=1\)，候选预测与原始 V14 逐元素一致。

## 2. 局部门输入与无泄漏约束

门控条件全部是推理时可获得的信息：

1. `h_main` 的低维投影；
2. 原始最终校正 `abs(delta_ctf)`；
3. C2F预测与base预测的差异 `abs(x_ctf-x_base)`；
4. 局部观测 mask 密度；
5. 仅在已观测位置计算的 base 误差图。

缺失位置 `x_f_gt` 不进入门控。门控条件默认 `detach`，因此新增结构不会通过条件分支
反向改变 V14 主干；但最终乘法仍会正常训练局部门和残差分支。

局部门不在原分辨率逐像素自由预测：

- temporal：`[B,1,T,1,1]`，每个时间步一个倍率；
- regional：`[B,1,T,H/4,W/4]`，插值为平滑区域倍率。

这样能够获得样本内部自适应能力，同时降低像素级门控过拟合风险。

## 3. 六个单变量候选

| 候选 | 门控粒度 | 最大相对偏移 delta | 其他设置 |
|---|---|---:|---|
| B01 | temporal | 0.10 | hidden=16，detach输入 |
| B02 | temporal | 0.20 | 同上 |
| B03 | temporal | 0.30 | 同上 |
| B04 | regional | 0.10 | spatial divisor=4 |
| B05 | regional | 0.20 | spatial divisor=4 |
| B06 | regional | 0.30 | spatial divisor=4 |

该矩阵只考察两个因素：门控粒度与有界幅度。B05 是事前理论首选，但所有候选必须按
相同验证标准排序，不能优先读取 B05 测试结果。

## 4. Core-6

每个候选完整运行：

1. TaxiBJ fixed@0.4
2. TaxiBJ random@0.4
3. BikeNYC fixed@0.6
4. BikeNYC random@0.8
5. CHAP fixed@0.2
6. CHAP random@0.4

总任务量：`6 × 6 = 36` 个完整训练。使用数据集原始预算：

- TaxiBJ：160 epoch，val_epoch=5；
- BikeNYC：140 epoch，val_epoch=2；
- CHAP：150 epoch，val_epoch=5。

根据最近实测耗时，单卡估算：

\[
6\times(2\times1.364+2\times0.099+2\times0.847)
=27.72\text{小时}.
\]

每次训练定期验证，只覆盖保存一个验证 MAE最佳检查点，训练结束加载该检查点测试一次。

## 5. 执行

先运行管线 smoke：

```bash
conda activate difftdi
python scripts/v14-exploration/run_brlg_exploration.py \
  --gpu 0 --phase screen --candidates B05 --epochs 1
```

smoke 不能作为正式结果。正式 Core-6：

```bash
python scripts/v14-exploration/run_brlg_exploration.py \
  --gpu 0 --phase screen
```

正式运行不要添加 `--epochs`。脚本单卡串行，每个任务超时上限为 4 小时；重启后执行
同一命令会检查配置、完整测试、有限 MAE/RMSE、`metrics.jsonl` 和 `best.pt`，自动跳过
已完成任务。

36 组结束后，仅用验证集生成冻结判定：

```bash
python scripts/v14-exploration/summarize_brlg_screen.py
```

## 6. 预注册晋级标准

候选必须同时满足：

1. Core-6 至少 4/6 点验证 MAE改善不少于 0.5%；
2. 验证 MAE宏平均改善不少于 1%；
3. 验证 RMSE宏平均退化不超过 0.5%；
4. 任一点验证 MAE退化不超过 3%；
5. 任一点验证 RMSE退化不超过 5%；
6. 每个数据集两点平均 MAE退化不超过 0.5%；
7. 全部指标有限，日志和检查点完整；
8. `local_gate_modulation` 有限且不长期固定在边界。

多个候选合格时，先按验证 MAE宏平均、再按验证 RMSE宏平均排序，只晋级一个候选。
全部失败则关闭 BRLG，不根据测试结果新增 delta 插值。

## 7. 晋级后才能运行

以下 `B05` 仅为命令示例，必须替换为验证集冻结的唯一候选：

```bash
python scripts/v14-exploration/run_brlg_exploration.py \
  --gpu 0 --phase multiseed --candidates B05 \
  --seeds 42 2026 3407

python scripts/v14-exploration/run_brlg_exploration.py \
  --gpu 0 --phase all24 --candidates B05
```

不得在 Core-6 判定前运行上述命令。

