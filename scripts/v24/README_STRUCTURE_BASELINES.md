# B1/B2/B4/B5 结构基线补充实验

2026-09-29：用户放弃 C3，已停止其训练并保留历史输出。已完成的 B3 继续作为对照，不重跑。本队列按 **B1 → B2 → B4 → B5** 单卡顺序运行。

| 标识 | 层/轮数 | 每轮激活数 | 专家参数 | 专家实例总数 | 每样本累计激活数 |
| --- | ---: | ---: | --- | ---: | ---: |
| B1 | 1 | 8 | 单层独立 | 8 | 8 |
| B2 | 4 | 2 | 每层独立八专家 | 32 | 8 |
| B3（已完成） | 4 | 2 | 四轮共享八专家 | 8 | 8 |
| B4 | 8 | 1 | 每层独立八专家 | 64 | 8 |
| B5 | 8 | 1 | 八轮共享八专家 | 8 | 8 |

八种专家均为 T/S/TD/SD/TA/ST/TL/SL。每轮路由器独立，router_state/expert_state=dynamic，直接传递专家输出（state_update_mode=direct），completion_feedback=false。关闭局部路由、残差累加、额外 shared 分支、搭档选择、接受门、多尺度、上轮专家编码、噪声和预热；融合使用 original。

共同训练条件与已完成的 B3 一致：TaxiBJ random0.4，train/val/test 九种混合 mask，同分布但独立固定评估 mask；seed7，batch_size32，单卡100epoch，val_epoch5，AdamW，lr1e-3 余弦下降至第100轮3e-4，weight_decay1e-4，grad_clip_norm1，AMP，关闭早停。L1 主损失、balance0.01、mid0、z0。保存 best.pth 和最后一轮 last.pth，使用 best 做最终测试。

## Top-1 路由约定

B4/B5 显式设置 `top1_selection=argmax_st`：训练和评估均选择最高分专家，前向权重为1，每个样本每轮只执行该专家。训练使用 `one_hot(argmax(p)) + (p - stop_gradient(p))` 作为直通估计；稀疏分发只执行选中的专家，因此任务梯度是基于选中输出的代理梯度，并不是执行全部专家后的完整 Soft MoE 梯度。

不能简单把单个选中概率归一化成 `p/p` 来训练路由，否则主任务到路由的梯度为零。旧 Top-1 配置默认仍为 `gumbel`，保持历史实验行为；B1/B2/B3 原生 Top-K 行为不变。

激活次数相同不等于 FLOPs、耗时、参数量相同：专家本身异构，八层还有更多路由和投影开销。Top-1 的代理梯度也与 Top-2 组内 softmax 不同，结果分析需明确这一差异。

## 启动与重启

```bash
python -u scripts/v24/run_structure_baselines.py --dataset taxibj --gpu 0 --epochs 100 --batch-size 32
# 自行创建新会话时：
tmux new-session -s v24-structure-baselines \
  'python -u scripts/v24/run_structure_baselines.py --dataset taxibj --gpu 0 --epochs 100 --batch-size 32'
```

本次沿用已有 `v24-b3-c3` 会话，观察命令为 `tmux attach -t v24-b3-c3`。支持 `--dataset bikenyc`、`--epochs`、`--batch-size`、`--dry-run`；默认只跑上述四组。相同源码、配置与数据指纹下，重启自动跳过回执、测试结果与 best/last 均完整的组；中断组从头重跑，不自动续训。

控制台保留 batch 级 `train epoch 当前/总数` tqdm，显示 train loss/mae/rmse；详细诊断写入日志。冻结计划、源码、回执和汇总位于 `outputs/v24-COE/experiments/structure_baselines/<dataset>/<fingerprint>/`。训练输出仍按 `{年月日_时分秒}_{核心描述}_seed7/random/rate0.4/...` 命名。

## 已完成 B3 参照

- 冻结实验：`outputs/v24-COE/experiments/b3_c3/taxibj/05ca97361742edea/`
- 回执：上述目录下 `results/B3.attempt1.json`
- 训练输出：`outputs/v24-COE/TaxiBJ/custom/20260929_134519_nofeedback_base_seed7/random/rate0.4/20260929_134519_seed7_bs32/`

本次没有改变 B3 权重或历史日志。比较时使用其已完成100epoch的结果。
