# 八专家深度、宽度与全部专家对评分对照

本队列按以下三组顺序运行。均使用 TaxiBJ（可切换 BikeNYC）、训练/验证/测试同分布九类混合 random 0.4 mask、T=12、dim=64、八种异构专家 T/S/TD/SD/TA/ST/TL/SL、seed 7、全局 batch 16、默认 80 epoch、每 5 epoch 验证。训练 mask 逐轮重采样，验证/测试 mask 各自固定。双卡 DDP 同时用于**一个**训练任务，三组之间串行，不在两张卡上各跑一个实验。按验证 MAE 选最优状态并测试，不落地保存 `best.pt`。

| 顺序 | 组名 | 结构与选择 | 每窗口路由专家调用次数 |
|---|---|---|---:|
| 1 | `depthpair_moe_s1_top8` | 单轮独立八专家池，全部激活，组内归一化融合 | 8 |
| 2 | `depthpair_moe_s2_top4` | 两轮各有独立八专家池，每轮选择得分最高的四个 | 8 |
| 3 | `depthpair_coe_s4_pair28` | 四轮共享八专家池，每轮对全部 C(8,2)=28 个组合计算 `z_i+z_j+b_ij`，选择最高分组合并只执行其中两个专家 | 8 |

第三组的 `b_ij` 是按当前状态学习的交互分数，零初始化时回退到原生 Top-2。它只在 Router 上评分全部 28 对，不预先执行全部专家。三组均保留共享状态更新、候选重要性 balance=0.01、无显式补全反馈、无接受门和无多尺度。路由专家调用次数相同，但轮数、共享状态更新次数、路由器次数及异构算子成本不同，因此这不是严格等 FLOPs 或等时长实验。

```bash
python -u scripts/v24/run_depth_pair8.py --dataset taxibj --gpus 0,1 --epochs 80 --dry-run
tmux new-session -s v24-depth-pair8 \
  'python -u scripts/v24/run_depth_pair8.py --dataset taxibj --gpus 0,1 --epochs 80'
```

单卡可用 `--gpu 0`，切换数据集用 `--dataset bikenyc`。完成回执会被核对后跳过，中断组从 epoch 1 重跑。结果分别写到 `outputs/v24-COE/<数据集>/custom/<YYYYMMDD_HHMMSS_核心名_seedN>/random/rate0.4/`，队列摘要位于 `outputs/v24-COE/experiments/coe_depth_pair8/`。
