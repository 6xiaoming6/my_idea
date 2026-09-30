# 时空数据补全实验项目（v24-COE）

当前 `v24-COE` 分支研究 **Temporal-Spatial Chain-of-Experts（TS-CoE）**：在原始时空网格上逐层更新补全状态，每层路由器按当前观测和状态从共享专家池中选择专家。`top_k` 表示**每层同时激活的专家数**；被选专家的权重归一化后融合。当前 v24 模型只使用 fine 尺度，不构造 mid/coarse 特征。早期多尺度模型和 v14-single 实验代码仍在仓库中，但它们不是 v24 训练入口的默认架构。

当前 B3 由输入编码、逐层独立路由器、跨层共享专家池和预测头组成，关闭额外点级共享分支，关闭 completion_feedback 并直接传递每轮专家融合输出。当前主骨干采用四轮、共享八专家池、每轮稀疏激活 Top-2：`T`（时间卷积）、`S`（空间卷积）、`TD`（空洞时间卷积）、`SD`（空洞空间卷积）、`TA`（时间注意力）、`ST`（局部时空联合）、`TL`（较大时间核）、`SL`（较大空间核）。每层可读取上层更新后的状态，固定链、Soft 路由和 Top-K 均可由配置控制。详细设计见 [v24 方案](model_designs/v24_Temporal-Spatial_Chain-of-Experts_详细方案.md)，实现见 [temporal_spatial_coe.py](src/stmoe_imputer/models/temporal_spatial_coe.py)。

当前夜间探索使用 B 系列派生的 **N1–N7**：四基础模式训练、未见组合/几何测试、初始状态路由对照与固定/自适应双尺度及跨轮状态记忆，入口为 `scripts/v24/run_n_exploration.py`。精确协议、尺度预算和运行方式见 [N1–N7 实验说明](scripts/v24/README_N_EXPLORATION.md)。

## 后续实验默认训练协议（2026-09-29）

后续新实验统一使用 **100 epoch，初始学习率 `1e-3`，余弦下限 `3e-4`，每 5 epoch 验证一次**。按验证 MAE 保存 `checkpoints/best.pth`；训练正常结束或触发早停时，另存实际最后一轮的 `checkpoints/last.pth`，最终测试仍使用 best。

当前 v24 四轮八专家主入口、三种直接传递结构基准、v25 通用消融入口已同步这些默认值。历史专用实验计划及已有输出中的配置保留原协议；新建实验时应使用本节协议，不能直接沿用历史脚本的旧学习率。

调度配置为 `type=cosine, total_epochs=100, eta_min=3e-4, reach_min_at_last_epoch=true`。训练器在每轮结束后调用调度器，因此该选项使用 99 次衰减间隔：第 1 轮实际使用 `1e-3`，第 100 轮实际使用 `3e-4`，之后保持下限。`--epochs N` 会同步调整周期；单轮冒烟测试仅使用初始学习率。

`last.pth` 在恢复 best 权重之前原子写入，包含最后一轮模型、优化器、学习率调度器、AMP scaler、各进程随机状态、DataLoader 随机状态、配置和轮次；`training_state` 另记录下一轮编号、最佳验证指标及早停计数。它保存完整训练状态，不只是模型权重。默认 `save_last_checkpoint=true`，显式关闭可用于不落盘的冒烟测试；本次新增的是状态保存，命令行训练入口目前尚未提供 `--resume`。已结束的历史实验不能据 best 权重补造 last。

## 当前 CoE 基础基线与新实验

后续结构改进统一对照 **B3：四轮共享八专家、原生 Top-2、direct 更新、completion_feedback=false**。批量大小默认全局32，单卡顺序运行。前次开启反馈的 B3/C3 已中断，按关闭反馈的新配置从头训练；不继承旧权重。B3 已完成；C3 已放弃并停止。B1→B2→B4→B5 结构对照已完成：单层独立 Top-8、四层独立 Top-2、八层独立 Top-1、八轮共享 Top-1，运行与路由约定见 [结构基线实验说明](scripts/v24/README_STRUCTURE_BASELINES.md)。历史 C3 定义保留在 [B3/C3 实验说明](scripts/v24/README_B3_C3.md)。

## 安装

先激活项目使用的 Python 环境（例如 `conda activate difftdi`），再在项目根目录执行。代码支持 Python 3.9 及以上；GPU 训练需要与本机 CUDA/驱动兼容的 PyTorch。若环境里已有合适的 PyTorch，下面的安装会保留满足版本约束的版本；否则先按本机环境安装 PyTorch，再安装项目依赖。

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` 包含训练/评估与 TaxiBJ、CHAP 预处理所需的直接依赖；不需要 torchvision。`pyproject.toml` 将预处理依赖分别提供为 `taxibj`、`chap` 可选项。Linux 上的 v24 顺序实验脚本使用 `fcntl` 和 `nvidia-smi` 检查单卡队列，正式训练需要可用的 NVIDIA GPU。

先用合成数据检查安装、前向和训练流程：

```bash
python -u scripts/train.py -c configs/v24/smoke.json --synthetic --no_plot -n v24_smoke
```

## TaxiBJ 数据与 mask

真实数据训练读取 `data/TaxiBJ/taxibj_{train,val,test}.npz`。NPZ 至少包含 `x_f_gt`（或 `x_f`），形状为 `[N,C,T,H,W]`；当前 TaxiBJ 配置使用 `C=2, T=12, H=W=32`。原始 random 0.4 协议读取 `data/TaxiBJ/random_mask/0.4/{train,val,test}.csv`：每个样本一行空间 mask，沿时间维广播。九类混合缺失模式由配置中的 `train_mask_diversity` / `eval_mask_diversity` 在线生成；两种协议不能直接混排比较。

数据文件不随 Git 提交。若已准备好分割后的 NPZ，但缺少原始 random 0.4 CSV，可生成：

```bash
python scripts/generate_fixed_masks.py \
  --train_npz data/TaxiBJ/taxibj_train.npz \
  --val_npz data/TaxiBJ/taxibj_val.npz \
  --test_npz data/TaxiBJ/taxibj_test.npz \
  --pattern random --mask_rate 0.4 --seed 7
```

已有 mask 文件时先核对其生成设置；重新生成会改变实验输入。其他缺失模式和双 mask 对照见 [双 mask 实验说明](scripts/v24/README_DUAL_MASK.md)。

## 运行历史 v24 实验入口

以下早期四轮八专家入口使用 **TaxiBJ/BikeNYC random 0.4 九类混合 mask**、共享专家池、每轮 Top-2，前 3 个 epoch 软预热、随后 3 个 epoch 过渡。根据最近的共享池对照结果，主配置关闭显式补全反馈；路由仍逐轮读取更新后的隐藏状态。默认 100 epoch，可用 `--epochs` 调整。先检查计划，再在本地启动（单卡或双卡 DDP）：

```bash
python -u scripts/v24/run_coe_main_s4_e8.py --dataset taxibj --gpu 0 --dry-run
tmux new-session -s v24-coe-main \
  'python -u scripts/v24/run_coe_main_s4_e8.py --dataset taxibj --gpu 0 --epochs 100'
# 双卡时把 --gpu 0 改为 --gpus 0,1；BikeNYC 时使用 --dataset bikenyc。
```

输出目录的实验层采用 `YYYYMMDD_HHMMSS_核心名_seedN`，例如 `20260921_095818_depth_s1_top6_seed7`；mask 类型和缺失率保留在下层目录。旧的 v24 实验目录已按相同规则迁移，索引和完成回执同步更新。

主配置与数据集入口见 [四轮八专家主骨干说明](scripts/v24/README_COE_MAIN_S4_E8.md)。旧的 [coe_main_base.json](configs/v24/coe_main_base.json) 是历史四轮六专家配置，保留供原实验复现。

四轮八专家的独立 MoE、旧搭档评分、残差搭档评分、共享原生 Top-2 与两种改进搭档路由的六组对照，见 [残差搭档对照说明](scripts/v24/README_COE_PARTNER_RESIDUAL4.md)；入口为 [run_coe_partner_residual4.py](scripts/v24/run_coe_partner_residual4.py)。

八专家单轮 Top-8、两轮 Top-4 与四轮共享池全部 28 对评分的对照，见 [深度与专家对实验说明](scripts/v24/README_DEPTH_PAIR8.md)，入口为 [run_depth_pair8.py](scripts/v24/run_depth_pair8.py)。

五组共享与反馈精简对照使用 **TaxiBJ random 0.4 九类混合 mask** 和双卡 DDP；`--dataset bikenyc` 可切换数据集。M0/M1/C0/C1 分离专家共享与补全反馈的影响，P1 检验基于主专家实际输出选择搭档是否优于原生 Top-2。五组串行、每组全局 batch 16（每卡 8），默认 80 epoch；推理只执行选中的两个专家。先检查计划，再在本地启动：

```bash
python -u scripts/v24/run_coe_partner.py --dataset taxibj --gpus 0,1 --dry-run
tmux new-session -s v24-coe-partner \
  'python -u scripts/v24/run_coe_partner.py --dataset taxibj --gpus 0,1 --epochs 80'
```

详见 [五组实验说明](scripts/v24/README_COE_PARTNER.md)。旧 [六组 focus 对照](scripts/v24/README_COE_FOCUS.md) 保留供历史复现；新五组的路由均衡项定义已修订，不能与旧日志当作严格配对。以下 TaxiBJ 命令是更早的实验入口。

单次真实数据训练可从 [TaxiBJ 配置](configs/v24/taxibj.json) 启动；该配置是早期的两层固定 T→S 示例。历史四层六专家软预热配置在 [coe_main_base.json](configs/v24/coe_main_base.json)，需由对应的历史实验脚本补齐具体数据与 mask 协议。

```bash
python -u scripts/train.py \
  -c configs/v24/taxibj.json \
  --train_npz data/TaxiBJ/taxibj_train.npz \
  --val_npz data/TaxiBJ/taxibj_val.npz \
  --test_npz data/TaxiBJ/taxibj_test.npz \
  --no_plot -n v24_taxibj_example
```

历史的深度与专家池对照使用 **TaxiBJ、原始 random 0.4、seed 7、batch 16、单卡顺序训练**。其中两组严格匹配训练协议的 Top-2 对照为三层六专家和四层八专家，默认各 30 epoch：

```bash
python scripts/v24/run_top2_pair.py --dry-run
# 确认没有其他 GPU 计算任务后，单卡顺序运行：
tmux new-session -s v24-top2-pair \
  'python -u scripts/v24/run_top2_pair.py --gpu 0'
```

脚本位于 [run_top2_pair.py](scripts/v24/run_top2_pair.py)，计划位于 [top2_pair_experiments.json](configs/v24/top2_pair_experiments.json)。`--epochs N` 可同时覆盖两组 epoch 和余弦学习率周期。已有完整结果回执的任务会跳过，中断但未完成的任务从头重跑。两组只改变链路层数和专家池；由于参数量与计算量也随之变化，这一对照不能单独区分深度收益和专家池收益。

更多可复用实验入口：

| 实验 | 入口与说明 | 主要协议 |
| --- | --- | --- |
| 八/十二专家深度与 Top-K 网格 | [run_depth_pool.py](scripts/v24/run_depth_pool.py) · [说明](scripts/v24/README_DEPTH_POOL.md) | 原始 random 0.4；默认 20 epoch，可用 `--epochs` 覆盖 |
| 软预热主方案与机制消融 | [run_coe_validation.py](scripts/v24/run_coe_validation.py) · [说明](scripts/v24/README_COE_VALIDATION.md) | 九类混合 mask；另含原始 mask 组 |
| 自由路由与固定链的双 mask 对照 | [run_dual_mask.py](scripts/v24/run_dual_mask.py) · [说明](scripts/v24/README_DUAL_MASK.md) | 原始 random 0.4 与九类混合 mask 各一组 |
| A0/A2 缺失率对照 | [run_rate_compare.py](scripts/v24/run_rate_compare.py) · [说明](scripts/v24/README_RATE_COMPARE.md) | 原始 random，缺失率 0.2–0.8 |

更早的 A–E 与八组机制实验分别见 [A–E 说明](scripts/v24/README_ABCDE.md) 和 [机制实验说明](scripts/v24/README_MECHANISM1.md)。不要同时启动多个 GPU 队列；这些入口会检查现有 GPU 计算进程。训练控制台只显示 batch 级 `train epoch 当前/总数` tqdm，以及 train 的 `loss`、`mae`、`rmse`；验证、测试和路由诊断写入日志。

## 输出与评估

训练结果按数据集、实验名、mask 和缺失率存放：

```text
outputs/v24-COE/TaxiBJ/custom/<YYYYMMDD_HHMMSS_核心名_seedN>/random/rate0.4/<时间_seed_bs>/
├── config.json
└── logs/
    ├── train.log
    ├── val.log
    ├── test.log
    └── metrics.jsonl
```

队列另在 `outputs/v24-COE/experiments/<队列名>/` 保存生成的配置、启动日志和完成回执 JSON。每次按**验证 MAE** 选最佳轮次，再用对应模型测试一次。历史 Top-2 和深度池实验设为 `save_best_checkpoint=false`：最佳状态只在训练进程的 CPU 内存中保留，不写 `best.pt`。五组精简实验为了在最佳权重上完成验证集 Oracle 诊断和复现，统一保存 `checkpoints/best.pt`；当前四轮八专家主骨干不运行 Oracle，保存 `best.pth` 与 `last.pth`。路由使用率、每层选择与梯度等详细指标保存在日志中。单种子短训结果适合筛选候选，不能单独证明结构优越性。

## 代码位置

- [src/stmoe_imputer/models/](src/stmoe_imputer/models/)：v24 CoE 与历史骨干网络；[registry.py](src/stmoe_imputer/models/registry.py) 按配置选择架构。
- [src/stmoe_imputer/data/](src/stmoe_imputer/data/)：NPZ、离线 CSV mask 与九类在线混合 mask。
- [scripts/train.py](scripts/train.py)：统一训练、验证、最佳轮次测试和结果记录。
- [scripts/v24/](scripts/v24/) 与 [configs/v24/](configs/v24/)：v24 顺序实验入口与配置。
- [experments_report/](experments_report/)：已有实验分析报告；[scripts/v14-exploration/](scripts/v14-exploration/) 与 [configs/v14-single/](configs/v14-single/) 保留历史实验。
