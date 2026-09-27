# 时空数据补全实验项目（v24-COE）

当前 `v24-COE` 分支研究 **Temporal-Spatial Chain-of-Experts（TS-CoE）**：在原始时空网格上逐层更新补全状态，每层路由器按当前观测和状态从共享专家池中选择专家。`top_k` 表示**每层同时激活的专家数**；被选专家的权重归一化后融合。当前 v24 模型只使用 fine 尺度，不构造 mid/coarse 特征。早期多尺度模型和 v14-single 实验代码仍在仓库中，但它们不是 v24 训练入口的默认架构。

模型由输入编码、逐层独立路由器、跨层共享的专家池、点级共享分支和预测头组成。常用六专家池是 `T`（时间卷积）、`S`（空间卷积）、`TD`（空洞时间卷积）、`SD`（空洞空间卷积）、`TA`（时间注意力）、`ST`（局部时空联合）；八专家池再加入较大时间核 `TL` 和空间核 `SL`。每层可读取上层更新后的状态，固定链、Soft 路由和 Top-K 均可由配置控制。详细设计见 [v24 方案](model_designs/v24_Temporal-Spatial_Chain-of-Experts_详细方案.md)，实现见 [temporal_spatial_coe.py](src/stmoe_imputer/models/temporal_spatial_coe.py)。

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

## 运行 v24 实验

当前后续实验默认使用 **BikeNYC random 0.4 九类混合 mask** 和双卡 DDP。六组精简对照依次运行，每组内部用两张卡，全局 batch 16（每卡 8）；hard Top-2 每个样本只计算选中的两个专家。先运行 `python -u scripts/v24/run_coe_focus.py --gpus 0,1 --dry-run` 核对计划，再在本地启动：

```bash
tmux new-session -s v24-coe-focus \
  'python -u scripts/v24/run_coe_focus.py --gpus 0,1 --epochs 50'
```

详见 [六组 CoE 对照实验说明](scripts/v24/README_COE_FOCUS.md)。以下 TaxiBJ 命令是历史实验入口，保留供复现。

单次真实数据训练可从 [TaxiBJ 配置](configs/v24/taxibj.json) 启动；该配置是早期的两层固定 T→S 示例。当前四层六专家软预热主配置在 [coe_main_base.json](configs/v24/coe_main_base.json)，需由相应实验脚本补齐具体数据与 mask 协议。

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
outputs/v24-COE/TaxiBJ/custom/<实验名>/random/rate0.4/<时间_seed_bs>/
├── config.json
└── logs/
    ├── train.log
    ├── val.log
    ├── test.log
    └── metrics.jsonl
```

队列另在 `outputs/v24-COE/experiments/<队列名>/` 保存生成的配置、启动日志和完成回执 JSON。每次按**验证 MAE** 选最佳轮次，再用对应模型测试一次。近期 Top-2 和深度池实验设为 `save_best_checkpoint=false`：最佳状态只在训练进程的 CPU 内存中保留，不写 `best.pt`；启用该选项的其他实验才会生成 `checkpoints/best.pt`。路由使用率、每层选择与梯度等详细指标保存在日志中。单种子短训结果适合筛选候选，不能单独证明结构优越性。

## 代码位置

- [src/stmoe_imputer/models/](src/stmoe_imputer/models/)：v24 CoE 与历史骨干网络；[registry.py](src/stmoe_imputer/models/registry.py) 按配置选择架构。
- [src/stmoe_imputer/data/](src/stmoe_imputer/data/)：NPZ、离线 CSV mask 与九类在线混合 mask。
- [scripts/train.py](scripts/train.py)：统一训练、验证、最佳轮次测试和结果记录。
- [scripts/v24/](scripts/v24/) 与 [configs/v24/](configs/v24/)：v24 顺序实验入口与配置。
- [experments_report/](experments_report/)：已有实验分析报告；[scripts/v14-exploration/](scripts/v14-exploration/) 与 [configs/v14-single/](configs/v14-single/) 保留历史实验。
