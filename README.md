# v25 云 GPU 训练分支

本分支从 `v25`（`d8c1e60`）创建，包含迁移时尚未提交的 v25 实现。云端入口是 **`scripts/cloud_gpu/run.py`**；模型、训练、六组 ID/OOD 评估沿用服务器实现。本分支只含代码、配置及数据校验清单，数据包另行传输，不包含旧检查点或训练输出。

## 1. 下载与环境

浅克隆只下载本分支最新内容，避免下载父分支历史中的大量实验产物：

```bash
git clone --depth 1 --single-branch --branch cloud_gpu git@github.com:6xiaoming6/my_idea.git
cd my_idea
conda create -n v25_cloud python=3.11 -y
conda activate v25_cloud
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-cloud.txt
python -m pip install -e . --no-deps
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

上述 PyTorch/CUDA 组合对应原服务器环境；云 GPU 驱动/架构必须支持所选 wheel。其他 CUDA 版本参考 [PyTorch 官方版本安装说明](https://pytorch.org/get-started/previous-versions/)。若换环境需记录版本；不承诺跨 GPU 逐位复现。仅训练已预处理数据无需 pandas、xarray 或原始 TaxiBJ/CHAP 文件。环境记录见 `cloud_environment.json`。

## 2. 单独传输数据

将 `v25_cloud_data_20261013.tar` 和同名 `.sha256` 文件传到云端，在文件所在目录运行 `sha256sum -c v25_cloud_data_20261013.tar.sha256`，然后在仓库目录执行：

```bash
python scripts/cloud_gpu/unpack_data.py --archive /实际路径/v25_cloud_data_20261013.tar
# 只解包 TaxiBJ 也可以：在上面命令末尾加 --datasets taxibj
# 已解包后重新核验：
python scripts/cloud_gpu/unpack_data.py
```

解包位置为 `data/cloud_gpu/{taxibj,bikenyc,chap}/{train,val,test}.npz`。脚本逐文件校验 SHA-256，已有不同内容时拒绝覆盖；不会重新划分窗口或重建尺度。`datasets/manifest.json` 是精确字节清单，`datasets/manifests/` 保存原预处理来源。可通过解包与训练两处相同的 `--data-root /大容量目录` 改变数据根目录。

保留原数据语义：F/M/C 是预构造数据输入，TaxiBJ/BikeNYC 空间求和，CHAP 求均值；各尺度独立遮挡。粗尺度观测可能包含 F 缺失区域的聚合信号，这是任务的额外观测条件，不是完全相同信息量的 F-only 任务。BikeNYC T=12、网格补齐行仍按既有协议评分；CHAP T=7，原始缺测来源限制见 manifest。

## 3. 启动实验

当前新增的三组 TaxiBJ 实验，从头训练：

```bash
python -u scripts/cloud_gpu/run.py --dataset taxibj --methods I01 I02 I03 --missing-rate 0.4 --epochs 100 --gpu 0
```

需要同环境公平对照时，把 `--methods` 改为 `B1 B2 I01 I02 I03`。默认 seed7、batch32、四类混合 mask（random_point/node_outage/temporal_gap/spatial_region）、每5epoch验证、lr1e-3余弦至3e-4。**这里的 0.4 不是纯 random 点缺失。** 默认单卡串行，云端一般用 GPU0，可显式 `--gpu 1`。不自动等待 GPU 空闲，也不停止其他进程；同仓库重复队列/同卡云入口由文件锁阻止。

后台训练（需安装 tmux）：

```bash
bash scripts/cloud_gpu/start_tmux.sh v25-cloud-taxibj --dataset taxibj --methods I01 I02 I03 --missing-rate 0.4 --epochs 100 --gpu 0
tmux attach -t v25-cloud-taxibj
```

`Ctrl-b d` 分离。训练、评估和报告结束后会话自动关闭；错误也退出并留日志，无常驻空壳会话。可在 `outputs/cloud_gpu/experiments/<数据集>/<批次>/launcher_logs/` 查看训练/评估/启动日志。每个GPU训练run保存 `logs/gpu_status.jsonl`，15秒只读采样，失败不影响训练。不会自动启动本仓库之外的监控或修改其他训练。

BikeNYC 尺度/差分九组探索（200epoch；两个缺失率分开启动，前一批结束后启动下一批）：

```bash
python -u scripts/cloud_gpu/run.py --dataset bikenyc --methods B1 B2 S01 S02 S03 S04 S05 S06 S07 S08 S09 --missing-rate 0.4 --epochs 200 --gpu 0
python -u scripts/cloud_gpu/run.py --dataset bikenyc --methods B1 B2 S01 S02 S03 S04 S05 S06 S07 S08 S09 --missing-rate 0.6 --epochs 200 --gpu 0
```

CHAP 使用 `--dataset chap --methods A1 A2 B1 B2 --epochs 100`，仅在显式启动时训练。快速筛选可指定 `--epochs 50 --scheduler constant --lr 0.001`，必须连同对应参考使用同预算；constant 由余弦调度器上下限均为lr实现，便于兼容原恢复格式。

## 4. 方法对应关系

所有方法保留共享八种异构专家；默认四轮 CMFF、原生 Top-2。固定尺度调度不等于专家路由。

|方法|真实观测及 H0|轮次尺度/Top-k|额外通信|
|---|---|---|---|
|A1|仅 F 可见；M/C掩码归零|CMFF / 2,2,2,2|无|
|A2|同 A1|同 A1|受控差分|
|B1|FMC 可见，三尺度融合 H0|CMFF / 2,2,2,2|无|
|B2|同 B1|同 B1|受控差分|
|I01|FMC 可见，H0 仅 F；轮次再注入 C/M/F/F|CMFF / 2,2,2,2|受控差分|
|I02|FMC 可见，三尺度 H0|每轮硬选一个尺度 / 2,2,2,2|受控差分|
|I03|FMC 可见，三尺度 H0|CMFF / 2,2,2,2|受控差分 + 自适应预测历史池|
|S01|同 B1|每轮硬选一个尺度 / 2,2,2,2|无|
|S02|同 B1|CMFF / 4,4,4,4|无|
|S03|同 B1|CMFF / 1,2,4,8|无|
|S04|同 B1|CMFF / 2,2,2,2|全部受控差分；独立从头 B2 复验|
|S05/S06/S07|同 B1|CMFF / 2,2,2,2|仅在第2/3/4轮输入使用差分|
|S08|同 B1|CFFF / 2,2,2,2|无|
|S09|同 B1|CFFF / 2,2,2,2|仅第4轮同尺度差分|

差分经过 RMS 幅度匹配及 ±0.1 条件门后修正专家输入；无差分依旧通过上一轮输出传递隐藏状态。I01保留融合层参数形状，M/C的H0槽位置零，不是取消轮次的真实多尺度输入。I03池内是已发生轮次的预测（0/1/2/3条），可见F回填，历史detach；不包含乘积交互。自由尺度选择没有软预热，使用选中分支的ST梯度；S03的Top-1也使用选中概率ST梯度。不能声称逐轮误差必然下降。

## 5. 冻结、恢复与结果

首次启动生成不可变计划、源码快照、数据哈希、配置与公共初始化审计；预检用实际batch做前向/反向及重专家检查，不写训练权重。实际训练保存 `best.pth`、`last.pth` 和完整恢复状态，仅用 ID 验证 MAE 选 best，然后自动评估六个 ID/OOD 协议并在 `experments_report/` 输出正式分析报告。报告包括全部组、分族结果、训练曲线、成本及失败/缺失。阶段报告不会冒充最终报告。

终端会打印批次目录；可只准备不训练：

```bash
python scripts/cloud_gpu/run.py --dataset taxibj --methods I01 I02 I03 --epochs 100 --gpu 0 --prepare-only
python -u scripts/cloud_gpu/run.py --resume /实际批次目录 --gpu 0
# 后台恢复：
bash scripts/cloud_gpu/start_tmux.sh v25-cloud-resume --resume /实际批次目录 --gpu 0
```

恢复沿用冻结配置，从最后完整epoch继续，完成组跳过，缺少评估则补评估。`--resume` 只允许同时指定 `--gpu` 与启动控制参数；训练配置需创建新批次，不得修改冻结文件。本分支提供的是云端从头复现；数据包不含服务器已训练检查点，不能直接接续服务器原批次。

`configs/cloud_gpu/` 是可移植模板和六协议定义，`reference_results.json` 仅保存少量旧指标供查阅，不作为云端已完成回执。保留的 `scripts/v24/`、`scripts/v25/` 用于训练组件与原实现溯源，其中旧队列启动器有原服务器路径，**云端统一用 scripts/cloud_gpu 入口**。

## 6. 安装检查（CPU，小型合成数据）

```bash
python -m unittest discover -s tests/cloud_gpu -v
```

测试只检查迁移与恢复实现，不产生科学实验结论。正式训练仍需实际 GPU 预检；实际运行环境记录在每个批次中。
