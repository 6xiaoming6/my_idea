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

原三组 TaxiBJ 输入/通信实验，从头训练：

```bash
python -u scripts/cloud_gpu/run.py --dataset taxibj --methods I01 I02 I03 --missing-rate 0.4 --epochs 100 --gpu 0
```

需要同环境公平对照时，把 `--methods` 改为 `B1 B2 I01 I02 I03`。默认 seed7、batch32、四类混合 mask（random_point/node_outage/temporal_gap/spatial_region）、每5epoch验证、lr1e-3余弦至3e-4。**这里的 0.4 不是纯 random 点缺失。** 默认单卡串行，云端一般用 GPU0，可显式 `--gpu 1`。不自动等待 GPU 空闲，也不停止其他进程；同仓库重复队列/同卡云入口由文件锁阻止。

新增的 **I04/I05 一致性两组**接在原三组之后，用独立冻结批次运行。以下仅提供后续启动命令；不要向旧批次的 `plan.json` 或 `source_snapshot/` 写入新组。原三组正在运行时不重复启动它们，等其训练、评估和报告完成后，再执行：

```bash
python -u scripts/cloud_gpu/run.py --dataset taxibj --methods I04 I05 --missing-rate 0.4 --epochs 100 --gpu 0
```

如果本机没有同环境同预算的B2参考，后续公平验证可显式新建 `--methods B2 I04 I05` 批次；这会从头训练B2，不会假定旧回执属于新计划。两组单独运行时，自动报告会计算I05/I04，并明确缺少同批B2，避免自动混入旧服务器指标。I04/I05均不添加历史池，方法和监督差异见第4节。

后台训练（需安装 tmux）：

```bash
bash scripts/cloud_gpu/start_tmux.sh v25-cloud-taxibj --dataset taxibj --methods I01 I02 I03 --missing-rate 0.4 --epochs 100 --gpu 0
tmux attach -t v25-cloud-taxibj
```

原三组结束后，新增两组也可使用独立的自动退出会话：

```bash
bash scripts/cloud_gpu/start_tmux.sh v25-cloud-consistency --dataset taxibj --methods I04 I05 --missing-rate 0.4 --epochs 100 --gpu 0
tmux attach -t v25-cloud-consistency
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

|方法|真实观测及 H0|轮次尺度/Top-k|额外通信或输出约束|
|---|---|---|---|
|A1|仅 F 可见；M/C掩码归零|CMFF / 2,2,2,2|无|
|A2|同 A1|同 A1|受控差分|
|B1|FMC 可见，三尺度融合 H0|CMFF / 2,2,2,2|无|
|B2|同 B1|同 B1|受控差分|
|I01|FMC 可见，H0 仅 F；轮次再注入 C/M/F/F|CMFF / 2,2,2,2|受控差分|
|I02|FMC 可见，三尺度 H0|每轮硬选一个尺度 / 2,2,2,2|受控差分|
|I03|FMC 可见，三尺度 H0|CMFF / 2,2,2,2|受控差分 + 自适应预测历史池|
|I04|FMC 可见，三尺度 H0|CMFF / 2,2,2,2|受控差分 + 完整M/C目标一致性损失；无历史池|
|I05|FMC 可见，三尺度 H0|CMFF / 2,2,2,2|受控差分 + 可见观测下的自适应剩余区域投影；无历史池|
|S01|同 B1|每轮硬选一个尺度 / 2,2,2,2|无|
|S02|同 B1|CMFF / 4,4,4,4|无|
|S03|同 B1|CMFF / 1,2,4,8|无|
|S04|同 B1|CMFF / 2,2,2,2|全部受控差分；独立从头 B2 复验|
|S05/S06/S07|同 B1|CMFF / 2,2,2,2|仅在第2/3/4轮输入使用差分|
|S08|同 B1|CFFF / 2,2,2,2|无|
|S09|同 B1|CFFF / 2,2,2,2|仅第4轮同尺度差分|

差分经过 RMS 幅度匹配及 ±0.1 条件门后修正专家输入；无差分依旧通过上一轮输出传递隐藏状态。I01保留融合层参数形状，M/C的H0槽位置零，不是取消轮次的真实多尺度输入。I03池内是已发生轮次的预测（0/1/2/3条），可见F回填，历史detach；不包含乘积交互。自由尺度选择没有软预热，使用选中分支的ST梯度；S03的Top-1也使用选中概率ST梯度。不能声称逐轮误差必然下降。

### I04：完整粗尺度目标损失基线

沿用B2骨干，只对最后一轮增加辅助监督。最终F预测回填可见F值，再按预处理时相同的空间2×2、4×4聚合得到M/C预测。分别与预处理数据中的完整 `x_m_gt`、`x_c_gt` 计算**所有位置**L1，求和数据的误差分别除面积4、16，再对两个尺度取平均；均值型数据不额外除面积。辅助损失系数为 `loss.lambda_observation_consistency=0.1`，与原F缺失位置L1、路由均衡项共同训练。

配置为 `model.coe.observation_consistency={"enabled":true,"mode":"full_target_loss"}`，同时显式打开 `data.return_multiscale_targets=true`。完整M/C标签只用于损失，模型前向仍只接收可见F/M/C及mask；不能把完整标签作为测试输入或推理校正信息。此组不增加参数，不强制输出精确满足聚合等式，不给中间轮增加监督。

### I05：可见观测下的自适应剩余区域约束

配置为 `model.coe.observation_consistency={"enabled":true,"mode":"residual_projection","weight_bound":2.0}`；不打开额外目标标签，不增加M/C辅助损失。

- 可见M总量减去其内部可见F，得到未知位置的剩余总量。
- 可见C总量先减已观测M整块总量，再减其余区域的可见F，只约束尚未被M解释的未知区域，避免重复扣除和重复修正。
- 在互不重叠的剩余区域内，对原骨干预测执行带非负条件的加权最小改动投影，满足对应剩余总量。已观测F保持原值；没有粗尺度约束的缺失位置保留原预测。均值型观测先按面积转成相应总量。
- 轻量权重头读取最终隐藏状态、按观测RMS归一化的F预测及F/M/C覆盖mask，输出 `exp(2*tanh(logit))`，由原F任务损失学习校正分配。末层零初始化对应权重全1的固定欧氏投影，**并非零校正，初始输出不等于B2**。新增参数使用独立随机数上下文，公共骨干初始化不变。

I05校正只读取可见观测，推理不使用 `x_f_gt/x_m_gt/x_c_gt`；训练与测试执行同一投影。当前支持精确、对齐、非负且聚合相容的F/M/C观测，不能视作任意带噪或非对齐数据的通用求解器。

**I04与I05不是等监督的纯软约束/硬约束对照。**I04通过全部粗尺度标签增加监督目标，I05仅使用可见观测校正；虽然完整粗尺度标签由既有F真值聚合得到、没有新增原始数据，监督形式和权重仍不同。报告分别比较I04/B2、I05/B2、I05/I04（限同批双方已有完整评估），披露ID/OOD误差、物理聚合残差和成本。聚合残差降低不等于补全MAE或泛化改善；后续仍需可见观测软损失、固定联合投影与等容量输出修正等独立控制。

## 5. 冻结、恢复与结果

首次启动生成不可变计划、源码快照、数据哈希、配置与公共初始化审计；预检用实际batch做前向/反向及重专家检查，不写训练权重。实际训练保存 `best.pth`、`last.pth` 和完整恢复状态，仅用 ID 验证 MAE 选 best，然后自动评估六个 ID/OOD 协议并在 `experments_report/` 输出正式分析报告。报告包括全部组、分族结果、训练曲线、成本及失败/缺失。阶段报告不会冒充最终报告。

终端会打印批次目录；可只准备不训练：

```bash
python scripts/cloud_gpu/run.py --dataset taxibj --methods I01 I02 I03 --epochs 100 --gpu 0 --prepare-only
# 原三组结束后的新增两组，只准备新计划，不启动：
python scripts/cloud_gpu/run.py --dataset taxibj --methods I04 I05 --missing-rate 0.4 --epochs 100 --gpu 0 --prepare-only
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
