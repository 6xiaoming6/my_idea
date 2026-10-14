# BikeNYC / CHAP 四因子格复验（共八组）

2026-10-13 用户确认：两数据集各 A1/A2/B1/B2，沿用四类混合 mask 0.4；不是纯随机点缺失。GPU1 单卡串行。保留旧 TaxiBJ 九组队列的中断状态，不同时恢复。

|每个数据集的组|输入观测|差分|对应问题|
|---|---|---|---|
|A1|F，M/C 值和掩码置零|无|F-only 参考|
|A2|F，M/C 值和掩码置零|RMS 幅度匹配及 ±0.1 条件门|相对 A1 的差分效应|
|B1|F/M/C 独立稀疏观测|无|相对 A1 的额外尺度观测效应|
|B2|F/M/C 独立稀疏观测|RMS 幅度匹配及 ±0.1 条件门|相对 B1 的差分效应；与 A2/A1 是否不同|

## 固定配置与数据

- 四轮 CMFF，八种共享专家 T/S/TD/SD/TA/ST/TL/SL，每轮只执行原生 Top-2；无 FFN、历史池、乘积交互、自由尺度、第二视图或一致性。
- 同数据集四组完整初始参数相同；无差分组门头不训练。不同通道数的数据集分别初始化，不伪造跨数据集参数一致。
- seed7、100 epoch、batch32、val 每5轮、AdamW 1e-3 余弦至3e-4、weight_decay1e-4、clip1、AMP、strict replay、ID val MAE 选 best。保存 best/last 完整检查点。
- 四训练族 random_point/node_outage/temporal_gap/spatial_region，rate=.4，训练每轮重采样；验证固定。复用 TaxiBJ 的六套测试协议与种子。测试 mask 复测不是训练种子重复。
- BikeNYC F=[B,2,12,24,12]，M=[B,2,12,12,6]，C=[B,2,12,6,3]。区域求和，编码时除以区域面积（4/16）。原始 flow_data.npy 为21×12，窗口底部额外3行零填充；本批沿用原mask/评分范围，绝对指标包含填充区域，四组一致。当前保留12步以隔离机制；12步不是已经验证的最佳时间长度，不能将本地24×12数据的采样间隔直接当作文献16×8版本的1小时。
- CHAP_Beijing F=[B,1,7,32,32]，M=[B,1,7,16,16]，C=[B,1,7,8,8]。PM2.5 区域均值，编码时不再除面积。输入最终数值都对应细尺度单位，诊断残差保留各自物理单位。
- 原始文件只读。新副本按 split 顺序剔除含之前保留 split 重复帧的窗口；BikeNYC 原始两边界各6帧重叠，CHAP无跨split重复帧。保留索引/文件SHA/形状/嵌套聚合校验写入manifest。该哈希检查不证明原始时间连续性；CHAP旧预处理有全序列缺失插值代码，当前缺乏当时缺失计数记录，不能据有限值断言插值步骤未执行。
- 完整 F 先聚合，再各尺度独立采样观测；模型只读各自可见值和掩码。F 的隐藏目标可以由独立 M/C 观测提供区域约束，这是额外信息，不是等观测预算比较。

## 启动与复现

```bash
python scripts/v25/run_cross_dataset_validation.py --prepare-only
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u scripts/v25/run_cross_dataset_validation.py --suite <冻结目录>
```

先通过 CPU 测试，再冻结源码/配置/数据/TaxiBJ四组参考回执与评估；每组 GPU preflight 包括真实 batch32 native 与强制 TA/ST 两条路径、3次成功 AMP 更新。训练监控 logs/gpu_status.jsonl 每15秒写两卡及进程/主机摘要；额外 thread_stacks.log 的定时 Timeout 是诊断触发，不等于训练异常。专用 tmux 主进程运行完整队列，训练/六套评估/正式报告完成后退出；中断从 last 全状态恢复，已完成跳过，失败停止。

## 预先规定分析

逐数据集逐协议报告 A2/A1、B1/A1、B2/B1、B2/A2 的 MAE/RMSE，相对改变量及差中之差；与冻结 TaxiBJ 参考比较方向，不跨数据集平均原始 MAE。重点检验 B1 的泛化收益，以及 B2 是否 ID 改善而 OOD 退步。差异接近零仅为弱信号，单 seed 不建立稳定结论。不因结果不符合 TaxiBJ 而省略或调整测试。

各组完成生成阶段报告，全8组完成且证据审计通过后正式报告；包含所有缺失族、曲线、门和逐轮诊断、参数/时间/显存及负结果。历史 TaxiBJ B2 有中断与 GPU 迁移，独立复验尚未完成。若新数据集结论不同，应保留这种异质性，不按有利结果选择论文数据集。是否把后续研究重心转向 BikeNYC/CHAP，应同时考虑任务代表性、稳定性和成本。

## BikeNYC 200 epoch 独立复验（2026-10-13 用户新增）

CHAP 已暂停，A1 last=28、best=20，后续组未启动；旧套件 `65564fa00949e268/pause_receipt.json` 保存暂停证据。不自动恢复 CHAP。

新增 BikeNYC A1/A2/B1/B2 四组，从头训练 200 epoch，余弦周期也设为 200，其余数据、初始化、seed7、batch32、12时间步、混合mask0.4及六套评估均保持一致。冻结旧100轮各组回执、评估、配置、曲线及初始化哈希；逐组检查200轮与100轮初始权重相同。200轮同时改变预算与余弦轨迹，不冒充原100轮之后的简单续训。旧100轮全部结果保留。

```bash
python scripts/v25/run_cross_dataset_validation.py --datasets bikenyc --epochs 200 --prepare-only
CUDA_VISIBLE_DEVICES=1 python -u scripts/v25/run_cross_dataset_validation.py --suite <新冻结目录>
```

新建专用 `v25-bikenyc-200` tmux，GPU1串行，100/200轮六套指标配对写入正式报告。已完成跳过、未完成从自身last恢复，不加载旧100轮的最终权重。
