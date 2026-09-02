# V14 证据校准自适应多尺度路径（ESAP）完整探索方案

## 1. 目标与边界

本轮继续以原始 V14 为唯一基线，探索 **Evidence-calibrated Scale-Adaptive
Pathways（ESAP）**。它不增加专家、不复制编码器、不改变 Top-K、不使用多阶段训练，
只改变现有细/中/粗尺度融合门的概率校准方式。训练仍是一次端到端训练，验证集选择
并覆盖保存一个 `best.pt`，训练结束后只加载该检查点测试一次。

上一轮 CSAS 表明固定 epoch 比例无法同时适配三个数据集：C03 在最佳检查点处的辅助
监督 scale 分别约为 TaxiBJ 0.085、BikeNYC 0.852、CHAP 0。因此本轮不再用数据集固定
时间表，而使用每个输入自身的观测证据控制多尺度路径。

## 2. 文献依据

| 工作 | 可迁移思想 | 本轮采用方式 |
|---|---|---|
| [Pathformer, ICLR 2024](https://openreview.net/forum?id=lJkOCMP2aW) | 不同输入具有不同时间动态，应自适应选择多尺度路径 | 让尺度融合权重由当前样本的可用观测证据校准 |
| [TimeMixer, ICLR 2024](https://openreview.net/forum?id=7oLshfEIC2) | 细尺度和粗尺度承载互补的微观/宏观变化 | 保留 V14 全部尺度与原有融合，仅调整使用强度 |
| [MSGNet, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28991) | 跨变量关系会随尺度变化 | 不用一个固定偏好对待细/中/粗路径 |
| [NAOMI, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/50c1f44e426560f3f2cdcb3e19e39903-Abstract.html) | 多分辨率从粗到细有利于长缺失段重建 | 保留 V14 coarse-to-fine 主体，不删除粗尺度 |
| [SPIN, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cf70320e93c08b39b1b29a348097a376-Abstract-Conference.html) | 稀疏时空观测要求传播机制与插补任务对齐 | 所有校准量只由可见 mask 得出，不读取缺失目标 |
| [Switch Transformer, JMLR 2022](https://www.jmlr.org/beta/papers/v23/21-0998.html) | 稀疏路由需要稳定、简单的校准 | 采用无参数 log-prior、温度和概率下限，不新增路由网络 |
| [LIMoE, 2022](https://arxiv.org/abs/2206.02770) | 路由塌缩和利用不平衡需要显式稳定机制 | 两个候选加入 active-scale uniform floor |

这些论文支持设计原则，不表示其原模型被直接复制到本项目。

## 3. 方法

对每个样本和尺度 \(s\in\{f,m,c\}\)，定义：

- 可用度 \(a_s\)：该尺度非空观测单元比例；
- 聚合纯度 \(p_s\)：一个非空聚合单元平均保留的原始细网格观测比例；
- 混合证据 \(e_s=(a_s+p_s)/2\)。

当前实现可由 fine/mid/coarse mask 的统计量精确得到：

\[
a=(q_f,q_m,q_c),\qquad
p=(1,q_f/q_m,q_f/q_c).
\]

这里没有使用 `x_f_gt`，因此训练、验证和测试都不存在目标泄漏。原尺度门 logits
记为 \(z_s\)，ESAP 使用：

\[
\tilde z_s=z_s/\tau+\gamma\log(\max(e_s,\epsilon)),\qquad
w=\operatorname{softmax}(\tilde z).
\]

可选防塌缩下限为：

\[
w'=(1-\rho)w+\rho u,
\]

其中 \(u\) 是 active scales 上的均匀分布。`legacy + gamma=0 + tau=1 + rho=0`
与原始 V14 逐元素一致。

## 4. 18 个预注册候选

| 候选 | 证据 | gamma | tau | uniform floor | 目的 |
|---|---|---:|---:|---:|---|
| E01–E04 | availability | 0.25/0.5/1/2 | 1 | 0 | 检验优先使用可用聚合单元是否有效 |
| E05–E08 | purity | 0.25/0.5/1/2 | 1 | 0 | 检验优先保留细粒度真实证据是否有效 |
| E09–E12 | hybrid | 0.25/0.5/1/2 | 1 | 0 | 检验覆盖度与纯度的折中 |
| E13 | legacy | 0 | 0.75 | 0 | 单独检验更尖锐的学习门 |
| E14 | legacy | 0 | 1.25 | 0 | 单独检验更平滑的学习门 |
| E15–E16 | hybrid | 0.5 | 0.75/1.25 | 0 | 检验中等证据先验与温度交互 |
| E17–E18 | hybrid | 0.5 | 1 | 0.05/0.10 | 检验防止尺度路径塌缩 |

E15–E18 的组合值在运行前已固定，不能根据 E01–E14 测试集临时修改。

## 5. 长程预筛任务与耗时

每个候选完整运行以下 Core-6：

1. TaxiBJ fixed@0.4
2. TaxiBJ random@0.4
3. BikeNYC fixed@0.6
4. BikeNYC random@0.8
5. CHAP fixed@0.2
6. CHAP random@0.4

总量为 `18 × 6 = 108` 个完整训练任务。根据本机最近 CSAS 实测中位耗时：

- TaxiBJ：1.364 小时/任务；
- BikeNYC：0.099 小时/任务；
- CHAP：0.847 小时/任务。

估算总时长为：

\[
18\times(2\times1.364+2\times0.099+2\times0.847)
=83.16\text{ 小时}=3.47\text{ 天}.
\]

这是单 GPU、不中断、不包含异常重跑的估计。脚本逐任务串行执行，并设置每任务 4 小时
超时；已完成任务可自动跳过，服务器重启后执行同一命令即可续跑。

## 6. 执行顺序

先做代码管线 smoke（不是正式结果）：

```bash
conda activate difftdi
python scripts/v14-exploration/run_esap_exploration.py \
  --gpu 0 \
  --phase screen \
  --candidates E01 \
  --epochs 1
```

确认 smoke 后执行约 3.47 天的正式预筛：

```bash
python scripts/v14-exploration/run_esap_exploration.py \
  --gpu 0 \
  --phase screen
```

不要添加 `--epochs`，否则不属于完整实验。默认使用 TaxiBJ 160、BikeNYC 140、
CHAP 150 epoch，以及各数据集原有 `val_epoch`。

## 7. 冻结的筛选规则

候选选择只读取最佳检查点对应的验证集指标。一个候选必须同时满足：

1. Core-6 中至少 4/6 个点的验证 MAE 相对 V14 改善不少于 0.5%；
2. 六点验证 MAE 宏平均改善不少于 1.0%；
3. 六点验证 RMSE 宏平均退化不超过 0.5%；
4. 任一点验证 MAE 退化不超过 3%，RMSE 退化不超过 5%；
5. 每个数据集的两个点平均 MAE 退化不超过 0.5%；
6. 指标全部有限，检查点、配置和日志完整。

若多个候选合格，先按验证 MAE 宏平均排序，再用验证 RMSE 排序，只允许一个进入三种子。
所有候选均不合格则关闭 ESAP，不从测试结果挑单点，不追加 gamma/tau/floor 插值。

## 8. 晋级后（现在不要运行）

冻结唯一候选后才允许：

```bash
python scripts/v14-exploration/run_esap_exploration.py \
  --gpu 0 --phase multiseed --candidates E10 \
  --seeds 42 2026 3407

python scripts/v14-exploration/run_esap_exploration.py \
  --gpu 0 --phase all24 --candidates E10
```

命令中的 `E10` 只是格式示例，不能在预筛完成前视为已选候选。

