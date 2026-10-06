# R11/R12：R7尺度软启动与双尺度加权融合

## 实验做法

两组从R7配置继承，均从头训练，seed7；不加载R7训练权重。

| 实验 | 训练 | 验证/测试 | 问题 |
| --- | --- | --- | --- |
| R11 | epoch1–5两个尺度按softmax融合；6–10逐渐转为Top-1；11起只执行Top-1 | 全程Top-1 | 先训练两个尺度能否缓解R7早期单路径锁定 |
| R12 | 全程两个尺度按尺度路由softmax融合 | 同训练，全程双尺度 | 保留两个尺度的连续贡献是否比硬选一个更好 |

F为32×32，C为16×16，T保持12（BikeNYC切换时按对应网格缩小2倍）。每轮仍用原生专家Router选择同一组两个专家，粗/细分支复用这个专家选择和组内权重，专家参数跨尺度、跨轮共享。双尺度阶段是所选两位专家各在两个尺度执行一次，即每轮4次专家执行；不是8个专家全执行，也不是分别从各尺度重新路由专家。

设尺度头分数为z，p=softmax(z)，h=one_hot(argmax(p))。粗分支输出上采样后与细分支加权融合，再按direct方式传给下一轮。

- R11前向权重 `w = s*p + (1-s)*h`。epoch1–5：s=1；epoch6–10：s=5/6、4/6、3/6、2/6、1/6；epoch11起s=0。训练中h沿用R7的 `h+p-stop_gradient(p)` 直通梯度。验证/测试始终s=0，不消耗两个尺度预算。
- R12：w=p，训练和评估均执行两个尺度。两个候选尺度的Top-2就是都保留，权重可不相等；并非根据专家logits直接计算尺度分数。
- 不引入温度调度、均衡尺度惩罚、新辅助损失或专家软预热。尺度头零输出层与R7的CCFF微小偏置相同。尺度历史特征记录累计粗尺度融合权重比例；硬阶段等价于粗尺度选择次数比例。

R11的渐变改变贡献权重；在第10epoch及之前仍实际执行两个尺度，第11epoch开始才降为只执行一个。不能把过渡阶段误称为计算量逐步线性减少。

## 共同条件

TaxiBJ random0.4、四种基础缺失训练（随机点/节点/时间/空间）、每epoch重采样；100epoch，batch32，GPU0单卡串行；每5epoch验证；AdamW、lr1e-3余弦到3e-4、weight_decay1e-4、grad_clip1、AMP。L1＋0.01 candidate balance，关闭feedback、partner、acceptance及额外共享分支；四轮共享八专家，原始融合、direct更新。保存best.pth和last.pth，按ID验证MAE选best。保留R系列相同六套固定评估协议和mask种子。

## 指标解释

新增 `scale_weights` 和 `scale_executed`，按样本和缺失类型累计：逐轮fine/coarse融合权重、实际执行比例、双尺度执行比例、权重熵，以及实际专家执行次数/专家网格面积代理。

兼容字段 `selected_scales`、`scale_path_*`、`coarse_fraction` 表示**权重最大的尺度及其路径**；在双尺度阶段不表示只执行该尺度。判断是否真正使用双尺度应同时看权重和execution指标。

双尺度四轮实际专家调用数16，网格面积代理10；硬Top-1调用数8，面积代理范围2–8。网格代理不等于精确FLOPs。R12的收益必须结合额外计算评估，不能称为与R7等计算预算。

## 入口、输出和复现

```bash
python -u scripts/v24/run_r_scale_followup.py --dataset taxibj --gpu 0 --epochs 100 --batch-size 32
python scripts/v24/run_r_scale_followup.py --dataset taxibj --dry-run
python scripts/v24/run_r_scale_followup.py --dataset taxibj --summary-only
# 指定组：--variants R11 / --variants R12；切换数据：--dataset bikenyc
```

配置：`configs/v24/r_scale_followup/R11.json`、`R12.json`。顺序R11→R12，每组训练后完成六套评估。输出队列：`outputs/v24-COE/experiments/r_scale_followup/<dataset>/<fingerprint>/`，保存冻结源码、配置、协议、数据指纹、训练回执、评估、summary和日志。旧R1–R10不重跑、不覆盖。

同一代码和命令再次启动跳过已完成任务；未完成训练仍从头重跑，不把last自动当断点。控制台保持简短任务切换及`train epoch 当前/总数`的batch级tqdm，只显示train loss/mae/rmse。

历史对照位于 `outputs/v24-COE/experiments/r_exploration/taxibj/b21d7928e67f4565/`，R7自由硬尺度、R5早期随机路径、参考N5固定路径。不同方法如有改善，需同时报告实际计算量与单种子局限。
