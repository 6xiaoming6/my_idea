# V20-single 实现与验收报告

日期：2026-09-02  
基线：V14-single（提交 `1f348d4`）  
模型：GMSV-MoE（Geometry-Matched Self-Validated Mixture-of-Experts）

## 1. 核心创新的实现结论

V20 没有改动 V14 的专家结构、共享/路由双分支、渐进式融合、可靠性尺度门、Safe C2F、Correction Adapter 或 Safety Controller。新增内容只改变专家选择依据：

1. GMP（Missing-Geometry-Matched Probe）根据真实缺失区域的局部空间可用率、长程空间可用率、时间可用率和尺度可靠性，从已观测位置中确定性地选择几何条件最相似的 Probe。
2. Expert On-Site Exam 先从输入和掩码中删除 Probe，再重新经过尺度编码器及全部四个专家，避免专家提前看到考试答案。
3. 共享、跨尺度的零初始化 Probe Decoder 将专家隐特征映射回数据空间，以 Probe MAE 测量当前样本、当前尺度上的专家能力。
4. SVER（Self-Validated Expert Routing）采用置信度自适应的概率几何融合：

   ```text
   p_comp = softmax(-normalized_probe_error / tau)
   eta = eta_max * clamp(1 - min(normalized_probe_error), 0, 1)
   p_final = softmax((1-eta) log(p_prior) + eta log(p_comp))
   ```

5. Probe 特征及路由证据均 detach。Probe loss 只训练 Probe Decoder；主任务仍正常训练 V14 Router 和 Experts，杜绝专家通过主任务梯度操纵考试成绩。
6. Probe Decoder 最后一层零初始化，使初始 `competence=uniform`、`eta=0`、`final_gate=prior_gate`，因此无需预训练或多阶段训练，初始化时严格退化到 V14。
7. 正式 forward 不接收或读取隐藏位置真值；隐藏真值只允许由独立的测试后离线分析脚本使用。

## 2. 代码变更

核心新增：

- `src/stmoe_imputer/models/v_single/v20_probe_mask.py`
- `src/stmoe_imputer/models/v_single/v20_probe_routing.py`
- `src/stmoe_imputer/models/v_single/v20_probe_validated_c2f_moe.py`
- `configs/v20-single/`：三数据集正式配置、smoke 配置及五组正式消融配置
- `scripts/v20-single/`：单点训练、三数据集 smoke、八点筛选、三种子、全量 24 点、消融、Probe 排序离线分析和结果汇总脚本
- `tests/test_v20_*.py` 与 `tests/_v20_utils.py`

兼容修改：

- `experts.py` 将原专家计算安全拆为 `forward_all()` 与 `mix_from_outputs()`。
- `main_branch.py` 增加可选 `routing_evidence` 及 prior/competence 几何融合。
- V14 wrapper 只增加可选证据透传；不传证据时保持原行为。
- `losses.py` 增加 `lambda_v20_probe`。
- `engine.py` 增加 V20 参数组及 Probe、能力、置信度、路由变化诊断。
- 训练入口对版本配置路径做通用化，V14 原调用方式和输出命名保持不变。

## 3. 配置与训练协议

三套配置完整继承对应 V14 配置，仅增加 V20 模块、Probe loss 和 V20 独立参数组；统一使用相同的 GMP、能力计算和 SVER 公式，不做数据集特制 Router。

| 数据集 | Epoch | `val_epoch` | Batch size | V20 LR | Probe ratio | `lambda_v20_probe` |
|---|---:|---:|---:|---:|---:|---:|
| TaxiBJ | 160 | 5 | 32 | 0.001 | 0.08 | 0.05 |
| BikeNYC | 140 | 2 | 16 | 0.001 | 0.08 | 0.05 |
| CHAP | 150 | 5 | 32 | 0.001 | 0.08 | 0.05 |

训练仍为一次端到端优化：定期验证，仅覆盖保存验证 MAE 最佳的 `best.pt`，训练结束加载该 checkpoint 完成一次测试。

## 4. 自动化测试结果

- V20 单元测试：16/16 通过。
- V14 原回归测试：19/19 通过。
- Expert Pool 的 dense、topk、soft_topk 重构前后最大绝对误差满足 `<=1e-7`。
- V20 零初始化与同权重 V14 的 prediction、gates、Top-K indices/weights、selected masks 和 V14 correction gates 满足 `<=1e-6`。
- 已覆盖 Probe 只能来自 observed、考试输入删除答案、隐藏真值无泄漏、无目标几何/候选不足回退、inactive scale、Probe 梯度隔离、主损失梯度连通、checkpoint round-trip 和三数据集动态 shape。

## 5. 三数据集真实 smoke

每个数据集都使用真实数据完成 2 个 epoch，并完整执行训练、验证、最佳 checkpoint 覆盖与测试。该结果只证明工程链路和数值稳定性，不用于判断正式模型精度。

| 数据集/点位 | 最佳轮 | Test MAE | Test RMSE | Test WAPE | Probe loss（第1→2轮） | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| TaxiBJ fixed@0.4 | 2 | 37.1201 | 62.6397 | 0.3737 | 109.2242→77.7757 | 11.736 GB |
| BikeNYC random@0.4 | 2 | 5.9158 | 18.6652 | 0.5734 | 7.4166→6.9027 | 1.827 GB |
| CHAP fixed@0.4 | 2 | 4.5831 | 6.6672 | 0.1559 | 28.5512→20.9026 | 7.195 GB |

三个点位均未出现异常或 NaN。Probe loss 均下降；测试时有效尺度的 `eta` 已由初始化的 0 变为非零，证明 Decoder 已开始形成可区分能力证据并实际参与路由。TaxiBJ 的 coarse 尺度按 V14 的 active-scale 规则不参与，符合设计。

## 6. 参数量与资源增量

| 数据集 | V14 参数量 | V20 参数量 | 增量 | 前2轮平均耗时 V14→V20 | 耗时增量 | 显存增量 |
|---|---:|---:|---:|---:|---:|---:|
| TaxiBJ | 4,947,168 | 5,002,562 | +1.120% | 30.269s→36.210s | +19.63% | +4.32% |
| BikeNYC | 4,885,344 | 4,940,738 | +1.134% | 2.705s→3.330s | +23.11% | +3.57% |
| CHAP | 4,940,441 | 4,995,802 | +1.121% | 20.179s→24.635s | +22.09% | +4.17% |

实测低于设计目标：训练时间不超过 V14 的 1.50 倍，峰值显存不超过 1.25 倍，因此第一版不需要引入 Prior Top-M 等额外近似。

## 7. Probe 排序离线链路验收

`analyze_probe_ranking.py` 已在 BikeNYC smoke best checkpoint 的一个测试 batch 上成功生成：

- `analysis/probe_oracle_summary.json`
- `analysis/probe_oracle_per_sample.csv`
- `analysis/probe_oracle_per_scale.csv`

链路验收样本的整体 Probe↔Oracle Spearman 为 0.4800、Top-1 agreement 为 0.4667、Top-2 overlap 为 0.7556。由于这里只分析一个 smoke batch，不能作为论文结论；正式结论必须在完整训练后的完整测试集上，对 Geometry Probe 与 Random Probe 分别统计。

## 8. 当前结论与下一步

V20 的代码实现、V14 数值兼容、无泄漏约束、梯度隔离、三数据集真实训练链路及离线科学诊断链路均已通过。现阶段能确认“实现正确且可完整训练”，尚不能仅凭 2-epoch smoke 宣称精度优于 V14。

正式实验顺序应为：八点筛选 → Probe 排序离线分析及 Random/Geometry 对照 → 核心四点三随机种子 → 通过准入后再跑全量 24 点。重点判断三个问题：Probe 排名是否预测真实缺失能力、几何匹配是否优于随机 Probe、能力证据是否最终改善 V14 的 MAE/RMSE。
