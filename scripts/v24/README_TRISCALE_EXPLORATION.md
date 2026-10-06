# 三尺度自由路由 T1–T9

## 每组做法与复现

| 实验 | 具体方法 | seed |
| --- | --- | --- |
| T1 | 固定CMFF，direct | 7 |
| T2 | 每轮三尺度自由Top-1，direct | 7 |
| T3 | 固定CMFF，细节保留 | 7 |
| T4 | 自由Top-1，细节保留 | 7 |
| T5 | T4，训练前8epoch每窗口均匀随机选择CMFF的12种排列，第9epoch起自由 | 7 |
| T6 | T1/T3中ID验证MAE最小的固定参考B复验 | 17 |
| T7 | T2/T4/T5选出的自由候选A复验 | 17 |
| T8 | A加最近历史，R9零初始化signed tanh门；Router仍看原状态 | 7 |
| T9 | A改为每层独立专家池，其他设置不变 | 7 |

F/M/C为空间分辨率32/16/8，时间12不变；旧双尺度C=16现在叫M。BikeNYC按输入网格缩小2/4倍，尺寸必须被4整除。四轮八专家T/S/TD/SD/TA/ST/TL/SL，每样本每轮仅所选尺度的两个专家执行。原生专家Top-2及组内softmax不变。自由尺度无配额、无最后一轮必须fine的限制，有81条路径。

尺度头输入为原R7路由特征、此前F/M/C次数除以总轮数4、剩余轮数比例。零输出层和1e-3的CMFF偏置初始化；训练用hard+p-stop_gradient(p)代理梯度。训练探索用独立CPU RNG seed20261001，路径与RNG保存为buffer；不消耗mask、loader及全局随机流。验证/测试不探索。

粗尺度输入采用观测mask加权池化，空格用已有初始估计占位，coverage仍为0。direct用上采样专家输出替换hidden；detail_preserving采用H−U(D(H))+U(E)。F直接返回E，避免减加抵消误差。不称残差为严格正交高频分量。T8的专家输入使用H+g(H_previous−H)，g∈(-1,1)，历史不detach但只在当前forward内存在。细节保留的H仍是原始未修正状态。未使用query/key不实例化。

共同配置：默认TaxiBJ random0.4、GPU0单卡、batch32、100epoch、每5epoch验证；AdamW lr1e-3余弦至3e-4、wd1e-4、clip1、AMP。L1+0.01 candidate balance；无新辅助损失。共享专家（T9除外）、direct主框架、original fusion，关闭feedback、接受门、partner、额外共享分支和C3。训练四基础mask逐epoch重采样，验证固定，沿用R系列六套测试与mask seed。模型/loader seed同时变更，mask seed不变。best仅由ID验证MAE选出，保存best/last完整检查点。

```bash
python -u scripts/v24/run_triscale_exploration.py --dataset taxibj --gpu 0 --epochs 100 --batch-size 32
python scripts/v24/run_triscale_exploration.py --dataset taxibj --dry-run
python scripts/v24/run_triscale_exploration.py --dataset taxibj --summary-only
# --variants T1 T3 指定静态组；请求T6–T9会自动加入T1–T5依赖。
# --deadline 2026-10-01T12:30:00+08:00 仅限制是否启动T8/T9，不杀死运行中的训练。
```

新入口/配置与旧R、N分开；旧输出/冻结源码/检查点保留。相同命令与源码恢复，已完成训练跳过，缺失评估补跑；未完成训练仍从头重跑，不暗中续训。冻结候选记录、动态配置发生变化时拒绝运行。修改参数或源码产生新队列指纹。

## 冻结选择与评估

B是T1/T3中最低ID验证MAE者，平局按编号。A优先从ID验证MAE不超过B的102%的T2/T4/T5中选J最小者；无合格组仍选J最小者并标记不通过门槛。J=0.5×ID验证MAE比值+0.5×原两元/形态/三元OOD测试MAE比值的均值。mask复测不重复计权。平局按编号。旧OOD数据已用于研发，不能把选择后的它们再包装成独立确认。

选择证据及其SHA256写入selection.json，恢复只验证证据，不重选。T6/T7及T8/T9配置继承冻结A/B，从头训练，不载入其权重。T8/T9只作为单种子探索，不替代主候选复验。

T7完成后，A/B seed7、T6/T7 seed17及旧N5/R5分别用20261003/04/05的新mask seed确认两元/形态/三元结果，仍有test+30000偏移。它们是相同测试时间窗口的新mask，不是新数据集。确认指标不进入候选选择。旧N5/R5/R6六套评估复制到本队列comparison_evaluations供强参考比较；旧文件不修改。BikeNYC没有这些固定旧参考时只做A/B确认，不能声称已比较TaxiBJ参考。

## 计算与预算

每窗口四轮共8次实际专家执行。专家网格面积代理=2×Σ(1/factor²)；CMFF为4.625，全F为8，全M为2，全C为0.5。该代理不含路由、重建、投影、记忆开销，不是精确FLOPs。日志按样本/缺失类型记录尺度选择、概率、81路径及实际专家调用量。

优先T1–T7。T8/T9估计采用已完成运行epoch耗时90分位×目标epoch，加实测评估时间和120秒保存余量，再乘1.15；T8另留15%记忆开销。无实测时分别估计90/80分钟。仅当能在截止前完成才启动，否则标记deferred_budget；不缩短epoch或启用另一张GPU。

## 输出与判断

outputs/v24-COE/experiments/triscale_exploration/<dataset>/<fingerprint>/保存plan、协议、source_snapshot、动态resolved_jobs、selection、configs、训练回执、评估、confirmation、budget_decisions、summary、自动analysis.md。训练输出保持日期时间_简短方法_seed目录规则。控制台只有切换及train epoch 当前/总数batch tqdm的train loss/mae/rmse。

自动分析遵循具体方法→实测结果→分析建议；每种子ID用best检查点的ID测试MAE，OOD用三个主要协议MAE比值。两种子均ID和平均OOD改善为优先；ID退步≤2%、平均OOD降低≥5%且各主要OOD退步≤5%可保留权衡。必须另看旧N5/R5/R6的误差与成本以及新mask确认。没有稳定收益就如实报告，R5保持当前泛化候选；不以多样性本身判成功，不宣称两个种子统计显著。
