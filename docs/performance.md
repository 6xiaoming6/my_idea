# best-perf：保持数值结果的执行优化

候选分支从原best提交`6caef6740bc00013a65d70dd0a98083648e1f88d`建立。原best和正在执行的复跑不改动。已完成本机GPU验证和独立100轮复跑：所有逐轮主指标、best/last模型及完整可恢复状态、六套MAE/RMSE均逐位一致。原best仍保持不变。

## 开关与兼容

默认`configs/i37_taxibj.json`不启用性能路径；模型和优化器参数组、初始化、state_dict键保持。
候选配置为`configs/i37_performance.json`，仅新增`train.performance.enabled=true`和`diagnostic_interval=20`。
可逐项设置布尔字段：`skip_redundant_init`、`reuse_fine_support`、`no_grad_intermediate`、`compact_outputs`、`cache_constants`、`pack_scalar_logs`、`skip_last_history`。启用后未指定字段默认true；无enabled则所有功能关闭。
模型默认前向仍完整返回。训练器在首批、末批和编号0/20/40/60等批次调用完整诊断；其余批次`model(batch,collect_diagnostics=False)`跳过可省诊断。日志增加诊断采样批数、总批数和间隔，主任务损失和完整划分MAE/RMSE/WAPE/计数每batch仍累积。验证和测试保持完整诊断。

## 保持的数值路径

- 初始化仅去掉被F/M/C融合预测覆盖的父类解码；真实P0仍完整建立反向图。
- P1/P2/P3仍执行解码、回填可见F并写入detach历史，只移除这些解码的无用反向图。P4和隐藏状态链、历史评分/注入器梯度不截断；必须通过AMP测试，避免权重缓存破坏Decoder梯度。
- F support仅复用本次forward已有值；不缓存跨样本mask、预测、隐藏状态或学习后位置特征。
- 常量缓存仅含原始坐标和support固定滤波器/分母，按设备、dtype和空间时间形状区分，最多16条；不写入模型state_dict。
- 统计打包先保持原FP32归约，再转为精确表示这些标量的FP64进行一次CPU传输，仍按原batch顺序以Python float累积。没有改成跨batch GPU归约。
- 末轮未读取的P4历史写入不再生成；非诊断batch不构造重复逐轮补全数组。每步日志读取结束后释放输出引用。

## 验收与测量

CPU：原6项发布测试及5项性能测试，包括单因素/组合的零容差前向、梯度、优化器、RNG、三batch包含非诊断batch的实际训练器、连续/恢复对齐与空/全观测。
GPU1等原版复跑结束后独占验证：真实batch32及末批20、F/M/C路径、TA/ST、空/全观测，各至少3次有效AMP更新；逐位比对结果、loss、梯度、裁剪后梯度、模型、优化器、scheduler/scaler及RNG。GPU探针从原始best源码和候选源码分别在隔离进程运行。
每个改动独立核对；不一致的项明确拒绝。组合若不一致，按既定顺序逐项加入，只保留每次严格通过的前缀扩展，不进行模型结构或超参搜索。
性能benchmark固定真实数据batch32、workers0、线程2，以原I37 best权重仅作计时起点；各3次重复，每次预热5批、测量20批，包含真实训练器和CPU诊断开销。报告训练步秒数、allocated/reserved峰值及真实样本数。benchmark权重不用于正式复跑。
预定实际收益门槛：中位步耗时至少降低3%或峰值allocated至少降低5%，同时另一项退步不超过2%；三个重复的主指标、最终模型、优化器及RNG必须一致。无收益不强行采纳。
通过后从未训练初始化用最终配置再独立跑100轮及原六套测试；要求逐轮主指标、best轮次、best/last模型和优化器/scheduler/scaler/RNG一致。耗时、源路径、配置中的性能开关和已标注的诊断采样可以不同。

源码修改如实列入`provenance.json`的performance字段，原完整来源清单保留为`provenance_original_i37.json`；没有把修改文件仍标为“原样未改”。不会因性能研究启用TF32、Flash Attention、fused optimizer、编译重排或改变数据加载进程数。

```bash
python -m unittest discover -s tests -v
# 仅供通过预检后的优化独立run；不要接管原run。
python scripts/train.py --data-dir /path/to/taxibj --run-dir outputs/i37_perf --config configs/i37_performance.json --gpu 1 --threads 2 --evaluate
```

## 已完成的实测验收（2026-10-26）

- 七项优化全部通过逐项及组合GPU检查，无拒绝项；80个GPU场景含完整AMP恢复重放。
- 优化版完整100轮与未优化best复跑完全一致，best仍为epoch95，ID MAE=8.051041961938529。
- 100轮训练/验证累计时间77.412→72.679分钟，减少6.114%；allocated峰值11.146292→9.846848GiB，减少11.658%。
- 短基准中位训练步612.678→577.591ms，减少5.727%；allocated中位峰值9.545609→8.279984GiB，减少13.259%。
- 以上耗时不包含检查点及额外六套评价，不是整进程墙钟时间。
- 小型完整证据见[performance_validation.json](performance_validation.json)。验证的执行源码提交为d920258a；后续文档更正不改变该冻结源码。
