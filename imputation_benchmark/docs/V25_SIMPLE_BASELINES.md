# v25 简单传统与循环网络基线

当前入口只做数据适配，不增加原方法之外的模块、投影或损失。

本批对齐 I25：TaxiBJ 两通道、T=12、混合四族缺失0.4、seed7、原数据切分与六套评估。
用户明确暂不运行 Transformer、SAITS 和扩散模型。

## 方法

1. ChannelMean：训练可见位置按通道均值。
2. NodeMean：训练可见位置按通道/网格均值，不假设窗口对应固定日历时段。
3. TemporalLinear：窗口内时间插值，边界最近值，整段缺失回退 NodeMean。
4. SpatialMean：可见空间邻域均值，窗口3/5/9按ID验证选择，空邻域回退NodeMean。
5. SoftImpute：窗口内12×2048残差矩阵；训练NodeMean中心化；相对最大观测残差奇异值阈值0.01/0.05/0.1/0.2，仅ID验证选择；50次上限、容差1e-4。
6. RITS_patch8：复用 SAITS 仓库提供的 RITS 实现。
7. BRITS_patch8：复用同仓库的 BRITS 实现。

循环模型源码不变，**不是运行SAITS模型**。旧BRITS目录存在固定时间长度/特征数入口，因此选择已有可配置实现。仅适配数据：32×32网格分为16个8×8块，每块保留12步和两通道；输入12×128、hidden32。无跨块通信，不能称为全网格BRITS论文最优结果。

神经基线100epoch、batch32完整窗口、FP32、AdamW、1e-3余弦到3e-4、weight_decay1e-4、clip1，每5轮按ID验证原始预测MAE选best。关闭上游可选MIT，仅保留该实现的原生观测重建项和BRITS原生双向一致性项，不额外监督缺失位置。训练前向和损失只读取F可见观测；完整标签只用于验证/测试指标。统计量仅用训练epoch1可见F；mask每epoch按I25规则重采样。

所有方法只评估原始预测，不添加可见M/C投影。M/C字段只随公共数据协议保留及用于掩码指纹核对，不参与基线前向、训练损失或输出后处理。

## 旧批次纠正（2026-10-15）

`20261015_161454` 是历史批次：曾额外评估 `+P`，且两个循环网络曾使用额外缺失位置L1。原模型源码没有修改，但“仅做数据适配”的旧描述不准确。五种传统法的 `raw` 结果保留为数据适配参考；旧RITS/BRITS（包括raw）不再列入严格原生基线主表，去掉推理投影不能消除训练损失的影响。所有 `+P` 仅作为历史记录，不作正式baseline。

冻结源码、检查点、原始JSON不修改。当前入口使用新协议 `f_only_native_loss_no_projection_v2`，拒绝恢复旧协议的套件、结果和检查点。循环网络需在新套件从头训练后才能得到修正结果；本次纠正未启动训练。

## 执行与恢复

入口 `imputation_benchmark/scripts/train/run_v25_simple_baselines.py`，先 `--prepare --root <project> --suite <new-suite>` 冻结当前脚本、I25源码/数据协议、上游实现、测试及参考结果。

通过CPU单元测试后，在 GPU1 `--preflight` 检查原尺寸batch32前向/反向/优化器恢复，峰值限制4GiB。预检不计入训练。

运行冻结入口必须 `CUDA_VISIBLE_DEVICES=1`，并设置 `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLCONFIGDIR=/tmp/v25-simple-mpl`。

队列按上述顺序运行；GPU1及本套件双锁避免并发；完成方法跳过；RNN每轮完整保存last（模型、优化器、调度器、RNG、loader状态）；失败保留日志并退出、不自动重试。传统方法不需要优化器检查点。

专用tmux将队列命令作为主进程，完整完成训练/六套评估/正式报告后自动关闭。GPU训练每15秒输出logs/gpu_status.jsonl，传统方法在CPU运行。

## 产物与边界

套件manifest记录数据/源码哈希、配置与方法；statistics保存拟合统计量；runs保留训练/验证记录与best/last；results保留六协议MAE/RMSE、各缺失族、缺失计数和mask指纹；正式报告写入experments_report，并在未完成时明确标为阶段报告。

不复用旧单通道/不同mask结果。使用I25/I27现有best评估结果作只读参考，不重训。单seed探索；mask复测不算训练种子复验；F-only与FMC信息量不同，不能把全部差距归因于模型结构。
