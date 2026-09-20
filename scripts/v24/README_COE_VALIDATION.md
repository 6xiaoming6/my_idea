# v24 主方案及机制验证

当前主方案固定为 `configs/v24/coe_main_base.json`：单尺度四层链、六专家 T/S/TD/SD/TA/ST、每层独立 router、专家池跨层共享、router 和专家读取当前状态、共享点级专家开启。训练 epoch 1–3 软融合，4–5 过渡，第6起完全硬路由；初始均匀混合0.5、采样温度2；balance=0.01，无输入噪声，不固定第二层。验证/测试确定性 argmax。

这是现阶段主候选，不代表已经证明所有机制有效或最终收敛。

## 顺序与假设

|组|相对主方案的改变|检验内容|
|---|---|---|
|M0 coe_main|无，重新运行同期参照|主方案精度及路由分工|
|M1 coe_initial_router|router_state=initial|各层读取更新后的状态是否有帮助；仍有四个独立router，并非重复一个router|
|M2 coe_initial_expert|expert_state=initial|专家输入逐层更新是否有帮助；残差仍累加，router仍读取当前状态|
|M3 coe_fixed_chain|固定TA→ST→S→TA；关闭无意义的预热和均衡项|同样四层链，动态选择方案是否比预指定路径更好|
|M4 coe_no_balance|balance=0|软预热下均衡辅助损失的额外作用|
|M5 coe_original_mask|恢复原始 random_point mask，训练/验证/测试使用对应 CSV|九类混合 mask 相对原始缺失构造是否改变主方案效果|

M1/M2保持深度、专家集合、预热、损失、共享支路一致，比单纯增加深度更能检验链式计算机制。M3是一条预指定固定链，不是最优固定路径；其训练方式与动态方案不同，比较的是整个动态选择方案。M2保留残差累加，不等同于删除所有链式结构。M5只改变 mask 来源，应单独与 M0 比较，不能把两种 mask 分布混为严格配对。 本轮不做深度×专家池网格，不添加多尺度，也不涉及baseline适配。

## 统一协议

TaxiBJ清洁数据，seed7，batch16，20 epoch，cosine周期20，每2 epoch验证，无早停。M0–M4的train/val/test使用九类缺失混合，缺失率0.4；M5恢复原始 random_point CSV，缺失率仍为0.4。按验证MAE在CPU内存保留最佳模型，结束后测试一次，不落盘best.pt。旧结果保留，新结果单独输出到 `outputs/v24-COE/experiments/coe_validation/`。

本阶段直接采用完整主配置，绕过旧 `experiments/full.json` 的两专家默认覆盖。测试检查最终生成配置，要求除声明的干预外model/loss/data/train均相同，并检查每组软阶段、硬阶段前向反向及评估。

## 运行

```bash
conda activate difftdi
python scripts/v24/run_coe_validation.py --dry-run
tmux new-session -s v24-coe-validation 'python -u scripts/v24/run_coe_validation.py --gpu 0'
```

在项目根目录运行。入口保留GPU占用检查，单卡顺序执行。五个动态组按此前约32分钟/组，固定链约10分钟，总计暂估2.8小时，建议预留3小时。没有自动启动训练。

## 分析要求

以M0为统一参照，汇总MAE/RMSE、每层选择率、路径分布、按缺失族路由和末5 epoch趋势。M1/M2若变差，支持状态更新机制，但不能只用更多路径证明精度收益。M3若接近M0，则动态路由的额外收益证据不足。M4用于决定是否继续保留均衡项；不要以均匀使用专家本身为成功标准。

20 epoch单seed是机制筛选。下一步先为胜出方案补按缺失族误差及更完整训练验证；当前日志中的按族选择率不能代替按族MAE。不同配置导致随机数消耗不同，即使seed相同也不是完全相同随机轨迹。
