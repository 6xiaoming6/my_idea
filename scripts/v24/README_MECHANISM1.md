# v24 mechanism1：状态反馈、固定链与软路由对照

本轮按 `model_designs/v24_20260920_COE_可行性评估与下一轮实验方案.md` 执行八组对照，单卡按表中顺序串行运行。所有组统一使用 4 层、6 个专家、TaxiBJ 清洁数据、九类混合 mask、缺失率 0.4、seed 7、batch 16 和 60 epoch。

|组|设置|目的|
|---|---|---|
|A0 `coe_mech_main`|四层六专家、软预热、当前状态 router/专家、balance=0.01|主方案参照；保存 best.pt 供后续路径干预|
|A1 `coe_mech_initial_router`|仅 `router_state=initial`，专家仍读更新状态|验证逐轮更新后的状态是否必须用于下一层决策|
|F1 `coe_mech_fixed_s_ta_sd_st`|固定 S→TA→SD→ST，关闭软预热和 balance|强固定链基线，控制四层和专家池|
|F2 `coe_mech_fixed_ta_sd_st_ta`|固定 TA→SD→ST→TA，关闭软预热和 balance|改变含 SD 链的专家顺序，检验顺序和传播范围|
|A2 `coe_mech_conditional_soft`|逐样本 conditional Soft，关闭预热，balance=0.01|比较硬路由和连续专家混合|
|A3 `coe_mech_global_soft`|每层一个全局可学习 Soft 权重，balance=0.01|检验样本无关的全局混合是否已经足够|
|A4 `coe_mech_no_balance`|主方案设置但 balance=0|判断均衡辅助项是否是主要收益来源|
|A5 `coe_mech_initial_expert`|仅 `expert_state=initial`，router 仍读更新状态|验证专家状态反馈是否影响后续路由|

训练每 epoch 重采样，val/test 使用固定的独立混合 mask；每 2 epoch 验证，cosine scheduler 总周期为 60，八组均保存 `best.pt`。固定链关闭软预热和均衡项，避免没有可学习 router 时保留无意义的路由损失。A2 是逐样本连续混合，A3 是每层共享的全局可学习混合权重。

F1/F2 是结构候选，不是验证集已经选出的最优路径；正式论文比较时只能用验证结果选择固定路径，不能用测试结果挑选。它们加入 SD 是为排除“主方案收益只是空间感受野更大”的解释。

## 运行

```bash
cd /home/students/HuangMingYu/code/py/my_idea/my_idea
conda activate difftdi
python scripts/v24/run_mechanism1.py --dry-run
tmux new-session -s v24-mechanism1 'python -u scripts/v24/run_mechanism1.py --gpu 0'
```

单卡顺序运行，入口会拒绝已有 GPU 计算进程。根据此前 20 epoch 的实测速度，60 epoch 的八组预计约 9–10 小时，建议预留 11 小时；实际时间会随 GPU 负载和验证耗时变化。脚本只做队列启动，不会自动后台启动训练。

## 判定规则

- A0 相对 F1/F2：比较 MAE、RMSE、按 mask family 的误差、路径分布和训练成本。
- A0 相对 A1：若 A0 更好，支持状态反馈 router；若接近，则把主张收窄为初始条件化自由选择。
- F1/F2 若追平 A0，需要补更强固定链或停止把动态选择作为主要收益来源。
- 只看路径数不作成功标准；同一协议内先看验证 MAE，再只测试一次。
