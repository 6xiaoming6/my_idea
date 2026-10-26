# 当前实验基线：I37

确认日期：2026-10-26。I42–I75已完成，本轮I64–I75没有同时改善ID与OOD的配置，继续固定原I37，不将I65的ID收益包装为综合优化。

## 模型及训练定义

三份真实F/M/C可见观测分别编码、对齐后形成H0；四轮共享T/S/TD/SD/TA/ST/TL/SL八种专家，每轮只执行原生Top-2。前三轮自由选择一个空间尺度，第四轮固定F。不是固定CMFF；seed7的MMFF是路由学到的路径。

轮间保留原RMS幅度匹配及±0.1门控差分、当前样本内因果历史预测池。最终H4经Decoder形成原始F预测，再以可见F/M/C的嵌套区域约束执行可学习非负分配投影。可见F原值保留。没有观测约束的缺失区域保留骨干预测，不保证每轮误差下降。

TaxiBJ，双通道、12步、F/M/C空间32/16/8；四类混合mask0.4，seed7，100epoch，batch32，每5轮ID验证MAE选best；AdamW1e-3余弦至3e-4，wd1e-4、clip1、AMP。参数347,930。

损失固定为投影前/后缺失F各0.5，可见rawF0.1，仅可见M/C一致性0.1，候选均衡0.01；不含I25的可学习前后损失标量。只读可见观测，不向前向传入隐藏真值。

## 已有参考结果

|指标|MAE|
|---|---:|
|ID|8.051042|
|二元组合（两次mask平均）|8.297262|
|未见形态（两次mask平均）|10.833551|
|三元组合|7.892074|
|三类别OOD均值|9.007629|

ID RMSE为14.660256。以上是固定原测试协议、seed7的结果，不代表跨种子稳定性或全局最优。新mask复测不等于新训练种子。

原配置：[i37_baseline.json](../../configs/v25/i37_baseline.json)；原作业、完成回执与原六套主要指标：[档案](../../experiment_records/v25/i37/)。这些配置保留原服务器路径供溯源。

## 在新机器复现

使用已经发布的独立best分支，它包含不依赖历史outputs的训练、数据准备、恢复及评价入口；数据和权重另外传输。

```bash
git clone --branch best --single-branch git@github.com:6xiaoming6/my_idea.git i37
cd i37
pip install -e .
python -u scripts/train.py --data-dir /path/to/taxibj --run-dir outputs/i37_seed7 --gpu 1 --evaluate
```

数据目录需有taxibj_train/val/test.npz，F/M/C完整数组来自同一分割、按既定空间sum聚合构造，模型输入时各尺度独立构建mask。不要把非均匀窗口强行拼成连续时间序列。实际依赖与环境要求以best分支README为准。

本v25分支保留完整研究实现及历史队列；不会因本次发布启动训练或恢复已暂停baseline。
