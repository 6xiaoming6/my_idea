# 精简日志与路由诊断

通用入口 `scripts/train.py` 和新队列使用的 `scripts/v24/train_four_direction.py` 共用日志精简策略。控制台保持 batch 级 `train epoch 当前/总数` tqdm，只显示 train loss/mae/rmse。

|文件|内容与频率|
|---|---|
|`logs/train.log`|每epoch的loss、MAE、RMSE、学习率、耗时及best标记；不写完整路由字典|
|`logs/val.log`|实际验证epoch的主要指标|
|`logs/metrics.jsonl`|每epoch的机器可读指标、成本、门控/梯度等标量摘要；没有逐路径明细或分条件路由大表|
|`diagnostics/epoch_00005.json.gz`|独立压缩诊断；默认第1轮、每5轮、配置的最后一轮保存|
|最终评估|保留完整路径分布；通用训练写`diagnostics/test.json.gz`，v24队列写`logs/test.json`，多协议评估继续沿用原评估JSON格式|

诊断每种路径分布（含分缺失类型）只记录概率最高的10条正概率路径，并保留所有路径的唯一数、熵、最大占比和未列出路径的总占比`other_fraction`。Top-10是截断展示，不重新归一化；不能把10条路径当成模型只有10条路径。标量日志保留候选筛选所需的验证MAE、面积代理等字段，现有报告训练曲线和筛选入口继续可用。

可以在训练配置中调整：

```json
{
  "train": {
    "logging": {
      "diagnostic_every": 5,
      "path_top_k": 10
    }
  }
}
```

`diagnostic_every`、`path_top_k`必须为正整数。此配置只控制记录，不改变mask、模型前向、损失或优化；路由统计目前仍在内存中正常计算，主要节省写盘、文件体积和检查点累计历史。

断点恢复以`last.pth`的epoch为准，重建精简train/val/metrics日志并移除尚未提交epoch的诊断。新检查点只累计精简历史；模型、优化器、调度器、AMP和随机状态照常完整保存。旧检查点含完整历史时可读取并转换。若新检查点对应的诊断文件被外部删除，只能恢复标量摘要，文件会明确标记`detail_source=checkpoint_summary_only`，不会假装重建完整明细。

需要合并低频诊断读取时：

```python
from stmoe_imputer.utils.metric_logging import read_epoch_history
rows = list(read_epoch_history("运行目录/logs/metrics.jsonl", include_diagnostics=True))
```

该读取器也兼容旧的完整JSONL。最终测试完整分布保留，不受Top-10限制。

已经结束的实验、旧日志及冻结源码快照不会自动改写；从旧快照恢复会继续使用旧日志格式。新启动并冻结的队列使用本策略。输出目录Git白名单仍仅包含普通`.log`，压缩诊断、JSON与检查点留在本地。
