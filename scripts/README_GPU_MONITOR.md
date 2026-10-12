# 训练 GPU 状态日志

`scripts/train.py`（包括复用它的 V14/V25 入口）和 `scripts/v24/train_four_direction.py` 的新 GPU 训练自动启动独立、只读监控；DDP 仅 rank0 启动。CPU 训练不启动。日志位于每次 run 的 `logs/gpu_status.jsonl`，恢复时追加，文件锁防止同一 run 重复监控。

每 15 秒查询一次所有主机 GPU 的编号、UUID、PCI 地址、驱动版本、核心温度、功耗/上限、利用率、显存使用/总量，以及计算进程的 GPU UUID、PID、显存。额外保存训练进程的线程数/RSS和主机可用内存/交换空间摘要，帮助区分 GPU 与主机资源压力。温度不代表所有显存/热点温度；计算进程列表不含所有图形进程；不支持的字段保留 N/A。

日志使用带时区的时间戳、单调时间及启动 boot ID；每条写入后 flush/fsync。事件为 start/sample/stop/monitor_error，nvidia-smi 非零返回或 2 秒超时保存在 sample 中。正常间隔约为 15 秒加查询时间，双卡通常每小时数百 KB，实际取决于进程数。查询不初始化 CUDA、不修改功耗/时钟、不重置 GPU。没有后台训练时监控自动退出。

旧冻结源码不会自动得到新功能。可在不修改快照、不暂停队列的情况下挂接：

```bash
# 监控一个仍存活的训练进程
python src/stmoe_imputer/utils/gpu_monitor.py --pid TRAIN_PID --run-dir /absolute/run/path
# 跟随指定队列的训练子进程，Z06/Z07/Z08 各自写入各自 run 目录
python src/stmoe_imputer/utils/gpu_monitor.py --queue-pid QUEUE_PID
```

队列模式只跟踪指定 PID 的后代 `train_four_direction.py --run-dir ...`，覆盖该训练入口内的验证/最终测试，单独启动的 OOD 评估进程不属于训练日志范围。PID 加进程启动时刻防止 PID 重用。SSH 挂断后需继续运行时，用专用 tmux 主命令或 nohup 保存到队列的 launcher 日志；监控随队列退出，不留空闲 shell。重启后重新挂接，不能补录启动前数据。

本日志不持续订阅内核 Xid；低频采样可能漏掉瞬时故障，断电可能丢失最后一条。查询错误或温度异常是线索，不能单独证明 GPU 导致整机宕机。保留原日志并结合管理员提供的内核/硬件日志排查。
