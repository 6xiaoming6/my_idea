# I29：I25严格F-only输入对照

用户授权：只输入F尺度，与不增加模块/后处理的单尺度baseline比较。GPU1，不使用GPU0。

- TaxiBJ两通道、12步、四类混合mask0.4、seed7、100epoch、batch32、每5轮验证；AdamW 1e-3余弦至3e-4，AMP，保留I25六套评估及F掩码。
- `data.input_scales=F`：训练/验证/测试均将M/C观测值和mask置零，不返回完整粗尺度标签。磁盘原NPZ保留公共数据格式，但M/C数值不进入模型或损失。
- `initial_observation_scales=F`：H0用原融合投影的F槽，M/C槽为零。不同于I01：后续各轮也没有M/C原生观测。
- 保留前三轮自由计算尺度、第四轮固定F，共享八专家、原生Top2、受控差分、detach历史池。内部C/M分支只处理由F状态得到的粗粒度特征，不代表额外M/C观测输入；不强制FFFF。
- 去掉最终M/C学习投影及分配头、可见M/C辅助损失。没有投影时两个F任务损失相同，合并为单项F缺失L1，移除无意义的可学习混合标量。保留0.01候选均衡。不增加非负截断等后处理。
- 从公共未训练初始化独立训练；逐项核对保留参数与I25公共初值一致。不是载入I25的best微调。参数减少须披露。
- 只比较5种传统法raw；旧RITS/BRITS训练额外含L1，不作为合规原生结果。修正后的RNN需另行训练，本组不替用户启动。

准备：`python scripts/v25/run_taxibj_input_communication.py --methods I29 --prepare-only`。
运行冻结入口：`CUDA_VISIBLE_DEVICES=1 python <suite>/source_snapshot/scripts/v25/run_taxibj_input_communication.py --suite <suite>`。
使用专用tmux主进程；预检、训练、六套评估和正式报告依次执行，失败即停止，结束自动退出。GPU监控15秒采样，完整best/last恢复，旧实验不改动。

CPU检查F掩码一致、M/C污染无影响、与屏蔽M/C后的I25原始预测一致、每窗口8次专家调用、历史读取0/1/2/3及正确F任务损失。GPU1原尺寸batch32检查AMP更新及完整状态恢复。
