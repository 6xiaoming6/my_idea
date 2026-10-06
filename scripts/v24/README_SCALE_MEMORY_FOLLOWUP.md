# 固定尺度与跨轮差分：八组受控复验

2026-10-05，目标上午10点左右完成。沿用100epoch，不临时改epoch或双卡。

|组|seed|固定尺度路径|跨轮输入|参考|
|---|---|---|---|---|
|D1|7|CMFF|M07归一化差分|旧S01|
|D2|7|MCFF|原生direct|旧S01|
|D3|7|MCFF|M07归一化差分|D2及D1|
|D4|17|CMFF|M07归一化差分|旧S11|
|D5|17|MCFF|原生direct|旧S11|
|D6|17|MCFF|M07归一化差分|D5及D4|
|D7|27|CMFF|M07归一化差分|旧S12|
|D8|27|MCFF|原生direct|旧S12|

C/M/F=8/16/32，T12。上述组均四轮共享八专家、原生Top2、original融合、direct输出，无completion feedback、尺度软启动、教师或其他额外损失。差分仅修正专家输入；完整公式沿用已验证four_direction的memory=delta。每窗口8次专家执行、面积代理4.625，不能称精确等FLOPs。

TaxiBJ random0.4，四基础mask、原切分和mask种子，逐epoch训练重采样，验证固定。GPU0单卡串行，batch32，100epoch，val5。AdamW 1e-3余弦至3e-4，L1+0.01candidate balance。best按ID验证；best/last及完整恢复状态保留。原六套测试＋新确认mask20261025/26/27（test偏移30000）＋0.2/0.6/0.8缺失率迁移。旧S01/S11/S12/G12补新评估，不重新训练。

D3/D6仅双seed，不能声称三seed复验。G12历史实际100%MCFF，但自由训练和固定训练的差别不可视为纯推理干预。所有配置预先冻结，无候选选择。完成训练跳过、缺评估补跑；配置变化产生新指纹，失败停止，完整epoch恢复。

```bash
python -u scripts/v24/run_scale_memory_followup.py --gpu 0
python scripts/v24/run_scale_memory_followup.py --dry-run
python scripts/v24/run_scale_memory_followup.py --summary-only
```

自动逐组更新阶段报告，统一评估完成导出正式报告到experments_report，顺序：具体做法→完整结果/曲线/诊断→配对分析和建议。控制台只任务切换及batch级train epoch当前/总数tqdm。
