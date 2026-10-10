# 双数据集四缺失率尺度对比（16组）

```bash
python -u scripts/v24/run_scale_rate_compare.py --gpu 0 --epochs 100 --batch-size 32
```

默认TaxiBJ再BikeNYC，各自random0.2/0.4/0.6/0.8，每个率依次CMFF、自由三尺度Top1。
这里random保持现有四基础缺失族混合，不改为纯随机点。
每个数据集/率各训练两个新模型；不是测试时改mask率。

- 同seed7公共参数初始化、同mask种子20260917和loader顺序。
- 四轮共享八专家，尺度内原生Top2，direct，关闭completion feedback。
- 两组均保留±0.1 RMS匹配差分，默认`--communication delta`。
- CMFF固定路径；Top1无配额、无末轮F限制，三尺度softmax预热5epoch＋过渡5epoch。
- 训练epoch1–5软融合三尺度；6–10软系数为5/6、4/6、3/6、2/6、1/6；11起仅Top1。
- 验证/测试始终Top1。尺度头全程训练，历史计数按融合贡献累计；预热最多24次专家执行，之后8次。无教师或额外尺度损失。
- 严格确定性池化/插值为显式配置开关，旧U/B/N/R/T/W行为不改。
- F/M/C为原尺寸/2/4；TaxiBJ=32×32/16×16/8×8；BikeNYC=24×12/12×6/6×3。
- 100epoch、batch32、val5、AdamW1e-3余弦至3e-4、L1+.01candidate balance、best/last。
- 单GPU串行，失败即停，last恢复，完成训练跳过；缺少六套评估则只补评估。
- 冻结源码、静态配置、数据指纹、公共初始化；`--suite <path>`使用冻结预算和配置。
- 输出仍为`outputs/v24-COE/{dataset}/custom/{datetime}_scale_{cmff|soft_top1}_seed7/random/rate*/...`。
- 队列目录`outputs/v24-COE/experiments/scale_rate_compare/{fingerprint}`。
- 每组结束更新阶段报告，全部结束自动导出正式报告至`experments_report/`。
- 控制台仅任务切换与batch级train epoch当前/总数tqdm（loss/mae/rmse）。

可选`--dataset taxibj bikenyc`、`--rates 0.2 0.4 0.6 0.8`、`--dry-run`、`--summary-only`。
恢复时以`--suite`内冻结配置为准，不通过命令行静默变更原训练预算。
报告先方法/复现，再结果/曲线/诊断，最后配对分析。ID为主要比较，OOD完整保留。
