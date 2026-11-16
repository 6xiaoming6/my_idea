# I37训练预算实验：启动前验收

验收日期：2026-11-16。这里只记录实现验证，不代表新超参数取得精度或泛化收益。

- 14项CPU测试通过，包含11配置隔离/相同初始化、全部余弦学习率首末值与单调性、调度器恢复、队列完成收据/报告覆盖，以及原发布CLI恢复与性能路径等价测试。
- 所有组仅修改train.epochs、scheduler.total_epochs和scheduler.eta_min；模型总参数347930、可训练333019，公共初始哈希2baf7361f1aa2308e5f2a55c07fd657e7d5feea87dd296ca071406aec3179f99。
- GPU1实测E200_L，真实TaxiBJ batch32、AMP，最后一轮强制TA/ST。前向、反向、一次AdamW更新以及模型/优化器/调度器/AMP/RNG检查点往返通过；每窗口8次专家执行，末轮F。
- GPU预检峰值allocated约7.905 GiB，前后向及往返流程约2.34秒。只是该预检路径，不当作完整训练峰值或吞吐。
- 模型、数据/损失实现及原训练/评估脚本保持best基准逐文件一致；冻结脚本在启动前再次核对。原测试协议及数据摘要与100轮I37参考一致。

完整收据：outputs/v25/experiments/epoch_budget/20261116_seed7/cpu_tests.log 和 preflight_gpu/preflight.json。每run由原训练入口写入15秒GPU状态日志，队列仅使用GPU1。
