# Configuration layout

- `policies/`：人工维护的训练策略 JSON，是 epoch、batch、val_epoch 等设置的来源。
- `training/`：由配置生成器写出的正式模型配置，不手工编辑。
- `testing/`：集成测试专用生成配置，与正式配置隔离。
- `legacy_smoke/`：旧 smoke 系统配置的只读归档，不再被入口脚本调用。

生成器：`../scripts/config/generate_train_configs.py`。
