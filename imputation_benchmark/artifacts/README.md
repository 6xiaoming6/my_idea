# Runtime artifacts

- `runs/`：当前维护的 paper、TaxiBJ 和 overview one-epoch 启动记录。
- `selftests/`：启动器自身测试结果。
- `native_experiments/`：上游模型训练期间产生的原生临时文件。
- `legacy/`：旧 smoke 系统的历史日志与结果，只读保留。

新的运行产物只能写入本目录或项目级 `outputs/`，不要写到 benchmark 根目录。
