# Baseline source repositories

本目录只存放上游模型源码及其原生资源。外层数据适配、训练策略、日志和启动逻辑
分别位于 `../scripts/` 与 `../configs/`，不要在本目录新增统一调度脚本。

- 主表：BRITS、CSDI、GAIN、GRIN、ImputeFormer、LATC、PAST、PriSTI、
  SAITS、STAMImputer、STCPA。
- 附录候选：AGCRN、ASTGNN、E2GAN、GCASTN、IGNNK、LAST、mTAN、SSTBAN。
- `GenerateData` 是上游项目附带的旧数据工具，仅作兼容保留。

目录名称保持上游仓库原名（例如 `grin`、`imputeformer`、`STAMImupter`），
避免破坏原生相对导入。
