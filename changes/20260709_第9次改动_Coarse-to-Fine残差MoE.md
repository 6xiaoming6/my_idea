# 20260709_第9次改动_Coarse-to-Fine残差MoE

## 修改目标

按照 `model_designs/v9-single.md` 实现第九版 `v9-single`：使用粗到细残差多分辨率 MoE，让粗尺度先预测全局低频结构，中尺度和细尺度逐级预测残差。

## 开发分支

v9-single

## 修改前提交

4a4933b

## 修改后提交

待本次修改提交后填写

## 核心修改

1. 新增 `CoarseToFineResidualMoE`：
   - coarse MoE 输出 `x_hat_coarse`；
   - mid MoE 输出 `delta_m`，得到 `x_hat_mid = up(x_hat_coarse) + alpha_m * delta_m`；
   - fine MoE 输出 `delta_f`，得到 `x_hat_main = up(x_hat_mid) + alpha_f * delta_f`；
   - `alpha_m/alpha_f` 支持 learnable sigmoid gate，默认初始化为较小残差强度；
   - 保留专家池、QualityRouter、Top-K 路由、专家 selected mask 和 gate 日志。

2. 新增多尺度监督：
   - `l_mid_supervision`；
   - `l_coarse_supervision`；
   - 监督目标在 loss 内由 `x_f_gt` masked pooling 得到，forward 仍只读取观测输入，不泄漏测试信息。

3. 新增 v9 配置和脚本：
   - `configs/v9-single/{taxibj,bikenyc,chap}.json`；
   - `configs/v9-single/ablations/`；
   - `configs/v9-single/policies/quick_5epoch.json`；
   - `scripts/v9-single/train.py`；
   - `scripts/v9-single/run_validation_matrix.py`；
   - `scripts/v9-single/run_full_experiments.py`。

4. `scripts/run_experiments.py` 新增 `--model-config-dir`，方便后续版本通过独立配置目录切换模型，不改变默认行为。

## 验证建议

先运行：

```bash
python scripts/v9-single/run_validation_matrix.py --gpu 0 --dry-run
python scripts/v9-single/run_validation_matrix.py --gpu 0
```

5 个验证点通过后再运行完整实验：

```bash
python scripts/v9-single/run_full_experiments.py --gpu 0
```
