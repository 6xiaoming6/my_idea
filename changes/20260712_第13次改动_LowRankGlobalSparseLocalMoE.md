# 第13次改动：Low-rank Global + Sparse Local MoE

- 日期：2026-07-12
- 开发分支：v13-single
- 修改前提交：4a4933b
- 修改后提交：`35df846a9d5a41be2b8afc6b8f4708580dca44b7`
- 版本状态：实现完成，端到端 smoke 通过，等待正式实验

## 1. 修改目标

第13版从稳定 main 独立开发，不继承第12版代码。目标是把时空补全显式拆分为：

1. 低秩全局分支：学习大范围、稳定的时空结构；
2. 稀疏局部 MoE：处理多尺度局部异质残差；
3. 小系数残差融合：在训练初期保护全局主路径。

旧版本文件、配置、脚本和输出目录均未覆盖。

## 2. 实际模型结构

### 2.1 多尺度编码

沿用 main 的 fine/mid/coarse 输入和 `ScaleTokenEncoder`。三个尺度分别经过 `QualityRouter` 与共享 `TopKRoutedExpertPool`，每个样本选择 Top-K 局部专家，再由 `ProgressiveRouteFusion` 逐级聚合为细尺度局部表示 `h_local`。

### 2.2 输入相关的低秩全局混合器

设计文档示例仅让输入 token 查询静态 anchors，静态 anchors 的 key/value 不包含当前样本信息，无法真正汇聚输入的全局结构。实际实现改为两阶段交叉注意力：

```text
input tokens --collect--> R 个输入相关 latent
R 个 latent --broadcast--> 每个输入 token 的全局上下文
```

两次注意力复杂度均为 `O(L×R)`，默认 `R=16`，没有使用 `O(L²)` 全局自注意力。全局更新采用 `sigmoid(global_gamma)` 小残差，输出投影使用 `std=1e-3` 小初始化。

### 2.3 全局—局部安全融合

主实验采用：

```text
h_global = h_f + sigmoid(global_gamma) * global_update
h_main = h_global + sigmoid(local_alpha) * local_proj(h_local)
```

`local_alpha_init=-3`，初值约为 0.0474；`local_proj` 默认零初始化。这样第一步近似从全局细尺度主路径开始，随后局部投影先学会有效映射，再逐渐向局部专家传播梯度。

支持以下可回退模式：

- `global_plus_local`：V13 主实验；
- `global_only`：只保留低秩全局分支；
- `local_only`：只保留多尺度局部 MoE；
- `none`：完全回到稳定 main 的原始共享+路由双分支。

### 2.4 损失

没有增加 nuclear norm、频域损失或新的低秩正则。V13 主模型保留 Smooth L1 主重建损失、跨尺度一致性损失和 MoE 重要性/负载均衡损失。由于主模型不再使用旧共享分支，其共享/路由辅助头与互补损失自然关闭；`main_fallback` 会恢复原 main 的双分支辅助训练。

## 3. 配置参数

V13 默认参数：

```json
{
  "global_local_mode": "global_plus_local",
  "use_lowrank_global": true,
  "use_sparse_local_moe": true,
  "lowrank_mode": "anchor_attention",
  "lowrank_rank": 16,
  "lowrank_num_heads": 4,
  "lowrank_dropout": 0.1,
  "global_gamma_init": -3.0,
  "global_out_init_scale": 0.001,
  "local_alpha_init": -3.0,
  "local_alpha_trainable": true,
  "local_alpha_fixed": 0.05,
  "local_proj_zero_init": true
}
```

三个数据集的模型配置分别位于：

- `configs/v13-single/taxibj.json`
- `configs/v13-single/bikenyc.json`
- `configs/v13-single/chap.json`

所有正式输出写入 `outputs/v13-single/`。

## 4. 消融配置

- `main_fallback.json`：稳定 main 对照；
- `global_only.json`：仅低秩全局；
- `local_moe_only.json`：仅稀疏局部 MoE；
- `rank_8.json`：低容量 bottleneck；
- 主配置：rank=16；
- `rank_32.json`：高容量 bottleneck；
- `alpha_fixed_0.05.json`：固定局部残差系数；
- 主配置：可学习局部残差系数。

## 5. 新增日志

训练、验证和测试日志增加：

- `v13_rank`
- `v13_global_gamma`
- `v13_alpha_local`
- `v13_rank_attention_entropy`
- `v13_global_update_norm`
- `v13_global_feature_norm`
- `v13_local_feature_norm`
- `v13_local_projected_norm`
- `v13_fused_feature_norm`

低秩与局部融合标量进入无 weight decay 的 scalar optimizer group，并使用配置中的 `scalar_lr_mult`。

## 6. 文件变更

新增模型：

- `src/stmoe_imputer/models/v_single/low_rank_mixer.py`
- `src/stmoe_imputer/models/v_single/v13_lowrank_mr_moe.py`
- `src/stmoe_imputer/models/v_single/__init__.py`

修改接入：

- `src/stmoe_imputer/models/main_branch.py`
- `src/stmoe_imputer/models/__init__.py`
- `src/stmoe_imputer/engine.py`
- `scripts/run_experiments.py`

新增配置与脚本：

- `configs/v13-single/`
- `scripts/v13-single/train.py`
- `scripts/v13-single/run_validation_matrix.py`
- `scripts/v13-single/run_full_experiments.py`

## 7. 验证结果

已完成：

1. Python 源码编译；
2. 全部 V13 JSON 解析；
3. `global_plus_local/global_only/local_only/none` 四种模式的前向与反向；
4. TaxiBJ、BikeNYC、CHAP 的真实输入尺寸前向；
5. synthetic 一轮完整训练、验证、最佳模型保存、重新加载和测试；
6. V13 诊断字段写入 train/val/test/metrics 日志；
7. 版本脚本与统一实验脚本 dry-run。

synthetic smoke 输出：

```text
outputs/v13-single/unknown/debug/smoke_v13_single/random/rate0.45/
```

## 8. 运行方法

五点一轮验证：

```bash
python scripts/v13-single/run_validation_matrix.py --gpu 0 --epochs 1
```

单个实验：

```bash
python scripts/v13-single/train.py \
  --dataset TaxiBJ \
  --mask fixed \
  --rate 0.2 \
  --gpu 0
```

单卡依次运行三个数据集全部 fixed，再运行全部 random：

```bash
python scripts/v13-single/run_full_experiments.py --gpu 0
```

