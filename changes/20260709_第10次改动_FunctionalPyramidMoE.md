# 第10次改动：Functional Pyramid MoE

- 开发分支：v10-single
- 修改前提交：4a4933b
- 修改后提交：待提交后补充

## 修改目标

根据 `model_designs/v10-single.md`，将原本同构 MoE 专家池扩展为具有明确职责划分的 Functional Pyramid Expert Pool。该版本仍保持“多分辨率 + MoE 路由”的主线，但让专家具备更强可解释性：

1. Smooth Expert：平滑低频结构；
2. Local Spatial Expert：局部空间邻域相关；
3. Temporal Expert：时间趋势；
4. Missing Pattern Expert：缺失形态与边界；
5. Dynamic Expert：突变与高频动态残差。

## 主要代码变化

### 1. 新增功能型专家池

新增 `src/stmoe_imputer/models/v_single/functional_experts.py`：

- `SmoothExpert`
- `LocalSpatialExpert`
- `TemporalExpert`
- `MissingPatternExpert`
- `DynamicExpert`
- `FunctionalExpertPool`

`FunctionalExpertPool` 的返回接口保持和原 `TopKRoutedExpertPool` 一致：

```text
z, top_indices, top_weights, selected_mask
```

因此可以无缝接入现有主干、loss、训练和评估流程。

### 2. 保留原模型回退

`MultiScaleMoEBackbone` 新增配置项：

```json
{
  "model": {
    "main": {
      "expert_pool_type": "functional"
    }
  }
}
```

当 `expert_pool_type="homogeneous"` 或不设置时，仍使用原始 `TopKRoutedExpertPool`，旧实验配置不受影响。

### 3. 新增专家使用率日志

训练、验证、测试日志中新增：

- `expert_usage_smooth`
- `expert_usage_local`
- `expert_usage_temporal`
- `expert_usage_missing`
- `expert_usage_dynamic`
- `expert_usage_entropy`

这些指标用于观察 Router 是否真的在不同数据集、缺失率和缺失模式下形成可解释专家分工。

### 4. 新增 v10 配置和脚本

新增：

- `configs/v10-single/taxibj.json`
- `configs/v10-single/bikenyc.json`
- `configs/v10-single/chap.json`
- `configs/v10-single/ablations/*.json`
- `configs/v10-single/policies/quick_1epoch.json`
- `configs/v10-single/policies/quick_5epoch.json`
- `scripts/v10-single/train.py`
- `scripts/v10-single/run_validation_matrix.py`
- `scripts/v10-single/run_full_experiments.py`

统一输出目录：

```text
outputs/v10-single
```

## 训练策略

第一版 v10 不新增 loss，继续使用 main loss。原因是本次核心变量是专家池结构，如果同时引入新的专家正则或辅助损失，会难以判断收益来自结构还是训练目标。

## 风险与回退

潜在风险：功能专家的人为归纳偏置可能限制同构专家自由学习，尤其是 BikeNYC 这类数据量和空间形态较特殊的数据。

保护措施：

1. 所有功能专家最后一层卷积零初始化，使初始状态接近残差恒等映射；
2. 保留 `expert_pool_type="homogeneous"` 回退配置；
3. 默认只使用 5 个专家，`top_k=2`，不进一步扩大专家池；
4. BikeNYC 默认 dropout 提高到 `0.15`，降低过拟合和专家过度激活风险。
