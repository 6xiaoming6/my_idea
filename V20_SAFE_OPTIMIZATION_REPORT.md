# V20 保守接纳优化说明

## 优化目标

原始 V20 在 8 个筛选点上的平均 MAE 相对 V14 退化 23.51%。离线诊断显示，Probe 的整体专家排序能力较弱，且原几何融合会在能力分布接近均匀时仍削弱 V14 先验，因此不能继续把三尺度 Probe 作为强制路由依据。

本次优化保留“缺失几何匹配自验证”的核心机制，但将它改为可拒绝的保守校正器：没有足够证据时严格回退到 V14，只有验证集确认有益时才接纳 Probe。

## 已完成修改

1. **证据中性融合**：改用以均匀专家分布为零证据的乘法校正。Probe 无辨识度时，最终路由与 V14 先验严格一致。
2. **Fine-only 证据**：保留全部尺度的 Probe 训练和诊断，但只有 Fine 尺度可以影响路由。此前离线 Probe→Oracle 诊断表明 Fine 明显强于 Mid/Coarse。
3. **熵阈值拒绝**：能力分布确定性低于 0.02 时 `eta=0`；在 0.02–0.08 之间逐步增加接纳强度，最大值降至 0.25。
4. **保护 V14 训练路径**：训练态不把 Probe 证据注入主路由，只训练 Probe 解码器；验证和测试态才允许候选校正。
5. **梯度裁剪隔离**：Probe 参数单独裁剪，避免 Probe 梯度通过全局梯度范数间接缩小 V14 主干更新。
6. **验证集选择接纳强度**：每个验证周期比较 `eta ∈ {0, 0.05, 0.10, 0.15, 0.25}`，以验证 MAE 选择当前 epoch 的值。`eta=0` 始终是 V14 路由回退项。最佳 epoch 与对应 eta 一起用于最终测试。
7. **保留旧版本复现入口**：`legacy_geometry_hybrid` 消融配置保留原始 V20 的三尺度、训练期、非中性融合行为。
8. **保存有效配置**：验证选中的 eta 写入最佳 checkpoint，并导出 `effective_config.json`，保证后续分析和测试恢复的是实际使用的校准强度。

## 验证状态

- V20 专项单元测试：22/22 通过。
- TaxiBJ、BikeNYC、CHAP：真实 1-epoch 训练—验证—最佳 checkpoint—测试链路通过。
- 验证校准 smoke：能够记录每个候选 eta 的验证 MAE、选择 eta、写入最佳 checkpoint 配置，并用同一 eta 测试。

这些检查证明实现和保护机制有效，但不能代替完整训练后的数值结论。

## 推荐验证命令

复用已完成的 `v20_vs_v14_screening` 中 V14 结果，只训练优化后的 8 个 V20 点：

```bash
python scripts/v20-single/run_compare_v14_v20.py \
  --profile screening \
  --tag v20_safe_optimized_screening \
  --v14-reference-tag v20_vs_v14_screening \
  --models v20 \
  --gpu 0 \
  --stop-on-error
```

脚本可断点续跑；重复同一命令会跳过新 tag 下已完整结束的 V20 点。结果汇总到：

`outputs/v20-single/comparison/v20_safe_optimized_screening/comparison.md`

只有优化版在筛选点上达到预期后，才应扩大到 24 点或多随机种子比较。
