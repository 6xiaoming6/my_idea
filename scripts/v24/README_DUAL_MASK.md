# v24 主方案与多固定链双 mask 验证

本阶段以当前软预热主方案 `coe_main` 为自由选择参照，同时测试 7 条固定四层链；每个模型都在两种 mask 协议上运行：

1. `mixed9_rate0.4`：九类缺失模式混合，缺失率 0.4；训练每 epoch 重采样，验证/测试固定。
2. `original_random_rate0.4`：恢复原始 random_point CSV，训练、验证、测试分别使用对应 CSV，缺失率 0.4。

|组|固定链|目的|
|---|---|---|
|`coe_main`|动态自由选择|主方案参照|
|`coe_fixed_tast`|TA-ST-S-TA|旧实验固定链参照|
|`coe_fixed_tsts`|T-S-T-S|时间/空间交替|
|`coe_fixed_stst`|S-T-S-T|空间/时间交替|
|`coe_fixed_tata`|TA-TA-TA-TA|重复时间注意力|
|`coe_fixed_ssss`|S-S-S-S|重复空间|
|`coe_fixed_tttt`|T-T-T-T|重复时间|
|`coe_fixed_stst_alt`|ST-ST-ST-ST|重复时空专家|

所有组均为四层六专家、单尺度、seed 7、20 epoch、batch 16、相同优化器和软预热主配置。固定链关闭路由均衡项，因为没有可学习路由；自由选择组保留 balance=0.01。当前阶段不保存 `best.pt`，仍在 CPU 内存中保留验证最优状态并进行最终测试。

这组实验检验的是自由选择能否稳定超过多种有代表性的固定链，而不是追求固定链的最优搜索。每个固定链都要在两种 mask 协议内分别与同协议的 `coe_main` 比较；不能把不同 mask 的误差直接混合排名。主要记录 MAE/RMSE、每层使用率、路径分布、按缺失族选择和最后五个 epoch。

## 启动

```bash
cd /home/students/HuangMingYu/code/py/my_idea/my_idea
conda activate difftdi
python scripts/v24/run_dual_mask.py --dry-run
tmux new-session -s v24-dual-mask 'python -u scripts/v24/run_dual_mask.py --gpu 0'
```

共 16 个任务。单卡顺序运行，预计约 5–6 小时，建议预留 6.5 小时。入口检测 GPU 占用并拒绝并行启动。
