# CoE 深度与专家池对照

八组统一使用 TaxiBJ legacy `random` mask、缺失率 0.4、seed 7、batch 16、20 epoch，每 2 epoch 验证；不保存 `best.pt`，仅在内存中保留最佳状态：

|顺序|实验|链路轮数|Top-K|专家池|目的|
|---:|---|---:|---:|---:|---|
|1|`hetero_s1_top8`|1|8|8|异构八专家，单轮同时激活 8 个|
|2|`hetero_s2_top4`|2|4|8|异构八专家，两轮同时激活 4 个|
|3|`hetero_s4_top2`|4|2|8|异构八专家，四轮同时激活 2 个|
|4|`hetero_s4_top8`|4|8|8|异构八专家，四轮同时激活 8 个|
|5|`dup12_s1_top8`|1|8|12|六类专家各两份独立实例，单轮 Top-8|
|6|`dup12_s2_top4`|2|4|12|六类专家各两份独立实例，两轮 Top-4|
|7|`dup12_s4_top2`|4|2|12|六类专家各两份独立实例，四轮 Top-2|
|8|`dup12_s4_top8`|4|8|12|六类专家各两份独立实例，四轮 Top-8|

模型结果写入 `outputs/v24-COE/TaxiBJ/custom/<YYYYMMDD_HHMMSS_核心名_seedN>/random/rate0.4/<时间_seed_bs>/`；队列配置和详细 launcher 日志写入 `outputs/v24-COE/experiments/coe_depth_pool/`。控制台实时显示 batch 级训练 tqdm，postfix 只包含 train 的 `loss`、`mae`、`rmse`。

```bash
cd /home/students/HuangMingYu/code/py/my_idea/my_idea
conda activate difftdi
python scripts/v24/run_depth_pool.py --dry-run
tmux new-session -s v24-depth-pool 'python -u scripts/v24/run_depth_pool.py --gpu 0 --epochs 20'
```

将 `--epochs` 改成任意正整数即可覆盖八组实验的训练轮数和 scheduler 周期；epoch 数会写入实验名，避免不同轮数的结果被自动跳过。

Top-K 决定每轮同时选择并归一化融合几个专家。八组串行预计约 3–5 小时，建议预留 6 小时；十二专家组的显存和耗时会明显高于八专家组。由于未保存 `best.pt`，最终最佳状态仅在单次进程内使用，进程结束后不保留最佳权重文件。
