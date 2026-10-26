# I37：多尺度观测驱动的共享专家补全

本分支为 **best-perf候选**：默认保留原计算，优化需显式启用，并通过严格逐位和性能验收。[优化说明](docs/performance.md)。原best提交及其源码另行保留。

这是 **I37 的独立复现发布版**，从实验冻结源码整理而来，提供数据准备、训练与断点恢复、六套缺失协议评估和独立推理。`best` 是发布分支名，不表示已经证明全局最优。

输入为同一时空现象的 F/M/C 三个空间尺度的可见观测及各自掩码。模型融合三尺度输入，经四轮共享专家计算，再通过解码和可学习分配投影输出细尺度补全结果。

- 前三轮自由选择一个尺度，第四轮固定 F；每轮只执行选中的两个专家。
- 八种专家跨轮共享参数；受控差分与历史预测池参与跨轮通信。
- 保留可见 F；利用可见 M/C 的区域总量，在缺失位置进行非负、可学习分配的校正。
- 固定投影前后 F 缺失损失权重 **0.5 / 0.5**，另加可见 F 重建和可见 M/C 一致性项。

默认配置共 **347,930 个参数**，使用 TaxiBJ 双通道、12 时间步。具体方法和边界见 [方法说明](docs/method.md)。

## 安装

```bash
git clone --branch best --single-branch git@github.com:6xiaoming6/my_idea.git i37
cd i37
python -m venv .venv
source .venv/bin/activate
# 请先按服务器 CUDA 环境安装兼容的 PyTorch，再安装本项目。
pip install -e .
python -m unittest discover -s tests -v
```

验收环境为 Python 3.11、PyTorch 2.6.0、NumPy 1.26.4；完整环境记录见 [验收记录](docs/validation.md)。不需要安装原研究仓库、其他 baseline 或历史实验脚本。

## 准备数据

数据和训练权重单独传输，不纳入 Git。已有 v25 预处理数据可以直接使用，目录中应有：

```text
data/taxibj/
  taxibj_train.npz
  taxibj_val.npz
  taxibj_test.npz
```

每个文件包含 `x_f_gt`、`x_m_gt`、`x_c_gt`，形状分别为 `[N,2,12,32,32]`、`[N,2,12,16,16]`、`[N,2,12,8,8]`，以物理流量总量存储。模型只接收按协议遮挡后的观测与掩码，完整 F 用于训练/评估标签，完整 M/C 不作为 I37 辅助损失标签。

若只有已划分好的完整 F 窗口：

```bash
python scripts/prepare_data.py --source /path/to/fine_splits --destination data/taxibj
```

此命令保持现有样本与划分，用不重叠的 2×2、4×4 求和构造 M/C，不从不完整 F 推造粗尺度观测。数据格式、时间划分和参考校验值见 [数据说明](docs/data.md)。

## 训练及恢复

```bash
python -u scripts/train.py \
  --data-dir data/taxibj --run-dir outputs/i37_seed7 \
  --gpu 1 --evaluate
```

默认 100 epoch、batch32、seed7、四类混合缺失 0.4，每 5 epoch 验证；AdamW，学习率从 1e-3 余弦下降至 3e-4。只按 **ID 验证 MAE** 选择 best。`--gpu 1` 指物理 GPU1，单进程单卡；在其他服务器可显式指定另一张卡。

中断后执行相同命令即可从完整 `last.pth` 恢复。训练状态包含模型、优化器、调度器、AMP scaler 和随机数状态；从最后完成的 epoch 继续，未提交 epoch 重新执行。源码、数据或配置变化会拒绝恢复，已完成训练自动跳过。不要让两个命令共用一个输出目录。

需要后台运行时：

```bash
python scripts/run_tmux.py --data-dir data/taxibj --run-dir outputs/i37_seed7 --gpu 1
tmux attach -t i37-release
```

tmux 中直接显示并保存输出，训练和六套评估结束后自动关闭；失败退出，不无限重试。GPU 训练每 15 秒在 `logs/gpu_status.jsonl` 记录轻量监控，恢复时追加。

主要产物：

```text
outputs/i37_seed7/
  config.json, job.json, run_provenance.json
  initialization.json, runtime.json, training_metadata.json
  checkpoints/best.pth, checkpoints/last.pth
  logs/                         # 曲线、诊断、GPU 状态
  result.json                   # 训练完成凭据
  evaluation.json, summary.md   # 六套统一测试及简表
```

## 单独评估和推理

```bash
python scripts/evaluate.py \
  --checkpoint outputs/i37_seed7/checkpoints/best.pth \
  --data-dir data/taxibj --output outputs/i37_seed7/evaluation.json --gpu 1

python scripts/infer.py \
  --checkpoint outputs/i37_seed7/checkpoints/best.pth \
  --input observations.npz --output completed.npz --gpu 1
```

推理 NPZ 只需 `x_f_obs, m_f, x_m_obs, m_m, x_c_obs, m_c`，均为带 batch 维的 `B,C,T,H,W`；掩码 1 表示可见，0 表示缺失；三尺度的 B/C/T 相同，空间比例 1:2:4。缺失值建议填 0，观测应为相容的非负物理总量。输出键 `F_complete` 与 F 同形状。支持 `--device cpu`。

## 已有参考结果

以下来自原冻结 I37 seed7、epoch95 的 best，**没有因发布代码重新训练或重新选模型**：

| 协议 | MAE | RMSE |
| --- | ---: | ---: |
| ID | 8.0510 | 14.6603 |
| 二元组合 | 8.3105 | 14.7534 |
| 未见形态 | 10.8212 | 18.4874 |
| 三元组合 | 7.8921 | 13.9171 |
| 二元组合 mask 复测 | 8.2840 | 14.5778 |
| 未见形态 mask 复测 | 10.8459 | 18.4938 |

六套协议与随机种子见 [test_protocol.json](configs/test_protocol.json)，完整数值见 [reference_results.json](docs/reference_results.json)。OOD 汇总先合并同类别 mask 复测，再对二元组合、未见形态、三元组合等权平均。单训练种子不支持统计显著性结论；mask 复测不是训练种子复验。单尺度 baseline 与多尺度输入的信息条件不同，应分开解释。

## 实现与可追溯性

- [配置](configs/i37_taxibj.json)：定版 I37 的模型、损失和训练设置。
- [方法与代码导览](docs/method.md)：输入、路由、通信、投影的真实顺序及限制。
- [来源清单](docs/provenance.json)：原冻结套件、逐文件哈希、整理范围。
- [验收记录](docs/validation.md)：初始化、真实检查点数值对齐、CPU 恢复测试与测试限制。

为保持权重键和计算路径一致，必要的父类及兼容函数保留原命名；对外只提供 I37 入口，不包含旧实验队列、数据、训练日志或第三方 baseline。
