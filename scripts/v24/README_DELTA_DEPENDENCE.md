# Z09–Z12：历史预测与专家交互对受控差分的依赖性

状态：已实现独立批次入口；训练进度以冻结suite的state/results/evaluations为准。2026-10-12核查X/Y/Z冻结计划，未发现下面四个同配置对照；已有X06/Y07仅为无交互历史池，不能替代。本批仅四次新训练，旧实验只读复用，不追加修改已完成的Z01–Z08冻结计划。

## 1. 四组具体做法

共同骨架为四轮共享八专家T/S/TD/SD/TA/ST/TL/SL、固定CMFF、每轮原生动态Top-2、direct隐状态更新。四组均不启用受控差分，上一轮输出仍作为下一轮输入。

| 执行顺序/编号 | 复制的实际模型 | 本组结构 | 唯一方法变化 | 主要问题 |
|---|---|---|---|---|
| 1 / Z09 | Y12 | CMFF＋自适应绝对历史＋乘积/差值交互 | 关闭受控差分 | 完整历史/交互组合是否依赖差分，差分有无边际收益 |
| 2 / Z10 | Z06 | CMFF＋自适应相对历史＋乘积/差值交互 | 关闭受控差分 | 相对历史背景下，额外相邻轮hidden差分是否必要 |
| 3 / Z11 | X10 | CMFF＋乘积/差值交互，无历史池 | 关闭受控差分 | 乘积/差值交互在纯CMFF上的增量及其与差分的相互作用 |
| 4 / Z12 | X09 | CMFF＋普通拼接MLP，无历史池 | 关闭受控差分 | 无差分条件下的等容量拼接参考 |

Z09保存观测硬约束、detach的历轮双通道补全预测；逐位置对历史评分并softmax读取R。注入器读取归一化R和真实mask，使用0.1×hidden RMS×tanh并乘缺失比例。Z10仅将读取数值换为R−P_last；评分、mask输入、幅度限制均继承Z06。一条历史时数值为零但mask可产生非零输出，不声称整个第二轮历史分支为零。

Z09/Z10/Z11的交互读取两个已执行专家归一化输出的[Abar*Bbar,Abar−Bbar]；Z12读取[Abar,Bbar]。均为共享128→32→64、6240参数、末层零初始化的有界分支，叠加原Top-2加权和。按专家编号排序，保留差值方向。无新专家调用。Z11/Z12不引入历史池。

特别区分三种“差”：本批去掉的是跨轮hidden受控差分；Z10的R−P_last历史表示，以及交互中的Abar−Bbar都保留。不能因名字含差值而一起删除。

## 2. 公平训练协议

- TaxiBJ，模型/loader seed7，random0.4，100epoch，batch32，GPU0单卡串行，num_workers=0。random沿用四基础族random_point/node_outage/temporal_gap/spatial_region，不是仅独立点缺失。
- 同一冻结train/val/test数据与split（2452/342/707窗口）、mask seed20260917、相同epoch重采样顺序与评估mask；AdamW lr1e-3余弦至3e-4、weight_decay1e-4、clip1、AMP、strict replay，沿用各父模型初始化。
- 全部从原始初始化训练100epoch，不载入Y12/Z06/X10/X09训练后的best或last；仅同一新run意外中断时允许从本组last恢复。
- 复制各父组的实际配置与公共/分支初始权重。方法因素只将core_validation.communication从conditional改为none。标识、输出路径等元数据另改，保存配置diff和初始化SHA。
- 保留差分门参数键与初始值，冻结且不执行门/RMS差分分支。总参数不变，但可训练参数减少；不宣称同有效容量。差分门冻结不能冻结历史评分器或注入器。
- 每5epoch验证，仅按同率ID验证MAE选best；保存best.pth、last.pth及完整优化器、调度器、AMP、RNG、loader恢复状态。
- 损失仅末轮缺失区L1＋0.01候选负载均衡；不加入中间监督、额外FFN、三平面、第二视图、一致性或自由尺度。

## 3. 旧参考与可回答的问题

父记录：Y12位于history_followup/47fea7c076e9a355；Z06位于history_mechanism/a964cb1715943e5a；X09/X10/X01/X06位于history_plane_interaction/86284122bd9a18c2；K04位于core_validation/684cad05994af9a4（均为outputs/v24-COE/experiments下）。每组引用原config、metrics、评估协议、结果及best checkpoint指纹。

| 对照 | 只改变的主要因素 | 解释范围 |
|---|---|---|
| Z09 / Y12 | 绝对历史＋交互背景下去差分 | 检验差分边际价值，不直接证明两种信息数学独立 |
| Z10 / Z06 | 相对历史＋交互背景下去差分 | 检验差分必要性，不等于相对历史可替代所有通信 |
| Z11 / X10 | 无历史乘积/差值背景下去差分 | 检验交互与差分依赖 |
| Z12 / X09 | 无历史拼接背景下去差分 | 检验拼接与差分依赖 |
| Z09 / X06 | 无差分绝对历史上加乘积/差值 | 检验交互在无差分历史背景的独立增量 |
| Z09 / Z11 | 无差分乘积/差值上加绝对历史 | 检验历史在无差分交互背景的增量 |
| Z10 / Z09 | 无差分条件下相对/绝对历史 | 检验历史表示取舍 |
| Z11 / Z12 | 无历史、无差分时交互输入表示 | 等容量乘积/差值对普通拼接 |
| Z11 / X01、Z12 / X01 | 纯CMFF上加交互 | 判断交互本身是否改善目标指标 |

不新增训练即可形成三个2×2对照：

| 差分 | 无历史、无交互 | 无历史＋乘积/差值 | 无历史＋拼接 |
|---|---|---|---|
| 无 | X01 | Z11 | Z12 |
| 有 | K04 | X10 | X09 |

| 差分 | 绝对历史、无交互 | 绝对历史＋乘积/差值 |
|---|---|---|
| 无 | X06 | Z09 |
| 有 | X05 | Y12 |

分别报告有/无差分时交互带来的配对变化；同时可报告原始MAE/RMSE的差中之差作为描述性交互量，不将单种子差中之差写成显著交互或因果证明。Z10缺少“相对历史、无交互”组，不能独立拆分相对历史与交互贡献；本批不因此扩大训练数量。

## 4. 评估、判读与证据边界

每组复用六套协议：ID、两元组合、未见几何、三元组合、两元组合mask复测、几何mask复测。完整报告MAE/RMSE、分缺失族、成本、逐轮误差/修好/破坏、历史权重、交互幅度、路由分布与AMP非有限诊断。复测不充当训练种子，ID/OOD分别列示，不用OOD挑best。

以候选/对照−1的百分比表示变化，负数为改善；列出每套协议，主三类OOD平均仅作补充，不重复计入两套mask复测。任一MAE/RMSE退步>5%显式标记。没有预先定义等价区间与多种子证据，不把“数值相近”写成统计等价。

- 去差分后ID/OOD均改善：支持在该结构和本协议下差分可能多余或有负作用，仍需新种子确认。
- 去差分后ID/OOD均退步：支持差分仍有边际价值；不能据此单独证明其不可替代。
- ID与OOD方向相反或不同缺失族方向不一：保留取舍，不宣称完全冗余或完全互补。
- Z09若接近/优于Y12，同时优于X06与Z11，才更有依据说历史＋交互在没有额外hidden差分时也有效。只看Z09/Y12不够。
- Z10若接近/优于Z06，说明该组合可能不需要额外受控差分；模型仍有正常hidden链和双专家交互，不能把结果全归功于相对历史。
- 本批只解释0.4、TaxiBJ、seed7；不能解决Z08在0.6下的退步，也不替代跨数据集/训练种子验证。主线表述为多轮协同补全，不要求逐轮误差下降。

## 5. 实现前需处理的约束与验收

当前HistoryFollowupCoE显式要求conditional，直接改Y12/Z06配置为none会报错。实现时应通过新批次专用、显式启用的构造路径支持none，保留旧Y/Z默认约束和冻结源码；不得只改标签或在已训练模型推理时关门。

验收应包括：父模型conditional路径输出/梯度兼容；各新组公共与分支初始化逐值匹配；无差分时差分门/RMS修正确实不执行且门无梯度；历史/交互仍有有限梯度且原生Top-2执行次数不变；Z10 mask通道保留；真实batch32 AMP显存/有限性检查；恢复RNG与完成跳过。预检与正式训练使用不同输出，不把预检计入四组结果。

新建独立冻结suite；GPU0串行Z09→Z10→Z11→Z12，避免与其他队列重复运行。每组自动写logs/gpu_status.jsonl，轻量采样15秒；保留启动、失败、恢复日志。队列训练、评估与正式报告全部完成后，专用tmux主进程退出、会话自动关闭。队列入口为scripts/v24/run_delta_dependence.py；--prepare-only只冻结计划，--suite恢复同一队列。


## 6. 运行入口与实现说明

```bash
/home/students/HuangMingYu/anaconda3/envs/project/bin/python -m unittest discover -s tests -p test_v24_delta_dependence.py
/home/students/HuangMingYu/anaconda3/envs/project/bin/python scripts/v24/run_delta_dependence.py --prepare-only
# 对上一步输出的suite，专用tmux主进程执行；包含GPU预检、训练、六套评估和报告
/home/students/HuangMingYu/anaconda3/envs/project/bin/python -u scripts/v24/run_delta_dependence.py --suite <suite>
```

DeltaDependenceCoE为显式构造入口：在CPU上按父模型原路径完成初始化，再在任何forward/optimizer之前设置communication=none并冻结history_gate。它不读取训练后权重；旧Y/Z构造器仍拒绝直接none配置。当前工作区旧默认行为保持不变，历史冻结快照完全只读。

新增每组的完整初始化与冻结父组initialization.pth逐值/哈希一致；另保存父配置diff，方法改动仅差分开关和显式构造标记。真实GPU验收4组×原生Top-2/强制TA-ST共8例，真实batch32，每例至少3次成功AMP更新；差分门和RMS差分函数若被调用立即报错，同时检查有界历史/交互分支的有限非零梯度。CPU恢复试验单独写临时目录。

所有run自动记录logs/gpu_status.jsonl，每15秒一次、训练退出即关闭；GPU预检日志位于suite/acceptance/logs/gpu_status.jsonl（通过相应监控run目录记录）。验收成功只是实现证据，不代表实验收益。
