# X01–X10：历史池、ST三平面与双专家交互

用户授权的10次新训练，复用冻结核心批次K04 seed7为参考；X01与旧K02核对。主线仍为CMFF受控差分，本批候选不预先确立第三贡献。

## 方法与复现设置

| 编号 | 差分 | 扩展 | 关键对照 |
|---|---|---|---|
| X01 | 无 | 原direct链 | 旧K02：实现复现 |
| X02 | 无 | 每轮Hnext=H+U | X01：简单残差 |
| X03 | 有 | 历史只读上一轮 | K04：预测反馈 |
| X04 | 有 | 全部历史等权读取 | X03：完整历史 |
| X05 | 有 | 全部历史逐位置自适应读取 | X04：历史选择；X03：总效应 |
| X06 | 无 | 同X05的自适应历史 | X05：差分；X01：历史独立贡献 |
| X07 | 有 | ST三平面，有效视角等权 | K04：新增三平面分支 |
| X08 | 有 | 同X07，覆盖加权融合 | X07：覆盖融合 |
| X09 | 有 | 双专家拼接MLP | K04：新增非线性容量 |
| X10 | 有 | 双专家乘积/差值MLP | X09：交互表示 |

所有组TaxiBJ、seed7、random0.4；沿用四基础mask族random_point/node_outage/temporal_gap/spatial_region，训练逐epoch重采样，不改成纯随机点。输入[B,2,12,32,32]，hidden64，固定C/M/F/F=8/16/32/32网格，时间长度始终12。共享T/S/TD/SD/TA/ST/TL/SL八专家池，每轮原生动态Top2，组内softmax。无新增轮后FFN、额外视图、一致性、自由尺度、路由预热或搭档选择。尺度调度固定；专家选择动态。

100epoch，batch32，GPU0单卡串行，每5epoch验证，best只按ID验证MAE选，保留best/last，无早停。AdamW lr1e-3余弦至3e-4，wd1e-4，clip1，AMP，严格确定性。L1＋0.01 candidate负载均衡，无新增逐轮监督。模型/loader seed7，mask基础seed20260917，val+20000/test+30000；原数据时间切分不变。

差分与K04一致：H减上一轮输入Hprev，按样本/通道匹配THW RMS，比例detach、分母下限1e-6；共享条件门0.1*tanh且输出层零初始化。只修正池化前专家输入，Router读原H。无差分不是没有隐藏状态传递。X02在融合恢复细网格后执行H+U，四轮均相加，无可学习保留门、额外LN或缩放。

### 历史池：X03–X06

每轮解码后保存P_i=M*Xobs+(1-M)*prediction_i，detach。每次前向池子为空，第r轮只读第1至r-1轮；不跨样本/批次，不改变原始mask，不回填主干原completion通道，不把预测当真实观测。末轮输出仍取第四轮解码。主hidden链保持正常梯度，历史内容断梯度，不增加中间监督。

X03读取最近一个；X04等权平均；X05/X06按每个时空位置跨历史softmax。评分器8→16→1(1×1×1 Conv/GELU/Conv)，输入为P_i/s、最近P/s、绝对差、年龄(r-i)/3和双通道观测mask的均值。s=max(真实观测RMS,1)，每样本/数据通道计算；无观测时s=1。评分器末层零初始化，初始权重均匀。

四组同一历史注入器：[读取结果/s,原mask]→4→32→64，GELU，末层零初始化；delta_pool=0.1*RMS(H)*tanh(inject)*(1-mean_channel(M))，RMSdetach。空池完全跳过。注入池化前专家输入；有差分组加在差分修正后，Router规则不改。评分器跨轮共享、注入器跨轮共享。注入器2272参数，自适应组额外161参数；不宣称三种读取严格等有效容量。旧预测可能保留错误，自适应读取不保证逐轮更准确。

### ST三平面：X07/X08

仅替换ST的实现包装，保留原network的全部权重和网格分支。原前缀64→128/GELU后的特征旁路降至16通道；以当前尺度真实mask覆盖率的双通道均值q，分别沿W/H/T做覆盖加权投影，形成TH/TW/HW平面。分母为该轴覆盖总和，无覆盖投影置零。每平面独立3×3 depthwise Conv2D/GELU/1×1 Conv2D(16通道)，跨轮共享；邻域支持为平面覆盖图3×3平均，padding不计入分母。无邻域支持的平面位置输出置零。

结果沿压缩轴广播回native grid；X07对支持>0的视角等权，X08按支持归一化加权。全无支持权重全零。融合16通道映射回128，输出层零初始化，乘固定0.1后加在原ST的3D depthwise输出上，再经过原LN/GELU/输出投影。原网格分支始终保留。不是三个新路由专家；只有ST实际被选的样本才执行。X07/X08参数逐值同初始化，唯一差别是融合权重；三平面结构增益仍包含额外容量，不能仅凭这两组证明投影优于等容量3D扩展。

借鉴EG3D的平面表示组织方式，不移植生成/渲染任务：https://nvlabs.github.io/eg3d/ 。覆盖权重代表观测支持，不等于校准置信度。

### 双专家交互：X09/X10

两专家每样本只执行一次，按专家编号确定A/B顺序。保留原组内softmax加权和U和其累加精度。A/B分别无仿射通道LayerNorm；X09输入[Abar,Bbar]，X10输入[Abar*Bbar,Abar-Bbar]。共享跨轮MLP128→32→64，GELU，末层零初始化，每组新增6240参数。delta_pair=0.1*RMS(U)*tanh(MLP)，RMSdetach，Unew=U+delta_pair。交互发生在native scale，之后恢复细网格。初始等于原模型；不改融合权重、不重选搭档、不增加第三次专家执行。

借鉴乘性交互：https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Lin_Bilinear_CNN_Models_ICCV_2015_paper.pdf 。这是逐元素乘积与差值，不是完整外积双线性池化。参数匹配不代表所有算子FLOPs完全相同。

## 评估、报告与解释边界

原六套固定测试：ID、两元组合、未见形态、三元组合、两元mask复测、形态mask复测。基础种子20260917/18/19/30及20261001/02。有效缺失点MAE/RMSE与每族结果全部保留，MAPE不作为主要结论。综合分析同时列相对K04和X01的ID变化；三类主要OOD采用相对同一参考的平均变化，另列最差协议；两次mask复测单列，不重复计入三类平均。>5%退步显式标记。未设ID退步2%的硬门槛；不按测试挑checkpoint或改方案。

逐轮诊断在forward之后的指标计算中使用标签，不输入模型：每轮缺失区MAE/RMSE、改善/退步/持平比例(绝对误差差值阈值1e-6原单位)。修好/破坏比例用误差≤0.05*max(观测RMS,1)作诊断阈值，均以全部有效缺失点为分母；不是优化目标或单调保证。原harm/benefit幅度继续记录。旧参考没有新增逐轮指标时显示缺失，不捏造。

历史报告轮次权重和注入幅度；三平面报告ST选择率、覆盖、权重和旁路幅度(后三项仅在ST被选批次统计)；交互报告专家对占比及无条件交互幅度矩，除以占比可得到条件平均。路径、参数、训练/验证时间、推理时间和allocated显存分别报告。每窗口8次专家调用/面积代理4.625不包含额外分支算力。日志标量按现有batch平均约定，不冒充逐点精确统计；逐轮误差和改善比例按缺失点累计。

只有TaxiBJ seed7，不能证明跨训练种子/数据集稳定泛化或第三贡献成立。现有测试已被探索性使用，候选后续需独立确认。不把X05总收益全归因于差分；X05/X06才隔离差分。

## 队列、初始化与复现

公共参数逐值核对旧K04 seed7冻结初始化。新增历史/平面/交互各用隔离seed+105001/105002/105003；配对组同模块初值相同。X01/X02/X06冻结无用差分门。旧参考目录：outputs/v24-COE/experiments/core_validation/684cad05994af9a4。复制K04结果、配置、日志和初始化，以及旧K02结果/评估作冻结参考，不覆盖原输出。

```bash
conda activate project
python -m unittest discover -s tests -p test_v24_history_plane_interaction.py
CUDA_VISIBLE_DEVICES=0 python scripts/v24/preflight_history_plane_interaction.py --output /tmp/x_preflight.json
python scripts/v24/run_history_plane_interaction.py --preflight /tmp/x_preflight.json --dry-run
python -u scripts/v24/run_history_plane_interaction.py --preflight /tmp/x_preflight.json --gpu 0
# 已启动后只按冻结suite恢复，完整训练跳过、中断从last恢复，缺评估补齐：
python -u scripts/v24/run_history_plane_interaction.py --suite <suite> --gpu 0
```

固定X01→X10顺序，GPU0空闲且完整batch32验收通过后启动，不停止其他队列、不占GPU1、不自动减batch。冻结源码/数据指纹/配置/初始化/preflight。suite锁防重复启动。每组训练及六套评估后自动生成阶段报告；十组全部完成核验后按AGENTS.md和export-experiments-report技能生成正式报告到experments_report，顺序为方法复现→完整结果/曲线/诊断→分析/负结果/边界。
