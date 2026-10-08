import json,os,math,shutil,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/students/HuangMingYu/code/py/my_idea/my_idea');S=R/'outputs/v24-COE/experiments/backbone_exploration/taxibj/2abad0134c468ed3';O=R/'experments_report'
D=json.load(open('/tmp/w_analysis_data.json'));A=json.load(open('/tmp/w_audit_summary.json'));P=json.loads((S/'plan.json').read_text())
name='20261008_v24_W01-W30固定骨干探索_核验分析报告';assets=O/(name+'_assets');assets.mkdir(exist_ok=True);dest=O/(name+'.md')
keys=['in_distribution','unseen_combinations','unseen_geometry','unseen_triple'];labels=['ID','两元组合','未见形态','三元组合'];B=D['W01']
mae=lambda n,k='in_distribution':D[n]['evaluations']['sets'][k]['metrics']['mae']
pct=lambda a,b:100*(a/b-1)
f=lambda x:'—' if x is None else f'{x:.4f}' if isinstance(x,float) else str(x)
rel=lambda p:os.path.relpath(p,O)
L=[]
def add(t=''):L.append(t)
def tab(h,rr):
 add('| '+' | '.join(h)+' |');add('| '+' | '.join(['---']*len(h))+' |')
 for row in rr:add('| '+' | '.join(f(v) for v in row)+' |')
 add()
def fig(name):add(f'![{name}]({assets.name}/{name}.png)');add()
# Keep every original method/config/result/family/diagnostic table, replace generic analysis.
auto=Path((S/'report_path.txt').read_text().strip());text=auto.read_text().split('## 3. 结果分析、局限与建议')[0]
text=text.replace(text.splitlines()[0],'# 20261008_v24_W01–W30固定CMFF＋有界差分探索：完整核验与分析',1)
text=text.replace('训练/六协议测试完成30/30；确认评估完成。','日期：2026-10-08（北京时间）。队列`2abad0134c468ed3`；30/30训练完成，180/180常规协议评估及24/24确认评估完成。本报告直接核验冻结源码、配置、原始日志和best/last检查点。')
questions=[
'确定性基准性能是什么？','同seed全过程能否精确重放？','尺度身份是否足够？','轮次身份相对尺度身份如何？','尺度通道仿射是否改善共享？','轮次通道仿射是否更好？','统一额外容量是否已有效？','尺度专用容量是否有效？','轮次专用容量是否有效？','适配是否应依赖局部缺失程度？',
'可见位置监督本身是否有效？','可见误差点反馈是否有效？','邻域传播相对点反馈是否有益？','仅修正缺失位置是否更好？','反馈内容的梯度有何作用？','残差信息是否应服务于Router？','输入与Router反馈能否叠加？','可见监督能否帮助反馈？',
'额外缺失视图本身的作用？','一致性约束带来哪些额外收益？','EMA教师是否比同模型教师好？','观测误差加权能否保住精度？','区域等权结构监督是否有益？','按缺失点数加权有何影响？','时间邻边监督是否有效？','空间邻边监督是否有效？',
'尺度适配与反馈是否互补？','尺度适配与区域监督是否互补？','反馈与区域监督是否互补？','三个模块同时启用有何影响？']
start=text.index('| 组 | 具体方法');end=text.index('\n\n',start)
oldtab=text[start:end];new=['| 组 | 具体方法 | 直接参考 | 要检验的问题 |','| --- | --- | --- | --- |']
for i,n in enumerate(D):
 cfg=D[n]['config'];method=cfg['experiment_plan']['method'];ref=cfg['experiment_plan'].get('reference','W01')
 if int(n[1:])>=27:ref='/'.join(cfg['experiment_plan']['sources'])+'及W01'
 new.append(f'| {n} | {method} | {ref} | {questions[i]} |')
text=text[:start]+'\n'.join(new)+text[end:]
# Add reproducibility details before all numeric results.
marker='## 2. 全部结果、曲线、诊断与完整性'
extra='''### 1.1 实际骨干与数据流补充

输入是`[B,2,12,32,32]`可见流量及二值mask，隐藏通道64。仅读取F分辨率输入，在网络内部构建C8/M16/F32；保存配置中的`data.multiscale=false`关闭的是旧外部多尺度输入，并不表示W系列没有内部多尺度执行。

编码器读取可见值、mask、观测支撑与位置特征。第2轮起，以当前和上一轮状态的全局/缺失池化摘要产生零初始化的有符号门；差分按每窗口每通道RMS匹配当前状态，归一化比例停止梯度，门限制±0.1。第一轮无历史。随后进入选中尺度的状态投影与归一化、可选W适配、两位专家融合。粗输出上采样到F后直接传给下一轮。初始预测形成的completion输入保持不更新；`completion_feedback=false`并不删除初始completion条件。

T/S为3邻域时间/空间卷积，TD/SD为dilation2的相应卷积，TA为窗口内时间注意力，ST为联合时空操作，TL/SL为5邻域时间/空间卷积。八专家跨轮共享，四个Router分别读取动态状态。当前W实验没有自由尺度选择，所有窗口始终CMFF。

训练/验证/测试窗口数分别2452/342/707，batch32时分别77/11/23个batch，训练不丢弃末batch。使用既有文件划分，未重新切分时间；原始日历边界不由本轮日志直接给出。四基础族按均衡调度分配，训练每epoch重新排列/生成mask，测试固定。

每轮原生Top2及组内softmax融合，candidate负载均衡系数0.01，关闭中间补全监督和z-loss。W16/W17的balance采用实际修正后的个体logits，公式不变。W19–W22第二视图仅额外计算补全/一致性，不重复计入balance或结构损失。W20/W22教师是同模型原视图输出detach；只有W21使用独立EMA参数。

所有组从同一公共初始化模板从头训练，不加载入选组的训练权重。组合选择事前按ID验证冻结为A=W10、B=W13、C=W23；四项测试平均是用户后续提出的描述性比较指标，没有倒过来参与模型、epoch或组合选择。

'''
text=text.replace(marker,extra+marker)
add(text)
add('### 2.1 逐项独立核验与数值事件')
tab(['核验项目','实测结果'],[
 ['源码与数据指纹',f"冻结{A['source_count']}份文件全部匹配；数据文件SHA256全部匹配；本次检查当前源码与冻结版本无差异"],
 ['训练完整性','30×100=3000 epoch；每组20次验证；60份best/last检查点均存在、可读取，epoch和config匹配'],
 ['指标完整性','180项常规评估、24项确认评估；主训练/验证指标、全部评估数值、最终模型参数均有限'],
 ['公共初始化','30组公共状态SHA256相同；新增模块使用隔离初始化'],
 ['数据次序','30组每epoch原数据/mask顺序摘要完全一致；W19–W22额外视图mask摘要也逐epoch相同'],
 ['重放审计','W01/W02全部100 epoch的模型、优化器、AMP、调度器及主指标审计完全一致'],
 ['恢复状态','各组last中的模型/optimizer哈希匹配末epoch审计；W21 last保存EMA教师'],
 ['推理执行','所有常规评估均为每窗8次专家执行、CMFF路径比例1、面积代理4.625'],
 ['AMP数值事件',f"每组跳过{int(A['amp_skips_range'][0])}–{int(A['amp_skips_range'][1])}次更新，占7700次尝试约0.14%–0.17%"],
 ['唯一非有限诊断','W19 epoch2：coe_step4_router_grad_norm=inf；该epoch记录1次AMP跳步。主指标及最终参数正常；不能写成全过程无数值事件'],
 ['总耗时',f"训练/验证计时合计{A['training_hours']:.2f}小时；另有常规六协议评估约{A['test_hours']*60:.1f}分钟、确认评估约{A['confirmation_hours']*60:.1f}分钟。训练末自带test、保存和启动时间未全部包含在此求和中"]])
add('完整审计见['+assets.name+'/verification.json]('+assets.name+'/verification.json)。本次分析不启动、停止或重跑任何训练/评估。')
add('### 2.2 两种评价口径：ID优先与四协议平均')
add('四项平均A4=(ID MAE+两元MAE+形态MAE+三元MAE)/4，每个协议等权；两个复测mask不重复计权。这是协议宏平均，不是把四套样本混在一起重算MAE。ID占25%，三个迁移协议合计75%；平均RMSE也是各协议RMSE算术平均，不是合并样本RMSE。')
add('同时提供N4=mean(MAE方法,d/MAE基准,d)作为按基准归一化的敏感性检查，越低越好。它没有改变冻结的ID验证筛选规则。')
rank=sorted(D,key=lambda n:D[n]['mean4'])
tab(['综合名次','组','ID MAE','A4平均MAE','相对W01变化','四项平均RMSE','N4'],[[i,n,mae(n),D[n]['mean4'],f"{pct(D[n]['mean4'],B['mean4']):+.2f}%",D[n]['mean4_rmse'],D[n]['relative4']] for i,n in enumerate(rank,1)])
add('按A4、四项平均RMSE、N4三种口径，第一名均为W20，第二名均为W22。相对W20，W22的A4高3.60%，W30高8.82%。')
fig('four_protocol_ranking');fig('id_ood_tradeoff')
add('### 2.3 关键训练轨迹与最优epoch分布')
add('最优epoch分布：'+str(A['best_epoch_distribution'])+'；21/30组在95或100达到验证最优。后期仍有变化，但这并不自动证明延长训练或改变退火一定有益。')
selected=['W01','W10','W13','W19','W20','W21','W22','W23','W24','W26','W27','W28','W29','W30']
tab(['组','best epoch','best ID val','last ID val','last训练MAE','last相对best val退步'],[[n,D[n]['receipt']['best_epoch'],D[n]['receipt']['best_val_mae'],D[n]['history'][-1]['val']['mae'],D[n]['history'][-1]['train']['mae'],f"{pct(D[n]['history'][-1]['val']['mae'],D[n]['receipt']['best_val_mae']):+.2f}%"] for n in selected])
add('W20在80 epoch验证最优，100 epoch为14.586；W22在100 epoch最优14.023；W30到100 epoch才达到14.013。W08在85 epoch最优14.636，而last为17.526，存在明显后期波动。报告全部测试使用best，不能将last波动直接套到best测试成绩。不同组总loss含不同附加项，比较学习曲线主要使用MAE。')
fig('key_validation_curves')
add('### 2.4 专家路径、差分门与反馈实测')
pathrows=[]
for n in ['W01','W10','W13','W19','W20','W22','W23','W26','W27','W28','W29','W30']:
 m=D[n]['evaluations']['sets']['in_distribution']['metrics'];pairs=[(k[14:-9],v) for k,v in m.items() if k.startswith('coe_pair_path_') and k.endswith('_fraction') and k!='coe_pair_path_max_fraction'];top=max(pairs,key=lambda x:x[1])
 pathrows.append([n,top[0],f'{top[1]*100:.2f}%']+[f"{D[n]['evaluations']['sets'][k]['metrics']['coe_pair_path_max_fraction']*100:.2f}%" for k in keys[1:]])
tab(['组','ID最常见完整专家对路径','ID路径占比','两元最大路径占比','形态最大路径占比','三元最大路径占比'],pathrows)
add('W20的主路径为SD+TL → S+TA → TD+ST → S+SL；ID覆盖703/707个窗口，两元与三元协议覆盖707/707。此处统计的是离散专家对选择，不能据此认定组内softmax融合权重也完全恒定。不同协议表中的最大路径不保证同名，完整键值见原始评估。')
add('W22的ID最大路径占比38.05%，W30为91.65%；W30三元组合为99.43%。更多路径不自动代表更优补全；本批最高综合分恰好伴随高度集中的离散路由。')
fig('dominant_path_fraction')
add('基准后3轮差分门绝对均值为0.0897/0.0989/0.0949，接近0.1上限；W20为0.0823/0.0954/0.0972，W22为0.0878/0.0949/0.0865。门确实参与计算，但门接近上限不是差分有效性的独立证明，本批没有重新做无差分消融。')
add('W10各轮适配增量绝对均值约0.050/0.111/0.139/0.118；W13后3轮反馈增量约0.052/0.025/0.026，首轮为0。它们说明模块被使用；不能从这些数值直接推出隐藏信息更可靠或错误被正确修复。去均值MAE诊断已按绝对值修正，旧日志未改写；W系列未输出旧协调模块的分解诊断，因此本报告不据该项声称细节恢复。')
add('### 2.5 同分布逐缺失族对照')
families=['random_point','node_outage','temporal_gap','spatial_region']
tab(['组']+families,[[n]+[D[n]['evaluations']['sets']['in_distribution']['metrics'].get('coe_family_'+f+'_mae') for f in families] for n in ['W01','W09','W10','W13','W19','W20','W22','W23','W26','W30']])
add('ID指标对相同测试mask的有效缺失点统计，原始表包含每族误差与计数；不能把上述四族未经计数加权的均值当成总体ID的定义。W10/W19/W22四族均低于W01；W20随机点和时间缺失高于W01，节点和区域低于W01。W30节点缺失降至10.658，但随机点、时间缺失分别升至9.361、15.746。')
add('### 2.6 新mask确认及覆盖边界')
add('确认种子为20261111/12/13，测试时沿用+30000偏移；评估的是相同测试时间窗口上的不同mask。确认对象由冻结的ID验证选择决定：W01、W10、W13、W23及W27–W30。W19/W20/W22不在这次新种子确认对象中；它们有原六协议中的两套复测，但没有本轮新确认或第二训练seed。')
confirmed=[n for n in D if D[n]['confirmation']]
cb=D['W01']['confirmation']['sets']
tab(['组']+[x+'新mask MAE' for x in labels[1:]]+[x+'相对W01' for x in labels[1:]],[[n]+[D[n]['confirmation']['sets'][k]['metrics']['mae'] for k in keys[1:]]+[f"{pct(D[n]['confirmation']['sets'][k]['metrics']['mae'],cb[k]['metrics']['mae']):+.2f}%" for k in keys[1:]] for n in confirmed])
add('W30新mask上的两元/形态/三元MAE为16.356/20.413/15.325，相对W01改善约26.64%/20.30%/30.99%，保留了原测试上的改善方向。W10/W13/W23的组合缺失劣势也没有被新mask消除。')
add('## 3. 结果分析、证据边界与下一步建议')
add('### 3.1 本批回答了哪些问题')
add('**综合表现的最强实测信号来自第二视图训练，ID精度的最强结构信号来自覆盖率尺度适配。二者尚未被本批实验组合验证。** 本批没有产生三训练种子复验，所有新结果均为seed7；“可继续研究”不能写成“已确认稳定创新”。')
add('**W01/W02：工程复现已解决。** 100 epoch全过程一致，排除了本批已记录轨迹上的同seed重放差异。它不等于统计重复，不保证换GPU/软件栈仍逐值一致，也不能用来估计独立seed方差。')
add('**W03–W10：共享专家确实存在可改善的输入适配空间，但收益形式不同。** W03/W04轻量身份带来ID改善2.18%/1.70%；W06/W08改善主要OOD但ID退步。统一rank8适配W07的ID与主要OOD同时变差，因此“单纯加一个适配器”没有获得支持。W09轮次rank8使ID MAE改善3.96%、RMSE改善5.41%，但组合缺失明显变差。')
add('W10是本批ID MAE第一：12.820，验证改善4.30%，测试改善5.13%，四个基础缺失族均有改善。与W08相比仅新增6个覆盖率调制参数，却降低ID MAE8.53%；这支持进一步复验缺失程度条件化适配。与此同时，相对W08两元/三元MAE分别升高69.11%/90.30%，不能把它描述为普遍优于尺度适配；相对W01 A4反而高13.57%。W10与W09的MAE/RMSE排序不同，不能笼统声称W10所有ID指标均最好。')
add('**W11–W18：观测误差的作用位置很重要，当前反馈证据有限。** W13局部传播反馈是此族唯一达到本轮ID标准的组，ID改善2.17%；W12点反馈、W14仅缺失修正、W16路由反馈、W17双反馈分别偏向OOD改善但损失ID。W15打开残差内容梯度相对W14挽回部分ID，却让OOD转差；不能推断detach或可微其中一个普遍更好。W11可见重建监督ID退步16.42%，W18监督＋反馈退步16.95%，本权重和训练预算下不宜优先推进。')
add('**W19–W22：先分开数据增强与一致性的贡献。** W19相对W01，ID改善2.47%，主要OOD改善15.58%–27.19%，说明第二视图补全监督本身已经贡献很大。W20相对W19，三个主要OOD进一步改善12.53%–17.21%，但ID退步2.41%。因此一致性具有超出额外视图的迁移收益信号，但并未保留W19的最佳ID水平。W20相对W01的ID只改善0.11%，验证反而高0.27%，不满足原定ID信号标准。')
add('W22相对W20，ID改善1.98%，三项OOD约退步4.97%–5.32%；相对W19，ID略差0.39%，OOD改善7.88%–12.84%。所以W22提供的是精度/迁移权衡，不是全面击败普通一致性。其A4为15.630，相对W01降低25.53%，仅比W20高3.60%。当前证据不证明可见误差是隐藏预测置信度，也不证明启发式权重优于随机权重或调小普通一致性强度；这些是下一步需要隔离的问题。')
add('W21 EMA在当前0.99系数、无额外预热设置下明显失败：ID16.446，比W01高21.71%，同时弱于W20的全部四项。epoch100教师/学生共同缺失差异8.226，大于W20的3.957；这一观测与教师滞后假设相容，但没有干预证据确认原因，不应推广成“EMA一般不适合”。')
add('**W23–W26：区域和空间结构监督值得保留，时间邻边监督当前不成立。** W23区域等权使ID改善3.49%，但A4高于基准7.11%；W24改为按缺失数量加权后ID退步，OOD改善，说明区域/点权重并非等价。W26空间邻边监督无需额外模型参数/训练视图，ID改善1.38%，三个主要OOD改善约14%–15%，A4降低12.34%，是较简单且较平衡的候选。W25时间差分ID高5.58%左右、A4高38.28%，不建议沿当前系数继续叠加。')
add('### 3.2 最后四组：ID收益不能直接相加，OOD出现组合效应')
combo=['W01','W10','W13','W23','W27','W28','W29','W30']
tab(['配置','组','ID MAE','相对W01 ID','A4','相对W01 A4'],[[v,n,mae(n),f"{pct(mae(n),mae('W01')):+.2f}%",D[n]['mean4'],f"{pct(D[n]['mean4'],B['mean4']):+.2f}%"] for v,n in zip(['无','A','B','C','A+B','A+C','B+C','A+B+C'],combo)])
add('A=W10、B=W13、C=W23。以上八项在相同seed和共同初始化下形成完整的模块开关对照；它们没有独立重复，不能给交互效应赋予统计显著性。')
fig('combination_comparison')
add('W27/W29分别损失ID4.40%/5.41%，但A4改善21.05%/19.11%；W28仅改善ID0.89%，A4退步4.50%。最终W30的ID13.389，仅比W01改善0.92%，未达到预先设定的1%测试门槛，也弱于三个单模块的ID。应明确结论：**没有证据表明这三个模块叠加能进一步提升ID精度。**')
add('W30的A4为16.418，综合第三；相对W01改善21.78%，主要OOD改善19.88%–31.41%，且新mask确认保持。这与三个单模块的OOD劣势形成对照，提示组合会改变模型对缺失结构的适应方式。这里是实测组合效应，尚不能归因为“反馈纠错”“专家分工”或特定信息传播机制。')
add('W22在原四项MAE上全部低于W30，但W22使用额外视图、训练约95.8分钟；W30不扩展训练缺失视图，约81.2分钟，推理约5.77ms/窗，高于W22约5.40ms/窗。因此不能只看四项误差就声称W22在所有训练/推理成本上也全面占优。W30可以作为“原四基础族训练、纯结构改动”的备选，当前不必优先堆叠更多模块。')
add('### 3.3 最容易被误读的三件事')
add('1. **高综合分不证明自由路由是收益来源。** W20离散路径高度集中但综合第一。W22路径更分散却略逊于W20，两者训练损失不同，这不构成路径多样性的因果对照。固定主路径消融、相同训练机制下的共享CoE/独立MoE对照尚未完成。')
add('2. **第二视图不等于完全未见缺失泛化。** W19–W22会在基础缺失之上追加另一个基础族，可能暴露组合结构。实际测试mask没有泄漏给训练，但研究问题变成包含训练分布扩展的鲁棒性，不能再笼统声称所有组合均为零暴露。W30/W26未增加第二视图，比较时应分层报告。')
add('3. **事后四项平均不能倒写成事前选择目标。** 组合仍按ID验证选W10/W13/W23，因此没有包含W20/W22；这解释了最终组合为何没有沿综合第一的方向构建。不能根据测试平均再挑组合后，继续将同一套测试称作未使用的确认集。未来若转向综合目标，应先在验证时间段定义对应协议和权重，再冻结候选。')
add('### 3.4 候选分级与研究优先级')
passed=[n for n in D if n not in ('W01','W02') and D[n]['receipt']['best_val_mae']<B['receipt']['best_val_mae'] and mae(n)<=.99*mae('W01')]
add('按本轮原定“验证改善且ID测试MAE至少降低1%”标准，单种子候选为：'+', '.join(passed)+'。其中相对W01三个主要OOD都改善的是W19、W22、W26。W20/W30有综合价值，但不通过这一原ID门槛，不能更改门槛后假装事前达标。')
tab(['候选','支持推进的实测理由','主要限制','建议定位'],[
 ['W20','A4与平均RMSE第一；普通一致性强对照','ID未达1%；训练暴露扩展；离散路径几乎固定','综合表现的强基线，必须保留'],
 ['W22','ID与全部主要OOD改善；A4第二；接近W20','权重机制因果证据缺失；未做新确认/独立seed','优先验证的训练机制候选'],
 ['W10','ID MAE第一，四基础族均改善','组合缺失明显退步','ID结构分支候选'],
 ['W19','简单增强同时改善ID与OOD','不能把收益归给一致性；额外训练计算','必要的数据增强对照'],
 ['W26','不加参数/视图，ID和OOD均改善','综合收益低于视图组；单seed','低复杂度结构监督备选'],
 ['W30','无第二视图，A4第三，新mask保留OOD收益','ID弱于单模块，结构更复杂','纯结构/组合泛化备选'],
 ['W07/W18/W25','当前预算下ID和三项主要OOD均弱于W01','没有收益证据','不沿原配置继续扩展'],
 ['W21','当前EMA实现弱于W20全部四项','不能推广到所有EMA设置','暂不优先']])
add('没有任何组在本批中达到“跨训练种子稳定推进”的证据级别；也没有同协议外部baseline结果，不能声称胜过全部baseline、达到SOTA或创新性已获认可。当前数据支持选研究候选，而非预先保证论文创新成立。')
add('### 3.5 下一步：先验证收益来源，再扩大模型')
add('建议优先级为：')
add('1. **配对复验W01/W19/W20/W22，保留W10/W26作为结构分支。** 使用独立训练seed17/27、固定相同mask及公共初始化规则；同时报告逐seed差值。补充W19/W20/W22的新确认mask，但不重新按确认结果筛选。这是建议，当前未启动。')
add('2. **拆解W22权重信息。** 相同第二视图、相同损失系数和训练预算，比较均匀权重、窗口内打乱的原权重、仅覆盖率权重及当前观测误差权重；再比较普通一致性不同系数，避免把有效正则强度差异误当成权重信息收益。离线用验证隐藏标签分析权重与实际错误关系，不能把该标签输入权重头。')
add('3. **检验CoE是否必要。** 在匹配激活预算的共享CoE和独立多层MoE上分别加入W19/W20/W22，并报告参数与实测成本。对W20增加主路径固定化的受控对照；仅做测试时干预只能说明当前模型依赖，若要证明学习动态路径的优势还需要从头训练的固定路径对照。')
add('4. **若转向综合目标，先重设验证协议。** 在验证时间段固定ID/两元/形态/三元评估，以事前等权或明确业务权重选模型；测试保持独立。不能因为W20当前测试均值第一，就把它当作未经选择偏差影响的最终证据。')
add('5. **W10与W20/W22的组合可探索，但尚无结果。** 它与本轮W27–W30不同。必须分别对照两个单模块、保持教师/视图一致；只有超过对应单模块才讨论叠加收益。考虑先做小规模候选验证，不继续无目标铺开几十组。')
add('本轮报告至此完成。原始自动完整附表保留：['+auto.name+']('+auto.name+')。该文件及本报告均不替代原始日志和冻结配置。')
dest.write_text('\n'.join(L));print(dest)
# Compact artifacts, no model output mutation.
(assets/'verification.json').write_text(json.dumps(A,ensure_ascii=False,indent=2))
summary={n:{'val_mae':d['receipt']['best_val_mae'],'best_epoch':d['receipt']['best_epoch'],'mean4_mae':d['mean4'],'mean4_rmse':d['mean4_rmse'],'mean4_relative':d['relative4'],'train_seconds':d['seconds'],'peak_gib':d['peak'],'params':d['meta']['trainable_params'],'checks':d['checks'],'nonfinite':d['nonfinite'],'sets':{k:{a:e['metrics'][a] for a in ('mae','rmse')} for k,e in d['evaluations']['sets'].items()}} for n,d in D.items()}
(assets/'results_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(assets/'family_metrics.json').write_text(json.dumps({n:{k:{a:v for a,v in e['metrics'].items() if a.startswith('coe_family_')} for k,e in d['evaluations']['sets'].items()} for n,d in D.items()},ensure_ascii=False,indent=2))
(assets/'checkpoint_audit.json').write_text(json.dumps({n:d['checkpoints'] for n,d in D.items()},ensure_ascii=False,indent=2))
# Plots use English labels for portable fonts; method identifiers match all tables.
plt.rcParams.update({'font.size':9})
fig1,ax=plt.subplots(figsize=(9,9));rank=sorted(D,key=lambda n:D[n]['mean4']);colors=['#c64e3b' if n=='W01' else '#238b75' if n in ('W20','W22','W30') else '#7a9fba' for n in rank];ax.barh(rank,[D[n]['mean4'] for n in rank],color=colors);ax.invert_yaxis();ax.axvline(B['mean4'],ls='--',color='#c64e3b');ax.set_xlabel('Equal-weight mean MAE across four protocols');ax.grid(axis='x',alpha=.2);fig1.tight_layout();fig1.savefig(assets/'four_protocol_ranking.png',dpi=160);plt.close(fig1)
fig1,ax=plt.subplots(figsize=(9,6))
for n,d in D.items():
 x=mae(n);y=sum(mae(n,k) for k in keys[1:])/3;color='#238b75' if d['config']['model']['coe']['backbone_exploration'].get('view','none')!='none' else '#557fa1';ax.scatter(x,y,color=color)
 if n in ('W01','W10','W19','W20','W21','W22','W23','W25','W26','W27','W29','W30'):ax.annotate(n,(x,y),xytext=(5,4),textcoords='offset points')
ax.set_xlabel('ID test MAE (lower is better)');ax.set_ylabel('Mean MAE of three shifted-mask protocols');ax.grid(alpha=.2);ax.set_title('Green: second-view training; blue: original four-family exposure');fig1.tight_layout();fig1.savefig(assets/'id_ood_tradeoff.png',dpi=160);plt.close(fig1)
fig1,axs=plt.subplots(2,2,figsize=(13,8))
for ax,names in zip(axs.flat,[['W01','W03','W09','W10'],['W01','W19','W20','W21','W22'],['W01','W23','W24','W25','W26'],['W01','W27','W28','W29','W30']]):
 for n in names:
  h=[x for x in D[n]['history'] if x['val']];ax.plot([x['epoch'] for x in h],[x['val']['mae'] for x in h],label=n)
 ax.set_xlim(35,100);ax.set_ylim(12.5,23);ax.set_xlabel('Epoch');ax.set_ylabel('ID validation MAE');ax.grid(alpha=.2);ax.legend()
fig1.tight_layout();fig1.savefig(assets/'key_validation_curves.png',dpi=160);plt.close(fig1)
fig1,ax=plt.subplots(figsize=(10,5));x=np.arange(8);ax.bar(x-.18,[mae(n) for n in combo],.36,label='ID MAE');ax.bar(x+.18,[sum(mae(n,k) for k in keys[1:])/3 for n in combo],.36,label='Mean shifted-mask MAE');ax.set_xticks(x);ax.set_xticklabels([a+'\n'+b for a,b in zip(combo,['Base','A','B','C','AB','AC','BC','ABC'])]);ax.legend();ax.grid(axis='y',alpha=.2);fig1.tight_layout();fig1.savefig(assets/'combination_comparison.png',dpi=160);plt.close(fig1)
fig1,ax=plt.subplots(figsize=(8,4));names=['W01','W10','W20','W22','W30'];matrix=np.array([[D[n]['evaluations']['sets'][k]['metrics']['coe_pair_path_max_fraction'] for k in keys] for n in names]);im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlOrRd');ax.set_yticks(range(5));ax.set_yticklabels(names);ax.set_xticks(range(4));ax.set_xticklabels(['ID','Pair masks','Geometry','Triple masks'])
for i in range(5):
 for j in range(4):ax.text(j,i,f'{matrix[i,j]*100:.1f}%',ha='center',va='center',color='white' if matrix[i,j]>.75 else 'black')
fig1.colorbar(im,ax=ax);ax.set_title('Dominant complete expert-pair path fraction');fig1.tight_layout();fig1.savefig(assets/'dominant_path_fraction.png',dpi=160);plt.close(fig1)
