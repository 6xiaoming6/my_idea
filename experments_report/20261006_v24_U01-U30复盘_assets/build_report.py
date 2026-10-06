import json,statistics,collections,os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];A=Path(__file__).resolve().parent;S=ROOT/'outputs/v24-COE/experiments/id_priority_exploration/taxibj/501f761e40b7e937';OUT=ROOT/'experments_report'
D=json.load(open(A/'review_data.json'));AUD=json.load(open(A/'audit.json'));SELECT=json.loads((S/'selection.json').read_text());J=json.loads((S/'resolved_jobs.json').read_text())
assert (S/'final_evaluations_complete.json').exists()
assert len(D)==30 and all(len(v['history'])==100 for v in D.values())
for n,v in D.items():
 assert v['evaluations']['status']=='finished' and len(v['evaluations']['sets'])==6
for n in json.loads((S/'final_evaluations_complete.json').read_text())['names']:
 assert D[n]['confirmation']['status']=='finished' and len(D[n]['confirmation']['sets'])==3
 assert D[n]['rate_transfer']['status']=='finished' and len(D[n]['rate_transfer']['sets'])==12
assert not AUD['source_mismatches'] and not AUD['data_changes'] and AUD['selection_hash_ok']
assert all(v['config_match'] and v['best_matches_val'] and not v['nonfinite_main'] and not v['nonfinite_diagnostics'] for v in AUD['groups'].values())
for n,a in AUD['groups'].items():
 for cp,z in a['checkpoints'].items():
  assert z['exists'] and z['config_match'] and z['stage_epoch']==z['epoch']
 for folder,z in a['protocols'].items():
  assert z['status']=='finished' and z['sets']==z['expected_sets'] and z['hash_match'] and z['config_match'] and z['checkpoint_match'] and z['best_matches'] and z['samples']==[707]
K=['in_distribution','unseen_combinations','unseen_geometry','unseen_triple'];REF=['U01','U29','U30'];FAMILIES={'受控差分':['U02','U19','U20'],'操作变化→Router':['U07','U21','U22'],'后两轮F的Top1':['U10','U23','U24'],'单调尺度教师':['U15','U25','U26'],'差分＋Top1':['U17','U27','U28']}
def metric(n,k=K[0],folder='evaluations',field='mae'):return D[n][folder]['sets'][k]['metrics'][field]
def change(n,r,k=K[0],folder='evaluations',field='mae'):return 100*(metric(n,k,folder,field)/metric(r,k,folder,field)-1)
def fmt(x):return f'{x:.3f}'
def pct(x):return f'{x:+.2f}%'
def mean(x):return statistics.mean(x)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n\n'
base=Path((S/'report_path.txt').read_text().strip()).read_text();assert '完整实验分析' in base.splitlines()[0]
prefix=base.split('## 3. ID优先分析')[0]
title='20261006_v24_U01-U30跨轮通信与尺度路由实验分析'
prefix='# '+title+'\n'+prefix.split('\n',1)[1]
prefix=prefix.replace('## 1. 各组具体做法与复现信息','## 1. 各组具体做法与复现信息')
questions={'U01':'新批次固定强参考','U02':'限制历史修正幅度是否有效','U03':'0.25与0.1的强度差异','U04':'RMS幅度匹配是否必要','U05':'操作内部变化能否替代状态差分','U06':'消息内容反传是否必要','U07':'信息更适合输入还是Router','U08':'保留末两轮F能否使Top1有效','U09':'固定训练后再开放能否稳定路由','U10':'固定阶段和末轮约束的共同作用','U11':'教师替代尺度任务直通是否更好','U12':'受约束路径下教师是否更好','U13':'学习双尺度选择和权重是否有效','U14':'匹配双尺度调用次数的固定参考','U15':'分辨率单调约束是否有效','U16':'尺度身份能否帮助共享专家','U17':'两个入选模块是否能共同改善','U18':'小幅执行后融合修正是否有效'}
def method(n):
 c=D[n]['config']['model']['coe']['id_priority'];comm=c.get('communication','none');p=c.get('scale_policy','fixed');co=c.get('constraint','none');warm=c.get('fixed_epochs',0)
 text={'fixed':'固定CMFF','st':'尺度Top1＋任务直通','teacher':'尺度Top1，仅教师训练头','top2':'每轮三选二尺度，组内softmax融合','fixed2':'固定CM/MF/CF/MF双尺度等权'}[p]
 if warm:text=f'前{warm}epoch CMFF，之后'+text
 if co=='last_two_fine':text+='；后两轮固定F（9路径）'
 if co=='monotone':text+='；分辨率不降、末轮F（10路径）'
 if comm!='none':text+='；'+{'delta':'RMS匹配H−Hprev','raw_delta':'原始H−Hprev','innovation':'RMS匹配上一轮U(E−Z)','innovation_detach':'同前消息内容detach','innovation_router':'操作变化摘要只修正Router'}[comm]+f'，幅度±{c.get("bound",.1)}'
 if c.get('scale_identity'):text+='；尺度身份向量'
 if c.get('response_fusion'):text+='；执行后双响应融合logit差±0.1'
 return text
start=prefix.index('| 组 | seed |');end=prefix.index('实际公共预算',start)
methods=[]
for n,d in D.items():
 q=questions.get(n,'同方法独立训练seed复验；检验稳定性')
 methods.append([n,d['config']['seed'],method(n),J[n]['reference'],q])
prefix=prefix[:start]+table(['组','seed','实际做法','配对参考','检验问题'],methods)+prefix[end:]
# Original table may call all contrasts single-factor; nuance is in analysis below.
sections=[]
def add(s=''):sections.append(s+'\n')
add('### 2.补充A 完整性复核与主结果配对')
add('30/30组均完成100epoch、20次验证，best/last共60个检查点存在且配置和阶段epoch相符。原六协议180项；18模型×3新mask确认=54项；18模型×12缺失率迁移=216项，合计450项模型—协议评估记录。所有评估均完成且配置、协议哈希、检查点路径和best epoch核对一致。训练/验证主指标与诊断无NaN/Inf记录，源码快照与数据size/mtime指纹未变化，未发现队列失败记录。')
add(f'AMP动态缩放累计跳过{sum(v["amp_skips"] for v in AUD["groups"].values()):.0f}个更新，每组11–13次，约占全部231000个训练batch的0.15%；这是有限的AMP跳步，不能把每组100epoch写成优化器更新次数严格完全相同。训练及常规验证日志计时合计{sum(v["receipt"]["total_time_sec"] for v in D.values())/3600:.2f}小时，不含全部写盘和额外450项测试的独立耗时。')
rows=[];paired={}
for name,ns in FAMILIES.items():
 vals=[change(n,r) for n,r in zip(ns,REF)];paired[name]={'groups':ns,'id_changes':vals,'id_mean_change':mean(vals)}
 rows.append([name,*[pct(x) for x in vals],pct(mean(vals)),fmt(mean(metric(n) for n in ns)),fmt(mean(metric(n,field='rmse') for n in ns)),'达到ID标准' if all(v<0 for v in vals) and mean(vals)<=-1 else '未达ID标准'])
add(table(['方法','seed7 ID变化','seed17','seed27','平均相对变化','平均MAE','平均RMSE','判定'],rows))
add(f'参考U01/U29/U30平均ID MAE={mean(metric(n) for n in REF):.3f}。相对变化先逐seed配对再平均；负数为改善。仅受控差分达到三seed均改善且平均至少1%的预注册ID标准，三seed不直接作统计显著性宣称。')
rows=[]
for name,ns in FAMILIES.items():
 for seed,n,r in zip((7,17,27),ns,REF):
  x=[change(n,r,k) for k in K[1:]];y=[change(n,r,k,'confirmation') for k in K[1:]]
  rows.append([name,seed,*[pct(v) for v in x],pct(mean(x)),*[pct(v) for v in y],pct(mean(y))])
add(table(['方法','seed','原两元','原形态','原三元','原平均相对','新两元','新形态','新三元','新平均相对'],rows))
add('原两元/形态复测种子20261001/02已在完整结果区列出；此表的新mask特指候选冻结后的20261101/02/03确认，不能与训练seed17/27复验混为一谈。平均相对OOD为三个协议MAE比值的等权均值。')
add('### 2.补充B 受控差分的分布边界')
rows=[]
for seed,n,r in zip((7,17,27),FAMILIES['受控差分'],REF):
 for rate in (.2,.6,.8):rows.append([seed,rate]+[pct(change(n,r,k+f'_rate{rate}','rate_transfer')) for k in K])
add(table(['seed','迁移缺失率','ID变化','两元变化','形态变化','三元变化'],rows))
rows=[]
for seed,n,r in zip((7,17,27),FAMILIES['受控差分'],REF):
 for k in K:
  x=D[n]['evaluations']['sets'][k]['metrics'];y=D[r]['evaluations']['sets'][k]['metrics']
  for key,v in x.items():
   if key.startswith('coe_family_') and key.endswith('_mae'):rows.append([seed,k,key[11:-4],fmt(y[key]),fmt(v),pct(100*(v/y[key]-1))])
add(table(['seed','协议','缺失族','参考MAE','受控差分MAE','变化'],rows))
add('### 2.补充C 历史强参考检查')
rows=[]
for new,old in zip(REF,['S01','S11','S12']):
 h=json.loads((S/'historical_evaluations'/f'{old}.json').read_text())
 rows.append([new,old]+[pct(100*(metric(new,k)/h['sets'][k]['metrics']['mae']-1)) for k in K])
for new,old in zip(FAMILIES['受控差分'],['S01','S11','S12']):
 h=json.loads((S/'historical_evaluations'/f'{old}.json').read_text())
 rows.append([new,old]+[pct(100*(metric(new,k)/h['sets'][k]['metrics']['mae']-1)) for k in K])
add(table(['当前组','历史同seed组','ID变化','两元变化','形态变化','三元变化'],rows))
add('旧S01/S11/S12与本批U01/U29/U30均为固定CMFF，但不是同一次训练。核对两个冻结目录发现：公共src文件差异仅registry，新U模型为新增模块，训练器与评估器源码相同；配置除id_priority入口开关和实验元数据外一致。新模块使用隔离初始化随机流，并新增阶段状态；固定前向回归已通过，但未保证GPU训练数值轨迹逐值相同。本次没有定位历史OOD差异的因果来源，不能将它简单归结为mask变了，也不能据此抹去本批同协议配对结果。')
legacy=[]
for name,relative in [('N5','reference_evaluations/reference_N5.json'),('R5','evaluations/R5.json'),('R6','evaluations/R6.json')]:
 p=ROOT/'outputs/v24-COE/experiments/r_exploration/taxibj/b21d7928e67f4565'/relative
 x=json.loads(p.read_text());legacy.append([name,7]+[fmt(x['sets'][k]['metrics']['mae']) for k in K]+['[原始记录]('+os.path.relpath(p,OUT)+')'])
add(table(['历史方法','seed','ID MAE','两元','形态','三元','来源'],legacy))
add('N5是双尺度固定MMFF（旧命名CCFF），R5是早期双尺度路径扰动，R6是独立专家固定双尺度。这里只列统一补评后的seed7背景，不能替代本批三seed参考或当前新mask确认。受控差分seed7的ID比N5低，但不及R6；R5仍有较强的两元/三元成绩。')
add('### 2.补充D 训练轨迹与计算成本')
rows=[]
for n in ['U01','U02','U03','U04','U10','U11','U12','U13','U14','U15','U16','U17','U18','U19','U20','U27','U28','U29','U30']:
 z=D[n];m=z['evaluations']['sets'][K[0]]['metrics'];rows.append([n,fmt(z['receipt']['total_time_sec']/60),fmt(m['forward_ms_per_sample_per_rank']),fmt(m['coe_expert_grid_equivalents']),int(m['coe_expert_execution_count']),z['metadata']['total_params'],z['metadata']['trainable_params']])
add(table(['组','训练含验证min','测试前向ms/窗口','面积代理','专家次数','总参数','可训练参数'],rows))
add('前向ms来自同批记录，受具体专家组合、运行时硬件状态影响，不是严格隔离的性能基准。总参数包括固定组注册但冻结的尺度头；评估参数成本时同时看可训练参数。面积代理不含通信/路由/教师成本，不等于FLOPs。')
fig,ax=plt.subplots(1,2,figsize=(12,4.8));names=list(FAMILIES);positions=list(range(5))
for i,seed in enumerate((7,17,27)):
 ax[0].bar([p+(i-1)*.24 for p in positions],[paired[name]['id_changes'][i] for name in names],width=.24,label=f'seed {seed}')
ax[0].set_xticks(positions,['Delta .1','Innovation router','Top1 FF','Monotone teacher','Delta + Top1'],rotation=15);ax[0].axhline(0,color='black',lw=.7);ax[0].set_ylabel('Paired ID MAE change (%)');ax[0].legend();ax[0].grid(axis='y',alpha=.2)
for seed,n,r in zip((7,17,27),FAMILIES['受控差分'],REF):
 ax[1].plot([.2,.4,.6,.8],[change(n,r,'in_distribution_rate0.2','rate_transfer'),change(n,r),change(n,r,'in_distribution_rate0.6','rate_transfer'),change(n,r,'in_distribution_rate0.8','rate_transfer')],marker='o',label=f'seed {seed}')
ax[1].axhline(0,color='black',lw=.7);ax[1].set_xlabel('Missing rate');ax[1].set_ylabel('Delta .1: ID MAE change (%)');ax[1].legend();ax[1].grid(alpha=.2);fig.tight_layout();fig.savefig(A/'paired_id_and_transfer.png',dpi=160);fig.savefig(A/'paired_id_and_transfer.svg');plt.close(fig)
add(f'![三种子ID配对与差分迁移]({A.name}/paired_id_and_transfer.png)')
fig,axs=plt.subplots(2,3,figsize=(14,8))
for i,(n,r) in enumerate(zip(FAMILIES['受控差分'],REF)):
 for z,style in [(r,'--'),(n,'-')]:
  hs=D[z]['history'];axs[0,i].plot([h['epoch'] for h in hs if h['val']],[h['val']['mae'] for h in hs if h['val']],style,label=z)
 hs=D[n]['history']
 for step in (2,3,4):axs[1,i].plot([h['epoch'] for h in hs],[h['train'].get(f'coe_step{step}_memory_gate_abs_mean',0) for h in hs],label=f'round {step}')
 axs[0,i].set_title(f'seed {(7,17,27)[i]}: ID validation');axs[1,i].set_title(n+': absolute gate');axs[1,i].axhline(.1,color='black',ls=':',lw=1)
 for row in (0,1):axs[row,i].set_xlabel('Epoch');axs[row,i].grid(alpha=.2);axs[row,i].legend()
fig.tight_layout();fig.savefig(A/'delta_curves_and_gates.png',dpi=160);plt.close(fig)
add(f'![受控差分验证曲线与门幅度]({A.name}/delta_curves_and_gates.png)')
add('best epoch分布为100轮13组、95轮11组、90轮3组、85轮3组，全部晚于尺度开放的第21轮。因此本批没有“best发生在开放前却误称路由收益”的问题。部分组在末期仍改善，部分已经回升，例如U19在85轮best val13.664、100轮14.317；U12在85轮15.814、100轮16.365；不能统一把100epoch末轮当最优。')
add('另一个重要实测：同seed、前20epoch同为固定CMFF的组，epoch20验证仍不同，例如U01=28.744、U09=26.973、U10=30.084。初始化一致和相同数据协议不等于GPU训练全程数值确定；这些组并非从同一个20epoch检查点分叉。后文对阶段训练的机制解释因此仅作受控实验线索，不作严格因果归因。')
add('## 3. 结果分析、证据边界与下一步建议')
add('### 3.1 本轮确立了什么：一个通过三种子ID标准的通信候选，而非全面胜出的自由尺度方案')
add('最明确的正结果是U02/U19/U20：固定CMFF上加入RMS匹配差分、将有符号门幅度限制到±0.1，三个seed ID分别改善5.02%/4.48%/2.64%，平均相对改善4.04%，MAE均值从14.012降至13.447。RMSE也逐seed降低。它达到了本轮事先确定的ID推进标准，是可以保留为下一轮主方法的候选。')
add('但这句话的范围必须明确：它稳定超过本批新参考，并不等于已经超过所有历史强模型，也不等于在所有缺失模式和缺失率下都改善。自由Top1、单调尺度教师、操作消息Router和通信＋自由尺度均没有通过三seed ID标准。')
add('### 3.2 为什么受控差分值得保留，哪些机制还没被证明')
add('U02比U03的ID更好（13.356对13.762），三个主要OOD也更好；两者只改门幅度0.1/0.25。这是当前强度控制有价值的单seed证据。U04取消RMS匹配后ID为13.828，弱于U02，但两元/三元MAE16.394/15.834优于U02的19.396/19.768。因此RMS匹配更有利于这次ID目标，不是所有指标的普遍最优选择，U03/U04均未做三seed复验。')
add('门的绝对值在三个seed的后续轮次约0.086–0.097，接近0.1上限；有符号均值约0.017–0.058，没有证明所有通道都取正值或所有窗口都使用相同方向。不能将其笼统称为统一动量，也不能仅凭门非零就宣称输入自适应有贡献。由于残差先按H的RMS匹配，±0.1约束的是每通道整体RMS意义下的历史修正幅度，而不是每个位置的逐点相对误差上界。')
add('下一步最需要区分：收益来自小幅历史残差本身，还是来自条件门、归一化或历史梯度。现有实验没有固定小系数和输入无关可学习门的配对对照，所以不能把复杂门网络的必要性作为已证实创新。')
add('### 3.3 泛化收益是真实的，但仍存在明确代价')
add('受控差分在rate0.4原三项OOD平均相对改善23.67%/29.73%/2.56%，新mask确认改善23.48%/28.66%/3.13%，复测并未推翻总体方向。但seed27未见形态从28.291升至30.673（+8.42%），新mask仍+6.91%，超过本轮必须披露的5%风险线。平均OOD改善不能隐藏这一子协议退步。')
add('同分布四基础族也不均匀：seed27随机点缺失MAE退步14.93%，节点/时间/区域则改善1.92%/5.28%/8.80%；seed7随机点小幅退步1.27%。因此本轮ID总体改善并不保证各个ID缺失族都改善。')
add('缺失率迁移进一步收窄适用范围：seed7在0.2/0.6/0.8的ID均改善；seed17在0.6/0.8反而退步4.40%/3.57%；seed27在0.6/0.8退步6.46%/6.66%，且0.2下两元/形态/三元分别退步28.60%/10.78%/12.60%。目前可以提出“训练缺失率0.4附近的稳定ID改进”，不能写成跨缺失率全面鲁棒。')
add('### 3.4 操作内部变化暂未优于简单状态差分')
add('U05/U06把消息改成上一轮实际专家融合输出减真实输入，即U(E−Z)，ID为14.865/14.890，相对U01分别退步约5.71%/5.90%；detach没有恢复ID收益。其OOD比新基准好，但本轮ID优先，不能靠另换指标把它们重新包装成赢家。')
add('U07只将消息用于Router，seed7 ID几乎持平；复验seed17/27分别退步1.49%/17.85%，三seed平均退步6.34%。这一实现没有显示比修改专家输入更可靠。U07因要求第二通信候选来自不同机制族而补位入选，ID验证初筛本来就没有通过，报告必须保留这一事实。结果否定的是当前消息和注入方式，不是所有操作级通信。')
add('### 3.5 自由尺度Top1和尺度教师仍未形成稳定的输入条件选择')
add('U10/U23/U24的ID变化为−1.70%/−1.21%/+11.95%，平均退步3.01%；U15/U25/U26为−3.03%/+7.43%/+15.73%，平均退步6.71%。这两类已经做完预定复验，不能继续只强调seed7的改善。')
add('ID测试路径更说明问题：U10/U23均100%走CMFF，U24全走CCFF；U15/U25/U26均100%走CMFF。U08全走MMFF，U09全走CMMF。最终表现大多是每个模型固定一条路径，而非针对输入选择尺度。U15的10条合法路径与U10的9条路径也必须称为受约束路由，不能描述为完整81路径自由选择。')
add('教师并未普遍改善：U11对无约束U09的ID更好（14.228对15.062），但在后两轮F条件下U12反而弱于U10（15.051对13.822）；U16加尺度身份后14.420，虽比U12好仍弱于基准14.061。单调教师在seed7有信号，复验失败。头有梯度、合法候选标签正确只能证明实现工作，不能证明该监督找到了有效尺度。')
add('相同最终CMFF并不保证相同训练模型。阶段开放、候选教师和代理梯度改变了训练过程；同时前20epoch的非完全一致轨迹构成归因限制。若仍研究自由尺度，先从同一20epoch检查点分叉固定/可学习分支，记录全过程路径和梯度，再考虑继续扩大结构搜索。')
add('### 3.6 组合没有稳定超过单独通信')
add('U17的seed7 ID13.026比U01改善7.36%，也比U02好2.47%；但U27的seed17为14.252，比基准退步5.09%，显著弱于单独通信12.954；U28的seed27为14.765，比基准退步2.45%，也弱于单独通信14.032。三seed平均ID相对变化约+0.06%，基本没有净收益。')
add('三个组合模型ID测试都100%走CMFF。因此不能把seed7的好结果归因于测试时条件化尺度选择，也不能声称通信和尺度路由存在稳定协同。组合在seed7/27仍有OOD改善，但本轮以ID优先，最终主方法应先保留单独通信。')
add('### 3.7 双尺度组合与响应融合：保留独立信号，不提升为稳定主线')
add('U13三选二尺度学习，相对U14固定尺度对等权的ID改善8.81%，两元/形态/三元改善17.28%/23.19%/20.22%。相对U01，ID改善1.53%，主要OOD明显改善，是本轮值得保留的第二类正信号。ID路径约84.16%为FC→FM→FC→FC，约15.42%为FC→MC→FC→FC，有真实的执行尺度组合差异。')
add('代价也明确：16次专家执行对8次，实际面积代理8.572对4.625；测试前向约8.56对5.05ms/窗口，训练75.6对51.4分钟，训练峰值约14.13对8.28GiB。U13相对U14虽然调用次数相同，面积仍为8.572对7.750；而且同时改变尺度对和融合权重，所以不能单独归因为“选对更好”或“权重更好”。目前只有seed7，应归为精度/成本权衡候选，需要匹配成本参考后复验。')
add('U18有界响应融合的ID13.406，相对U01改善4.66%，三个主要OOD也改善，但没有进入本轮复验安排，不能据此确定其稳定性。它改变的是已选两专家的融合，不是找到优于原生Top2的专家组合。')
add('### 3.8 历史参考差异是重要的证据边界')
add('本批新CMFF参考的OOD比历史同seed S01/S11/S12明显差：两元约+49%至+74%，形态+32%至+45%，三元+50%至+71%。新参考的ID波动则约−4.82%至+7.07%。这说明OOD对训练轨迹有很强敏感性，也是必须保留历史强参考的原因。')
add('受控差分相对历史CMFF的ID变化为−2.75%/−9.08%/+4.25%，不是三个seed全胜；seed27三个主要OOD均大幅差于历史S12。故目前能确定的是本批配对ID改进，不能宣称全项目最好或已经完全解决缺失泛化。原N5/R6等旧强模型也不能因本轮结果好而省略，但不同历史预算/初始化对照不应冒充同批公平比较。')
add('本次已复核冻结源码、配置、mask种子、输入数据指纹及评估检查点，未发现测试协议被改写。新初始化随机流管理和GPU数值轨迹与历史运行并不等同；没有直接因果实验，不能把差异全归咎于某一个实现细节或简单称为随机误差。后续在引入新机制前，应增加固定CMFF的可重复性与严格数值控制检查。')
add('### 3.9 当前论文主线与最小下一步')
add(table(['分级','方法','当前可说的结论'],[['稳定ID推进候选','CMFF＋±0.1 RMS差分','三seed ID均改善，平均4.04%；有形态/迁移风险，尚未全面超过历史强参考'],['单seed精度/成本信号','U13三选二尺度组合','优于固定双尺度等权，但激活与面积成本更高，待复验'],['单seed融合信号','U18有界响应融合','改善当前seed7，不等于改进专家选择'],['未见稳定ID收益','U07 Router、U10 Top1、U15教师、U17组合及各自复验','不纳入当前默认模型'],['未支持的机制主张','逐轮单调补全、按输入自由尺度、自适应门必需、天然泛化','需要专门对照，不能由当前指标直接推出']]))
add('建议将当前方法候选固定为“四轮共享Top2＋CMFF＋受控跨轮状态差分”，论文问题收敛到“在多尺度迭代补全中，怎样传递有限幅度的历史变化来改善下一轮计算”。这是一条有实验支持的研究方向，但关键创新究竟来自条件通信、幅度控制还是一般跳连，仍要通过以下小规模消融确定；不以复杂模块数量代替新颖性论证。')
add(table(['优先级','具体对照','隔离的问题'],[['1','同批CMFF重复运行；记录初始化、loader/mask、AMP跳步与GPU确定性','核查当前与历史参考漂移，建立可靠比较底座'],['2','±0.1条件门 vs 固定小系数 vs 输入无关可学习通道门，保持差分定义和共享初始化','条件化通信是否必要，还是小幅跳连就足够'],['3','U02完整梯度 vs 只detach差分内容；门特征梯度保持','历史梯度是否参与收益'],['4','U02与U04配对seed17/27；再考虑幅度0.05/0.1/0.25的有限复验','RMS匹配和强度的ID/OOD权衡，避免同时改多个因素'],['5','U13跨seed复验＋固定学习到的尺度对/可学习权重分离对照','是否值得用额外尺度执行换收益，以及选择/融合各自贡献']]))
add('不要立即把自由尺度与通信重新打包成主模型，也不需要继续大范围搭档评分/GRU搜索。只有简单通信通过上述机制消融后，再扩展到第二数据集、额外时间切分与未参与研发的新缺失协议。新mask确认仍使用同一测试时间窗，缺失率迁移没有重新训练，也不是跨数据集泛化。三seed只作配对重复证据，不宣称统计显著；MAPE不作为主结论。')
add('本次只分析既有结果和导出报告，没有启动、停止或重跑任何实验。原队列自动完整报告保留。')
add(f'复核附件：[审计]({A.name}/audit.json)、[配对统计]({A.name}/paired_results.json)、[数据提取脚本]({A.name}/collect_review.py)、[报告生成脚本]({A.name}/build_report.py)。')
json.dump(paired,open(A/'paired_results.json','w'),ensure_ascii=False,indent=2)
# Preserve original full results and diagnostics; append reviewed tables before analysis.
result=prefix+'\n'+'\n'.join(sections)
dest=OUT/(title+'.md');dest.write_text(result)
print(dest)
