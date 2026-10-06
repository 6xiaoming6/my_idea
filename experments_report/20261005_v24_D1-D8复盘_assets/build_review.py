from pathlib import Path
import json,math,hashlib,statistics as st,re
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experments_report';A=Path(__file__).parent
P=ROOT/'outputs/v24-COE/experiments/scale_memory_followup/taxibj/018c3ec72556bd07'
rows=json.loads((P/'summary.json').read_text());R={r['variant']:r for r in rows};K=['in_distribution','unseen_combinations','unseen_geometry','unseen_triple']
for f in (P/'references').glob('*.json'):
 d=json.loads(f.read_text());R[f.stem]={'evaluations':{k:v['metrics'] for k,v in d['evaluation']['sets'].items()}}
def table(h,rs):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rs])+'\n\n'
def met(n,folder=None):return R[n]['evaluations'] if folder is None else {k:v['metrics'] for k,v in json.loads((P/folder/f'{n}.json').read_text())['sets'].items()}
def change(a,b,folder=None,rate=None):
 x,y=met(a,folder),met(b,folder);keys=K[1:] if folder=='confirmation' else K
 if rate:keys=[k+'_rate'+rate for k in keys]
 return [100*(x[k]['mae']/y[k]['mae']-1) for k in keys]
pairs=[('D1','S01'),('D4','S11'),('D7','S12'),('D2','S01'),('D5','S11'),('D8','S12'),('D3','D2'),('D6','D5'),('D3','S01'),('D6','S11'),('D3','D1'),('D6','D4'),('D3','G12'),('D2','G12')]
paired=[{'a':a,'b':b,'test':change(a,b),'confirmation':change(a,b,'confirmation'),'rates':{v:change(a,b,'rate_transfer',v) for v in ['0.2','0.6','0.8']}} for a,b in pairs]
(A/'paired_results.json').write_text(json.dumps(paired,ensure_ascii=False,indent=2))
H={};audit={};ev={}
for r in rows:
 n=r['variant'];run=Path(r['run_dir']);hs=[json.loads(l) for l in (run/'logs/metrics.jsonl').read_text().splitlines()];H[n]=hs
 ev[n]=json.loads((P/'evaluations'/f'{n}.json').read_text())['sets']['in_distribution']['metrics']
 bad=[]
 for h in hs:
  for sp in ['train','val']:
   for k,v in (h.get(sp) or {}).items():
    if (isinstance(v,str) and v in ['inf','-inf','nan']) or (isinstance(v,float) and not math.isfinite(v)):bad.append([h['epoch'],sp,k,v])
 audit[n]={'epochs':len(hs),'validations':sum(bool(h['val']) for h in hs),'best_exists':(run/'checkpoints/best.pth').exists(),'last_exists':(run/'checkpoints/last.pth').exists(),'nonfinite':bad,'amp_skips':sum(h['train'].get('train_skipped_amp_steps',0) for h in hs)}
 assert len(hs)==100 and audit[n]['validations']==20
plan=json.loads((P/'plan.json').read_text());mismatches=[]
for name,sha in plan['source'].items():
 for label,base in [('snapshot',P/'source_snapshot'),('current',ROOT)]:
  f=base/name
  if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=sha:mismatches.append([label,name])
counts={}
for folder in ['evaluations','confirmation','rate_transfer']:
 files=list((P/folder).glob('*.json'));total=0
 for f in files:
  d=json.loads(f.read_text());assert d['status']=='finished'
  for k,s in d['sets'].items():
   total+=1
   for metric in ['mae','rmse']:assert math.isfinite(s['metrics'][metric])
 counts[folder]={'files':len(files),'sets':total}
(A/'audit.json').write_text(json.dumps({'runs':audit,'coverage':counts,'source_count':len(plan['source']),'source_mismatches':mismatches},ensure_ascii=False,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig,axs=plt.subplots(1,2,figsize=(13,4.8));selected=paired[:8];xx=np.arange(len(selected))
axs[0].bar(xx-.18,[z['test'][0] for z in selected],.36,label='ID test');axs[0].bar(xx+.18,[st.mean(z['test'][1:]) for z in selected],.36,label='Mean OOD');axs[1].bar(xx,[st.mean(z['confirmation']) for z in selected],.6,color='#ca7040',label='New-mask mean OOD')
for ax in axs:
 ax.set_xticks(xx);ax.set_xticklabels([z['a']+'/'+z['b'] for z in selected],rotation=40,ha='right');ax.axhline(0,color='black',lw=.8);ax.set_ylabel('MAE change (%); lower is better');ax.grid(axis='y',alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(A/'paired_changes.png',dpi=170);fig.savefig(A/'paired_changes.svg');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for n in ['D1','D3','D4','D6','D7']:
 h=H[n];axs[0].plot([v['epoch'] for v in h if v['val']],[v['val']['coe_step3_memory_gate_abs_mean'] for v in h if v['val']],label=n)
for n in ['D2','D3','D5','D6']:
 h=H[n];axs[1].plot([v['epoch'] for v in h if v['val']],[v['val']['mae'] for v in h if v['val']],label=n)
axs[0].set_title('Round 3 gate: mean absolute value');axs[0].set_ylim(0,1.05);axs[1].set_title('ID validation MAE');axs[1].set_ylim(10,45)
for ax in axs:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(A/'gates_and_curves.png',dpi=170);plt.close(fig)
old=OUT/'20261005_v24_固定尺度与跨轮差分8组_完整实验分析.md';txt=old.read_text();body=txt.split('## 3. 配对分析、局限与建议')[0]
body=body.replace('# 20261005_v24_固定尺度与跨轮差分8组_完整实验分析','# 20261005 v24 D1–D8：固定尺度顺序与跨轮差分复核',1)
method='''### 1.1 本轮要回答的问题与代码核对

| 问题 | 直接配对 | 证据覆盖 |
| --- | --- | --- |
| 单F上的M07收益能否迁移到强CMFF骨干？ | D1/S01、D4/S11、D7/S12 | 三训练seed |
| G12的MCFF顺序是否本身优于CMFF？ | D2/S01、D5/S11、D8/S12 | 三训练seed |
| MCFF中差分是否稳定有效？ | D3/D2、D6/D5 | 两训练seed，无seed27 |
| 最终同走MCFF是否就等于相同方法？ | D3/G12、D2/G12 | seed7，训练策略不同，非严格机制消融 |

实际输入为TaxiBJ双通道窗口[B,2,12,32,32]。原生Router从当前隐藏状态提取特征并选Top2；差分只校正专家输入，不改本轮Router输入，输出仍direct替换。历史Hprev为上一轮输入，H为上一轮专家执行后的状态，均存于细网格张量。粗尺度由当前状态及可见观测/覆盖率构造，未读隐藏真值。completion feedback关闭，初始补全通道并不随各轮预测更新。

差分门共享MLP读取当前/历史的全局和缺失位置摘要，每通道输出tanh门；没有额外小系数。Δnorm=(H−Hprev)×stopgrad(RMS(H)/max(RMS(H−Hprev),1e−6))。因此零初始化只保证起点不变，后续门允许幅度接近1；它不是始终维持小步长的残差模块。

原生固定尺度组总参数309687、可训练250043；差分组330807、可训练271163。固定尺度头虽冻结仍计入总参数，可训练增加21120（8.45%）。两类专家激活次数均8，网格面积代理均4.625；额外门控并非零计算成本。最优检查点均只由ID val选取，没有用OOD挑epoch。

复核368份冻结清单文件：源码快照及当前文件均匹配SHA256。数据指纹为文件大小/mtime；旧参考best.pth另保存内容哈希。此次只读取结果和写报告，没有新训练、重测或改模型。原报告保留，新增报告纠正其“未通过/未完成”合并措辞：本批确认评估已全完成，未通过是效果门槛未通过。

'''
body=body.replace('## 2. 结果、训练曲线与诊断',method+'## 2. 结果、训练曲线与诊断',1)
lines=[body,'### 2.1 完整性、耗时及异常复核\n',
'2026-10-05 01:47启动，09:03:55完成最后评估，墙钟约7小时17分，在上午10点前完成。训练程序计时合计6.893小时（训练+验证，不含写盘和额外评估）。8组各100epoch、20次验证，best/last均存在；8份六协议测试、12份三协议新mask确认、12份十二协议缺失率迁移全部finished。主指标和诊断未发现NaN/Inf；AMP共跳过93步，占61600训练batch约0.151%，不应描述为完全没有数值溢出。\n\n',
'### 2.2 按完整种子组汇总\n\n下表为三训练seed MAE均值。MCFF+差分只有两seed，另表单列，避免不等种子均值直接排名。\n\n']
groups=[('原生CMFF',['S01','S11','S12']),('CMFF+差分',['D1','D4','D7']),('原生MCFF',['D2','D5','D8'])]
lines.append(table(['结构','种子数','ID','两元组合','未见形态','三元组合'],[[name,len(ns)]+[f'{st.mean(met(n)[k]["mae"] for n in ns):.3f}' for k in K] for name,ns in groups]))
lines.append(table(['seed','结构','ID','两元组合','未见形态','三元组合'],[[seed,name]+[f'{met(n)[k]["mae"]:.3f}' for k in K] for seed,name,n in [(7,'MCFF','D2'),(7,'MCFF+差分','D3'),(17,'MCFF','D5'),(17,'MCFF+差分','D6')]]))
lines.append('### 2.3 配对变化与新mask复测\n\n负值为改善，正值为退步；平均OOD先求三个协议各自的相对变化再平均，不等于绝对MAE均值的变化。\n\n')
lines.append(table(['方法/参考','ID变化','两元变化','形态变化','三元变化','平均OOD变化','新mask平均OOD变化'],[[z['a']+'/'+z['b']]+[f'{v:+.2f}%' for v in z['test']]+[f'{st.mean(z["test"][1:]):+.2f}%',f'{st.mean(z["confirmation"]):+.2f}%'] for z in paired]))
lines.append(f'![配对变化]({A.name}/paired_changes.png)\n\n')
lines.append('### 2.4 缺失率迁移配对复核\n\n每格为ID变化 / 平均OOD变化；只改变测试mask，没有重训。\n\n')
lines.append(table(['方法/参考','rate0.2','rate0.6','rate0.8'],[[z['a']+'/'+z['b']]+[f'{z["rates"][v][0]:+.2f}% / {st.mean(z["rates"][v][1:]):+.2f}%' for v in ['0.2','0.6','0.8']] for z in paired]))
lines.append('### 2.5 门控和训练轨迹的补充记录\n\n')
lines.append(table(['组','第2轮mean(g)/mean(abs(g))','第3轮','第4轮'],[[n]+[f'{ev[n][f"coe_step{i}_memory_gate_signed_mean"]:.4f} / {ev[n][f"coe_step{i}_memory_gate_abs_mean"]:.4f}' for i in [2,3,4]] for n in ['D1','D3','D4','D6','D7']]))
lines.append('D3的门绝对值在第5epoch验证已约0.992/0.999/0.999，D1约0.996/1.000/0.999。D6同期约0.867/0.874/0.949。门不是始终保持零或微小值。D7第4轮有符号均值约0.0004，但绝对值均值0.9452，均值抵消不能解释为“没有使用差分”。\n\n')
lines.append(f'![门控与验证曲线]({A.name}/gates_and_curves.png)\n\n')
lines.append(table(['组','train MAE首→末','val20','val40','val60','val80','val100','best epoch'],[[n,f'{h[0]["train"]["mae"]:.3f}→{h[-1]["train"]["mae"]:.3f}']+[f'{next(v["val"]["mae"] for v in h if v["epoch"]==e):.3f}' for e in [20,40,60,80,100]]+[R[n]['best_epoch']] for n,h in H.items()]))
lines.append('8组中5组best在100epoch，其余3组在95epoch。D3/D6训练和ID验证在末期仍改善，因此没有证据把OOD差异简单归因为“训练未收敛”或“某一组没训练完”；未记录逐epoch OOD，不能断言OOD具体在哪轮开始变差。\n\n')
analysis=r'''
## 3. 结果分析、边界与下一步建议

### 3.1 核心判断：未找到稳定超过固定CMFF的新方案，需要收紧上一轮M07结论

本轮回答了两个具体问题：第一，CMFF换成MCFF没有稳定收益，三个seed的ID和平均OOD均变差；第二，M07差分直接叠加到CMFF不能保持单F上的泛化收益，三个seed平均OOD均变差，新mask确认方向一致。固定CMFF仍应作为共享多尺度主基准。

这不推翻上一轮单F上的M07测量，也不证明所有跨轮信息传递无效。它否定的是更具体、此前尚未验证的外推：**“单尺度上有效的归一化差分模块，可以直接迁移到强多尺度链并继续获益”。** 不应继续把该模块默认加入后续主模型。

### 3.2 CMFF加差分：三seed泛化一致退步，形态缺失最敏感

D1/D4/D7相对S01/S11/S12的ID变化为+8.18%/−5.17%/+2.39%，平均OOD变化为+9.06%/+29.02%/+16.42%；新mask平均变化+8.79%/+30.32%/+15.99%。原三项OOD全部逐seed退步，不是某一协议的平均口径导致。

形态缺失退步尤为突出：+18.36%/+47.17%/+23.72%。D4的形态MAE29.387，对照S11为19.968；其中移动区域和多块缺失分别28.777/29.998，对照19.350/20.587，两个子族都退步。D4的ID改善与OOD恶化并存，不能用ID最好来选择“泛化方案”。

缺失率迁移也没有挽回。D4在rate0.8的ID退步50.01%，平均OOD退步15.71%；D1在各迁移率ID/OOD均退步。D7在rate0.8有轻微改善，但不足以改变全局判断。当前定义的CMFF+差分不作为推进候选。

### 3.3 固定MCFF：seed7接近，seed27明显失败；CMFF在这组比较中更稳

仅将前两轮C/M交换，seed7 ID+1.15%、平均OOD+1.10%；seed17 +0.46%/+10.37%；seed27 +9.82%/+55.14%。新mask分别+1.04%/+10.09%/+55.28%，与原测试相符。MCFF没有显示对未见形态的优势，三个seed形态MAE都高于CMFF。

两个顺序使用同一尺度多重集合、同样8次专家执行和面积代理4.625，因此差异不能归结为MCFF执行更少尺度或训练epoch少。它提示尺度操作顺序会改变学习结果，但只验证了CMFF与MCFF，不等于CMFF是所有81条路径的最优者。相对MMFF、CCFF等路径是否稳定更好，仍需各自公平复验。

“先粗后细利于先建立大范围关系”是与结果相容的解释，尚未直接验证感受野利用、频谱或信息流。不能把解释写成已证实机制。

### 3.4 MCFF加差分：D6是值得保留的单seed权衡结果，D3阻止了稳定性结论

D3/D6相对各自原生MCFF都改善ID：−3.32%/−7.23%。但平均OOD方向相反：D3退步46.84%，D6改善8.51%；新mask为+47.35%/−7.37%。D6在0.2/0.6/0.8迁移中也同时改善ID和平均OOD，是实测有价值的单seed结果，不能抹去。

换成更强同seed CMFF，D6的ID改善6.80%，但rate0.4平均OOD退步1.03%，新mask退步2.02%；形态原测试退步5.33%，新mask退步6.68%。它更接近**ID精度换取小幅结构泛化退步**，不是全面胜出。D6在其它缺失率相对CMFF的平均OOD约改善2.62%–3.84%，仍只有一个训练seed。

D3相对CMFF虽ID改善2.21%，平均OOD退步47.70%。在两个seed已经呈现巨大方向冲突时，仅追加一个seed27无法自动确立方案，至少要先解释为何门控和训练路径如此敏感。此处未跑MCFF+差分seed27是预先计划的覆盖边界，不是失败遗漏。

### 3.5 G12为什么不能用“最终都是MCFF”来解释

旧G12是自由尺度+差分，最终测试所有窗口走MCFF；D3是从训练开始就固定MCFF+差分。D3相对G12 ID改善6.13%，平均OOD退步52.44%，新mask仍退步53.52%。固定MCFF原生D2则相对G12 ID改善2.90%、平均OOD退步3.94%。

因此，G12的泛化成绩不能仅由最终MCFF顺序与“存在差分”两个标签解释，训练过程及参数状态很重要。但不能反过来说差异证明自由尺度路由有益：两者训练轨迹、尺度头的任务梯度、额外模块参数初始化时机等不同；G12没有同配置第二训练seed。

尤其S02类自由尺度不仅可能改硬路径，还通过选中尺度的直通系数提供任务梯度；固定组不含这一梯度通道。即便硬路径最终相同，训练目标的梯度图也不相同。应通过受控共同初始化/共同检查点分叉或保留固定前向但移除代理梯度的消融，才能拆开这些因素。本次没有做这些干预，不能给出因果结论。

### 3.6 两条机制线索：门控强度与跨尺度差分语义

**实测线索一：零初始化很快不再是轻量修正。** 差分先按当前H的RMS放大到相近幅度，再乘tanh门；没有0.1等额外幅度限制。D1/D3在第5epoch门绝对值已经接近1，最终部分轮次仍约0.998–0.999。输入修正绝对均值约0.5–0.9，与记录的轮次状态变化量处于相近量级。

这说明模块已经在强烈改变专家输入。D7的门均值接近0却有约0.945的绝对值，再次说明只看有符号均值会漏掉正负通道相互抵消。但“门接近饱和导致泛化差”仍是待验证假设：D6也有较大的门，当前没有同初始状态下限制幅度的因果对照，不能单靠相关性认定原因。

**结构线索二：跨尺度差分不只包含补全进展。** 所有H张量虽被恢复到32×32，CMFF的三次有效历史差分分别涉及初始细网格→C、C→M、M→F的前后状态；MCFF类似为初始细网格→M、M→C、C→F。它们混合了专家变换、降采样/上采样及尺度转换影响。

由于只在执行前读取上一轮差分，最后一次F→F的输出差分产生后本轮就结束，尚没有下一轮读取它。因而这套四轮固定多尺度链里的历史门，实际主要在利用跨尺度状态差，而不是纯粹同尺度更新差。单F中有效不保证此时仍具有相同语义。归一化还会把这种混合差分匹配到当前状态幅度。这是代码层面的区别与可检验解释，并不是已经测得“混叠造成失败”。

### 3.7 结论边界和当前主线取舍

- **稳定参考保留：** 四轮共享、原生Top2、固定CMFF、direct、无额外差分。它的三seed结果由上一批训练，本批用相同新mask补评，仍保持优势。
- **暂不默认叠加：** 当前M07归一化跨轮差分。保留单F上的有效记录，同时承认多尺度迁移失败。
- **单seed精度/泛化权衡：** D6。不能将其推广为MCFF+差分的稳定结论，也不能遗漏D3反例。
- **未证明：** 自由尺度比固定尺度更好、更多轮必然更好、每轮误差单调降低、MoE天然更泛化。本轮没有改变轮数，不能增加这些主张。

全部结果基于TaxiBJ、一个时间切分，训练rate0.4与四基础mask混合；新mask仍复用原测试时间窗口。三seed或两seed只提供有限复现证据，不直接声称统计显著。历史参考与本轮为独立运行、未开启严格数值确定性；结果差异可能包含训练数值轨迹影响。MAPE在零/近零流量处不稳定，本报告以MAE/RMSE为主。

### 3.8 下一步建议：先做能判定机制的少量检查，不再直接叠加新模块

1. **冻结当前主基准。** 将无额外差分的CMFF保留为后续结构比较参考，不因为一个seed的ID更低而替换。当前可写的可靠实证是“特定粗到细执行顺序在本协议下更稳”，新颖性还需单独研究，不能把稳定结果本身等同于创新。
2. **先检验强差分是否是问题。** 在现有已训练模型上进行不训练的差分幅度干预，例如α=0/0.1/0.25/1；这是推理依赖/敏感性诊断，必须与重新训练区分，不能拿推理关门退步证明模块从头训练有益。若看到明确幅度规律，再以同初始公共参数比较固定小幅度与原定义，从头训练且配对seed。
3. **再检验跨尺度语义。** 若继续研究通信，应先区分“专家在当前尺度造成的更新”和“重采样造成的变化”，考虑传递E_s(H_s)−H_s等操作内变化摘要，而非直接把两份不同尺度历史恢复状态作差。先验证其可比性和数值幅度，不一次同时改路由、融合与损失。这里只提出可检验方向，不承诺有效。
4. **自由尺度单列训练因素对照。** G12最终固定MCFF仍不能说明它为何泛化好。可在共同骨干初始化或检查点上分离固定前向、自由硬路径、选中分支代理梯度；不要只按最终路径名称分组。只有在强固定参考上跨seed获得实际收益，才将自由选择上升为论文主线。

本次分析没有启动上述任何新实验。完成结果已足以作出阶段取舍：保留强多尺度结构，暂停把未经跨尺度验证的差分模块作为默认改进；后续把预算用在能区分原因的受控验证上。
'''
lines.append(analysis)
lines.append(f'\n附件：[原始记录审计]({A.name}/audit.json)、[配对及迁移统计]({A.name}/paired_results.json)、[复核生成脚本]({A.name}/build_review.py)。\n')
dest=OUT/'20261005_v24_D1-D8固定尺度与跨轮差分复盘.md';dest.write_text('\n'.join(lines))
bad=[l for l in re.findall(r'\]\(([^)]+)\)',dest.read_text()) if not l.startswith(('http:','https:','#')) and not (dest.parent/l).exists()]
assert not bad,bad
print(dest);print('characters',len(dest.read_text()),'source differences',mismatches,'coverage',counts)
