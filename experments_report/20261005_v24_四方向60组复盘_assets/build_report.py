from pathlib import Path
import json,statistics as st,hashlib,math
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'outputs/v24-COE/experiments/four_direction_exploration/taxibj/625dc23938a50059'
OUT=ROOT/'experments_report';ASSET=Path(__file__).parent
rows=json.loads((P/'summary.json').read_text());R={r['variant']:r for r in rows}
K=['in_distribution','unseen_combinations','unseen_geometry','unseen_triple']
D={n:json.loads((P/'evaluations'/f'{n}.json').read_text()) for n in R}
for f in (P/'reference_evaluations').glob('*.json'):
 d=json.loads(f.read_text());R[f.stem]={'evaluations':{k:v['metrics'] for k,v in d['sets'].items()}}
def table(head,rs):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rs])+'\n\n'
def metrics(n,folder=None):
 return R[n]['evaluations'] if folder is None else {k:v['metrics'] for k,v in json.loads((P/folder/f'{n}.json').read_text())['sets'].items()}
def changes(a,b,folder=None,rate=None):
 x,y=metrics(a,folder),metrics(b,folder);keys=K if folder!='confirmation' else K[1:]
 if rate:keys=[k+'_rate'+rate for k in keys]
 return [100*(x[k]['mae']/y[k]['mae']-1) for k in keys]
FAMS=[('归一化差分 M07',['M07','M15','M16'],['M01','M13','M14']),('有符号历史 M05',['M05','M17','M18'],['M01','M13','M14']),('自由Top1 S02',['S02','S13','S14'],['S01','S11','S12']),('学习三尺度融合 S09',['S09','S15','S16'],['S10','S17','S18']),('晚期扰动 G09',['G09','G10','G11'],['S01','S11','S12']),('组合教师 P08',['P08','P09','P10'],['M01','M13','M14']),('当前轮搭档教师 P06',['P06','P11','P12'],['M01','M13','M14']),('原R5',['R5','G01','G02'],['N5','R2','G03']),('固定CMFF对单F',['S01','S11','S12'],['M01','M13','M14'])]
H={r['variant']:[json.loads(l) for l in (Path(r['run_dir'])/'logs/metrics.jsonl').read_text().splitlines()] for r in rows}
# Revalidate raw records rather than assume automatic report coverage.
audit={};primary_bad=[]
for r in rows:
 n=r['variant'];run=Path(r['run_dir']);hs=H[n];meta=json.loads((run/'training_metadata.json').read_text());m=D[n]['sets'][K[0]]['metrics']
 bad=[(h['epoch'],sp,k,v) for h in hs for sp in ['train','val'] for k,v in (h.get(sp) or {}).items() if isinstance(v,str) and v in ['nan','inf','-inf']]
 audit[n]={'epochs':len(hs),'validations':sum(bool(h.get('val')) for h in hs),'best_exists':(run/'checkpoints/best.pth').exists(),'last_exists':(run/'checkpoints/last.pth').exists(),'amp_skips':sum(h['train'].get('train_skipped_amp_steps',0) for h in hs),'nonfinite_diagnostics':bad,'params':meta['total_params'],'trainable':meta['trainable_params'],'peak_gib':max(h['perf']['peak_memory_gb'] for h in hs),'expert_calls':m.get('coe_expert_execution_count'),'area_proxy':m.get('coe_expert_grid_equivalents')}
 for h in hs:
  for sp in ['train','val']:
   for key in ['loss','mae','rmse']:
    v=(h.get(sp) or {}).get(key)
    if v is not None and (not isinstance(v,(int,float)) or not math.isfinite(v)):primary_bad.append((n,h['epoch'],sp,key,v))
coverage={}
for folder in ['evaluations','development','confirmation','rate_transfer','reference_evaluations']:
 files=list((P/folder).glob('*.json'));sets=0
 for f in files:
  d=json.loads(f.read_text());assert d['status']=='finished'
  for k,s in d['sets'].items():
   sets+=1
   for key in ['mae','rmse']:assert math.isfinite(s['metrics'][key]),(f,k,key)
 coverage[folder]={'files':len(files),'protocol_results':sets}
assert not primary_bad
plan=json.loads((P/'plan.json').read_text());source_diff={}
for name,sha in plan['source'].items():
 for label,base in [('snapshot',P/'source_snapshot'),('current',ROOT)]:
  f=base/name
  if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=sha:source_diff.setdefault(label,[]).append(name)
(ASSET/'review_audit.json').write_text(json.dumps({'runs':audit,'coverage':coverage,'source_files':len(plan['source']),'source_differences':source_diff,'primary_bad':primary_bad},ensure_ascii=False,indent=2))
# Plots: paired changes retain seeds and avoid hiding outliers in means.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig,axs=plt.subplots(1,2,figsize=(14,6))
plot_fams=FAMS[:8];labels=['Delta','Signed history','Free Top1','Soft 3 scales','Late perturbation','Pair teacher','Partner teacher','R5']
for i,seed in enumerate([7,17,27]):
 x=np.arange(8)+(i-1)*.23
 a=[changes(aa[i],bb[i])[0] for _,aa,bb in plot_fams]
 b=[st.mean(changes(aa[i],bb[i],'confirmation')) for _,aa,bb in plot_fams]
 axs[0].bar(x,a,.23,label=f'seed {seed}');axs[1].bar(x,b,.23,label=f'seed {seed}')
for ax,title in zip(axs,['ID test MAE change (%)','New-mask mean OOD relative change (%)']):
 ax.axhline(0,color='black',lw=.8);ax.set_title(title);ax.set_xticks(range(8));ax.set_xticklabels(labels,rotation=40,ha='right');ax.grid(axis='y',alpha=.2);ax.legend()
fig.suptitle('Paired comparisons against each method\'s designated reference; lower is better')
fig.tight_layout();fig.savefig(ASSET/'paired_changes.png',dpi=170);fig.savefig(ASSET/'paired_changes.svg');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for n in ['S01','S02','S13','S14','G12']:
 hs=H[n];axs[0].plot([h['epoch'] for h in hs if h['val']],[h['val']['mae'] for h in hs if h['val']],label=n)
for n in ['G09','G10','G11']:
 hs=H[n];axs[1].plot([h['epoch'] for h in hs if h['val'] and h['epoch']>=70],[h['val']['mae'] for h in hs if h['val'] and h['epoch']>=70],marker='o',label=n)
axs[1].axvline(93,ls='--',color='black',label='perturbation begins');axs[0].set_ylim(10,65)
for ax in axs:ax.set_xlabel('Epoch');ax.set_ylabel('ID validation MAE');ax.grid(alpha=.2);ax.legend()
axs[0].set_title('Scale routes and combined method');axs[1].set_title('Best checkpoints precede late perturbation')
fig.tight_layout();fig.savefig(ASSET/'focused_curves.png',dpi=170);plt.close(fig)
old=OUT/'20261004_v24_四方向60组_625dc23938a50059_完整实验分析.md'
base=old.read_text().split('## 3. 结果分析、证据边界与建议')[0]
base=base.replace('# 20261004 v24 四方向60组：完整实验分析','# 20261005 v24 四方向60组：三种子复验与主线取舍报告',1)
base=base.replace('生成时间：2026-10-04T09:07:46；完成60/60。队列`625dc23938a50059`。','复核日期：2026-10-05；完成60/60。队列`625dc23938a50059`。本报告保留自动报告的完整方法、结果表和曲线，新增原始日志审计、逐种子配对复核、归因纠正与研究建议。')
base=base.replace('训练时间包含验证和检查点保存。','训练时间为程序计时的训练＋验证；计时截止在检查点写盘之前，不含检查点写盘和队列额外评估。')
method=r'''
### 1.1 实现补充与复现边界

数据窗口为 `[B,2,12,32,32]`（TaxiBJ双通道流量），训练/验证/测试分别2452/342/707个窗口，每epoch77个训练batch。先将不可见输入置零，使用观测值、mask、支撑特征及位置编码生成初始表示H0。每轮从当前H提取原生Router特征并选择专家；关闭completion feedback意味着专家附带的补全输入沿用初始补全，并不将每轮预测重新注入该通道。隐藏状态H仍逐轮变化。

尺度执行在F=32×32、M=16×16、C=8×8上进行，时间12不变。粗尺度输入使用观测加权池化、覆盖率mask以及当前状态；不读取隐藏真值。原生Router在当前细网格状态上选择一次专家对，多尺度同轮各分支执行同一对专家，输出上采样后融合。默认direct写回下一轮H；本轮没有启用T系列的细节保留公式。

| 方法族 | 精确操作与要检验的问题 |
| --- | --- |
| M03 | Hnext=kH+(1-k)E，逐轮sigmoid标量k初值0.1；检验普通保留当前状态是否足够。 |
| M04 | 专家输入H+tanh(g)(H0-H)，门末层零初始化；检验初始编码锚定。 |
| M05/M06 | H+g(Hprev-H)，g分别tanh/sigmoid且初始均0.05；初始前向一致，检验是否需要允许负向历史校正。 |
| M07/M08 | Δ=H-Hprev；按每窗口每通道THW的RMS将Δ匹配H幅度，归一化比值停止梯度，分母下限1e-6；专家输入H+tanh(g)Δnorm。M08额外detach差分内容，不是把整个门控分支detach。 |
| M09/M10 | 当前/历史/差分的全局与缺失位置摘要经零初始化头修正Router logits；M10另加M07专家输入修正。 |
| M11/M12 | M11传递上轮已执行两专家的全局/缺失摘要、one-hot身份和融合权重，64维MLP输出零初始化路由与FiLM修正；M12改用64维GRU累积状态/差分摘要。历史只在本次forward中存在。 |
| S02–S06 | 每轮尺度头读取原Router特征、三尺度累计次数/4及剩余轮数比例；1e-3的CMFF微偏置初始化。选中分支使用hard+p-stopgrad(p)，未选尺度不执行。S03前20epoch按0.8p+0.2/3采样；S04加教师；S05加6项历史统计；S06缓存各尺度D(H0)，仅更新执行尺度并记录年龄，零门融合缓存。 |
| S07–S10 | S07学习三选二，S08固定CM/MF/CF/MF各等权；S09全三尺度softmax，S10全三尺度等权。每轮分别4/4/6/6次专家执行，不能和Top1按等计算量比较。 |
| G01–G11 | G01–G06为旧双尺度MMFF/FMMF早期扰动及共享/独立复验；G07/G08/G09在1–8/1–100/93–100epoch均匀抽取CMFF的12种排列，评估固定CMFF；G10/G11实际复验G09。 |
| P02–P06/P08 | 组合分数z_i+z_j+d或主专家固定后的搭档分数z_j+d，d输出层零初始化，平分时保留原生Top2，仍用原始个体logits组内softmax。P05/P06搭档头仅读取路由特征及主专家身份，没有额外试执行主专家获取提议。 |
| P01/P07 | P01仅改等权融合；P07读取已执行专家响应摘要和原融合权重，摘要detach，对两专家融合logit差加0.5*tanh(d)，d零初始化。 |
| G12 | 实际组合为M07+S02，先修正细网格专家输入再执行尺度处理。只有seed7，没有新mask确认或缺失率迁移文件。 |

教师组每20个训练batch选最多4个有效窗口，轮次循环；S04比较3尺度，其余比较5个不同专家候选。干预轮之前的实际上下文停止梯度，后续正常重新路由；用最终缺失MAE形成softmax负误差标签，温度max(0.1×候选均值误差,1e-6)。P06仅改成当前轮解码误差。0.1倍交叉熵只更新修正头；P08另外取消任务损失经组合直通系数的梯度。头有梯度不等于选择质量改善。

实际复验映射：M15/M16=M07，M17/M18=M05；S13/S14=S02，S15/S16=S09，S17/S18=S10；G10/G11=G09；P09/P10=P08，P11/P12=P06。三种训练seed为7/17/27，mask seed不变。M02仅一个seed；不能据此宣布独立/共享的普遍胜负。

原启动命令（仅记录，本次分析没有重启训练）：

```bash
python -u scripts/v24/run_four_direction_exploration.py --dataset taxibj --gpu 0 --epochs 100 --batch-size 32
```

复核355份受冻结清单管理的源码/配置：快照及当前文件均匹配清单SHA256。数据指纹实际只保存文件大小与mtime，不是内容SHA256。训练未启用严格确定性算法；同seed不承诺逐batch轨迹完全一致。新模块在骨干之后初始化保留公共参数起点，不等于所有后续数值轨迹相同。G09/S01在扰动前已经出现微小数值差异并逐步分化，不能将扰动前差异归因于未来扰动。

'''
base=base.replace('## 2. 实验结果',method+'## 2. 实验结果',1)
parts=[base,'### 2.1 本次复核的完整性与关键对照\n']
parts.append('2026-10-01 11:15开始，2026-10-04约09:08完成统一评估，墙钟约69小时53分。60组均训练100epoch、验证20次，best/last均存在。训练程序累计计时67.59小时；剩余墙钟包括写盘、额外评估及队列开销。\n\n')
parts.append(table(['阶段','模型文件数','协议结果数'],[[f,x['files'],x['protocol_results']] for f,x in coverage.items()]))
parts.append('60×100个epoch的train/val主指标及全部评估MAE/RMSE均为有限值。AMP累计跳过675个优化步，约占462000个训练batch的0.146%，分布于多个epoch，并非完全没有溢出。P04 epoch2的第4轮Router梯度范数记录过一次inf；主指标有限、后续完成，不将其隐去，也不能仅据此归因全部效果差异。60组best epoch分布：100轮25组、95轮22组、90轮10组、85轮1组、80轮2组。当前100epoch仍有不少组末期改善，但不能因此推断延长训练必然修复路径问题。\n\n')
parts.append('### 2.2 三种子绝对值与成本概览\n\nOOD绝对值为两元、形态、三元MAE的算术平均，再跨三个训练seed平均；这里不是下表的平均相对变化，二者不能混用。\n\n')
groups=[('单尺度共享',['M01','M13','M14']),('归一化差分',['M07','M15','M16']),('有符号历史',['M05','M17','M18']),('固定CMFF',['S01','S11','S12']),('自由Top1',['S02','S13','S14']),('学习全三尺度',['S09','S15','S16']),('等权全三尺度',['S10','S17','S18'])]
parts.append(table(['方法','ID MAE三种子均值','OOD MAE均值','平均训练程序min','专家次数/窗口'],[[name,f"{st.mean(R[n]['evaluations'][K[0]]['mae'] for n in ns):.3f}",f"{st.mean(R[n]['evaluations'][k]['mae'] for n in ns for k in K[1:]):.3f}",f"{st.mean(R[n]['training_seconds']/60 for n in ns):.1f}",audit[ns[0]]['expert_calls']] for name,ns in groups]))
parts.append('### 2.3 逐种子配对变化及新mask确认\n\n变化=100×(方法MAE/同seed参考MAE−1)，负数为改善；OOD为三个协议分别求相对变化后取均值。原测试的同分布与新mask的OOD来自不同评估记录。新mask是同一测试时间窗口的mask复测，不是新的训练seed或跨数据集证据。\n\n')
pairrows=[];machine=[]
for name,aa,bb in FAMS:
 for seed,a,b in zip([7,17,27],aa,bb):
  ds=changes(a,b);cs=changes(a,b,'confirmation');rates={rate:changes(a,b,'rate_transfer',rate) for rate in ['0.2','0.6','0.8']}
  pairrows.append([name,seed,a+'/'+b,f'{ds[0]:+.2f}%',f'{st.mean(ds[1:]):+.2f}%',f'{max(ds[1:]):+.2f}%',f'{st.mean(cs):+.2f}%'])
  machine.append({'method':name,'seed':seed,'a':a,'b':b,'test_percent':ds,'confirmation_percent':cs,'rate_percent':rates})
parts.append(table(['方法','seed','比较','ID变化','平均OOD变化','最差OOD变化','新mask平均OOD变化'],pairrows))
parts.append(f'![逐种子配对变化]({ASSET.name}/paired_changes.png)\n\n')
parts.append('### 2.4 缺失率迁移配对变化\n\n每格为ID变化 / 平均OOD变化；均未重新训练。0.4的改善不能自动外推至0.2/0.6/0.8。\n\n')
parts.append(table(['方法','seed','rate0.2','rate0.6','rate0.8'],[[x['method'],x['seed']]+[f"{x['rate_percent'][rate][0]:+.2f}% / {st.mean(x['rate_percent'][rate][1:]):+.2f}%" for rate in ['0.2','0.6','0.8']] for x in machine]))
(ASSET/'paired_results.json').write_text(json.dumps(machine,ensure_ascii=False,indent=2))
parts.append('### 2.5 训练趋势、尺度路径和门控实测\n\n')
parts.append(f'![重点曲线]({ASSET.name}/focused_curves.png)\n\n')
parts.append('S01的val MAE在epoch20/40/60/80/100分别为26.181/17.480/15.193/14.406/14.732；S02为26.307/19.057/16.570/15.147/14.922。自由尺度复验S13/S14在epoch100仍为29.077/28.635，明显不是只在某次测试mask上偶然失利。\n\n')
pathrows=[]
for n in ['S01','S02','S03','S04','S05','S06','S13','S14','G12']:
 m=D[n]['sets'][K[0]]['metrics'];q=sorted([(k.removeprefix('coe_scale_execution_path_').removesuffix('_fraction'),v) for k,v in m.items() if k.startswith('coe_scale_execution_path_') and v>0],key=lambda x:-x[1]);pathrows.append([n,', '.join(f'{k.upper()} {v:.2%}' for k,v in q),f"{audit[n]['area_proxy']:.3f}"])
parts.append(table(['组','ID实际执行路径','面积代理'],pathrows))
parts.append('S02/S13/S14及G12在两元、形态、三元协议中同样100%采用各自上述固定路径。S06的ID有约3.5%走CFFF，其主要OOD全部走MFFF。S07虽出现多条尺度对路径，OOD仍差。\n\n')
weightrows=[]
for n in ['S09','S15','S16']:
 m=D[n]['sets'][K[0]]['metrics']
 for step in range(1,5):weightrows.append([n,step]+[f"{m[f'coe_step{step}_{s}_mixture_weight']:.4f}" for s in ['fine','mid','coarse']])
parts.append(table(['组','轮','F平均权重','M平均权重','C平均权重'],weightrows))
parts.append('三尺度全执行组的实际执行路径全部为FMC→FMC→FMC→FMC，即24次专家执行。上表为贡献权重，不代表未执行低权重分支。S09在前三轮几乎分别只依赖C/C/M，末轮约99.34%依赖F。\n\n')
gaterows=[]
for n in ['M05','M17','M18','M07','M15','M16','G12']:
 m=D[n]['sets'][K[0]]['metrics'];gaterows.append([n]+[f"{m[f'coe_step{s}_memory_gate_signed_mean']:.4f}" for s in [2,3,4]])
parts.append(table(['组','第2轮门均值','第3轮门均值','第4轮门均值'],gaterows))
parts.append('M05的负门乘(Hprev−H)在代数上对应顺着最近变化方向外推；但M17/M18门均值为正，M15的差分门为负，不能宣称所有种子都学到同一种“动量”方向。以上是通道/窗口均值，不是门控逐元素全同号。教师组末轮日志的修正头梯度范数均非零（例如S04约0.751，P03约1.879，P08约3.251；为教师诊断批次均值），只能排除“完全没训练该头”的解释。\n\n')
parts.append('### 2.6 晚期扰动检查点的时序核对\n\n')
parts.append(table(['组','best epoch','epoch90 val','epoch95 val','epoch100 val'],[[n,R[n]['best_epoch']]+[f"{next(h['val']['mae'] for h in H[n] if h['epoch']==epoch):.3f}" for epoch in [90,95,100]] for n in ['G09','G10','G11']]))
parts.append('扰动统一第93epoch才开始，而best分别在90/80/90epoch。测试使用best，所以测试的三个模型均未经过晚期扰动更新。95和100轮的ID验证明显高于扰动前。\n\n')
analysis=r'''
## 3. 结果分析、证据边界与建议

### 3.1 总体判断：已经找到可靠结构，也找到局部有效机制，但尚未获得全面胜出的自由路由创新

这60组的最大价值是缩小研究范围。固定三尺度CMFF在三个seed上同时改善ID和三个主要OOD，相对单F共享基准的ID下降15.98%/12.28%/22.24%，平均OOD下降39.17%/62.67%/60.00%；新mask确认和三个缺失率迁移也保持优势。它是本批最稳的共享结构参考。该结论支持这一具体三尺度顺序，不证明任意多尺度都好，更没有在这里完成等参数密集网络或独立多尺度的三种子对比。

新增机制中，M07在rate0.4的三种子与新mask上给出最干净的跨轮信息收益；S09给出“在全三尺度已执行条件下，学习贡献权重优于等权”的证据。但M07在缺失率迁移有失败，S09在精度和成本上又均不如固定CMFF。没有一个新的自由尺度/专家组合方法通过全部强参考和泛化边界检查。因此不能把本批写成“已经证明自由路由是论文核心创新”。

### 3.2 方向1：跨轮传递值得继续，优先传递归一化状态变化，暂不扩大复杂通信

**M07为最值得复验的局部机制。** 相对同seed单尺度共享参考，三个seed的ID变化−6.79%/+1.26%/−0.19%，平均OOD变化−16.17%/−21.61%/−27.02%；所有主要OOD协议逐seed均改善。新mask平均改善−16.84%/−23.03%/−26.91%。符合rate0.4下预先允许的精度/OOD权衡门槛。参数由250043增至271163（约8.45%），专家调用仍8次，不靠额外专家激活取得该收益。

但边界必须保留：seed17在rate0.6/0.8的ID分别退步23.33%/43.37%，平均OOD分别退步6.72%/25.19%。这是“针对训练缺失率附近的结构泛化收益”，还不能说全面鲁棒。它的绝对成绩也仍不如固定CMFF，最有价值的下一步是把M07放到强固定尺度基准上做受控检验。

**M05的ID更稳定，但泛化不如M07均衡。** 三种子ID均改善约6%–9%，三个OOD均值也都改善；不过seed27的未见形态MAE从47.895升到58.824，退步22.82%，新mask仍退步20.22%。按“任何主要OOD不退步超过5%”的稳健要求不能称为全面可推进。高缺失率的ID也在三个seed均退步，应作为精度/泛化权衡对照保留。

**复杂通信不是当前必要条件。** seed7中，M09仅修正Router的ID由16.344升到18.029；M12的ID16.525，三项OOD都比M01差。M11的ID14.403较好，但三元OOD38.871明显高于25.379；M10的ID14.119也较好，却伴随形态和三元退步。这提示“训练/ID更好”不等于“缺失组合泛化更好”。M07/M08中detach差分后ID更低（14.663），OOD却从21.811/27.143/21.310变为33.743/36.605/34.001；这为历史梯度有作用提供单seed线索，尚不足以证明机制因果。

M02独立四层的seed7 ID14.852、两元18.751，强于M01；参数740411是共享参考约2.96倍。它应继续作为强结构对照，不能只和较弱共享单F基准比就宣称共享结构优越。

### 3.3 方向4：固定尺度收益稳定；当前自由尺度训练仍失败；学习融合只赢了较弱等权参考

**自由Top1并未在样本间形成有效的自由选择。** S02/S13/S14在ID和主要OOD中100%分别采用MMFF/FMCM/MFCM，seed17/27的ID MAE26.996/26.878，较固定CMFF退步89.46%/99.68%。这种结果更像训练种子决定一条全局路径，而不是输入条件决定路径。不能将某个单路径本身称为错误；问题是它没有产生任务收益和稳健条件化选择。

S03随机探索最终收敛到MMMM，ID23.186；S04最终效果教师ID14.120、OOD27.622/30.105/26.435，没有超过S01；S05/S06的历史特征/缓存也没有解决问题。这次自由组初筛全部超过ID验证102%门槛，S02是规则要求补位的候选，并非合格胜者。S07三选二有多条路径，仍出现两元39.692、三元39.318；路径更多不等于更优。

**S09相对S10有真实、可复现的贡献加权收益。** 三种子ID变化−11.89%/−17.11%/+0.44%，平均OOD−46.19%/−20.00%/−19.03%，新mask与0.2/0.6/0.8迁移均保持平均OOD改善。这说明等权不是好的全尺度融合方式。

但与同seed固定CMFF相比，S09系列三个seed的ID退步7.53%/11.05%/29.73%，平均OOD退步6.13%/44.71%/46.80%；同时专家执行24对8次，面积代理10.5对4.625，平均训练程序时间99.7对50.2分钟，峰值显存约16.05对8.28GiB（seed7）。因此没有精度/成本上的推进理由。更不能把全三尺度执行叫稀疏Top1。

**一种待验证解释：** 当前direct粗输出可能丢失细节，selected-only直通梯度只能利用实际执行分支输出，没有未执行尺度完整反事实信号；这可能使路径在早期数值偏差后锁定。S13/S14末轮选M而不是F与较差精度相容，但不能凭相关性归因，也不能认定加“末轮必须F”即可解决。S04教师负结果已经表明简单补标签并不自动有效。

### 3.4 方向3：旧R5收益没有通过第三种子，扰动不应作为默认主线

原R5的seed7/17相对各自固定MMFF参考，平均OOD分别改善34.72%/25.74%，但ID退步5.48%/6.32%；seed27反过来ID改善9.05%，平均OOD退步13.19%，新mask仍退步12.83%。所以之前seed7的强泛化结果是真实测量，但不能再描述为稳定收益。独立池G04/G05也未给出稳定优势：seed7平均OOD几乎持平（−0.18%），seed17 ID退步7.51%。

三尺度早期G07、持续G08都没有超过固定CMFF。G08 ID21.857，说明训练全程随机路径、评估固定路径的这套定义并不合适；不能外推为所有路径正则无效。

**G09/G10/G11的best结果不是晚期扰动效果。** best在90/80/90轮，扰动93轮开始。它们与S01/S11/S12的best测试差异反映扰动前已形成的不同训练轨迹；不能说“晚期扰动导致这些best OOD退步”，也不能说其ID改善验证了晚期扰动。实际扰动后的95/100轮ID验证有明显恶化，但要严格评估扰动因果，应从同一92轮检查点分叉“继续固定/开始扰动”，以相同后续数据顺序比较末轮模型，且继续单列best选择结果。

### 3.5 方向2：暂未找到稳健超越原生Top2的选择器

P02的纯组合直通学习ID19.729；P03加最终效果教师改善到18.108，但仍差于原生16.344。P04历史条件化没有解决；P05残差搭档ID16.897，P06当前轮教师16.762，后者主要OOD平均更好，但seed7/17的ID仍超过预先2%退步门槛。

P08取消任务组合直通梯度、仅教师训练修正头，seed7有改善；但seed17 P09形态OOD95.078，对原生43.739退步117.37%，新mask仍退步110.86%。这是实质性不稳定，不能被seed27的改善平均掉。P06复验对平均OOD更一致，但整体未达到稳定推进标准，也未超过多尺度强结构。

P01原生选择改等权后的ID29.197，说明当前系统对融合方式敏感；等权也移除了任务损失通过选中权重训练个体Router的通道，不能简单总结成“等权在所有MoE都差”。P07有界响应修正ID16.744、形态36.326，也未形成净优势。教师头梯度非零、选择与原生Top2有分歧，只能说明模块在工作，不能说明其选的是更好的组合。

### 3.6 G12有组合线索，但不足以宣称协同或自适应尺度有效

G12是M07+S02。相对S02，平均OOD改善39.08%，但ID退步2.12%；相对M07，ID改善6.10%、平均OOD改善29.45%。然而强参考S01已经达到13.733/16.377/18.942/15.560，G12为14.306/15.437/19.305/14.864：ID退步4.17%，平均OOD仅改善2.77%，形态退步1.91%。不满足预设的ID≤2%、OOD至少改善5%的权衡标准。

G12实际100%走MCFF，S01走CMFF。因此相对S01同时改变了跨轮传递和尺度顺序，无法拆出哪项有效；也没有证明按输入自适应。它只有seed7、没有新mask确认和缺失率迁移；应归类为单种子组合线索。下一步必须增加固定MCFF及固定CMFF+M07等对照。

### 3.7 与旧强参考的关系和整体局限

旧N5(seed7)的ID13.581略优于本轮S01的13.733，但本轮S01的三项OOD16.377/18.942/15.560明显低于旧N5的25.371/28.262/23.893。旧R5在seed7两元/三元约15.613/14.928仍较强，但本轮S01的ID及形态更好；不能声称S01逐项击败所有旧方法。旧独立R6 ID13.296更好，OOD约26.066/25.434/25.931弱于S01。历史对照说明ID和OOD的权衡真实存在；没有旧/新统一三seed设计的地方要保留边界。

本轮只在TaxiBJ和一个时间切分、训练rate0.4开展训练。确认mask仍使用同一测试时间窗口，缺失率迁移也没有换数据集。三训练seed不足以直接宣称统计显著。候选从验证时段OOD筛选，确认集没有参与冻结选择；但本次看过确认结果后，下一轮再使用它就应将其视为已知研发数据。训练mask是四类混合，OOD为组合/新几何，不能把单纯mask复测写成训练模式之外所有分布都泛化。

本轮四轮基本固定，因此可以研究“跨轮信息传递”，不能新增宣称本轮证明“更多轮数必然更好”或“各轮补全单调改善”。中间解码没有专门监督。常规专家调用数不含教师试运行，面积代理也不是精确FLOPs；耗时与显存需要分别披露。参数更多、路径更丰富或门控非零都不是创新有效性的充分证据。

### 3.8 建议：停止扩散，围绕强固定三尺度与差分传递做小而清晰的验证

| 优先级 | 下一组对照 | 隔离的问题 | 通过条件 |
| --- | --- | --- | --- |
| 1 | 固定CMFF、固定CMFF+M07、固定MCFF、固定MCFF+M07；配对seed7/17/27 | 传递机制与尺度顺序哪个带来收益；解释G12 | 同seed强参考ID不退步超过2%，OOD稳定改善且各主要族不严重退步，迁移不过度恶化 |
| 2 | G12冻结强制MCFF推理与其原推理；另用固定MCFF从头训练作对照 | 是否实际需要条件化尺度路由，以及学习路径是否只有架构搜索作用 | 不将所有窗口同一路径包装为条件化；推理干预与重新训练结果分开报告 |
| 3 | 在上述胜出固定结构上，对M07做归一化/有符号门/历史内容detach的最小消融 | 是变化量、幅度控制还是梯度通道有效 | 至少配对复验，保留ID及各OOD族，不能只报告平均 |
| 4 | 若仍要自由尺度，先共享同一预训练检查点，再比较固定/可学习路径；预注册最终F约束与无约束对照 | 降低专家与尺度选择共同冷启动的不确定性，检验末层分辨率假设 | 超过固定强结构后才扩大搜索；若加约束须如实称受约束路由 |

不建议立刻继续几十组专家组合、GRU通信或任意路径扰动。方向优先级仍可保持1与4优先，但具体落点应是“强多尺度链中的受控状态变化传递”，而不是把全自由选择作为必须成立的前提。方向3继续用于检验泛化，不能预先假定MoE天然更泛化；方向2暂存为负结果及少量对照。

最终分级：固定CMFF为**稳定可推进的结构参考**；M07为**rate0.4下可复验的精度/泛化权衡，尚需强参考与缺失率审核**；M05为**ID收益与形态风险并存**；S09为**相对全尺度等权有效，但被固定结构在精度和成本上超过**；G12为**单种子组合线索**；当前自由Top1、R5三seed稳定性及新专家组合主线均为**未达到推进标准**。
'''
parts.append(analysis)
parts.append(f'\n复核附件：[完整性审计]({ASSET.name}/review_audit.json)、[逐种子与迁移统计]({ASSET.name}/paired_results.json)、[报告生成脚本]({ASSET.name}/build_report.py)。原自动报告及其曲线附件保留。\n')
dest=OUT/'20261005_v24_四方向60组实验复盘与主线建议.md';dest.write_text('\n'.join(parts))
print(dest);print('characters',len(dest.read_text()),'rows',len(rows),'source_diff',source_diff)
