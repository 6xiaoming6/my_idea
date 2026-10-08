"""Read-only V1-V4 audit/report; never starts training or evaluation."""
import json,hashlib,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/v24'))
from run_b3_c3 import digest
S=ROOT/'outputs/v24-COE/experiments/coordination_exploration/taxibj/cfc99a08cd16e2a6'
OUT=ROOT/'experments_report/20261006_v24_V1-V4三尺度协调实验复盘.md'
A=Path(__file__).parent
load=lambda p:json.loads(Path(p).read_text())
plan=load(S/'plan.json');rs=load(S/'summary.json');es={r['variant']:load(S/'evaluations'/f"{r['variant']}.json") for r in rs}
H={r['variant']:[json.loads(l) for l in (Path(r['run_dir'])/'logs/metrics.jsonl').read_text().splitlines()] for r in rs}
M={r['variant']:load(Path(r['run_dir'])/'training_metadata.json') for r in rs}
audit={'source_hash_mismatches':[],'runs':{},'invalid_diagnostics':'coord_*_demeaned_mae sums signed residuals without abs; invalid MAE, excluded','failures':[str(p) for p in (S/'failures').glob('*')]}
for p,sha in plan['source'].items():
 if hashlib.sha256((S/'source_snapshot'/p).read_bytes()).hexdigest()!=sha:audit['source_hash_mismatches'].append(p)
import torch
for r in rs:
 n=r['variant'];run=Path(r['run_dir']);cfg=load(run/'config.json');receipt=load(S/'results'/f'{n}.json');ev=es[n];hs=H[n];bad=[]
 def scan(x,path=''):
  if isinstance(x,dict):
   for k,v in x.items():scan(v,path+'/'+k)
  elif isinstance(x,list):
   for i,v in enumerate(x):scan(v,path+'/'+str(i))
  elif isinstance(x,float) and not math.isfinite(x) or isinstance(x,str) and x in ('nan','inf','-inf'):bad.append(path)
 scan(hs,'history');scan(ev,'evaluation')
 cp={}
 for name in ('best','last'):
  saved=torch.load(run/'checkpoints'/f'{name}.pth',map_location='cpu',weights_only=False)
  cp[name]={'epoch':saved['epoch'],'config_match':digest(saved['config'])==digest(cfg)};del saved
 best=min((h for h in hs if h['val']),key=lambda h:h['val']['mae'])
 audit['runs'][n]={'epochs':len(hs),'continuous_epochs':[h['epoch'] for h in hs]==list(range(1,101)),
 'val_count':sum(h['val'] is not None for h in hs),'receipt_config_match':receipt['config_sha256']==digest(cfg)==ev['config_sha256']==digest(plan['jobs'][n]['config']),
 'protocol_match':ev['protocol_sha256']==digest(plan['protocol']),'best_matches_log':best['epoch']==r['best_epoch'] and best['val']['mae']==r['val_mae'],
 'checkpoints':cp,'evaluation_count':len(ev['sets']),'nonfinite':bad,'amp_skips':sum(h['train'].get('train_skipped_amp_steps',0) for h in hs)}
 assert not bad and audit['runs'][n]['receipt_config_match'] and audit['runs'][n]['protocol_match'] and audit['runs'][n]['best_matches_log']
 assert cp['best']['epoch']==r['best_epoch'] and cp['last']['epoch']==100
assert not audit['source_hash_mismatches']
(A/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
lines=['# V1–V4：三尺度区域监督与输出协调实验复盘','', '复盘日期：2026-10-06；版本v24；队列`cfc99a08cd16e2a6`；4/4训练、24/24评估完成。以下核实自冻结源码、配置、检查点与原始日志。','', '## 1. 每组具体做法与复现信息','']
def add(s=''):lines.append(s)
def f(x):return f'{x:.4f}' if isinstance(x,float) else str(x)
def table(h,rows):
 add('| '+' | '.join(h)+' |');add('| '+' | '.join(['---']*len(h))+' |')
 for row in rows:add('| '+' | '.join(f(v) for v in row)+' |')
 add()
def rel(p):return str(Path(p).relative_to(ROOT/'experments_report')) if str(p).startswith(str(ROOT/'experments_report')) else '../'+str(Path(p).relative_to(ROOT))
table(['组','具体做法','对照','研究问题'],[[r['variant'],r['method'],r['reference'],q] for r,q in zip(rs,['本批U02参考','增加监督是否有效','协调本身是否有效','条件分配是否优于均匀分配'])])
add('### 共同设置与具体数据流')
add('TaxiBJ，输入[B,2,12,32,32]；四轮共享八专家T/S/TD/SD/TA/ST/TL/SL、每轮原生Top2、组内softmax、direct、关闭completion feedback。固定C→M→F→F，C/M/F=8/16/32；沿用U02的±0.1 RMS匹配H−Hprev通信，门读取当前/历史全局与缺失摘要，只修正专家输入。每窗口专家执行8次，面积代理4.625；不代表等FLOPs。')
add('训练2452、验证342、测试707个窗口；batch32，对应77/11/23个batch。四基础族random_point/node_outage/temporal_gap/spatial_region按原混合协议训练，random0.4、训练逐epoch重采样；验证与测试固定，模型/loader seed7、mask seed20260917。AdamW，100epoch，val5，1e-3余弦至3e-4（第100轮达到下限）、wd1e-4、clip1、AMP、无早停。best只取ID验证MAE最低epoch；保存best/last及完整恢复状态。')
add('原损失L1+0.01 candidate balance。V2–V4额外0.05L_C+0.05L_M：第一、二轮原生C/M专家融合输出分别经独立decoder副本，预测该区域缺失值均值。仅使用训练有效真值建立目标；缺失标签不完整的区域不参与该辅助监督。辅助loss按有效区域等权，最终MAE按缺失点等权，因此两者不能当作相同量比较。')
add('V3/V4在第四轮F输出后做C→M软修正：区域残差r=n_missing×预测缺失均值−当前缺失预测之和；每个缺失位置增加0.1×w×r，观测位置不改。V3均匀w=1/n；V4使用Conv1×1→GELU→Conv1×1、隐藏16的共享分配头，输入最终hidden/当前预测/mask/区域均值差/尺度factor，再在缺失位置softmax。零输出层使V4初始化等于V3；新增模块隔离随机流，公共骨干初始化一致。修正不是物理守恒或严格双层一致，0.1也不是单点绝对变动上限。')
add('V2辅助梯度训练区域头及骨干。V3/V4主任务梯度还经过修正进入区域头与骨干；因此跨模型成绩同时反映训练轨迹和推理修正。记录raw→after_c→after_m是同一检查点的输出端消融，不是独立训练或逐轮修复准确性的证据。')
add('六协议保持相同测试时间窗口：ID种子20260917、两元20260918、形态20260919、三元20260930、两元复测20261001、形态复测20261002，沿用test种子偏移。后两者是测试mask复测，不是训练种子复验；本批没有跨数据集或新缺失率评估。')
add('### 复现来源')
add('入口：`python -u scripts/v24/run_coordination_exploration.py --dataset taxibj --gpu 0 --epochs 100 --batch-size 32`。历史复现优先使用冻结源码；恢复指定`--suite outputs/v24-COE/experiments/coordination_exploration/taxibj/cfc99a08cd16e2a6`。此次分析未启动、重跑或改变训练。')
table(['组','配置','日志','评估'],[[r['variant'],f"[config]({rel(S/'configs'/(r['variant']+'.json'))})",f"[logs]({rel(Path(r['run_dir'])/'logs')})",f"[evaluation]({rel(S/'evaluations'/(r['variant']+'.json'))})"] for r in rs])
add(f"[冻结plan]({rel(S/'plan.json')})；[机制源码]({rel(S/'source_snapshot/src/stmoe_imputer/models/coordination_coe.py')})；[校正与指标源码]({rel(S/'source_snapshot/src/stmoe_imputer/coordination.py')})；[完整性审计]({rel(A/'audit.json')})。")
add('## 2. 完整结果、曲线与机制诊断')
table(['组','epoch/val次数','best epoch','ID val MAE','best/last','AMP跳步','非有限记录'],[[r['variant'],'100/20',r['best_epoch'],r['val_mae'],'均存在且配置一致',audit['runs'][r['variant']]['amp_skips'],len(audit['runs'][r['variant']]['nonfinite'])] for r in rs])
add('全部24个评估完成，无失败记录；保存的best与日志最优验证epoch相符，源码哈希一致。AMP各组有11–13个跳过更新，不等于训练失败，但不能说训练更新步数完全一致。')
keys=list(es['V1']['sets']);labels=['ID','两元组合','未见形态','三元组合','两元复测','形态复测']
table(['组']+[k+' MAE/RMSE' for k in labels],[[r['variant']]+[f"{r['evaluations'][k]['mae']:.4f}/{r['evaluations'][k]['rmse']:.4f}" for k in keys] for r in rs])
add('### 训练曲线与成本')
table(['组','train MAE首→末','val20','val40','val60','val80','val95','val100'],[[n,f"{hs[0]['train']['mae']:.3f}→{hs[-1]['train']['mae']:.3f}"]+[next(h['val']['mae'] for h in hs if h['epoch']==e) for e in [20,40,60,80,95,100]] for n,hs in H.items()])
add('V2在epoch20即领先V1（25.815对31.085），epoch100仍领先；V3/V4未超过V2。V1最优95，其他均100；晚期仍有验证改善，不支持“已明显过拟合所以需要提前停止”的结论。各组目标不同，因此用train MAE比较学习进展，不能直接把总loss较大解释为补全更差。')
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(12,4))
for n,hs in H.items():
 axs[0].plot([h['epoch'] for h in hs],[h['train']['mae'] for h in hs],label=n)
 vs=[h for h in hs if h['val']];axs[1].plot([h['epoch'] for h in vs],[h['val']['mae'] for h in vs],label=n)
for ax,title in zip(axs,['Train MAE','ID validation MAE']):ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(A/'curves.png',dpi=160);plt.close(fig);add(f'![训练曲线]({rel(A/"curves.png")})')
table(['组','总/可训练参数','训练+验证min','峰值GiB','ID推理ms/窗口'],[[r['variant'],f"{M[r['variant']]['total_params']}/{M[r['variant']]['trainable_params']}",r['training_seconds']/60,max(h['perf']['peak_memory_gb'] for h in H[r['variant']]),es[r['variant']]['sets']['in_distribution']['metrics']['forward_ms_per_sample_per_rank']] for r in rs])
add('四组训练与验证计时合计约207.3分钟（约3小时27分），额外评估/保存另计。V4可训练参数比V1增加5734（约2.11%）。耗时未显示明显额外负担，但单次顺序运行计时不能据此断言复杂模型更快。')
add('### 同检查点输出修正消融')
table(['组','协议','raw MAE','C后','M后','修正相对raw变化'],[[n,labels[i],m['coord_raw_mae'],m['coord_after_c_mae'],m['coord_after_m_mae'],f"{100*(m['coord_after_m_mae']/m['coord_raw_mae']-1):+.2f}%"] for n in ['V2','V3','V4'] for i,k in enumerate(keys) for m in [es[n]['sets'][k]['metrics']]])
add('### 区域预测可靠性与分配')
table(['组','协议','C头区域MAE','F聚合C区域MAE','M头区域MAE','F聚合M区域MAE'],[[n,labels[i]]+[m[z] for z in ['coord_c_head_region_mae','coord_c_raw_region_mae','coord_m_head_region_mae','coord_m_raw_region_mae']] for n in ['V2','V3','V4'] for i,k in enumerate(keys) for m in [es[n]['sets'][k]['metrics']]])
table(['组','ID平均修正幅度','C分配熵','M分配熵','观测值变动'],[[n]+[es[n]['sets']['in_distribution']['metrics'][k] for k in ['coord_correction_abs','coord_c_allocation_entropy','coord_m_allocation_entropy','coord_observed_change']] for n in ['V3','V4']])
add('ID上V3的C聚合误差为8.4143→8.2465→8.4094，V4为8.6970→8.4008→8.5180：后续M校正回撤了部分C级区域改善。软协调没有保证嵌套两级同时一致。V4分配熵低于V3均匀分配，说明有非均匀分工，但不等价于最终方法更优。')
add('### 分缺失族与路由诊断')
family=[]
for n,e in es.items():
 for protocol,entry in e['sets'].items():
  m=entry['metrics']
  for k,v in m.items():
   if k.startswith('coe_family_') and k.endswith('_mae'):family.append([n,protocol,k[11:-4],v,m[k[:-3]+'rmse']])
table(['组','协议','缺失族','MAE','RMSE'],family)
table(['组','ID路径','第2/3/4轮门绝对值'],[[n,', '.join(k.removeprefix('coe_scale_execution_path_').removesuffix('_fraction')+f' {v:.1%}' for k,v in e['sets']['in_distribution']['metrics'].items() if k.startswith('coe_scale_execution_path_') and k.endswith('_fraction') and v>0),'/'.join(f"{e['sets']['in_distribution']['metrics'][f'coe_step{i}_memory_gate_abs_mean']:.4f}" for i in [2,3,4])] for n,e in es.items()])
add('### 指标实现审计：去均值MAE无效')
add('冻结源码`CoordinationMetrics.update`对去均值残差直接求和，遗漏abs；区域内正负项抵消，日志coord_*_demeaned_mae因此近零。这不是细节完美恢复，全部该字段作废，不据此推断平滑或细节保持。它只用于no_grad诊断，不进入loss、best选择或最终MAE/RMSE；主结果与区域均值MAE有效。本次未重跑评估，旧日志原样保留；附录已标记字段无效。')
add('## 3. 分析、证据边界与下一步建议')
lookup={r['variant']:r for r in rs}
comparisons=[]
for a,b in [('V2','V1'),('V3','V2'),('V4','V3'),('V3','V1'),('V4','V1')]:comparisons.append([a,b]+[f"{100*(lookup[a]['evaluations'][k]['mae']/lookup[b]['evaluations'][k]['mae']-1):+.2f}%" for k in keys])
table(['方法','参考']+labels,comparisons)
add('### 3.1 可以支持什么')
add('本批ID收益首先来自额外区域监督：V2较V1降低4.77%，同时验证误差改善。加入均匀协调后V3较V2反而退步1.61%；条件分配V4较V3再退步1.28%。因此没有证据证明“协调＋条件分配”比单纯辅助监督更有利于ID。')
add('同检查点的修正并非无效：V3 ID从14.2796到14.1001（−1.26%），V4从14.8162到14.2804（−3.62%）；V3/V4六协议均有直接修正收益。但V4的raw比V3更差，修正幅度更大仍未获得更好的最终ID。这支持“模块对自身模型有用”，不能推出“整个模型比简单参考好”，也不能用3.62%与1.26%的差直接量化分配器优劣，因为骨干不同。')
add('### 3.2 OOD代价明确，测试mask复测保持相同趋势')
add('V2相对V1两元/形态/三元MAE退步74.14%/35.96%/85.69%；V3和V4虽修复部分V2泛化损失，仍未超过V1。V4相对V1三类分别退步25.92%/9.17%/15.96%。两元与形态的第二套mask同样退步，不能解释成单套mask偶然不利。没有方法同时取得最佳ID与OOD，V1仍是本批泛化参考。')
add('ID分族：三种改动对随机点都比V1更差；V2随机点9.8224对9.4390，但节点/时间/区域分别更优。V3区域缺失18.4392为本批最低；V4没有任一ID缺失族超过V2和V3中的最佳值。这说明收益更偏结构化缺失，不能把总ID改善描述为所有模式都改善。')
add('### 3.3 初始机制假设尚未得到支持')
add('C/M头的区域估计在V2–V4、六协议中均弱于同模型最终F预测的聚合。例如V4 ID：C头20.0660对F聚合8.6970，M头26.4490对11.7748。原先“粗尺度总量更准，再指导细尺度”的前提并未出现。小比例混合仍可能降低误差，因为两者误差并非完全同向、且模型联合适应修正；这些是待验证解释，不能将弱粗头包装为更可靠的区域先验。')
add('结构上早期C/M头只有1/2轮处理，最终F有4轮，信息加工深度不同；独立早期监督也改变骨干训练目标。现有数据没有任务/辅助梯度冲突或误差协方差测量，不能断言退步就是梯度冲突、粗粒度必然丢失信息或mask过拟合导致。')
add('### 3.4 同seed基准漂移限制创新结论')
old=ROOT/'outputs/v24-COE/experiments/id_priority_exploration/taxibj/501f761e40b7e937';u=load(old/'evaluations/U02.json');u_id=u['sets']['in_distribution']['metrics']['mae']
add(f'历史U02(seed7) ID={u_id:.4f}，本批同方法V1=14.5719，退步{100*(rs[0]["evaluations"][keys[0]]["mae"]/u_id-1):.2f}%；本批最佳V2=13.8768也比历史U02高{100*(rs[1]["evaluations"][keys[0]]["mae"]/u_id-1):.2f}%。核对配置，训练/模型设置相同，只有日志策略和实验元数据差异；不能拿V2对这次偏弱V1的4.77%就宣布刷新现有最优主线。历史成绩也不能替代本批直接对照。')
add('U02与V1 epoch1 train MAE已从108.733206到108.733112出现微小差异，epoch2扩大为95.5152对95.4072，说明并非只由最佳epoch挑选差异引起。GPU数值非确定性/执行环境可能参与，但尚未定位；不应归因于日志改动，也不能把相同seed视作完整可重复训练。')
add('### 3.5 建议与分级')
add('**V2：单种子ID信号，存在严重泛化代价。V3：同模型修正有效，未超过辅助监督ID参考。V4：条件分配未建立额外ID优势。当前不将跨尺度协调确立为核心创新。**')
add('优先先解决同配置复现波动，并以相同初始化、数据序列/环境做成对参考；若继续本方向，先复验V1/V2的低成本辅助监督收益及OOD代价，不能只复验最复杂V4。将0.05/0.05辅助权重降低或后期开启属于待验证的训练策略，不预设其能够恢复泛化。')
add('只有区域预测可靠性有改善后，再考虑更晚的区域读出；这需要区分更多加工深度和尺度本身的效果。诊断修复后再研究细节损失、误差相关性及C/M冲突。若对V4研究条件分配，宜冻结同一个已训练骨干比较均匀与学习分配，隔离联合训练轨迹；按ID验证选择，不扫描测试结果调强度。')
add('四组仅一个训练seed；两个mask复测不是训练复验，不声称统计显著或跨数据集泛化。尚未测过低/高缺失率，不能外推。保留已有U02配对种子证据作为主线基础，本批没有足够证据推翻此前主线，也没有足够证据增加一个稳定创新模块。')
(A/'valid_diagnostics.json').write_text(json.dumps({n:{k:{metric:v for metric,v in z['metrics'].items() if metric.startswith('coord_') and 'demeaned_mae' not in metric} for k,z in e['sets'].items()} for n,e in es.items()},ensure_ascii=False,indent=2))
add(f'附件：[审计]({rel(A/"audit.json")})；[有效协调诊断]({rel(A/"valid_diagnostics.json")})；[复盘脚本]({rel(A/"review.py")})。')
OUT.write_text('\n'.join(lines));print(OUT)
