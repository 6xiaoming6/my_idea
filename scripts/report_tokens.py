#!/usr/bin/env python3
"""Formal multi-run report from frozen configurations and raw run receipts."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,math,os
from release_utils import load,write
CATEGORIES={'二元组合':['unseen_combinations','unseen_combinations_repeat'],
            '未见形态':['unseen_geometry','unseen_geometry_repeat'],'三元组合':['unseen_triple']}
LABELS={'in_distribution':'ID','unseen_combinations':'二元组合','unseen_geometry':'未见形态',
        'unseen_triple':'三元组合','unseen_combinations_repeat':'二元复测','unseen_geometry_repeat':'形态复测'}
def scores(ev,ref):
 out={}
 for metric in ('mae','rmse'):
  cats={k:sum(ev['sets'][n]['metrics'][metric] for n in v)/len(v) for k,v in CATEGORIES.items()}
  refs={k:sum(ref['sets'][n]['metrics'][metric] for n in v)/len(v) for k,v in CATEGORIES.items()}
  ratios={k:cats[k]/refs[k] for k in cats}
  out[metric]=dict(categories=cats,mean=sum(cats.values())/3,ratio=sum(ratios.values())/3,worst_change_pct=100*(max(ratios.values())-1))
 return out

def report(suite):
 suite=Path(suite);plan=load(suite/'plan.json');source=Path(plan['source']);refdir=Path(plan['reference']);ref=load(refdir/'evaluation.json')
 protocol=load(source/'configs/test_protocol.json');date=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
 target=Path(plan['report_dir'])/f'{date}_v25_LT01-LT10独立latentToken探索_分析报告.md';target.parent.mkdir(parents=True,exist_ok=True)
 items=[]
 for job in plan['jobs']:
  run=suite/'runs'/job['name'];receipt=load(run/'result.json') if (run/'result.json').exists() else {}
  ev=load(run/'evaluation.json') if (run/'evaluation.json').exists() else {}
  history=[json.loads(s) for s in (run/'logs/metrics.jsonl').read_text().splitlines()] if (run/'logs/metrics.jsonl').exists() else []
  complete=receipt.get('completed_epochs')==100 and receipt.get('status')=='finished' and ev.get('status')=='finished' and set(ev.get('sets',{}))==set(protocol['evaluations']) and all((run/'checkpoints'/f'{n}.pth').exists() for n in ('best','last'))
  items.append(dict(**job,run=run,receipt=receipt,ev=ev,history=history,complete=complete,
                    meta=load(run/'training_metadata.json') if (run/'training_metadata.json').exists() else {}))
 complete=all(i['complete'] for i in items)
 title='完整报告' if complete else '阶段报告（尚有未完成或失败组）'
 lines=[f'# {date} LT01–LT10 独立 latent token 空间探索：{title}','',
        f'冻结源码 `{plan["source_commit"]}`，参考 best 提交 `{plan["base_commit"]}`；单训练种子 seed7。','',
        '## 一、实验具体做法与复现配置','','| 编号 | 直接对照 | 实际改动 |','| --- | --- | --- |',
        '| I37 | 已验证发布版 | F网格隐藏状态、原始固定损失；只读取已有结果，不重训 |']
 for i in items:lines.append(f'| {i["name"]} | {i["control"]} | {i["description"]} |')
 lines+=['','### 共同设置与数据流','',
 'TaxiBJ，双通道，12时间步；F/M/C空间大小32/16/8，完整F求和得到M/C后，各尺度独立生成缺失mask。四类混合缺失率0.4，训练逐epoch重采样，验证/测试固定；数据划分与I37一致。GPU1单卡串行，100epoch、batch32、AdamW、学习率1e-3余弦下降至3e-4、每5epoch验证；best只按ID验证MAE选取。','',
 'F/M/C先用原共享编码器编码，再通过学习的token查询分别读取三个原生网格，拼接融合为H0。公共latent布局为[B,T,K,64]，T=12，K默认128；LT02为64，LT03为256。K不对应H/W，不按网格分辨率设定，也不重排成二维网格。原专家只在真实原生网格上执行。','',
 '每轮维持相同token槽位：token空间受控差分及历史消息→选中尺度的位置查询从tokens解码原生网格→融合该尺度的可见观测、mask、support和位置→原生Top-2→用token查询写回。尺度默认前三轮自由、末轮F；LT09固定CMFF。每窗口仍8次专家执行。I37原有路由读取时机及H0预测补位规则未同时改动。','',
 '双向投影为单头交叉注意力，查询/键维度16，值/输出64，无dropout、无bias；按独立batch×time维分块计算，不能把分块理解为截断空间注意力。token槽位可学习、无空间坐标；原生网格的位置仅用于从tokens查询网格或编码网格的键。值/输出线性层初始化为恒等。LT04为每尺度独立映射，初始从同一映射复制，随后独立学习，跨轮仍共享；LT10组合独立映射与残差回写。','',
 'LT05/LT10将专家输出减去原生尺度输入，再编码成token增量并加回输入token。这不保证补全误差逐轮下降。原I37是F网格隐状态，本批是真正的无网格空间槽位，而不是M/C网格或只是修改通道数。保留时间轴，未探索把时空全部压成一条无时间结构的token序列。','',
 '历史池保留可见F回填后的因果、detach预测P1/P2/P3。原读取器产生F网格历史消息，再通过F→token映射注入；映射无bias，零消息严格为零，并将token消息RMS上限限制为隐藏RMS的0.1，避免投影放大幅度。中间Decoder仍服务历史和诊断，跨轮H始终是token状态。末轮以F位置查询tokens得到F隐藏特征，经原Decoder形成预测，再进行原可学习分配投影。','',
 '投影前/后F缺失L1固定0.5/0.5，可见F重建0.1，可见M/C投影前一致性0.1，候选均衡0.01。所有实验从公共未训练参数初始化；新增映射使用独立随机数流。去模块的组保留原参数值但冻结禁用模块，报告总参数和可训练参数；新增映射不保证等参数。','',
 f'冻结计划：[plan.json]({suite / "plan.json"})；源码：`{source}`；配置：`configs/latent_tokens/LT01.json`至`LT10.json`；运行目录：`{suite}/runs/LTxx`。数据、源码、配置、初始化和参考产物的SHA256见`manifest.json`。','',
 f'恢复命令：`{plan["python"]} {source / "scripts/queue_tokens.py"} --suite {suite}`。完成组跳过，中断组从完整last恢复；失败停止，不无限重试。','',
 '## 二、完整结果与诊断','','| 实验 | 状态/轮数 | best | 验证MAE | 总/可训练参数 | 训练验证分钟 | 峰值allocated GiB |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
 comparisons={}
 for i in items:
  r=i['receipt'];h=i['history'];meta=i['meta'];epoch=r.get('completed_epochs',h[-1]['epoch'] if h else 0)
  peak=max((v.get('perf',{}).get('peak_memory_gb',0) for v in h),default=0)
  lines.append(f'| {i["name"]} | {"完成" if i["complete"] else "未完成/失败"}/{epoch} | {r.get("best_epoch","—")} | {r.get("best_val_mae","—")} | {meta.get("total_params","—")}/{meta.get("trainable_params","—")} | {r.get("total_time_sec",0)/60:.2f} | {peak:.3f} |')
  if i['complete']:
   for row in i['ev']['sets'].values():
    if not all(math.isfinite(row['metrics'][m]) for m in ('mae','rmse')):raise ValueError('Nonfinite evaluation')
   comparisons[i['name']]=scores(i['ev'],ref)
 lines+=['','耗时为训练和验证累计，不含全部检查点写入与六套额外测试。显存为allocated，不能与nvidia-smi reserved直接等同。','','| 实验 | 协议 | MAE | RMSE |','| --- | --- | ---: | ---: |']
 evals=[('I37',ref)]+[(i['name'],i['ev']) for i in items]
 for name,ev in evals:
  for protocol_name in protocol['evaluations']:
   m=ev.get('sets',{}).get(protocol_name,{}).get('metrics',{})
   lines.append(f'| {name} | {LABELS[protocol_name]} | {m.get("mae","—")} | {m.get("rmse","—")} |')
 lines+=['','| 实验 | ID变化% | 三类等权OOD MAE | OOD相对I37类别比值 | OOD RMSE | 最差类别MAE变化% |','| --- | ---: | ---: | ---: | ---: | ---: |']
 eligible=[];refid=ref['sets']['in_distribution']['metrics']['mae']
 for i in items:
  if i['name'] not in comparisons:continue
  s=comparisons[i['name']];idmae=i['ev']['sets']['in_distribution']['metrics']['mae'];change=100*(idmae/refid-1)
  lines.append(f'| {i["name"]} | {change:+.3f} | {s["mae"]["mean"]:.6f} | {s["mae"]["ratio"]:.6f} | {s["rmse"]["mean"]:.6f} | {s["mae"]["worst_change_pct"]:+.3f} |')
  if idmae<=1.03*refid:eligible.append((s['mae']['ratio'],s['rmse']['ratio'],idmae,i['name']))
 lines+=['','二元组合与未见形态先各自平均两次mask测试，再对二元、形态、三元三类等权汇总，避免复测双倍加权。主筛选ID代价不超过I37的3%；完整结果不因门槛而删减。该测试筛选仅作为单seed探索，不视为独立确认。','',
 '### 训练曲线及诊断','']
 os.environ.setdefault('MPLCONFIGDIR',str(suite/'matplotlib_cache'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 assets=target.parent/(target.stem+'_assets');assets.mkdir(exist_ok=True)
 fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
 for i in items:
  h=i['history']
  if not h:continue
  axes[0].plot([r['epoch'] for r in h],[r['train']['loss'] for r in h],label=i['name'])
  v=[r for r in h if r.get('val')]
  axes[1].plot([r['epoch'] for r in v],[r['val']['mae'] for r in v],label=i['name'])
 for ax,title in zip(axes,['Train loss','ID validation MAE']):
  ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
  if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7,ncol=2)
 fig.savefig(assets/'curves.png',dpi=170);fig.savefig(assets/'curves.pdf');plt.close(fig)
 lines+=[f'![训练与验证曲线]({assets.name}/curves.png)','',
 '| 实验 | 原始F L1 | 投影后F L1 | 投影修正幅度 | 无约束F MAE/计数 | 主要尺度路径 |','| --- | ---: | ---: | ---: | --- | --- |']
 for name,ev in evals:
  m=ev.get('sets',{}).get('in_distribution',{}).get('metrics',{})
  paths=sorted(((k,v) for k,v in m.items() if k.startswith('coe_scale_execution_path_') and isinstance(v,(int,float))),key=lambda x:-x[1])[:3]
  path=', '.join(k.removeprefix('coe_scale_execution_path_').removesuffix('_fraction')+f':{v:.3f}' for k,v in paths)
  lines.append(f'| {name} | {m.get("l_raw_f","—")} | {m.get("l_final","—")} | {m.get("coe_v25_consistency_correction_abs","—")} | {m.get("combination_post_unconstrained_mae","—")}/{m.get("combination_post_unconstrained_count","—")} | {path or "—"} |')
 lines+=['','| 实验 | 轮1槽位差异/RMS | 轮2 | 轮3 | 轮4 | 轮4槽位余弦相似度 |','| --- | ---: | ---: | ---: | ---: | ---: |']
 for name,ev in evals:
  m=ev.get('sets',{}).get('in_distribution',{}).get('metrics',{})
  vals=[str(m.get(f'coe_token_step{j}_spread_rms_ratio','—')) for j in range(1,5)]
  lines.append('| '+name+' | '+' | '.join(vals)+' | '+str(m.get('coe_token_step4_mean_slot_cosine','—'))+' |')
 lines+=['','每组`evaluation.json`保留全部六套的路径、逐轮质量、通信、历史、投影前后及受约束/无约束区域诊断；`logs/diagnostics`保留训练采样与完整验证诊断，`logs/gpu_status.jsonl`为15秒GPU日志。空区域按有效计数解释，不以0误差代表有收益。','',
 '## 三、分析、边界与建议','']
 if eligible:
  best=sorted(eligible)[0];lines.append(f'ID代价≤3%的新候选中，OOD相对类别比值最低的是{best[3]}（{best[0]:.6f}）。'+('仍未优于I37的OOD汇总，不能强行推荐新模型。' if best[0]>=1 else '这仅说明本次seed7的汇总折中改善，仍需检查逐类负结果。'))
 else:lines.append('当前没有完整新组满足ID代价≤3%的筛选要求；不以OOD单项改善替代精度约束。')
 lines+=['','| 对照 | ID MAE变化% | 三类等权OOD MAE变化% |','| --- | ---: | ---: |']
 byname=dict(evals)
 for a,b in [('I37','LT01'),('LT01','LT02'),('LT01','LT03'),('LT01','LT04'),('LT01','LT05'),('LT01','LT06'),('LT01','LT07'),('LT06','LT08'),('LT07','LT08'),('LT01','LT09'),('LT03','LT10'),('LT05','LT10')]:
  if byname[a].get('status')!='finished' or byname[b].get('status')!='finished':continue
  x,y=byname[a],byname[b];sx,sy=scores(x,ref),scores(y,ref)
  lines.append(f'| {a}→{b} | {100*(y["sets"]["in_distribution"]["metrics"]["mae"]/x["sets"]["in_distribution"]["metrics"]["mae"]-1):+.3f} | {100*(sy["mae"]["mean"]/sx["mae"]["mean"]-1):+.3f} |')
 lines+=['','固定数量token可能形成信息瓶颈；交叉注意力若产生过于相似的槽位，也可能损失空间细节。token数量、尺度映射共享、残差写回和通信的对照用于检查这些风险，不能仅凭汇总改善确证机制。历史消息与路由摘要均针对无网格token适配，因此I37→LT01是完整架构改动，不能单独归因于token数量。可学习token和交叉注意力本身不是新的原创方法，需以真实多尺度缺失条件下的实测效果判断价值。','',
 '所有结果为同一TaxiBJ划分、seed7；mask复测不是独立训练种子，不能声称统计显著性、全局最优或每轮误差必降。正式模型仍保留I37，不根据未完成结果覆盖best分支。','',
 '后续先查看ID/OOD及最差类别是否共同可接受，再定位latent尺寸、映射容量、回写和通信的作用；若不存在更好的折中，保留原I37并如实记录负结果。','']
 target.write_text('\n'.join(lines));write(suite/'comparison.json',comparisons)
 write(suite/'report_receipt.json',dict(path=str(target),complete=complete));return target
