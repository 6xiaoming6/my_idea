"""Automatic method-first U-series reports; selection and interpretation stay ID-first."""
from __future__ import annotations
import json,math,os,statistics
from datetime import datetime
from pathlib import Path


def export_report(root,suite,rows,jobs,label):
    root=Path(root);suite=Path(suite);out=root/'experments_report';out.mkdir(exist_ok=True)
    done=sum(r['status']=='finished' for r in rows)
    final=label=='final' and done==30 and (suite/'final_evaluations_complete.json').exists()
    stem=datetime.now().strftime('%Y%m%d')+'_v24_U系列30组_'+suite.name+'_'+('完整实验分析' if final else '阶段分析_'+label)
    assets=out/(stem+'_assets');assets.mkdir(exist_ok=True)
    lines=[f'# {stem}','',f'生成时间：{datetime.now().isoformat(timespec="seconds")}；训练及原六协议测试完成{done}/{len(rows)}。最终确认评估'+('完成' if final else '尚未整体完成')+'。','', '## 1. 各组具体做法与复现信息','']
    def add(s=''):lines.append(s)
    def tab(h,rs):
        add('| '+' | '.join(h)+' |');add('| '+' | '.join(['---']*len(h))+' |')
        for r in rs:add('| '+' | '.join(map(str,r))+' |')
        add()
    def f(x):return '—' if x is None else f'{x:.3f}'
    def link(p,label):return f'[{label}]({os.path.relpath(p,out)})'
    tab(['组','seed','具体做法','对照','状态'],[[r['variant'],r['seed'],r['method'],r['reference'],r['status']] for r in rows])
    first=next(iter(jobs.values()))['config']
    add(f'实际公共预算：{first["data"]["dataset_name"]} random0.4，batch{first["data"]["batch_size"]}，{first["train"]["epochs"]}epoch，val{first["train"]["val_epoch"]}，单GPU串行。默认AdamW 1e-3余弦至3e-4、wd1e-4、clip1、AMP，无早停。四轮共享八专家T/S/TD/SD/TA/ST/TL/SL，每执行尺度原生Top2及组内softmax，direct，completion feedback关闭。L1+0.01candidate balance；教师组额外0.1交叉熵。best只按ID验证MAE，保存best/last及恢复状态。\n')
    add('F/M/C为空间原分辨率、1/2、1/4；TaxiBJ为32/16/8，时间12不变。输入观测值、mask、支撑特征和位置编码生成H0；原生Router从当前H路由。粗尺度观测加权池化、覆盖率mask及当前状态不访问隐藏真值。完成值通道维持初始补全，不随中间预测反馈。每轮所选尺度的专家输出恢复到细网格传递。\n')
    add('U02/U03/U04的专家输入为H+αtanh(g)Δ，门仍读取当前/历史全局及缺失摘要、零初始化。α分别0.1/0.25/0.1；U02/03的Δ按每窗口通道THW RMS匹配H，比例detach、分母下限1e-6；U04直接H−Hprev。U05改为上一轮U(E−Z)，Z是真实投影归一化后的专家输入、E为实际Top2融合输出；再做相同RMS匹配。U06仅detach消息内容，门输入不detach。U07只以当前/消息摘要修正专家logits，0.1tanh零头，不改专家输入。首轮无历史，历史不跨forward。\n')
    add('U08/U10/U12/U16前两轮自由后三四轮固定F，共9路径；U09/U11全部81路径；U15分辨率只升不降、末轮F，共10路径。U09–U16前20epoch固定CMFF，21起开放且评估遵守检查点阶段；阶段保存在model buffer中。U08/09/10使用选中分支直通梯度；U11/12/15/16仅教师更新尺度头。U13/14开放后每轮两尺度、各执行同一专家对，前者尺度softmax，后者固定CM/MF/CF/MF等权。U16在21起加入3个跨轮共享零初始化尺度向量。U18保留原生Top2，响应摘要detach后修正融合logit差±0.1。\n')
    add('教师每20训练batch、最多4有效监督窗口，轮次在可学习位置间循环；仅比较合法尺度，单候选窗口跳过。实际前缀停止梯度，从干预轮继续到最终预测，训练缺失MAE形成softmax标签，温度max(0.1候选平均误差,1e-6)。0.1交叉熵仅更新尺度头，不读取val/test标签，不推进主RNG、统计或历史。推理不试运行。\n')
    add('四基础mask（随机点/节点中断/时间缺口/空间区域）混合、训练逐epoch重采样、验证固定，沿用mask seed20260917。训练seed7，配对17/27，loader同seed；扩展初始化使用隔离CPU随机流，公共骨干和全局随机流保持相同起点。U17组合继承两模块各自时序。\n')
    add('选择只读训练回执的best ID val及同epoch验证日志，优先下降≥1%，再按val、面积代理、可训练参数、编号排序；通信两候选来自不同机制族。不用测试/OOD选候选。配置和证据哈希冻结；非默认预算使用本队列新参考，不将旧100epoch历史成绩作为公平基准。\n')
    add('所有组原六套测试；复验方法及参考再做新mask20261101/02/03（test+30000）和rate0.2/0.6/0.8迁移。它们仍是同一测试时间窗，不是跨数据集。历史S/M/G/D仅作背景。\n')
    add('完整配置与实现：'+link(suite/'plan.json','冻结plan')+'；'+link(suite/'source_snapshot/scripts/v24/README_ID_PRIORITY_EXPLORATION.md','复现说明')+'。入口`python -u scripts/v24/run_id_priority_exploration.py --dataset taxibj --gpu 0 --epochs 100 --batch-size 32`，其他dataset/预算见plan。\n')
    tab(['组','配置','日志'],[[r['variant'],link(suite/'configs'/f"{r['variant']}.json",'config') if (suite/'configs'/f"{r['variant']}.json").exists() else '待解析/执行',link(Path(r['run_dir'])/'logs','logs') if r['run_dir'] else '—'] for r in rows])
    add('## 2. 结果、曲线、完整性和机制诊断\n')
    keys=['in_distribution','unseen_combinations','unseen_geometry','unseen_triple','unseen_combinations_repeat','unseen_geometry_repeat']
    tab(['组','best epoch','val MAE']+[k+' MAE/RMSE' for k in keys],[[r['variant'],r['best_epoch'],f(r['val_mae'])]+[f(r['evaluations'].get(k,{}).get('mae'))+'/'+f(r['evaluations'].get(k,{}).get('rmse')) for k in keys] for r in rows])
    add('MAE/RMSE越低越好；MAPE受零/近零流量影响，不用作主结论。未完成项保留—，不从最终排名中静默删除。\n')
    histories={};audits={};diags={};cost=[];families=[];main_bad=[]
    for r in rows:
        if not r['run_dir']:continue
        n=r['variant'];run=Path(r['run_dir']);hs=[json.loads(l) for l in (run/'logs/metrics.jsonl').read_text().splitlines()];histories[n]=hs
        meta=json.loads((run/'training_metadata.json').read_text());e=suite/'evaluations'/f'{n}.json';ev=json.loads(e.read_text()) if e.exists() else {'sets':{}}
        m=ev['sets'].get('in_distribution',{}).get('metrics',{});diags[n]=m
        bad=[]
        for h in hs:
            for sp in ('train','val'):
                for k,v in (h.get(sp) or {}).items():
                    if (isinstance(v,str) and v in ('nan','inf','-inf')) or (isinstance(v,float) and not math.isfinite(v)):
                        bad.append([h['epoch'],sp,k,v])
                        if k in ('loss','mae','rmse'):main_bad.append([n,h['epoch'],sp,k,v])
        audits[n]={'epochs':len(hs),'validations':sum(bool(h['val']) for h in hs),'best':(run/'checkpoints/best.pth').exists(),'last':(run/'checkpoints/last.pth').exists(),'amp_skips':sum(h['train'].get('train_skipped_amp_steps',0) for h in hs),'nonfinite':bad}
        cost.append([n,meta['total_params'],meta['trainable_params'],f(r['training_seconds']/60),f(max(h['perf']['peak_memory_gb'] for h in hs)),f(m.get('coe_expert_execution_count')),f(m.get('coe_expert_grid_equivalents'))])
        for k,s in ev['sets'].items():
            for name,v in s['metrics'].items():
                if name.startswith('coe_family_') and name.endswith('_mae'):families.append([n,k,name[11:-4],f(v),f(s['metrics'].get(name[:-3]+'rmse'))])
    tab(['组','总参数','可训练参数','训练计时min','峰值GiB','专家次数','面积代理'],cost)
    add('面积代理非精确FLOPs；U13/U14开放后16次，其余8次，教师试运行不计入推理次数。训练计时含训练/验证，检查点写盘及额外评估另计。\n')
    tab(['组','epoch','验证次数','best/last','AMP跳步','非有限诊断数'],[[n,a['epochs'],a['validations'],str(a['best'])+'/'+str(a['last']),a['amp_skips'],len(a['nonfinite'])] for n,a in audits.items()])
    add('主指标异常：'+json.dumps(main_bad,ensure_ascii=False)+'。详细异常列表见审计附件。\n')
    tab(['组','val20','val40','val60','val80','val100','train MAE首→末'],[[n]+[f(next((h['val']['mae'] for h in hs if h['epoch']==e and h['val']),None)) for e in (20,40,60,80,100)]+[f(hs[0]['train']['mae'])+'→'+f(hs[-1]['train']['mae'])] for n,hs in histories.items()])
    tab(['组','阶段开放','第2/3/4轮门绝对值','第2/3/4轮消息RMS','主要实际执行路径'],[[n,f(m.get('coe_scale_phase_open')),','.join(f(m.get(f'coe_step{i}_memory_gate_abs_mean')) for i in (2,3,4)),','.join(f(m.get(f'coe_step{i}_memory_norm')) for i in (2,3,4)),', '.join(k.removeprefix('coe_scale_execution_path_').removesuffix('_fraction')+f' {v:.1%}' for k,v in sorted(((k,v) for k,v in m.items() if k.startswith('coe_scale_execution_path_') and v>0),key=lambda z:-z[1])[:3])] for n,m in diags.items()])
    tab(['组','协议','缺失族','MAE','RMSE'],families)
    tab(['组','epoch候选数指标之和','epoch教师计时指标之和','最大epoch头梯度指标'],[[n,f(sum(h['train'].get('four_probe_candidates',0) for h in hs)),f(sum(h['train'].get('four_probe_seconds',0) for h in hs)),f(max(h['train'].get('four_probe_head_grad_norm',0) for h in hs))] for n,hs in histories.items()])
    add('教师列由epoch日志聚合值汇总；若训练引擎对batch取平均，这些列为平均值的跨epoch合计，不能误称实际试运行总次数或总耗时。完整原始字段保存在日志。\n')
    if histories:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axs=plt.subplots(3,2,figsize=(14,13))
        blocks=[list(range(1,8)),list(range(8,19)),list(range(19,31))]
        for i,block in enumerate(blocks):
            for n,hs in histories.items():
                if int(n[1:]) not in block:continue
                axs[i,0].plot([h['epoch'] for h in hs],[h['train']['loss'] for h in hs],label=n)
                axs[i,1].plot([h['epoch'] for h in hs if h['val']],[h['val']['mae'] for h in hs if h['val']],label=n)
            for j,title in enumerate(('train loss','ID validation MAE')):
                ax=axs[i,j];ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
                if ax.lines:ax.legend(ncol=4,fontsize=7)
        fig.tight_layout();fig.savefig(assets/'curves.png',dpi=140);fig.savefig(assets/'curves.svg');plt.close(fig)
        add(f'![训练与验证曲线]({assets.name}/curves.png)\n')
    for folder,title in [('historical_evaluations','历史背景（非本批公平参考）'),('confirmation','新mask确认'),('rate_transfer','缺失率迁移')]:
        add('### '+title+'\n');records=[]
        for p in sorted((suite/folder).glob('*.json')):
            d=json.loads(p.read_text())
            for k,s in d.get('sets',{}).items():records.append([p.stem,d.get('status'),k,f(s['metrics'].get('mae')),f(s['metrics'].get('rmse'))])
        tab(['组','状态','协议','MAE','RMSE'],records)
        if folder!='historical_evaluations':
            details=[]
            for p in sorted((suite/folder).glob('*.json')):
                for k,entry in json.loads(p.read_text()).get('sets',{}).items():
                    for metric,value in entry['metrics'].items():
                        if metric.startswith('coe_family_') and metric.endswith('_mae'):
                            details.append([p.stem,k,metric[11:-4],f(value),f(entry['metrics'].get(metric[:-3]+'rmse'))])
            tab(['组','协议','缺失族','MAE','RMSE'],details)
    add('## 3. ID优先分析、泛化代价与建议\n')
    indexed={r['variant']:r for r in rows};computed={};pair_rows=[]
    for r in rows:
        b=indexed.get(r['reference']);a=r['variant']
        if not b or a==r['reference'] or any(k not in r['evaluations'] or k not in b['evaluations'] for k in keys[:4]):continue
        ratios=[r['evaluations'][k]['mae']/b['evaluations'][k]['mae'] for k in keys[:4]];computed[a]={'reference':b['variant'],'ratios':ratios}
        pair_rows.append([a,b['variant']]+[f'{100*(v-1):+.2f}%' for v in ratios]+['存在>5%单项OOD代价' if max(ratios[1:])>1.05 else '未触发5%标记'])
    tab(['方法','参考','ID变化','两元','形态','三元','泛化代价'],pair_rows)
    add('负值改善，正值退步。本轮按ID筛选，OOD退步完整披露，但不重新以OOD调整候选。\n')
    add('### 单因素对照\n')
    pairs=[('U03','U02','门幅度'),('U04','U02','RMS匹配'),('U05','U02','消息内容'),('U06','U05','消息detach'),('U07','U05','专家输入或Router'),('U08','U01','受约束自由尺度'),('U09','U01','固定后开放'),('U10','U08','固定20epoch阶段'),('U10','U09','末两轮F约束'),('U11','U09','教师替代尺度直通'),('U12','U10','约束下教师替代'),('U15','U11','单调路径约束'),('U16','U12','尺度身份'),('U13','U14','双尺度学习贡献'),('U18','U01','有界响应融合')]
    contrasts=[]
    for a,b,hypothesis in pairs:
        if a not in indexed or b not in indexed or not all(k in indexed[n]['evaluations'] for n in (a,b) for k in keys[:4]):continue
        ratios=[indexed[a]['evaluations'][k]['mae']/indexed[b]['evaluations'][k]['mae'] for k in keys[:4]]
        contrasts.append([a,b,hypothesis]+[f'{100*(v-1):+.2f}%' for v in ratios])
    tab(['方法','对照','所检验因素','ID','两元','形态','三元'],contrasts)
    add('该表包含单seed机制消融，只能定位信号；U07相对U05同时涉及信息作用位置和头形式，不能视为严格单因素。\n')
    failures={p.stem:json.loads(p.read_text()) for p in (suite/'failures').glob('*.json')}
    if failures:add('失败/中断记录（后续恢复成功也保留审计）：`'+json.dumps(failures,ensure_ascii=False)+'`\n')
    sfile=suite/'selection.json'
    if sfile.exists():
        selection=json.loads(sfile.read_text());add('冻结候选：通信'+str(selection['memory'])+'；尺度'+str(selection['scale'])+'。\n')
        tab(['候选','ID val','相对U01','通过1%初筛','验证面积','可训练参数'],[[n,f(s['val_mae']),f(s['id_ratio']),s['eligible'],f(s['area']),s['trainable_params']] for n,s in selection['scores'].items()])
        families=[('通信第一',[selection['memory'][0],'U19','U20']),('通信第二',[selection['memory'][1],'U21','U22']),('尺度第一',[selection['scale'][0],'U23','U24']),('尺度第二',[selection['scale'][1],'U25','U26']),('组合',['U17','U27','U28'])]
        records=[]
        for label,aa in families:
            if not all(n in indexed and keys[0] in indexed[n]['evaluations'] for n in aa+['U01','U29','U30']):records.append([label,'配对未齐','—','—']);continue
            vals=[indexed[a]['evaluations'][keys[0]]['mae']/indexed[b]['evaluations'][keys[0]]['mae'] for a,b in zip(aa,['U01','U29','U30'])]
            improve=100*(1-statistics.mean(vals));passed=all(v<1 for v in vals) and improve>=1
            records.append([label,'达到ID推进标准' if passed else '未达稳定ID标准',', '.join(f'{100*(v-1):+.2f}%' for v in vals),f'{improve:.2f}%'])
        tab(['方法族','分级','seed7/17/27 ID变化','平均ID改善'],records)
        # Confirmation and transfer comparisons retain every protocol and seed.
        for folder in ('confirmation','rate_transfer'):
            add('### '+folder+'配对变化\n');rs=[]
            for _,aa in families:
                for a,b in zip(aa,['U01','U29','U30']):
                    files=[suite/folder/f'{n}.json' for n in (a,b)]
                    if not all(f.exists() for f in files):continue
                    x,y=[json.loads(f.read_text()) for f in files]
                    if x.get('status')!=y.get('status') or x.get('status')!='finished':continue
                    for k,v in x['sets'].items():
                        ratio=v['metrics']['mae']/y['sets'][k]['metrics']['mae'];rs.append([a,b,k,f'{100*(ratio-1):+.2f}%', '退步>5%' if ratio>1.05 else ''])
            tab(['方法','同seed参考','协议','MAE变化','提示'],rs)
        for source in [selection['memory'][0],selection['scale'][0]]:
            a,b=indexed.get('U17'),indexed.get(source)
            if a and b and keys[0] in a['evaluations'] and keys[0] in b['evaluations']:
                add(f'U17对单模块{source}的seed7 ID变化：{100*(a["evaluations"][keys[0]]["mae"]/b["evaluations"][keys[0]]["mae"]-1):+.2f}%。不凭单种子组合结果声称协同。\n')
    else:add('候选尚未冻结，以上为阶段结果。\n')
    add('机制边界：零初始化、门未饱和、头梯度非零、路径更丰富都不等于更优。操作内变化E−Z不是已证实的修复量。受约束9/10路径方法不称为完全自由81路径。若best在20epoch或更早，该组最终模型尚未经历开放路由，不能把其成绩归因于后续阶段；阶段字段和best epoch须同时核对。U13/U14增加激活数，只能按同执行结构对照并另报精度/成本。U18只有seed7，不宣称稳定融合改进。\n')
    add('建议：优先保留三个seed均降低ID且平均≥1%的方法；附上所有OOD和迁移代价后再决定产品/论文取舍。若无方法达到标准，应保留CMFF，不将补位复验包装为初筛成功。通信有效再做幅度/内容/梯度最小消融；尺度有效再区分固定阶段、路径约束与教师信号。当前仍仅单数据集时间切分，新mask不等于跨数据集泛化，三seed不直接宣称统计显著。\n')
    (assets/'audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2));(assets/'diagnostics.json').write_text(json.dumps(diags,ensure_ascii=False));(assets/'paired.json').write_text(json.dumps(computed,ensure_ascii=False,indent=2));(assets/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    add('附件：'+link(assets/'audit.json','审计')+'；'+link(assets/'diagnostics.json','完整ID诊断')+'；'+link(assets/'paired.json','配对结果')+'。')
    dest=out/(stem+'.md');dest.write_text('\n'.join(lines));(suite/'report_path.txt').write_text(str(dest)+'\n');return dest
