"""Method-first formal stage/final reports, generated automatically by the queue."""
from __future__ import annotations
import json,math,os
from datetime import datetime
from pathlib import Path

MEMORY={'none':'原生direct','residual':'专家输出与当前状态凸组合，保留系数初始0.1','anchor':'零初始化H0锚定','signed':'最近历史tanh门，初始0.05','positive':'最近历史sigmoid门，初始0.05','delta':'RMS归一化最近差分','delta_detach':'RMS差分停止梯度','router':'历史/差分只修正Router','both':'历史同时修正Router和专家输入','message':'已选双专家响应、身份与权重传递','gru':'64维GRU跨轮压缩记忆'}
SCALE={'fine':'单F','fixed':'固定CMFF','free':'自由三选一ST','sample':'前20epoch混合均匀采样，之后argmax','teacher':'自由三选一＋最终效果监督','history':'历史条件化尺度头','bank':'三个尺度分别维护状态缓存','top2':'三选二softmax融合','fixed2':'固定CM/MF/CF/MF等权','soft3':'三尺度全执行条件化融合','equal3':'三尺度全执行等权','permutation':'指定epoch随机12种CMFF排列，其余固定'}
PAIR={'native':'原生Top2','equal':'原生Top2等权','joint':'原生组合分数＋零初始化修正','joint_history':'历史条件化组合修正','partner':'原生主专家＋残差搭档','bounded':'原生Top2＋±0.5有界响应融合修正','joint_teacher_only':'组合修正头仅候选监督，无任务ST梯度'}

def method(job):
    if not job:return '待冻结候选'
    cfg=job['config'];spec=cfg['model']['coe'].get('four_direction')
    if not spec:return cfg['experiment_plan']['method']
    return '；'.join([MEMORY[spec.get('memory','none')],SCALE[spec.get('scale','fine')],PAIR[spec.get('pair','native')],
                      '教师='+spec.get('teacher','none'), '专家池='+cfg['model']['coe']['expert_sharing'],
                      '扰动epoch='+str(spec.get('explore_range','无'))])


def export_report(root,suite,rows,jobs,label):
    root=Path(root);suite=Path(suite);out=root/'experments_report';out.mkdir(exist_ok=True)
    date=datetime.now().strftime('%Y%m%d');finished=sum(r['status']=='finished' for r in rows)
    suffix='完整实验分析' if label=='final' and finished==60 else f'阶段分析_{label}'
    stem=f'{date}_v24_四方向60组_{suite.name}_{suffix}';assets=out/(stem+'_assets');assets.mkdir(exist_ok=True)
    lines=[f'# {date} v24 四方向60组：{suffix}','',f'生成时间：{datetime.now().isoformat(timespec="seconds")}；完成{finished}/{len(rows)}。队列`{suite.name}`。','',
           '## 1. 每个实验的具体做法与复现信息','']
    def add(s=''):lines.append(s)
    def table(h,rs):
        add('| '+' | '.join(h)+' |');add('| '+' | '.join(['---']*len(h))+' |')
        for r in rs:add('| '+' | '.join(map(str,r))+' |')
        add()
    def link(p,text):return f'[{text}]({os.path.relpath(p,out)})'
    def f(x):return '—' if x is None else f'{x:.3f}'
    table(['编号','具体做法','参考','seed','状态'],[[r['variant'],method(jobs.get(r['variant'])),r['reference'],r['seed'],r['status']] for r in rows])
    add('共同条件：默认TaxiBJ random0.4、GPU0单卡、batch32、100epoch、val5；实际覆盖值见逐组config。AdamW 1e-3余弦至3e-4、wd1e-4、clip1、AMP；四轮八专家，原生组内softmax，默认direct、共享池、关闭completion feedback。F/M/C=原始网格/2/4倍降采样；TaxiBJ为32/16/8，时间12。旧双尺度C16在新记号中为M。每个激活尺度只执行所选两个专家。三选二和全三尺度的专家次数、面积及耗时必须单独比较。\n')
    add('训练为随机点/节点中断/时间缺口/空间区域四类，mask seed20260917，逐epoch重采样；模型和loader seed同时变更。best只按ID验证MAE保存，last每epoch保存完整恢复状态。候选依据验证时间段的OOD协议20261011/12/13筛选；测试协议、新mask20261021/22/23及缺失率迁移均不参与筛选。所有seed偏移遵守val+20000、test+30000。\n')
    add('默认损失L1+0.01原生candidate均衡。教师组每20batch、最多4有效窗口、轮次循环，尺度比较3候选/专家比较5候选，运行剩余轮次的最终误差生成标签（P06为当前轮），0.1倍交叉熵只更新选择头。试运行无梯度、不改变历史或主随机流，推理不执行候选。\n')
    add('精确公式、模块启用条件和运行命令见冻结README：'+link(suite/'source_snapshot/scripts/v24/README_FOUR_DIRECTION_EXPLORATION.md','方法与复现说明')+'；'+link(suite/'plan.json','源码/数据指纹')+'；'+link(suite/'resolved_jobs.json','动态配置')+'。\n')
    table(['实验','配置','运行日志','测试评估'],[[r['variant'],link(suite/'configs'/f"{r['variant']}.json",'config') if r['variant'] in jobs and (suite/'configs'/f"{r['variant']}.json").exists() else '待执行',link(Path(r['run_dir'])/'logs','logs') if r['run_dir'] else '—',link(suite/'evaluations'/f"{r['variant']}.json",'evaluation') if (suite/'evaluations'/f"{r['variant']}.json").exists() else '—'] for r in rows])
    add('## 2. 实验结果\n')
    protocols=['in_distribution','unseen_combinations','unseen_geometry','unseen_triple','unseen_combinations_repeat','unseen_geometry_repeat']
    table(['组','best epoch','best val MAE']+protocols,[[r['variant'],r['best_epoch'],f(r['val_mae'])]+[f"{f(r['evaluations'].get(k,{}).get('mae'))} / {f(r['evaluations'].get(k,{}).get('rmse'))}" for k in protocols] for r in rows])
    add('表中测试列为MAE / RMSE，越低越好；未完成组保留为—，不能纳入最终排名。MAPE受零/近零目标影响，不作为结论依据。\n')
    histories={};audits={};diagnostics={};cost=[];family=[]
    for r in rows:
        n=r['variant']
        if not r['run_dir']:continue
        run=Path(r['run_dir']);history=[json.loads(s) for s in (run/'logs/metrics.jsonl').read_text().splitlines()];histories[n]=history
        meta=json.loads((run/'training_metadata.json').read_text())
        epath=suite/'evaluations'/f'{n}.json'
        e=json.loads(epath.read_text()) if epath.exists() else {'sets':{}}
        m=e['sets'].get('in_distribution',{}).get('metrics',{})
        audits[n]={'epochs':len(history),'validations':sum(bool(h.get('val')) for h in history),'best_exists':(run/'checkpoints/best.pth').exists(),'last_exists':(run/'checkpoints/last.pth').exists(),
                   'amp_skips':sum(h['train'].get('train_skipped_amp_steps',0) for h in history),
                   'nonfinite_diagnostics':[(h['epoch'],sp,k,v) for h in history for sp in ('train','val') for k,v in (h.get(sp) or {}).items() if isinstance(v,str) and v in ('nan','inf','-inf')]}
        paths={k:v for k,v in m.items() if k.startswith('coe_scale_execution_path_') and v>0}
        if not paths:paths={k:v for k,v in m.items() if k.startswith('coe_scale_path_') and v>0}
        top=sorted(paths.items(),key=lambda x:-x[1])[:3]
        diagnostics[n]={'paths':paths,'rounds':{k:v for k,v in m.items() if k.startswith('coe_step')},'families':{k:v['metrics'] for k,v in e['sets'].items()}}
        cost.append([n,meta['total_params'],meta['trainable_params'],f(r['training_seconds']/60 if r['training_seconds'] else None),f(max(h['perf']['peak_memory_gb'] for h in history)),f(m.get('coe_expert_execution_count')),f(m.get('coe_expert_grid_equivalents')),', '.join(f'{k.removeprefix("coe_scale_execution_path_").removeprefix("coe_scale_path_").removesuffix("_fraction")} {v:.1%}' for k,v in top)])
        for protocol,v in e['sets'].items():
            for key,mae in v['metrics'].items():
                if key.startswith('coe_family_') and key.endswith('_mae'):family.append([n,protocol,key[len('coe_family_'):-4],f(mae),f(v['metrics'].get(key[:-3]+'rmse'))])
    table(['组','总参数','可训练参数','训练程序min','峰值GiB','专家次数','面积代理','主要尺度执行路径'],cost)
    add('面积代理非FLOPs；训练时间包含验证和检查点保存。教师试运行开销另见train日志four_probe_seconds/candidates；未计入正常前向专家次数。混合尺度路径以实际激活集合记录，不能用最高权重尺度代替实际执行。\n')
    table(['组','epoch','验证次数','best/last','AMP跳步','非有限诊断数'],[[n,a['epochs'],a['validations'],f"{a['best_exists']}/{a['last_exists']}",a['amp_skips'],len(a['nonfinite_diagnostics'])] for n,a in audits.items()])
    table(['组','train loss首→末','train MAE首→末','val20','val40','val60','val80','val100'],[[n,f"{f(h[0]['train']['loss'])}→{f(h[-1]['train']['loss'])}",f"{f(h[0]['train']['mae'])}→{f(h[-1]['train']['mae'])}"]+[f(next((v.get('val',{}).get('mae') for v in h if v['epoch']==i and v.get('val')),None)) for i in [20,40,60,80,100]] for n,h in histories.items()])
    if histories:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axs=plt.subplots(4,2,figsize=(14,16))
        for i,g in enumerate('MSGP'):
            for n,h in histories.items():
                if n[0]!=g:continue
                axs[i,0].plot([v['epoch'] for v in h],[v['train']['loss'] for v in h],label=n)
                axs[i,1].plot([v['epoch'] for v in h if v.get('val')],[v['val']['mae'] for v in h if v.get('val')],label=n)
            for j,title in enumerate(['train loss','ID validation MAE']):
                ax=axs[i,j];ax.set_title(g+' '+title);ax.grid(alpha=.25);ax.set_xlabel('epoch')
                if ax.lines:ax.legend(ncol=4,fontsize=7)
        fig.tight_layout();fig.savefig(assets/'curves.png',dpi=130);fig.savefig(assets/'curves.svg');plt.close(fig)
        add(f'![训练和验证曲线]({assets.name}/curves.png)\n')
    add('### 分缺失类型结果\n');table(['组','协议','缺失族','MAE','RMSE'],family)
    for folder,title in [('reference_evaluations','旧强参考'),('confirmation','新mask确认'),('rate_transfer','缺失率迁移')]:
        add('### '+title+'\n');items=[]
        for path in sorted((suite/folder).glob('*.json')):
            data=json.loads(path.read_text())
            for k,v in data.get('sets',{}).items():items.append([path.stem,data['status'],k,f(v['metrics']['mae']),f(v['metrics']['rmse'])])
        table(['模型','状态','协议','MAE','RMSE'],items)
    add('## 3. 结果分析、证据边界与建议\n')
    indexed={r['variant']:r for r in rows};pairs=[]
    for path in (suite/'reference_evaluations').glob('*.json'):
        ref=json.loads(path.read_text())
        if ref.get('status')=='finished':
            indexed[path.stem]={'variant':path.stem,'status':'finished','evaluations':{k:{m:v['metrics'][m] for m in ('mae','rmse')} for k,v in ref['sets'].items()}}

    for r in rows:
        b=indexed.get(r['reference'])
        if r['status']!='finished' or b is None or b['status']!='finished' or r['variant']==r['reference']:continue
        keys=protocols[:4]
        differences=[100*(r['evaluations'][k]['mae']/b['evaluations'][k]['mae']-1) for k in keys]
        pairs.append([r['variant'],r['reference']]+[f'{x:+.2f}%' for x in differences])
    table(['方法','指定参考','ID MAE变化','两元变化','形态变化','三元变化'],pairs)
    add('正值表示退步，负值表示改善。只比较同一数据协议；不能把不同OOD/ID的绝对误差比当作归一化泛化能力。\n')
    sp=suite/'selection.json'
    if sp.exists():
        selected=json.loads(sp.read_text());add('冻结候选：`'+json.dumps({k:v for k,v in selected.items() if k not in ('scores','evidence_sha256')},ensure_ascii=False)+'`。\n')
        table(['候选','参考','ID val比值','OOD val比值','J','通过ID门槛'],[[n,x['reference'],f(x['id_ratio']),f(x['ood_ratio']),f(x['J']),x['eligible']] for n,x in selected['scores'].items()])
        families=[('memory1',[selected['memory'][0],'M15','M16'],['M01','M13','M14']),('memory2',[selected['memory'][1],'M17','M18'],['M01','M13','M14']),('scale',[selected['scale'],'S13','S14'],['S01','S11','S12']),('mixture',[selected['mixture'],'S15','S16'],[selected['mixture_reference'],'S17','S18']),('paths',[selected['generalization'],'G10','G11'],['S01','S11','S12']),('pair1',[selected['pair'][0],'P09','P10'],['M01','M13','M14']),('pair2',[selected['pair'][1],'P11','P12'],['M01','M13','M14'])]
        families.append(('R5三种子',['R5','G01','G02'],['N5','R2','G03']))
        families.append(('R5独立池双种子',['G04','G05'],['R6','G06']))
        verdicts=[]
        for name,aa,bb in families:
            if any(n not in indexed or indexed[n]['status']!='finished' for n in aa+bb):verdicts.append([name,'等待三种子','—']);continue
            ratios=[{k:indexed[a]['evaluations'][k]['mae']/indexed[b]['evaluations'][k]['mae'] for k in protocols[:4]} for a,b in zip(aa,bb)]
            win=lambda r:r['in_distribution']<1 and sum(r[k] for k in protocols[1:4])/3<1
            trade=lambda r:r['in_distribution']<=1.02 and sum(r[k] for k in protocols[1:4])/3<=.95 and max(r[k] for k in protocols[1:4])<=1.05
            verdict='三种子研发测试可推进信号，仍须确认与成本审核' if all(map(win,ratios)) else '三种子精度/泛化权衡，仍须确认' if all(map(trade,ratios)) else '部分种子信号，尚未稳定' if any(map(win,ratios)) else '未达推进标准'
            confirmation_ratios=[]
            for a,b in zip(aa,bb):
                ap=suite/'confirmation'/f'{a}.json';bp=suite/'confirmation'/f'{b}.json'
                if ap.exists() and bp.exists():
                    ar=json.loads(ap.read_text());br=json.loads(bp.read_text())
                    if ar.get('status')==br.get('status')=='finished':confirmation_ratios.append([ar['sets'][k]['metrics']['mae']/br['sets'][k]['metrics']['mae'] for k in protocols[1:4]])
            if len(confirmation_ratios)==len(aa):
                confirmed=all(sum(v)/3<1 for v in confirmation_ratios)
                verdict += '；新mask平均OOD各seed均改善' if confirmed else '；新mask未在各seed均保持平均OOD改善'
            verdicts.append([name,verdict,'; '.join(f"ID{r['in_distribution']:.3f}/OOD{sum(r[k] for k in protocols[1:4])/3:.3f}" for r in ratios)])
        table(['候选族','分级','seed7/17/27误差比值'],verdicts)
    else:add('尚未冻结全部候选；以上只能作为阶段结果，不能宣布复验成功。\n')
    add('机制判断要求：历史门控或消息被使用不等于有益；尺度路径多样性不等于效果提升；增加尺度激活后的改善必须同时披露成本。无中间监督时，中间解码MAE不能证明各轮单调修复。M05/M06初始前向相同但门函数不同；M07/M08隔离历史梯度。P06与最终教师比较用于检验短期目标是否失配，不推广为所有贪心方法无效。\n')
    add('新mask确认与研发测试共享时间窗口，不是跨数据集验证；缺失率迁移未重训，仅代表该变化下的表现。只有三个训练种子，不直接宣称统计显著。G12单种子且组合多个机制，不能独立证明协同效应。若自由尺度仍固定路径，应如实注明未证明按输入条件化选择。\n')
    add('下一步建议按冻结的配对结果推进：优先保留在三种子及新mask下同时改善精度与泛化的机制；对只有单种子信号的配置继续成对复验，不直接新增模块。若收益仅来自更多尺度执行，归类为计算/精度权衡。与旧N5/R5/R6比较后再决定主线；没有稳定增益时保留已有强参考，不把复杂度当作成果。\n')
    (assets/'audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2));(assets/'diagnostics.json').write_text(json.dumps(diagnostics,ensure_ascii=False));(assets/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    add('附件：'+link(assets/'audit.json','完整性核验')+'；'+link(assets/'diagnostics.json','逐轮/路径/类型诊断')+'；'+link(assets/'summary.json','机器可读汇总')+'。')
    destination=out/(stem+'.md');destination.write_text('\n'.join(lines));(suite/'report_path.txt').write_text(str(destination)+'\n');return destination
