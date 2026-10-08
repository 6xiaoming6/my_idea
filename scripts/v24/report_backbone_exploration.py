"""Frozen W-series method-first reporting. Single-seed signals, no significance claims."""
import json,math,os
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path


def export_report(root,suite,rows,jobs):
    root=Path(root);suite=Path(suite);out=root/'experments_report';out.mkdir(exist_ok=True)
    done=sum(r['status']=='finished' for r in rows);final=done==30 and (suite/'confirmation_complete.json').exists()
    stem=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')+'_v24_W系列固定骨干30组_'+suite.name+('_完整实验分析' if final else '_阶段分析')
    assets=out/(stem+'_assets');assets.mkdir(exist_ok=True)
    lines=[f'# {stem}','',f'训练/六协议测试完成{done}/30；确认评估'+('完成' if final else '尚未全部完成')+'。','', '## 1. 各组具体做法与复现配置','']
    def add(s=''):lines.append(s)
    def f(x):return '—' if x is None else f'{x:.4f}' if isinstance(x,float) else str(x)
    def table(h,rr):
        add('| '+' | '.join(h)+' |');add('| '+' | '.join(['---']*len(h))+' |')
        for row in rr:add('| '+' | '.join(f(v) for v in row)+' |')
        add()
    def link(p,label):return f'[{label}]({os.path.relpath(p,out)})'
    table(['组','具体方法','直接参考','状态'],[[r['variant'],r['method'],r['reference'],r['status']] for r in rows])
    cfg=jobs['W01']['config']
    add(f"公共预算：{cfg['data']['dataset_name']} random0.4，batch{cfg['data']['batch_size']}，{cfg['train']['epochs']}epoch，val{cfg['train']['val_epoch']}，seed7模型/loader，mask20260917，四基础缺失，原切分，训练每epoch重采样。AdamW1e-3 cosine到3e-4、wd1e-4、clip1、AMP，无早停。best只按ID验证MAE，保存best/last和完整恢复状态。")
    add('主干固定U02：四轮CMFF，TaxiBJ尺度8/16/32、时间12；共享八专家T/S/TD/SD/TA/ST/TL/SL，每轮原生Top2与组内softmax，direct，±0.1 RMS匹配差分传递。原completion通道不更新。新增适配位于投影归一化后的Z；反馈先加到原差分修正后，再进入投影。每推理窗口8次专家执行，面积代理4.625，不是等FLOPs。')
    add('适配：身份/仿射零初始化；rank8为Z+0.1B GELU(AZ)，B零初始化；覆盖率调制2sigmoid(ac+b)，初始1。反馈第二轮起，R=M(Xobs−上一轮原始解码)/max(RMSobs(Xobs),1)；点反馈或3×3×3观测归一化传播，无观测邻域置零。输入修正0.05 RMS(H)tanh(两层32维头)，路由修正0.1tanh(64维头)，输出零初始化；W15保留残差梯度，其他detach。W11/W18增加四轮可见重建平均L1×0.02。')
    add('第二视图取每batch前8窗，独立几何随机流在原观测中追加遮挡到0.5，学生额外补全L1×0.25。W20+共同隐藏一致性×0.05；W21用0.99EMA教师，仅成功optimizer step后更新，仍评估学生；W22用教师可见误差形成停止梯度归一化权重。教师不读共同隐藏标签。W23/24只对最终预测做C/M均值监督，各0.025，区别为区域等权/缺失数量加权；W25/26用最终x_comp的时间/空间邻边误差×0.05，仅两端有效且至少一端有效隐藏。')
    add('W19–22及其组合可能暴露基础缺失的组合，原两元协议不能再全部解释为未见组合；所有组仍使用同一套固定测试mask，保留协议名便于比较并明确暴露差异。新确认mask仍共享测试时间窗口，不是新数据集。')
    add('全部使用严格确定性后端：reshape/reduce池化、可分离矩阵双线性上采样、确定性CUDA、禁用TF32与非数学SDPA。新增模块隔离初始化随机流，并从冻结common_initialization.pth复制公共参数。W01/W02是完全相同seed的工程重放，不是两个训练种子。旧成绩仅历史参考。')
    add('完整机制/命令：'+link(suite/'source_snapshot/scripts/v24/README_BACKBONE_EXPLORATION.md','冻结README')+'；'+link(suite/'plan.json','冻结源码、配置与数据指纹')+'。入口`python -u scripts/v24/run_backbone_exploration.py --dataset taxibj --gpu 0 --epochs 100 --batch-size 32`；恢复指定`--suite`原队列。')
    table(['组','配置','日志'],[[r['variant'],link(suite/'configs'/f"{r['variant']}.json",'config') if (suite/'configs'/f"{r['variant']}.json").exists() else '待解析',link(Path(r['run_dir'])/'logs','logs') if r['run_dir'] else '—'] for r in rows])
    add('## 2. 全部结果、曲线、诊断与完整性')
    keys=list(cfg['experiment_plan']['protocol']['evaluations'])
    table(['组','best epoch','ID val']+[k+' MAE/RMSE' for k in keys],[[r['variant'],r['best_epoch'],r['val_mae']]+[f(r['evaluations'].get(k,{}).get('mae'))+'/'+f(r['evaluations'].get(k,{}).get('rmse')) for k in keys] for r in rows])
    histories={};cost=[];audits={};details={};families=[];extra=[]
    for r in rows:
        if not r['run_dir']:continue
        n=r['variant'];run=Path(r['run_dir'])
        if not (run/'logs/metrics.jsonl').exists() or not (run/'training_metadata.json').exists():continue
        hs=[json.loads(l) for l in (run/'logs/metrics.jsonl').read_text().splitlines()]
        if not hs:continue
        histories[n]=hs
        meta=json.loads((run/'training_metadata.json').read_text());ep=suite/'evaluations'/f'{n}.json';ev=json.loads(ep.read_text()) if ep.exists() else {'sets':{}}
        m=ev['sets'].get('in_distribution',{}).get('metrics',{});details[n]=m
        bad=[(h['epoch'],sp,k) for h in hs for sp in ('train','val') for k,v in (h.get(sp) or {}).items() if isinstance(v,str) and v in ('nan','inf','-inf') or isinstance(v,float) and not math.isfinite(v)]
        audits[n]={'epochs':len(hs),'validations':sum(h['val'] is not None for h in hs),'best':(run/'checkpoints/best.pth').exists(),'last':(run/'checkpoints/last.pth').exists(),'nonfinite':bad,'amp_skips':sum(h['train'].get('train_skipped_amp_steps',0) for h in hs)}
        replay=json.loads((run/'replay_audit.json').read_text()) if (run/'replay_audit.json').exists() else [];sw=sum(a['extra_student_windows'] for a in replay);tw=sum(a['extra_teacher_windows'] for a in replay)
        cost.append([n,meta['total_params'],meta['trainable_params'],(r['training_seconds'] if r['training_seconds'] is not None else sum(h['perf']['epoch_time_sec'] for h in hs))/60,max(h['perf']['peak_memory_gb'] for h in hs),m.get('forward_ms_per_sample_per_rank'),sw,tw])
        for name,s in ev['sets'].items():
            for k,v in s['metrics'].items():
                if k.startswith('coe_family_') and k.endswith('_mae'):families.append([n,name,k[11:-4],v,s['metrics'].get(k[:-3]+'rmse')])
        extra.append([n]+[sum(h['train'].get(k,0) for h in hs)/len(hs) for k in ['l_w_observed','l_w_second_view','l_w_consistency','l_w_region_c','l_w_region_m','l_w_temporal','l_w_spatial']])
    table(['组','总参数','可训练参数','训练/验证min','峰值GiB','ID ms/窗','额外学生窗次数','额外教师窗次数'],cost)
    table(['组','epoch/val','best/last','AMP跳步','非有限数'],[[n,f"{a['epochs']}/{a['validations']}",f"{a['best']}/{a['last']}",a['amp_skips'],len(a['nonfinite'])] for n,a in audits.items()])
    table(['组','观察重建','二视图','一致性','C区域','M区域','时间差分','空间差分'],extra)
    add('上表附加损失为epoch均值的算术平均，只是训练诊断，不是精确全数据误差。额外前向按实际窗口次数累计；额外视图没有新增第二份balance或结构损失。')
    table(['组','协议','缺失族','MAE','RMSE'],families)
    table(['组','epoch20 val','epoch40 val','epoch60 val','epoch80 val','epoch100 val'],[[n]+[next((h['val']['mae'] for h in hs if h['epoch']==e and h['val']),None) for e in (20,40,60,80,100)] for n,hs in histories.items()])
    diagkeys=[f'coe_w_step{i}_{k}' for i in (2,3,4) for k in ('adapter_abs','feedback_abs','feedback_route_abs','visible_residual_abs','feedback_coverage')]
    table(['组']+diagkeys,[[n]+[m.get(k) for k in diagkeys] for n,m in details.items()])
    if histories:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axs=plt.subplots(4,2,figsize=(13,15))
        for idx,(lo,hi) in enumerate([(1,10),(11,18),(19,26),(27,30)]):
            for n,hs in histories.items():
                if not lo<=int(n[1:])<=hi:continue
                axs[idx,0].plot([h['epoch'] for h in hs],[h['train']['mae'] for h in hs],label=n)
                vs=[h for h in hs if h['val']];axs[idx,1].plot([h['epoch'] for h in vs],[h['val']['mae'] for h in vs],label=n)
            for ax,title in zip(axs[idx],['Train MAE','ID validation MAE']):
                ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
                if ax.lines:ax.legend(ncol=3,fontsize=8)
        fig.tight_layout();fig.savefig(assets/'curves.png',dpi=140);plt.close(fig);add(f'![曲线]({assets.name}/curves.png)')
    table(['组','实测专家调用/窗','面积代理'],[[n,m.get('coe_expert_execution_count'),m.get('coe_expert_grid_equivalents')] for n,m in details.items()])
    table(['组','轮']+['T','S','TD','SD','TA','ST','TL','SL'],[[n,i]+[m.get(f'coe_step{i}_{e}_selection_rate') for e in ('T','S','TD','SD','TA','ST','TL','SL')] for n,m in details.items() for i in range(1,5)])
    add('上述为专家被选中比例，每轮合计2；完整专家权重、路径和诊断保存于 '+link(assets/'id_diagnostics.json','ID诊断')+'。')
    gate=suite/'replay_gate.json';add('重放审计：`'+(gate.read_text() if gate.exists() else '尚未完成')+'`。')
    confirm=[]
    for p in sorted((suite/'confirmation').glob('*.json')):
        d=json.loads(p.read_text())
        for k,z in d.get('sets',{}).items():confirm.append([p.stem,d['status'],k,z['metrics']['mae'],z['metrics']['rmse']])
    table(['组','状态','新mask确认','MAE','RMSE'],confirm)
    add('## 3. 结果分析、局限与建议')
    index={r['variant']:r for r in rows};paired=[]
    for r in rows:
        b=index[r['reference']]
        if r is b or not all(k in r['evaluations'] and k in b['evaluations'] for k in keys[:4]):continue
        ratios=[r['evaluations'][k]['mae']/b['evaluations'][k]['mae'] for k in keys[:4]]
        paired.append([r['variant'],r['reference']]+[f'{100*(v-1):+.2f}%' for v in ratios]+['泛化代价>5%' if max(ratios[1:])>1.05 else '未触发5%标记'])
    table(['组','直接参考','ID','两元','形态','三元','提示'],paired)
    b=index['W01'];signals=[]
    for r in rows:
        if r['variant'] in ('W01','W02') or not all('in_distribution' in v['evaluations'] for v in (r,b)):continue
        ratio=r['evaluations']['in_distribution']['mae']/b['evaluations']['in_distribution']['mae'];passed=ratio<=.99 and r['val_mae']<b['val_mae']
        costs=[k for k in keys[1:4] if k in r['evaluations'] and k in b['evaluations'] and r['evaluations'][k]['mae']>1.05*b['evaluations'][k]['mae']]
        exposed=jobs[r['variant']]['config']['model']['coe']['backbone_exploration'].get('view','none')!='none'
        signals.append([r['variant'],'单种子候选' if passed else '未达预设ID信号标准',f'{100*(ratio-1):+.2f}%',','.join(costs) or '未触发5%标记','第二视图扩展训练分布' if exposed else '四基础族'])
    table(['组','分级','相对W01 ID','相对W01泛化代价>5%的协议','训练暴露'],signals)
    selection=suite/'selection.json'
    if selection.exists():
        sel=json.loads(selection.read_text());add('冻结候选：`'+json.dumps(sel['selected'],ensure_ascii=False)+'`，仅ID验证筛选，不读测试。')
        table(['组','验证MAE','通过1%初筛'],[[n,v['mae'],v['eligible']] for n,v in sel['scores'].items()])
        contrasts=[]
        for n in ('W27','W28','W29','W30'):
            if n not in jobs or 'in_distribution' not in index[n]['evaluations']:continue
            for source in jobs[n]['config']['experiment_plan']['sources']:
                contrasts.append([n,source,f"{100*(index[n]['evaluations']['in_distribution']['mae']/index[source]['evaluations']['in_distribution']['mae']-1):+.2f}%"])
        table(['组合','单模块','ID变化'],contrasts)
    else:add('候选尚未冻结；不提前解释组合收益。')
    failures={p.stem:json.loads(p.read_text()) for p in (suite/'failures').glob('*.json')};add('失败/中断审计（含恢复前记录）：`'+json.dumps(failures,ensure_ascii=False)+'`。')
    add('W01/W02仅工程重放，不构成独立训练种子。所有候选仅seed7，不宣称统计显著或稳定创新。辅助损失、模块非零梯度、覆盖率或路径变化均不是收益证据。可见误差不是隐藏置信度；零观测区不应伪造证据。结构监督与额外视图有不同训练目标/计算成本，不能只按epoch声称公平计算。')
    add('优先对同时降低ID验证、ID测试至少1%的方法再做独立种子配对复验；全部披露OOD代价及新mask结果。组合仅胜过基准而未超过对应单模块时，不声称协同。若无候选，保留CMFF＋有界差分骨干，停止无收益的模块扩张。本次只报告，不能为报告加跑/停止实验。')
    (assets/'audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2));(assets/'id_diagnostics.json').write_text(json.dumps(details,ensure_ascii=False));(assets/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    dest=out/(stem+'.md');dest.write_text('\n'.join(lines));(suite/'report_path.txt').write_text(str(dest)+'\n');return dest
