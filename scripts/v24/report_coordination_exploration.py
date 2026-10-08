"""Method-first V-series report, generated from frozen configs and raw evaluation logs."""
import json,math,os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def export_report(root,suite,rows,jobs):
    root=Path(root);suite=Path(suite);destdir=root/'experments_report';destdir.mkdir(exist_ok=True)
    done=sum(r['status']=='finished' for r in rows);final=done==4
    date=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
    stem=f'{date}_v24_V1-V4三尺度协调_{suite.name}_'+('完整实验分析' if final else '阶段分析')
    assets=destdir/(stem+'_assets');assets.mkdir(exist_ok=True)
    lines=[f'# {stem}','',f'完成{done}/4；'+('全部训练与六协议评估已完成。' if final else '队列尚未完成，以下为阶段结果。'),'', '## 1. 各组具体做法与复现信息','']
    def add(x=''):lines.append(x)
    def fmt(x):return '—' if x is None else f'{x:.4f}' if isinstance(x,float) else str(x)
    def table(headers,data):
        add('| '+' | '.join(headers)+' |');add('| '+' | '.join(['---']*len(headers))+' |')
        for row in data:add('| '+' | '.join(fmt(v) for v in row)+' |')
        add()
    def link(path,label):return f'[{label}]({os.path.relpath(path,destdir)})'
    table(['组','具体方法','直接对照','状态'],[[r['variant'],r['method'],r['reference'],r['status']] for r in rows])
    cfg=jobs['V1']['config']
    add(f"实际预算：{cfg['data']['dataset_name']} random0.4，batch{cfg['data']['batch_size']}，{cfg['train']['epochs']}epoch，每{cfg['train']['val_epoch']}epoch验证，seed{cfg['seed']}。AdamW1e-3余弦至3e-4、wd1e-4、clip1、AMP；best只按ID val MAE，保存best/last。")
    add('全部基于U02：四轮共享八专家T/S/TD/SD/TA/ST/TL/SL，原生Top2＋组内softmax，CMFF（TaxiBJ 8/16/32，T12），direct、无completion feedback，±0.1 RMS状态差分通信。四基础缺失训练每epoch重采样，原时间切分/固定验证和六套测试mask保持不变。训练seed和loader seed同为7。')
    add('V2–V4复制公共decoder为独立C/M区域头，分别读取第1/2轮原生尺度专家融合输出，预测缺失位置均值；各0.05区域等权L1。只监督缺失标签全部有效的区域。V3输出校正C→M，各修正10%的区域总量差异；V4使用共享16维1×1分配头（hidden/预测/mask/均值差/尺度身份），缺失位置softmax，零输出初始化使起点等于V3。没有物理守恒、额外专家、接受门或尺度路由。V3/V4主任务梯度通过修正回传；aux也更新骨干。')
    add('新增模块使用decoder复制/隔离随机流，不改变公共骨干初始化。V4增加可训练参数，不能声称严格等参数。所有组每窗口8次专家执行，面积代理4.625不等于FLOPs。原四轮诊断不包含最终协调修正，不把未监督中间解码误差称为逐轮单调改善。')
    add('复现：`python -u scripts/v24/run_coordination_exploration.py --dataset '+str(jobs['V1']['config']['data']['dataset_name']).lower()+' --gpu 0 --epochs '+str(cfg['train']['epochs'])+' --batch-size '+str(cfg['data']['batch_size'])+'`。恢复指定`--suite`原路径。')
    add(link(suite/'plan.json','冻结配置、源码哈希与数据指纹')+'；'+link(suite/'source_snapshot/scripts/v24/README_COORDINATION.md','完整机制定义')+'。')
    table(['组','配置','日志'],[[r['variant'],link(suite/'configs'/f"{r['variant']}.json",'config'),link(Path(r['run_dir'])/'logs','logs') if r['run_dir'] else '—'] for r in rows])
    add('## 2. 结果、训练曲线及机制诊断\n')
    keys=list(cfg['experiment_plan']['protocol']['evaluations'])
    table(['组','best epoch','ID val MAE']+[k+' MAE/RMSE' for k in keys],[[r['variant'],r['best_epoch'],r['val_mae']]+[fmt(r['evaluations'].get(k,{}).get('mae'))+'/'+fmt(r['evaluations'].get(k,{}).get('rmse')) for k in keys] for r in rows])
    histories={};metrics={};cost=[];audit=[];family=[];diagnostics=[]
    for r in rows:
        if not r['run_dir']:continue
        n=r['variant'];run=Path(r['run_dir']);hp=run/'logs/metrics.jsonl'
        hs=[json.loads(x) for x in hp.read_text().splitlines()] if hp.exists() else [];histories[n]=hs
        meta=json.loads((run/'training_metadata.json').read_text())
        ep=suite/'evaluations'/f'{n}.json';ev=json.loads(ep.read_text()) if ep.exists() else {'sets':{}}
        bad=[]
        for h in hs:
            for sp in ('train','val'):
                for k,v in (h.get(sp) or {}).items():
                    if (isinstance(v,str) and v in ('nan','inf','-inf')) or (isinstance(v,float) and not math.isfinite(v)):bad.append((h['epoch'],sp,k))
        audit.append([n,len(hs),sum(h.get('val') is not None for h in hs),(run/'checkpoints/best.pth').exists(),(run/'checkpoints/last.pth').exists(),len(bad),sum(h['train'].get('train_skipped_amp_steps',0) for h in hs)])
        m=ev['sets'].get('in_distribution',{}).get('metrics',{});metrics[n]=m
        cost.append([n,meta['total_params'],meta['trainable_params'],r['training_seconds']/60 if r['training_seconds'] else None,max((h['perf']['peak_memory_gb'] for h in hs),default=0),m.get('forward_ms_per_sample_per_rank'),m.get('coe_expert_execution_count'),m.get('coe_expert_grid_equivalents')])
        for protocol,entry in ev['sets'].items():
            for k,v in entry['metrics'].items():
                if k.startswith('coe_family_') and k.endswith('_mae'):family.append([n,protocol,k[11:-4],v,entry['metrics'].get(k[:-3]+'rmse')])
                if k.startswith('coord_'):diagnostics.append([n,protocol,k,v])
    table(['组','总参数','可训练参数','训练/验证min','峰值GiB','forward ms/sample','专家执行','面积代理'],cost)
    table(['组','epochs','val次数','best','last','非有限字段数','AMP跳步'],audit)
    table(['组','raw MAE','C后 MAE','M后 MAE','C头区域MAE','F原预测C区域MAE','M头区域MAE','F原预测M区域MAE','平均修正','观测变动'],[[n]+[m.get(k) for k in ('coord_raw_mae','coord_after_c_mae','coord_after_m_mae','coord_c_head_region_mae','coord_c_raw_region_mae','coord_m_head_region_mae','coord_m_raw_region_mae','coord_correction_abs','coord_observed_change')] for n,m in metrics.items()])
    add('raw是同一已训练检查点关闭输出协调的结果，非另一个独立训练模型。区域指标按完整有效区域等权，主误差/去均值细节误差按有效缺失点；没有标签的区域不生成虚假零分。V1无区域头/修正诊断，—表示不适用。')
    table(['组','协议','缺失族','MAE','RMSE'],family)
    table(['组','协议','完整协调诊断','数值'],diagnostics)
    if histories:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(11,4))
        for n,hs in histories.items():
            ax[0].plot([h['epoch'] for h in hs],[h['train']['mae'] for h in hs],label=n)
            vs=[h for h in hs if h.get('val')];ax[1].plot([h['epoch'] for h in vs],[h['val']['mae'] for h in vs],label=n)
        for a,title in zip(ax,['Training MAE','ID validation MAE']):a.set_title(title);a.set_xlabel('Epoch');a.legend();a.grid(alpha=.2)
        fig.tight_layout();fig.savefig(assets/'curves.png',dpi=140);plt.close(fig);add(f'![训练曲线]({assets.name}/curves.png)')
        table(['组','首轮train MAE','末轮train MAE','val20','val40','val60','val80','val100'],[[n,hs[0]['train']['mae'],hs[-1]['train']['mae']]+[next((h['val']['mae'] for h in hs if h['epoch']==e and h.get('val')),None) for e in (20,40,60,80,100)] for n,hs in histories.items() if hs])
    failures={p.stem:json.loads(p.read_text()) for p in (suite/'failures').glob('*.json')}
    add('中断/失败审计记录：`'+json.dumps(failures,ensure_ascii=False)+'`。已恢复的历史失败仍保留。')
    add('## 3. 分析、证据边界及建议\n')
    index={r['variant']:r for r in rows};comparisons=[]
    for a,b,label in [('V2','V1','区域辅助监督'),('V3','V2','均匀协调'),('V4','V3','条件分配'),('V3','V1','整体协调'),('V4','V1','整体条件协调')]:
        if all(k in index[n]['evaluations'] for n in (a,b) for k in keys):
            ratios=[100*(index[a]['evaluations'][k]['mae']/index[b]['evaluations'][k]['mae']-1) for k in keys]
            comparisons.append([a,b,label]+[f'{v:+.2f}%' for v in ratios])
            add(f'{a}相对{b}（{label}）ID MAE变化{ratios[0]:+.2f}%，'+('单种子改善信号。' if ratios[0]<0 else '未见ID收益。')+(' 有主要OOD协议退步超过5%。' if max(ratios[1:4])>5 else ''))
    table(['方法','对照','问题']+keys,comparisons)
    for n,m in metrics.items():
        if 'coord_raw_mae' in m:
            change=m['coord_after_m_mae']-m['coord_raw_mae']
            add(f'{n}同检查点协调前后MAE差{change:+.4f}。'+('推理修正直接降低误差。' if change<0 else '推理修正未降低误差；如跨模型改善，不应全部归因于推理校正。'))
    add('仅seed7；不能声称稳定创新或统计显著。V2改善仅支持额外区域监督；V3进一步改善才支持协调机制；V4进一步改善才支持条件分配，但仍有参数增加。C/M目标也会出错，粗头低误差不自动保证单点MAE改善。软协调不是严格跨层一致或物理守恒。')
    add('先检查区域估计是否优于细预测聚合，并联合分析去均值细节误差/分缺失族代价。只有有收益的方法再与其直接参考进行seed17/27配对复验；本批不自动加跑额外组。旧U02成绩仅作背景，公平比较以本批V1为准。')
    if not final:add('当前为阶段分析，不作四组最终排名或完整机制结论。')
    (assets/'audit.json').write_text(json.dumps({'audit':audit,'failures':failures},ensure_ascii=False,indent=2))
    dest=destdir/(stem+'.md');dest.write_text('\n'.join(lines));(suite/'report_path.txt').write_text(str(dest)+'\n');return dest
