"""Method-first reports for the eight fixed-scale/delta controlled experiments."""
import json,math,os,statistics
from pathlib import Path
from datetime import datetime

def export_report(root,suite,rows,jobs):
    root=Path(root);suite=Path(suite);out=root/'experments_report';out.mkdir(exist_ok=True)
    refs={p.stem:json.loads(p.read_text()) for p in (suite/'references').glob('*.json')}
    final=all(r['status']=='finished' for r in rows) and all((suite/g/f'{n}.json').exists() and json.loads((suite/g/f'{n}.json').read_text()).get('status')=='finished' for n in refs for g in ('confirmation','rate_transfer'))
    stem=datetime.now().strftime('%Y%m%d')+'_v24_固定尺度与跨轮差分8组_'+('完整实验分析' if final else '阶段报告')
    assets=out/(stem+'_assets');assets.mkdir(exist_ok=True)
    lines=[f'# {stem}','',f'队列：`{suite.name}`；训练及自身评估完成{sum(r["status"]=="finished" for r in rows)}/{len(rows)}；参考确认评估'+('完成' if final else '尚待核对/完成')+'。','', '## 1. 每组具体做法与复现信息','']
    def add(s=''):lines.append(s)
    def tab(h,rs):
        add('| '+' | '.join(h)+' |');add('| '+' | '.join(['---']*len(h))+' |')
        for r in rs:add('| '+' | '.join(map(str,r))+' |')
        add()
    def f(x):return '—' if x is None else f'{x:.3f}'
    def link(p,title):return f'[{title}]({os.path.relpath(p,out)})'
    tab(['组','seed','具体做法','主要参考'],[[r['variant'],r['seed'],r['method'],r['reference']] for r in rows])
    add('固定CMFF参考S01/S11/S12分别复用上一批seed7/17/27，均100epoch、batch32。D1/D4/D7检验CMFF中差分传递的三seed收益；D2/D5/D8检验MCFF与CMFF顺序的三seed差异；D3/D6与D2/D5构成MCFF中差分传递的双seed比较。MCFF+差分没有seed27，不能称为三seed复验。另保留旧G12（自由尺度+差分，seed7）作为比较，其实际测试路径原为100%MCFF。\n')
    add('共同设置：TaxiBJ random0.4，四基础mask训练/ID验证，训练逐epoch重采样；GPU0单卡串行，batch32，100epoch，val5，AdamW 1e-3余弦至3e-4、wd1e-4、clip1、AMP。四轮共享八专家，每轮原生Top2，原始融合、direct更新、关闭completion feedback、无尺度学习/软预热、无教师。L1+0.01 candidate balance；best只按ID验证MAE，保存best/last。模型及loader使用7/17/27，mask种子不变。\n')
    add('C/M/F=8/16/32空间网格，时间12不变；粗输出上采样后direct传递。差分方法只将专家输入改为H+tanh(g)·Δnorm，Δ=H−Hprev；每样本每通道THW的RMS匹配H幅度，比例detach、分母下限1e-6，门末层零初始化。首轮无历史，Router输入仍是当前原H，历史不跨batch。每窗口8次专家执行、面积代理4.625，面积代理不等于FLOPs。\n')
    add('新确认mask为20261025/26/27，沿用test+30000偏移；全部新组及旧参考用相同mask。原六套测试保留作为已知协议，新确认集只在训练完成后评估，不选择检查点。另评估缺失率0.2/0.6/0.8。确认仍在同一测试时间窗，不代表跨数据集。\n')
    add('入口：`python -u scripts/v24/run_scale_memory_followup.py --gpu 0`。冻结源码/配置与参考checkpoint哈希见'+link(suite/'plan.json','plan')+'；本批没有候选筛选或根据中间测试修改后续配置。\n')
    tab(['组','配置','日志'],[[r['variant'],link(suite/'configs'/f"{r['variant']}.json",'config') if (suite/'configs'/f"{r['variant']}.json").exists() else '待执行',link(Path(r['run_dir'])/'logs','logs') if r['run_dir'] else '待执行'] for r in rows])
    add('## 2. 结果、训练曲线与诊断\n')
    keys=['in_distribution','unseen_combinations','unseen_geometry','unseen_triple','unseen_combinations_repeat','unseen_geometry_repeat']
    index={r['variant']:r for r in rows};allrows=list(rows)
    for n,v in refs.items():
        r={'variant':n,'seed':v['config']['seed'],'status':'historical','best_epoch':v['receipt']['best_epoch'],'val_mae':v['receipt']['best_val_mae'],
           'run_dir':v['receipt']['run_dir'],'training_seconds':v['receipt']['total_time_sec'],'evaluations':{k:s['metrics'] for k,s in v['evaluation']['sets'].items()}}
        index[n]=r;allrows.append(r)
    tab(['组','状态','best epoch','val MAE']+[k+' MAE/RMSE' for k in keys],[[r['variant'],r['status'],r['best_epoch'],f(r['val_mae'])]+[f(r['evaluations'].get(k,{}).get('mae'))+'/'+f(r['evaluations'].get(k,{}).get('rmse')) for k in keys] for r in allrows])
    audits={};histories={};cost=[];diags={};family=[]
    for r in rows:
        if not r['run_dir']:continue
        n=r['variant'];run=Path(r['run_dir']);hist=[json.loads(s) for s in (run/'logs/metrics.jsonl').read_text().splitlines()];histories[n]=hist
        meta=json.loads((run/'training_metadata.json').read_text());ep=suite/'evaluations'/f'{n}.json';ev=json.loads(ep.read_text()) if ep.exists() else {'sets':{}}
        m=ev['sets'].get('in_distribution',{}).get('metrics',{});diags[n]=m
        audits[n]={'epochs':len(hist),'validations':sum(bool(h['val']) for h in hist),'best':(run/'checkpoints/best.pth').exists(),'last':(run/'checkpoints/last.pth').exists(),'amp_skips':sum(h['train'].get('train_skipped_amp_steps',0) for h in hist),
          'nonfinite':[(h['epoch'],sp,k,v) for h in hist for sp in ('train','val') for k,v in (h.get(sp) or {}).items() if isinstance(v,str) and v in ('nan','inf','-inf')]}
        cost.append([n,meta['total_params'],meta['trainable_params'],f(r['training_seconds']/60),f(max(h['perf']['peak_memory_gb'] for h in hist)),f(m.get('coe_expert_execution_count')),f(m.get('coe_expert_grid_equivalents'))])
        for k,s in ev['sets'].items():
            for key,v in s['metrics'].items():
                if key.startswith('coe_family_') and key.endswith('_mae'):family.append([n,k,key[11:-4],f(v),f(s['metrics'].get(key[:-3]+'rmse'))])
    tab(['组','总参数','可训练参数','训练计时min','峰值GiB','专家次数','面积代理'],cost)
    add('计时为训练与验证，检查点写盘及额外评估不在该计时内；峰值是训练过程记录。不同训练运行的分钟差异不可直接当严格效率测试。\n')
    tab(['组','epoch','验证次数','best/last','AMP跳步','非有限诊断数'],[[n,a['epochs'],a['validations'],str(a['best'])+'/'+str(a['last']),a['amp_skips'],len(a['nonfinite'])] for n,a in audits.items()])
    tab(['组','第2/3/4轮有符号门均值','实际尺度路径'],[[n,', '.join(f(m.get(f'coe_step{i}_memory_gate_signed_mean')) for i in (2,3,4)),', '.join(k.replace('coe_scale_execution_path_','').replace('_fraction','')+' '+f'{v:.1%}' for k,v in m.items() if k.startswith('coe_scale_execution_path_') and v>0)] for n,m in diags.items()])
    if histories:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,2,figsize=(12,4))
        for n,hs in histories.items():
            axs[0].plot([h['epoch'] for h in hs],[h['train']['loss'] for h in hs],label=n)
            axs[1].plot([h['epoch'] for h in hs if h['val']],[h['val']['mae'] for h in hs if h['val']],label=n)
        for ax,title in zip(axs,['Train loss','ID validation MAE']):ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=8)
        fig.tight_layout();fig.savefig(assets/'curves.png',dpi=150);plt.close(fig);add(f'![训练曲线]({assets.name}/curves.png)\n')
    tab(['组','协议','缺失族','MAE','RMSE'],family)
    for folder in ('confirmation','rate_transfer'):
        add('### '+folder+'\n');rs=[]
        for p in sorted((suite/folder).glob('*.json')):
            d=json.loads(p.read_text())
            for k,v in d.get('sets',{}).items():rs.append([p.stem,d['status'],k,f(v['metrics']['mae']),f(v['metrics']['rmse'])])
        tab(['组','状态','协议','MAE','RMSE'],rs)
    add('## 3. 配对分析、局限与建议\n')
    pairs=[('D1','S01'),('D4','S11'),('D7','S12'),('D2','S01'),('D5','S11'),('D8','S12'),('D3','D2'),('D6','D5'),('D3','D1'),('D6','D4'),('D3','G12')]
    computed={};pr=[]
    for a,b in pairs:
        x,y=index.get(a),index.get(b)
        if not x or not y or any(k not in x['evaluations'] or k not in y['evaluations'] for k in keys[:4]):continue
        ds=[100*(x['evaluations'][k]['mae']/y['evaluations'][k]['mae']-1) for k in keys[:4]]
        cs=[]
        if all((suite/'confirmation'/f'{n}.json').exists() for n in (a,b)):
            ca,cb=[json.loads((suite/'confirmation'/f'{n}.json').read_text()) for n in (a,b)]
            if ca['status']==cb['status']=='finished':cs=[100*(ca['sets'][k]['metrics']['mae']/cb['sets'][k]['metrics']['mae']-1) for k in keys[1:4]]
        computed[a+'/'+b]={'id':ds[0],'ood':statistics.mean(ds[1:]),'worst':max(ds[1:]),'confirmation':cs}
        pr.append([a,b]+[f'{v:+.2f}%' for v in ds]+[f'{statistics.mean(ds[1:]):+.2f}%',f'{statistics.mean(cs):+.2f}%' if cs else '待评估'])
    tab(['方法','参考','ID变化','两元','形态','三元','平均OOD','新mask平均OOD'],pr)
    add('负值改善、正值退步。OOD变化先逐协议求误差比再平均，不以绝对值混合替代。\n')
    for label,ps in [('CMFF差分三seed',['D1/S01','D4/S11','D7/S12']),('MCFF顺序三seed',['D2/S01','D5/S11','D8/S12']),('MCFF差分双seed',['D3/D2','D6/D5'])]:
        if not all(p in computed for p in ps):add(label+'：配对未齐，暂不下最终结论。\n');continue
        values=[computed[p] for p in ps]
        strict=all(v['id']<0 and v['ood']<0 and v['worst']<=5 for v in values)
        trade=all(v['id']<=2 and v['ood']<=-5 and v['worst']<=5 for v in values)
        confirmed=all(len(v['confirmation'])==3 and statistics.mean(v['confirmation'])<0 and max(v['confirmation'])<=5 for v in values)
        label_result='配对精度/泛化改善信号' if strict else '达到预设ID/OOD权衡门槛' if trade else '未达到一致推进门槛'
        add(f'{label}：{label_result}；新mask'+('保持平均收益且无单项>5%退步' if confirmed else '尚未全部通过/尚未完成')+'。仍需核对缺失率迁移，不以均值掩盖单seed失败。\n')
    add('D3/G12用于比较相同名义MCFF路径下固定训练与自由路径训练的结果，不是仅开关推理路由的严格干预；训练路径历史与参数构造不同，不能据此独立宣称路由有害。D3/D1、D6/D4可在有差分时比较顺序；结合原生MCFF/CMFF配对，才分析尺度顺序与信息传递的交互。\n')
    add('冻结尺度头不执行学习，差分门零初始化应保持对应固定基准的初始输出。训练并非严格数值确定，旧参考与本轮独立启动，三seed/双seed不等于统计显著。新mask与原测试共享窗口。只有当差分在强固定基准、新mask及缺失率迁移均无不可接受退步时才推进；若收益只来自MCFF顺序，应归因尺度结构，不能包装为跨轮通信创新。\n')
    add('下一步：先按上述配对和逐缺失族核对。CMFF差分稳定时再做归一化/门控/历史梯度最小消融；MCFF+差分仅双seed，若有信号再补seed27。本轮不根据中途测试改配置、不自动追加训练、不因时间接近10点缩短epoch。\n')
    (assets/'audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2));(assets/'paired.json').write_text(json.dumps(computed,ensure_ascii=False,indent=2));(assets/'diagnostics.json').write_text(json.dumps(diags,ensure_ascii=False))
    dest=out/(stem+'.md');dest.write_text('\n'.join(lines));(suite/'report_path.txt').write_text(str(dest)+'\n');return dest
