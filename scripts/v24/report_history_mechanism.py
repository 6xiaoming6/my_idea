"""Z reports: methods first, all evidence, then matched-control analysis."""
import csv
from datetime import datetime
import os
from pathlib import Path
from zoneinfo import ZoneInfo
from run_b3_c3 import load,write
from run_scale_rate_compare import filehash
from report_history_followup import collect,fmt,PRIMARY

def export_report(root,suite,payload):
    root,suite=Path(root),Path(suite)
    if (suite/'reference_manifest.json').exists():
        for rel,sha in load(suite/'reference_manifest.json').items():
            if filehash(suite/rel)!=sha:raise RuntimeError('Frozen reference changed: '+rel)
    defs=dict(payload['references']);defs.update({j['method']:j for j in payload['jobs'].values()})
    records={};histories={}
    for label,job in defs.items():records[label],histories[label]=collect(suite,label,job,label in payload['references'])
    final=all(r['status']=='finished' for r in records.values())
    date=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
    title=f'{date}_v24_历史交互机制八组_{suite.name}_{"完整分析" if final else "阶段分析"}'
    dest=root/'experments_report'/(title+'.md');assets=dest.with_suffix('');assets.mkdir(parents=True,exist_ok=True)
    write(suite/'summary.json',records);write(assets/'results.json',records);write(assets/'curves.json',histories)
    lines=['# '+title,'','8次新训练；11个已完成参考只读复用。'+('全部完成。' if final else '阶段报告，未全部完成。'),'',
        '## 1. 实验具体做法与复现信息','','| 组 | 来源 | 率 | 具体做法 | 对照 |','|---|---|---|---|---|']
    for label,job in defs.items():
        ep=job['config']['experiment_plan']
        lines.append(f'| {label} | {"引用" if label in payload["references"] else "新训练"} | {records[label]["rate"]} | {ep.get("method",label)} | {ep.get("reference","核心主线")} |')
    doc=suite/'source_snapshot/scripts/v24/README_HISTORY_MECHANISM.md'
    if not doc.exists():doc=Path(__file__).with_name('README_HISTORY_MECHANISM.md')
    lines += ['',doc.read_text().split('## 1. 方法与复现',1)[1].split('## 2. 分析预先约定',1)[0],
        f'[冻结计划]({suite/"plan.json"})；[初始化审计]({suite/"initialization_audit.json"})；[GPU验收]({suite/"acceptance/preflight.json"})。','',
        '| 组 | 配置 | run（日志及检查点） |','|---|---|---|']
    for label,job in defs.items():
        config=suite/'references'/label/'config.json' if label in payload['references'] else suite/'configs'/f'{job["variant"]}.json'
        lines.append(f'| {label} | [config]({config if config.exists() else suite/"plan.json"}) | {records[label]["run"] or "未启动"} |')
    lines += ['','## 2. 全部结果、训练曲线与诊断','','| 组 | 状态/epoch | best | val MAE | ID MAE/RMSE | 分钟 | GiB | 总/可训参数 | ID ms/窗口 | AMP跳步/非有限诊断 |','|---|---|---|---|---|---|---|---|---|---|']
    for label,r in records.items():
        m=r['evaluations'].get('in_distribution',{}).get('metrics',{})
        lines.append(f'| {label} | {r["status"]}/{r["epoch"]} | {r["best_epoch"]} | {fmt(r["val_mae"])} | {fmt(m.get("mae"))}/{fmt(m.get("rmse"))} | {fmt(r["minutes"])} | {fmt(r["peak_gib"])} | {r["parameters"]}/{r["trainable"]} | {fmt(m.get("forward_ms_per_sample_per_rank"))} | {r["amp_skips"]}/{r["nonfinite_diagnostics"]} |')
        if r['issues'] or r['failure_event']:lines += ['',f'{label} 核验问题/保留历史失败：{r["issues"]}；{r["failure_event"]}','']
    lines += ['','分钟仅已提交epoch训练验证，不计宕机、重放batch、评估与落盘；GiB为allocated峰值；推理是单次实测非重复基准。诊断非有限计数不等同主要loss非有限。Y07引用含一次早期router梯度inf及AMP跳步，原Y的协议冲突/服务器恢复记录保存在父suite。','',
        '| 组 | 协议 | MAE | RMSE |','|---|---|---|---|']
    rows=[];families=[]
    for label,r in records.items():
        for proto,e in r['evaluations'].items():
            m=e['metrics'];lines.append(f'| {label} | {proto} | {fmt(m["mae"])} | {fmt(m["rmse"])} |')
            rows.append(dict(group=label,rate=r['rate'],protocol=proto,mae=m['mae'],rmse=m['rmse']))
            for k,v in m.items():
                if k.startswith('coe_family_') and k.endswith('_mae'):
                    families.append(dict(group=label,rate=r['rate'],protocol=proto,family=k[11:-4],mae=v,rmse=m.get(k[:-3]+'rmse')))
    for filename,data in [('full_protocols.csv',rows),('full_families.csv',families)]:
        if data:
            with (assets/filename).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines += ['',f'[完整分缺失族MAE/RMSE]({assets.name}/full_families.csv)；[六套完整表]({assets.name}/full_protocols.csv)。','']
    os.environ.setdefault('MPLCONFIGDIR','/tmp/v24_z_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for rate in (.4,.6):
        fig,axs=plt.subplots(2,2,figsize=(12,9))
        for label,h in histories.items():
            if records[label]['rate']!=rate:continue
            for ax,(split,key) in zip(axs.flat,[('train','mae'),('val','mae'),('train','loss'),('val','rmse')]):
                pts=[v for v in h if v.get(split)]
                if pts:ax.plot([v['epoch'] for v in pts],[v[split][key] for v in pts],label=label)
                ax.set_title(f'{rate} {split} {key}');ax.set_xlabel('Epoch');ax.set_yscale('log');ax.grid(alpha=.2)
        for ax in axs.flat:
            if ax.lines:ax.legend(fontsize=7,ncol=3)
        filename=f'curves_rate{rate}.png';fig.tight_layout();fig.savefig(assets/filename,dpi=140);plt.close(fig)
        lines += ['',f'![训练曲线]({assets.name}/{filename})']
    lines += ['','| 组 | train MAE 首→末 | val MAE 首/best/末 |','|---|---|---|']
    for label,h in histories.items():
        vals=[v for v in h if v.get('val')]
        if h:lines.append(f'| {label} | {fmt(h[0]["train"]["mae"])}→{fmt(h[-1]["train"]["mae"])} | '+('/'.join(fmt(v) for v in [vals[0]['val']['mae'],records[label]['val_mae'],vals[-1]['val']['mae']]) if vals else '无验证')+' |')
    lines += ['','| 组 | 协议 | 轮 | MAE | RMSE | 改善 | 退步 | 修好 | 破坏 |','|---|---|---|---|---|---|---|---|---|']
    for label,r in records.items():
        for proto,e in r['evaluations'].items():
            for step in range(1,5):
                values=[fmt(e['metrics'].get(f'coe_x_step{step}_{k}')) for k in ['mae','rmse','improved_fraction','worsened_fraction','repaired_fraction','broken_fraction']]
                lines.append(f'| {label} | {proto} | {step} | '+' | '.join(values)+' |')
    lines += ['',f'[完整门控、历史权重、交互幅度与配对分布]({assets.name}/results.json)；[原始曲线数据]({assets.name}/curves.json)。',
        '修好/破坏比例分母为全部有效缺失点；ST_selected_fraction占位零不解释为未执行ST。','','## 3. 配对分析、证据边界与建议','',
        '| 候选/对照 | 协议 | MAE变化 | RMSE变化 |','|---|---|---|---|']
    for job in payload['jobs'].values():
        label=job['method'];r=records[label]
        for ref in job['config']['experiment_plan']['factors']['control']:
            b=records[ref]
            if r['status']!='finished' or b['status']!='finished':continue
            ds={}
            for proto,e in r['evaluations'].items():
                d=[100*(e['metrics'][m]/b['evaluations'][proto]['metrics'][m]-1) for m in ['mae','rmse']];ds[proto]=d[0]
                lines.append(f'| {label}/{ref} | {proto} | {d[0]:+.2f}% | {d[1]:+.2f}%'+(' **退步>5%**' if max(d)>5 else '')+' |')
            lines += ['',f'{label}/{ref}：ID MAE {ds["in_distribution"]:+.2f}%；三类主要OOD相对变化平均 {sum(ds[k] for k in PRIMARY)/3:+.2f}%，最差 {max(ds[k] for k in PRIMARY):+.2f}%。负数为改善，复测不重复纳入均值。','']
    lines += ['',doc.read_text().split('## 2. 分析预先约定',1)[1],
        '当前完成状态不代表机制成立。信息删除组总参数相同但有效首层数据梯度容量不同，固定读取组评分器冻结；只把Z01/Y12与Z07/Z08称为等容量输入表示对照。没有单调性保证，没有显著性检验；三平面或中间监督不在本批。所有负结果及失败保留。']
    dest.write_text('\n'.join(lines)+'\n');(suite/'report_path.txt').write_text(str(dest)+'\n');return dest
