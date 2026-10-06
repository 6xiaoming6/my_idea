#!/usr/bin/env python3
"""T1--T9: three-scale structure controls, frozen selection and seed replication."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_r_exploration as previous
import run_b3_c3 as baseline
from stmoe_imputer.config import deep_update
CONFIG_DIR=ROOT/'configs/v24/triscale_exploration'
VARIANTS=tuple(f'T{i}' for i in range(1,10))
DEV_SETS=('unseen_combinations','unseen_geometry','unseen_triple')
DEFAULT_DEADLINE='2026-10-01T12:30:00+08:00'


def jobs(dataset='taxibj',epochs=100,batch_size=32,variants=VARIANTS[:5]):
    result=[]
    for variant in variants:
        if variant not in VARIANTS[:5]:raise ValueError('T6--T9 require a frozen selection')
        spec=baseline.load(CONFIG_DIR/f'{variant}.json')
        job=previous.jobs(dataset,epochs,batch_size,('R7',))[0]
        cfg=copy.deepcopy(job['config'])
        cfg['model']['coe']['spatial_scale']={'enabled':False}
        cfg['model']['coe']['triscale']=copy.deepcopy(spec['triscale'])
        cfg['experiment_plan']={'suite':'triscale_exploration','variant':variant,'base_variant':'R7',
                                'protocol':baseline.load(CONFIG_DIR/'protocol.json')}
        result.append({**job,'variant':variant,'name':spec['name'],'config':cfg})
    return result


def required_variants(requested):
    wanted=set(requested)
    if wanted & set(VARIANTS[5:]):wanted.update(VARIANTS[:5])
    return [v for v in VARIANTS if v in wanted]


def select_candidates(evidence):
    """No confirmation masks are available to this deterministic selector."""
    if not all(k in evidence for k in VARIANTS[:5]):raise ValueError('Selection requires all T1--T5')
    for r in evidence.values():
        values=[r['val_mae']]+[r['evaluations'][s]['mae'] for s in DEV_SETS]
        if not all(math.isfinite(v) and v>0 for v in values):raise ValueError('Invalid selection metric')
    b=min(('T1','T3'),key=lambda k:(evidence[k]['val_mae'],k))
    scores={}
    for a in ('T2','T4','T5'):
        id_ratio=evidence[a]['val_mae']/evidence[b]['val_mae']
        ood=sum(evidence[a]['evaluations'][s]['mae']/evidence[b]['evaluations'][s]['mae'] for s in DEV_SETS)/3
        scores[a]={'id_val_ratio':id_ratio,'ood_ratio':ood,'J':.5*(id_ratio+ood),'eligible':id_ratio<=1.02}
    eligible=[k for k,v in scores.items() if v['eligible']]
    a=min(eligible or list(scores),key=lambda k:(scores[k]['J'],k))
    return {'A':a,'B':b,'scores':scores,'id_gate_passed':bool(eligible),
            'evidence_sha256':baseline.digest(evidence),'selection_sets':list(DEV_SETS),
            'checkpoint_selection':'ID validation MAE only'}


def resolve_job(variant,static_jobs,selection,templates):
    spec=templates[variant];source=selection[spec['source']]
    job=copy.deepcopy(next(j for j in static_jobs if j['variant']==source))
    job['config']=deep_update(job['config'],spec['override'])
    job['config']['experiment_plan'].update(variant=variant,base_variant=source,selection_sha256=baseline.digest(selection))
    job['variant']=variant;job['name']=job['name']+'_'+spec['suffix']
    return job


def frozen_json(path,value):
    if path.exists():
        if baseline.load(path)!=value:raise RuntimeError(f'Frozen record changed: {path}')
    else:baseline.write(path,value)


def manifest(dataset):
    source=previous.manifest(dataset)
    for p in [*CONFIG_DIR.glob('*.json'),ROOT/'scripts/v24/README_TRISCALE_EXPLORATION.md',ROOT/'tests/test_v24_triscale.py',ROOT/'tests/test_v24_triscale_queue.py']:
        source[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return source


def reference_records(dataset):
    if dataset!='taxibj':return []
    old=ROOT/'outputs/v24-COE/experiments/r_exploration/taxibj/b21d7928e67f4565'
    result=[]
    for name in ('N5','R5','R6'):
        evaluation=old/('reference_evaluations/reference_N5.json' if name=='N5' else f'evaluations/{name}.json')
        data=baseline.load(evaluation);ckpt=Path(data['checkpoint']);stat=ckpt.stat()
        result.append({'variant':name,'checkpoint':str(ckpt),'config_sha256':data['config_sha256'],
                       'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'evaluation':str(evaluation),
                       'evaluation_sha256':hashlib.sha256(evaluation.read_bytes()).hexdigest()})
    return result


def summarize(suite,resolved,requested):
    protocol=baseline.load(suite/'protocol.json')
    previous.summarize(suite,list(resolved.values()),protocol)
    rows={r['variant']:r for r in baseline.load(suite/'summary.json')}
    decisions=baseline.load(suite/'budget_decisions.json') if (suite/'budget_decisions.json').exists() else {}
    for v in requested:
        if v not in rows:rows[v]={'variant':v,'status':'awaiting_selection'}
        if decisions.get(v,{}).get('deferred') and rows[v]['status']=='pending':rows[v]['status']='deferred_budget'
    ordered=[rows[v] for v in VARIANTS if v in rows];baseline.write(suite/'summary.json',ordered)
    return ordered


def selection_evidence(suite,resolved):
    rows=summarize(suite,resolved,VARIANTS[:5]);result={}
    for row in rows:
        if row['variant'] not in VARIANTS[:5]:continue
        if row['status']!='finished':raise RuntimeError('Selection inputs are incomplete')
        name=row['variant'];receipt=baseline.completed(suite,resolved[name])
        result[name]={'val_mae':row['val_mae'],'evaluations':row['evaluations'],
                      'config_sha256':receipt['config_sha256'],
                      'evaluation_sha256':hashlib.sha256((suite/'evaluations'/f'{name}.json').read_bytes()).hexdigest()}
    return result


def load_or_select(suite,resolved):
    evidence=selection_evidence(suite,resolved);path=suite/'selection.json'
    if path.exists():
        selected=baseline.load(path)
        if selected['evidence_sha256']!=baseline.digest(evidence):raise RuntimeError('Frozen selection evidence changed')
        return selected
    selected=select_candidates(evidence);baseline.write(path,selected);return selected


def estimate_seconds(suite,resolved,epochs,variant):
    samples=[];eval_times=[]
    for job in resolved.values():
        receipt=baseline.completed(suite,job)
        if not receipt:continue
        for line in (Path(receipt['run_dir'])/'logs/metrics.jsonl').read_text().splitlines():
            row=json.loads(line)
            if 'train' in row:samples.append(row['perf']['epoch_time_sec'])
        ep=suite/'evaluations'/f'{job["variant"]}.json'
        if ep.exists():eval_times.append(sum(s.get('seconds',0) for s in baseline.load(ep).get('sets',{}).values()))
    if not samples:return 90*60 if variant=='T8' else 80*60
    p90=sorted(samples)[math.ceil(.9*len(samples))-1]
    # Memory is absent from the screening jobs; reserve an explicit overhead.
    module_factor=1.15 if variant=='T8' else 1.0
    return 1.15*(p90*epochs*module_factor+max(eval_times or [180])+120)


def can_start_optional(now,deadline,estimate):
    return (deadline-now).total_seconds()>=estimate


def confirm(suite,resolved,selection,refs,gpu,device='cuda'):
    destination=suite/'confirmation';destination.mkdir(exist_ok=True)
    if not (destination/'source_snapshot').exists():
        (destination/'source_snapshot').symlink_to('../source_snapshot',target_is_directory=True)
    protocol=baseline.load(suite/'confirmation_protocol.json');frozen_json(destination/'protocol.json',protocol)
    records=[]
    for name in (selection['A'],selection['B'],'T6','T7'):
        if name not in resolved:return
        receipt=baseline.completed(suite,resolved[name])
        if not receipt:return
        records.append({'variant':name,'checkpoint':str(Path(receipt['run_dir'])/'checkpoints/best.pth'),
                        'config_sha256':baseline.digest(resolved[name]['config'])})
    records.extend(r for r in refs if r['variant'] in ('N5','R5'))
    for r in records:
        previous.run_evaluation(destination,r['variant'],r['checkpoint'],r['config_sha256'],protocol,
                                resolved['T1']['sources']['test'],gpu,device=device)


def verdict(rows,selection):
    indexed={r['variant']:r for r in rows};pairs=[(selection['A'],selection['B']),('T7','T6')];ratios=[]
    for a,b in pairs:
        if a not in indexed or b not in indexed or any(indexed[k]['status']!='finished' for k in (a,b)):return '等待第二种子复验',[]
        ra=indexed[a];rb=indexed[b]
        ratios.append({'seed':7 if a==selection['A'] else 17,
                       'id_ratio':ra['evaluations']['in_distribution']['mae']/rb['evaluations']['in_distribution']['mae'],
                       'ood_ratios':{s:ra['evaluations'][s]['mae']/rb['evaluations'][s]['mae'] for s in DEV_SETS}})
    if all(r['id_ratio']<1 and sum(r['ood_ratios'].values())/3<1 for r in ratios):return '可继续推进：两种子ID及平均OOD均改善；仍需检查强参考与确认mask',ratios
    if all(r['id_ratio']<=1.02 and sum(r['ood_ratios'].values())/3<=.95 and max(r['ood_ratios'].values())<=1.05 for r in ratios):return '可保留的精度/泛化权衡；仍需检查强参考与确认mask',ratios
    if any(r['id_ratio']<1 and sum(r['ood_ratios'].values())/3<1 for r in ratios):return '只有单种子信号，尚未稳定复现',ratios
    return '未达到预设推进标准，不确立自由尺度主线',ratios


def write_report(suite,rows,resolved,selection):
    # Method-first format; this automatic receipt does not substitute for review.
    lines=['# 三尺度实验自动汇总','',f'队列：{suite.name}；完整性以summary.json为准。','',
           '## 1. 每组具体做法','', '| 实验 | 来源 | 尺度模式 | 更新 | 记忆 | 专家池 | seed | 状态 |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        name=row['variant'];job=resolved.get(name)
        if not job:lines.append(f'| {name} | 待选择 | — | — | — | — | — | {row["status"]} |');continue
        c=job['config'];sp=c['model']['coe']['triscale']
        lines.append(f'| {name} | {c["experiment_plan"]["base_variant"]} | {sp["mode"]} | {sp["update"]} | {sp["recent_memory"]} | {c["model"]["coe"]["expert_sharing"]} | {c["seed"]} | {row["status"]} |')
    lines+=['','F/M/C为空间原分辨率、1/2、1/4；固定路径CMFF。explore_free训练前8epoch随机12种排列，之后自由；评估始终自由。细节保留为H−U(D(H))+U(E)，F保持direct。参数、训练和数据精确配置见configs/，来源与指纹见plan.json，源码见source_snapshot/。',
            '','## 2. 实测结果','','| 实验 | best epoch | val MAE | ID MAE | 两元 | 形态 | 三元 | 分钟 | 网格代理 |','| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        if row.get('status')!='finished':continue
        name=row['variant'];e=row['evaluations'];area=baseline.load(suite/'evaluations'/f'{name}.json')['sets']['in_distribution']['metrics'].get('coe_expert_grid_equivalents',float('nan'))
        vals=[name,str(row['best_epoch']),f'{row["val_mae"]:.4f}']+[f'{e[s]["mae"]:.4f}' for s in ('in_distribution',*DEV_SETS)]+[f'{row["training_seconds"]/60:.1f}',f'{area:.3f}']
        lines.append('| '+' | '.join(vals)+' |')
    lines+=['','### 完整MAE / RMSE与尺度诊断','']
    for row in rows:
        if row.get('status')!='finished':continue
        name=row['variant'];sets=baseline.load(suite/'evaluations'/f'{name}.json')['sets']
        lines += [f'**{name}**','', '| 协议 | MAE | RMSE | 主尺度路径 |', '| --- | --- | --- | --- |']
        for key,value in sets.items():
            m=value['metrics'];paths={k[len('coe_scale_path_'):-len('_fraction')]:v for k,v in m.items() if k.startswith('coe_scale_path_') and v>0}
            top=sorted(paths.items(),key=lambda x:-x[1])[:3]
            lines.append(f'| {key} | {m["mae"]:.4f} | {m["rmse"]:.4f} | '+', '.join(f'{k.upper()} {v:.1%}' for k,v in top)+' |')
        lines+=['']
    lines+=['### 新mask确认（不用于选模型）','','| 检查点 | 两元MAE | 形态MAE | 三元MAE |','| --- | --- | --- | --- |']
    confirmed={}
    for path in sorted((suite/'confirmation/evaluations').glob('*.json')):
        record=baseline.load(path)
        if record.get('status')!='finished':continue
        confirmed[path.stem]=record['sets']
        lines.append('| '+path.stem+' | '+' | '.join(f'{record["sets"][key]["metrics"]["mae"]:.4f}' for key in DEV_SETS)+' |')
    lines+=['','### 旧强参考（原六套协议）','','| 模型 | ID MAE | 两元MAE | 形态MAE | 三元MAE | 网格代理 |','| --- | --- | --- | --- | --- | --- |']
    for path in sorted((suite/'comparison_evaluations').glob('*.json')):
        record=baseline.load(path)['sets'];m=record['in_distribution']['metrics']
        lines.append('| '+path.stem+' | '+' | '.join(f'{record[key]["metrics"]["mae"]:.4f}' for key in ('in_distribution',*DEV_SETS))+f' | {m.get("coe_expert_grid_equivalents",8):.3f} |')
    lines+=['','细分缺失类型、概率、训练曲线、参数配置及显存见原日志与configs；面积代理不是精确FLOPs。','', '## 3. 分析与建议','']
    if selection:
        label,ratios=verdict(rows,selection);lines += [f'冻结参考B={selection["B"]}，自由候选A={selection["A"]}。',label,'', '```json',json.dumps(ratios,ensure_ascii=False,indent=2),'```']
        # Surface stronger historical references and confirmation, not merely new weak baselines.
        cautions=[]
        a=next((r for r in rows if r['variant']==selection['A'] and r['status']=='finished'),None)
        if a:
            own=baseline.load(suite/'evaluations'/f'{selection["A"]}.json')['sets']
            for p in sorted((suite/'comparison_evaluations').glob('*.json')):
                ref=baseline.load(p)['sets'];metrics=('mae','rmse')
                dominated=all(ref[s]['metrics'][m]<=own[s]['metrics'][m] for s in ('in_distribution',*DEV_SETS) for m in metrics)
                area=ref['in_distribution']['metrics'].get('coe_expert_grid_equivalents',8)
                ownarea=own['in_distribution']['metrics']['coe_expert_grid_equivalents']
                if dominated and area<=ownarea:cautions.append(f'{p.stem}在主要测试MAE/RMSE及网格代理上均不差；不能仅据新基准改善确立主线。')
        lines+=['']+cautions+['','最终判断需结合确认mask与旧强参考；T8/T9只有单种子，不能当复现。若无稳定自由尺度收益，保留R5为当前泛化候选；不把路由多样性作为收益。']
        confirmation_ratios=[]
        for a,b in ((selection['A'],selection['B']),('T7','T6')):
            if a in confirmed and b in confirmed:
                confirmation_ratios.append({'candidate':a,'reference':b,'ood_ratios':{key:confirmed[a][key]['metrics']['mae']/confirmed[b][key]['metrics']['mae'] for key in DEV_SETS}})
        confirmation_improves=(len(confirmation_ratios)==2 and all(sum(r['ood_ratios'].values())/3<1 for r in confirmation_ratios))
        if len(confirmation_ratios)==2:
            lines+=['', '新mask确认：两个种子平均OOD均改善。' if confirmation_improves else '新mask确认未在两个种子上均保持平均OOD改善，不将研发协议上的收益视为已确认。']
        else:lines+=['','新mask确认尚未齐全，不能给出确认结论。']
        baseline.write(suite/'assessment.json',{'provisional_verdict':label,'seed_ratios':ratios,'historical_cautions':cautions,
            'confirmation_ratios':confirmation_ratios,'confirmation_mean_improves_both_seeds':confirmation_improves,
            'requires_confirmation_review':len(confirmation_ratios)!=2,'requires_manual_cost_and_tradeoff_review':True})
    else:lines.append('筛选/复验尚未完成，不提前确定研究主线。')
    (suite/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('taxibj','bikenyc'),default='taxibj');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    p.add_argument('--deadline',default=DEFAULT_DEADLINE,help='Timezone-aware ISO cutoff for optional T8/T9')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--summary-only',action='store_true')
    args=p.parse_args()
    if min(args.epochs,args.batch_size)<1 or args.gpu<0 or len(set(args.variants))!=len(args.variants):p.error('Invalid arguments')
    deadline=datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:p.error('--deadline must include a timezone')
    requested=required_variants(args.variants)
    static=jobs(args.dataset,args.epochs,args.batch_size,VARIANTS[:5])
    templates={v:baseline.load(CONFIG_DIR/f'{v}.json') for v in VARIANTS[5:]}
    source=manifest(args.dataset);refs=reference_records(args.dataset)
    data={v:{'size':Path(v).stat().st_size,'mtime_ns':Path(v).stat().st_mtime_ns} for v in static[0]['sources'].values()}
    payload={'static_jobs':static,'templates':templates,'requested':requested,'source':source,'data':data,
             'references':refs,'deadline':args.deadline,'protocol':baseline.load(CONFIG_DIR/'protocol.json'),
             'confirmation_protocol':baseline.load(CONFIG_DIR/'confirmation_protocol.json')}
    fingerprint=baseline.digest(payload)[:16];suite=ROOT/'outputs/v24-COE/experiments/triscale_exploration'/args.dataset/fingerprint
    if args.dry_run:print(json.dumps({'suite':str(suite),**payload},ensure_ascii=False,indent=2));return
    if args.summary_only:
        if not suite.exists():raise FileNotFoundError(suite)
        resolved=baseline.load(suite/'resolved_jobs.json');rows=summarize(suite,resolved,requested)
        print(json.dumps(rows,ensure_ascii=False,indent=2));return
    suite.parent.mkdir(parents=True,exist_ok=True)
    with (suite.parent/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);suite.mkdir(exist_ok=True)
        for name,sha in source.items():
            target=suite/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(ROOT/name,target)
            if hashlib.sha256(target.read_bytes()).hexdigest()!=sha:raise RuntimeError('Frozen source mismatch')
        frozen_json(suite/'plan.json',{**payload,'fingerprint':fingerprint})
        frozen_json(suite/'protocol.json',payload['protocol']);frozen_json(suite/'confirmation_protocol.json',payload['confirmation_protocol'])
        for ref in refs:
            original=Path(ref['evaluation'])
            if hashlib.sha256(original.read_bytes()).hexdigest()!=ref['evaluation_sha256']:raise RuntimeError('Reference evaluation changed')
            stat=Path(ref['checkpoint']).stat()
            if (stat.st_size,stat.st_mtime_ns)!=(ref['size'],ref['mtime_ns']):raise RuntimeError('Reference checkpoint changed')
            target=suite/'comparison_evaluations'/f'{ref["variant"]}.json';target.parent.mkdir(exist_ok=True)
            frozen_json(target,baseline.load(original))
        resolved=baseline.load(suite/'resolved_jobs.json') if (suite/'resolved_jobs.json').exists() else {}
        selection=baseline.load(suite/'selection.json') if (suite/'selection.json').exists() else None
        for variant in requested:
            if variant in VARIANTS[:5]:job=copy.deepcopy(next(j for j in static if j['variant']==variant))
            else:
                selection=load_or_select(suite,resolved)
                job=resolve_job(variant,static,selection,templates)
            if variant in resolved and resolved[variant]!=job:raise RuntimeError('Resolved job changed')
            resolved[variant]=job;baseline.write(suite/'resolved_jobs.json',resolved)
            frozen_json(suite/'configs'/f'{variant}.json',job['config'])
            for path,stamp in data.items():
                stat=Path(path).stat()
                if {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}!=stamp:raise RuntimeError('Dataset changed')
            if variant in ('T8','T9') and not baseline.completed(suite,job):
                estimate=estimate_seconds(suite,resolved,args.epochs,variant);now=datetime.now(ZoneInfo('Asia/Shanghai'))
                decision={'checked_at':now.isoformat(),'estimated_seconds':estimate,'deferred':not can_start_optional(now,deadline,estimate)}
                path=suite/'budget_decisions.json';decisions=baseline.load(path) if path.exists() else {};decisions[variant]=decision;baseline.write(path,decisions)
                if decision['deferred']:
                    print(f'Deferred {variant}: insufficient time before {args.deadline}',flush=True)
                    rows=summarize(suite,resolved,requested);write_report(suite,rows,resolved,selection);continue
            try:
                if not baseline.completed(suite,job):baseline.launch(suite,job,args.gpu)
                receipt=baseline.completed(suite,job)
                previous.run_evaluation(suite,variant,Path(receipt['run_dir'])/'checkpoints/best.pth',baseline.digest(job['config']),payload['protocol'],job['sources']['test'],args.gpu)
                if variant=='T7':confirm(suite,resolved,selection,refs,args.gpu)
            finally:
                rows=summarize(suite,resolved,requested);write_report(suite,rows,resolved,selection)
        print(f'Results: {suite / "summary.json"}',flush=True)

if __name__=='__main__':main()
