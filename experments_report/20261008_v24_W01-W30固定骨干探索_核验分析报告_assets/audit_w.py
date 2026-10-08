import json,hashlib,sys,math,collections
from pathlib import Path
import torch
R=Path('/home/students/HuangMingYu/code/py/my_idea/my_idea');S=R/'outputs/v24-COE/experiments/backbone_exploration/taxibj/2abad0134c468ed3'
sys.path[:0]=[str(S/'source_snapshot/src'),str(S/'source_snapshot/scripts/v24')]
from run_b3_c3 import digest
from stmoe_imputer.utils.deterministic import state_hash
read=lambda p:json.loads(p.read_text());plan=read(S/'plan.json');jobs=read(S/'resolved_jobs.json');data={};issues=[]
source_bad=[];current_changed=[]
for name,sha in plan['source'].items():
 if hashlib.sha256((S/'source_snapshot'/name).read_bytes()).hexdigest()!=sha:source_bad.append(name)
 if not (R/name).exists() or hashlib.sha256((R/name).read_bytes()).hexdigest()!=sha:current_changed.append(name)
data_bad=[]
for name,stamp in plan['data'].items():
 p=Path(name)
 if hashlib.sha256(p.read_bytes()).hexdigest()!=stamp['sha256']:data_bad.append(name)
for i in range(1,31):
 n=f'W{i:02}';receipt=read(S/'results'/f'{n}.json');ev=read(S/'evaluations'/f'{n}.json');run=Path(receipt['run_dir']);cfg=read(run/'config.json');h=[json.loads(x) for x in (run/'logs/metrics.jsonl').read_text().splitlines()];audit=read(run/'replay_audit.json');meta=read(run/'training_metadata.json');init=read(run/'initialization.json')
 checks={'epochs100':[x['epoch'] for x in h]==list(range(1,101)),'val20':sum(x['val'] is not None for x in h)==20,'receipt_finished':receipt['status']=='finished','config_match':digest(cfg)==receipt['config_sha256']==ev['config_sha256']==digest(jobs[n]['config']),'eval6':ev['status']=='finished' and len(ev['sets'])==6,'eval_protocol':ev['protocol_sha256']==digest(plan['protocol']),'eval_cp':Path(ev['checkpoint'])==run/'checkpoints/best.pth','best_metric':min(x['val']['mae'] for x in h if x['val'])==receipt['best_val_mae'],'audit100':len(audit)==100}
 ck={}
 for kind in ('best','last'):
  p=run/'checkpoints'/f'{kind}.pth';v=torch.load(p,map_location='cpu',weights_only=False)
  ck[kind]={'epoch':v['epoch'],'config_match':digest(v['config'])==digest(cfg),'model_finite':all(torch.isfinite(t).all().item() for t in v['model'].values() if t.is_floating_point())}
  if kind=='last':ck[kind]['audit_model_match']=state_hash(v['model'])==audit[-1]['model_sha256'];ck[kind]['optimizer_match']=state_hash(v['optimizer'])==audit[-1]['optimizer_sha256'];ck[kind]['ema_saved']=v['training_state']['w_training']['teacher'] is not None
  del v
 checks['checkpoints']=ck['best']['epoch']==receipt['best_epoch'] and ck['last']['epoch']==100 and all(x['model_finite'] and x['config_match'] for x in ck.values()) and ck['last']['audit_model_match'] and ck['last']['optimizer_match']
 bad=[];primary_bad=[]
 for row in h:
  for sp in ('train','val'):
   for k,v in (row[sp] or {}).items():
    if isinstance(v,str) and v.lower() in ('inf','-inf','nan') or isinstance(v,(int,float)) and not math.isfinite(v):
     bad.append((row['epoch'],sp,k,str(v)))
     if k in ('mae','rmse','loss'):primary_bad.append(bad[-1])
 eval_bad=[(name,k,v) for name,es in ev['sets'].items() for k,v in es['metrics'].items() if isinstance(v,(int,float)) and not math.isfinite(v) or isinstance(v,str) and v.lower() in ('inf','-inf','nan')]
 checks['primary_finite']=not primary_bad;checks['evaluation_finite']=not eval_bad
 m=ev['sets']['in_distribution']['metrics'];check_scales=all(q['metrics']['coe_expert_execution_count']==8 and q['metrics']['coe_expert_grid_equivalents']==4.625 and q['metrics']['coe_scale_execution_path_c__m__f__f_fraction']==1 for q in ev['sets'].values());checks['path_budget']=check_scales
 if not all(checks.values()):issues.append((n,checks))
 cpath=S/'confirmation'/f'{n}.json';confirm=read(cpath) if cpath.exists() else None
 data[n]={'receipt':receipt,'evaluations':ev,'config':cfg,'history':h,'audit':audit,'meta':meta,'initialization':init,'checks':checks,'checkpoints':ck,'nonfinite':bad,'eval_nonfinite':eval_bad,'confirmation':confirm}
keys=['in_distribution','unseen_combinations','unseen_geometry','unseen_triple']
base=data['W01']['evaluations']['sets']
for n,d in data.items():
 es=d['evaluations']['sets'];d['mean4']=sum(es[k]['metrics']['mae'] for k in keys)/4;d['mean4_rmse']=sum(es[k]['metrics']['rmse'] for k in keys)/4;d['relative4']=sum(es[k]['metrics']['mae']/base[k]['metrics']['mae'] for k in keys)/4
 d['amp_skips']=sum(x['train'].get('train_skipped_amp_steps',0) for x in d['history'])
 d['seconds']=sum(x['perf']['epoch_time_sec'] for x in d['history']);d['peak']=max(x['perf']['peak_memory_gb'] for x in d['history'])
 d['extra_student']=sum(x['extra_student_windows'] for x in d['audit']);d['extra_teacher']=sum(x['extra_teacher_windows'] for x in d['audit'])
orders={n:[a['data_order_sha256'] for a in d['audit']] for n,d in data.items()};extra={n:[a['extra_order_sha256'] for a in data[n]['audit']] for n in ('W19','W20','W21','W22')}
summary={'source_bad':source_bad,'source_count':len(plan['source']),'current_changed':current_changed,'data_bad':data_bad,'issues':issues,'common_initializations':len(set(d['initialization']['common_state_sha256'] for d in data.values())),'all_data_order_same':len(set(tuple(x) for x in orders.values()))==1,'all_second_view_same':len(set(tuple(x) for x in extra.values()))==1,'W01_W02_exact':data['W01']['audit']==data['W02']['audit'],'training_hours':sum(d['seconds'] for d in data.values())/3600,'test_hours':sum(x['seconds'] for d in data.values() for x in d['evaluations']['sets'].values())/3600,'confirmation_hours':sum(x['seconds'] for d in data.values() if d['confirmation'] for x in d['confirmation']['sets'].values())/3600,'amp_skips_range':[min(d['amp_skips'] for d in data.values()),max(d['amp_skips'] for d in data.values())],'best_epoch_distribution':dict(collections.Counter(d['receipt']['best_epoch'] for d in data.values()))}
Path('/tmp/w_audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));Path('/tmp/w_analysis_data.json').write_text(json.dumps(data,ensure_ascii=False));print(json.dumps(summary,ensure_ascii=False,indent=2))
for n,d in data.items():
 print(n,'avg4',round(d['mean4'],3),'avgRMSE',round(d['mean4_rmse'],3),'min',round(d['seconds']/60,1),'peak',round(d['peak'],2),'nf',len(d['nonfinite']),'badkeys',sorted(set(x[2] for x in d['nonfinite']))[:4],'extra',d['extra_student'],d['extra_teacher'])
