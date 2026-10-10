import json,hashlib,sys,math,copy
from pathlib import Path
import torch
ROOT=Path('/home/students/HuangMingYu/code/py/my_idea/my_idea')
sys.path.insert(0,str(ROOT/'scripts/v24'))
from run_b3_c3 import digest
S=ROOT/'outputs/v24-COE/experiments/scale_rate_compare/a8e91b58131e845d'
load=lambda p:json.loads(Path(p).read_text())
plan=load(S/'plan.json');audit={'issues':[],'source_mismatch':[],'live_source_differences':[],'data_mismatch':[],'runs':{},'pairs':{}}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for name,h in plan['source'].items():
 if sha(S/'source_snapshot'/name)!=h:audit['source_mismatch'].append(name)
 if not (ROOT/name).exists() or sha(ROOT/name)!=h:audit['live_source_differences'].append(name)
for p,stamp in plan['data'].items():
 if sha(p)!=stamp['sha256']:audit['data_mismatch'].append(p)
data={}
for n,j in plan['jobs'].items():
 rec=load(S/'results'/(n+'.json'));run=Path(rec['run_dir']);cfg=load(run/'config.json')
 hist=[json.loads(l) for l in (run/'logs/metrics.jsonl').read_text().splitlines()]
 e=load(S/f'evaluations_rate{j["rate"]:g}'/(n+'.json'));meta=load(run/'training_metadata.json')
 issues=[];nonfinite=[]
 if rec['config_sha256']!=digest(cfg) or cfg!=j['config']:issues.append('config mismatch')
 if len(hist)!=100 or [h['epoch'] for h in hist]!=list(range(1,101)):issues.append('history mismatch')
 if len([h for h in hist if h['val']])!=20:issues.append('validation count mismatch')
 vals=[h for h in hist if h['val']];best=min(vals,key=lambda h:h['val']['mae'])
 if best['epoch']!=rec['best_epoch'] or best['val']['mae']!=rec['best_val_mae']:issues.append('best mismatch')
 for h in hist:
  for split in ('train','val'):
   if h.get(split):
    for k,v in h[split].items():
     if v in ('inf','-inf','nan') or isinstance(v,float) and not math.isfinite(v):nonfinite.append((h['epoch'],split,k,str(v)))
    for k in ('loss','mae','rmse'):
     if not isinstance(h[split].get(k),(int,float)) or not math.isfinite(h[split][k]):issues.append('nonfinite primary')
  actual=h['train'].get('coe_expert_execution_count')
  want=24 if j['method']=='top1' and h['epoch']<=10 else 8
  if actual!=want:issues.append(f'execution count {h["epoch"]}: {actual}')
 cps={}
 for cname in ('best','last'):
  cp=torch.load(run/'checkpoints'/(cname+'.pth'),map_location='cpu',weights_only=False)
  ce=rec['best_epoch'] if cname=='best' else 100
  ok=cp['epoch']==ce and digest(cp['config'])==digest(cfg) and all(not t.is_floating_point() or torch.isfinite(t).all().item() for t in cp['model'].values() if torch.is_tensor(t))
  if not ok:issues.append(cname+' checkpoint failed')
  cps[cname]={'epoch':cp['epoch'],'finite_and_config_match':ok}
  del cp
 if e['status']!='finished' or len(e['sets'])!=6 or e['config_sha256']!=digest(cfg) or e['protocol_sha256']!=digest(cfg['experiment_plan']['protocol']):issues.append('evaluation receipt mismatch')
 for name,entry in e['sets'].items():
  if entry['mask_spec']['rates']!=[j['rate']]:issues.append('eval wrong rate')
  if entry['effective_mask_seed']!=entry['mask_spec']['seed']+30000:issues.append('eval wrong seed')
  if entry['metrics']['coe_expert_execution_count']!=8:issues.append('eval dense')
  if not all(math.isfinite(entry['metrics'][k]) for k in ('mae','rmse')):issues.append('eval nonfinite')
 delta=abs(rec['test']['mae']-e['sets']['in_distribution']['metrics']['mae'])
 if delta>1e-6:issues.append(f'ID repeated eval differs: {delta}')
 audit['runs'][n]={'issues':issues,'epochs':len(hist),'validations':len(vals),'checkpoints':cps,'nonfinite_diagnostics':nonfinite,'test_repeat_abs_diff':delta}
 data[n]={'job':j,'receipt':rec,'history':hist,'evaluation':e,'metadata':meta}
 if issues:audit['issues'].append({n:issues})
for n,d in data.items():
 j=d['job']
 if j['method']!='top1':continue
 ref=n.replace('_top1','_cmff');b=data[ref];cs=[copy.deepcopy(x['job']['config']) for x in (d,b)]
 for c in cs:
  c.pop('experiment_plan')
  for k in ('scale_policy','scale_soft_warmup_epochs','scale_soft_transition_epochs'):c['model']['coe']['id_priority'].pop(k)
 if cs[0]!=cs[1]:audit['issues'].append(n+' unfair configuration')
 ms=['in_distribution','unseen_combinations','unseen_geometry','unseen_triple']
 pairs={key:100*(d['evaluation']['sets'][key]['metrics']['mae']/b['evaluation']['sets'][key]['metrics']['mae']-1) for key in d['evaluation']['sets']}
 a4=[sum(x['evaluation']['sets'][key]['metrics']['mae'] for key in ms)/4 for x in (b,d)]
 audit['pairs'][n]={'mae_change_pct':pairs,'rmse_change_pct':100*(d['evaluation']['sets']['in_distribution']['metrics']['rmse']/b['evaluation']['sets']['in_distribution']['metrics']['rmse']-1),'val_change_pct':100*(d['receipt']['best_val_mae']/b['receipt']['best_val_mae']-1),'A4_cmff_top1':a4,'A4_change_pct':100*(a4[1]/a4[0]-1)}
audit['source_count']=len(plan['source']);audit['failure_files']=[str(p) for p in (S/'failures').glob('*.json')]
audit['started']=load(S/'started.json');audit['completed']=load(S/'completed.json')
audit['wall_hours']=(audit['completed']['time']-audit['started']['time'])/3600
Path('/tmp/scale16_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));Path('/tmp/scale16_data.json').write_text(json.dumps(data,ensure_ascii=False))
print(json.dumps({k:v for k,v in audit.items() if k not in ('runs','pairs')},ensure_ascii=False,indent=2))
print('diagnostic_nonfinite', {n:x['nonfinite_diagnostics'] for n,x in audit['runs'].items() if x['nonfinite_diagnostics']})
print('pairs',json.dumps(audit['pairs'],ensure_ascii=False,indent=2))
