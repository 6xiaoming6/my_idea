from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json, hashlib, math, sys, statistics, subprocess
import torch

ROOT = Path('/home/students/HuangMingYu/code/py/my_idea/my_idea')
SUITE = ROOT/'outputs/v24-COE/experiments/core_validation/684cad05994af9a4'
sys.path.insert(0, str(SUITE/'source_snapshot/src'))
from stmoe_imputer.utils.deterministic import state_hash

def read(p): return json.loads(Path(p).read_text())
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
def filehash(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

plan=read(SUITE/'plan.json'); init=read(SUITE/'initialization_audit.json')
verification={'checked_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
 'suite':str(SUITE),'suite_digest':digest(plan)[:16],
 'git_branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
 'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
 'completed':read(SUITE/'completed.json') if (SUITE/'completed.json').exists() else None,
 'source_count':len(plan['source']), 'source_mismatch':[], 'current_source_difference':[], 'data':{}, 'runs':{}}
for n,sha in plan['source'].items():
 if not (SUITE/'source_snapshot'/n).exists() or filehash(SUITE/'source_snapshot'/n)!=sha: verification['source_mismatch'].append(n)
 if not (ROOT/n).exists() or filehash(ROOT/n)!=sha: verification['current_source_difference'].append(n)
for n,stamp in plan['data'].items():
 q=Path(n); verification['data'][n]={'sha256_match':filehash(q)==stamp['sha256'],'stat_match':(q.stat().st_size,q.stat().st_mtime_ns)==(stamp['size'],stamp['mtime_ns']),**stamp}

rows={}; curves={}; evaluations={}
for n,j in plan['jobs'].items():
 issues=[];rpath=SUITE/'results'/f'{n}.json';epath=SUITE/'evaluations'/f'{n}.json'
 if not rpath.exists() or not epath.exists():
  verification['runs'][n]={'issues':['missing receipt/evaluation']};continue
 r=read(rpath);e=read(epath);run=Path(r['run_dir']);h=[json.loads(x) for x in (run/'logs/metrics.jsonl').read_text().splitlines()]
 cfgsha=digest(j['config']);protocol=j['config']['experiment_plan']['protocol'];v=[x for x in h if x.get('val')];best=min(v,key=lambda x:x['val']['mae'])
 for p in [SUITE/'configs'/f'{n}.json',SUITE/'jobs'/f'{n}.json',run/'config.json']:
  c=read(p); c=c['config'] if 'config' in c else c
  if digest(c)!=cfgsha:issues.append('config mismatch '+str(p))
 if r['status']!='finished' or r['completed_epochs']!=100 or [x['epoch'] for x in h]!=list(range(1,101)):issues.append('training incomplete')
 if [x['epoch'] for x in v]!=list(range(5,101,5)):issues.append('validation cadence')
 if r['config_sha256']!=cfgsha or (r['best_epoch'],r['best_val_mae'])!=(best['epoch'],best['val']['mae']):issues.append('receipt/best mismatch')
 if any(not isinstance(x[sp][m],(int,float)) or not math.isfinite(x[sp][m]) for x in h for sp in ['train','val'] if x.get(sp) for m in ['loss','mae','rmse']):issues.append('nonfinite training primary metrics')
 if e['status']!='finished' or set(e['sets'])!=set(protocol['evaluations']):issues.append('test incomplete')
 if e['config_sha256']!=cfgsha or e['protocol_sha256']!=digest(protocol):issues.append('evaluation hash')
 if e['checkpoint']!=str(run/'checkpoints/best.pth') or e['best_epoch']!=r['best_epoch'] or e['npz']!=j['sources']['test']:issues.append('evaluation source/checkpoint')
 for k,a in e['sets'].items():
  if any(not isinstance(a['metrics'][m],(int,float)) or not math.isfinite(a['metrics'][m]) for m in ['mae','rmse']):issues.append('nonfinite test '+k)
  spec=protocol['evaluations'][k]
  if a['effective_mask_seed']!=spec['seed']+30000 or a['mask_spec']['families']!=spec['families'] or a['mask_spec']['rates']!=[.4]:issues.append('mask protocol '+k)
  if a['metrics']['coe_expert_execution_count']!=8 or a['metrics']['coe_expert_grid_equivalents']!=(8 if j['method'] in ['K01','K03'] else 4.625):issues.append('execution cost '+k)
  path=j['config']['model']['coe']['core_validation']['path'].lower()
  if a['metrics'].get('coe_scale_path_'+path+'_fraction')!=1:issues.append('fixed scale path '+k)
 rawtest=read(run/'logs/test.json')
 if any(rawtest[m]!=e['sets']['in_distribution']['metrics'][m] or r['test'][m]!=rawtest[m] for m in ['mae','rmse']):issues.append('ID receipt/raw-test disagreement')
 cpmeta={};bhash=None;lastbest=None
 for kind in ['best','last']:
  cp=run/'checkpoints'/f'{kind}.pth'
  if not cp.exists():issues.append('missing '+kind);continue
  c=torch.load(cp,map_location='cpu',weights_only=False)
  if digest(c['config'])!=cfgsha or c['epoch']!=(r['best_epoch'] if kind=='best' else 100):issues.append('checkpoint epoch/config '+kind)
  if kind=='best':bhash=state_hash(c['model'])
  else:
   lastbest=state_hash(c['training_state']['best_model'])
   if any(c.get(k) is None for k in ['optimizer','scheduler','scaler','rng_states','training_state']):issues.append('missing continuation state')
  cpmeta[kind]={'epoch':c['epoch'],'config_sha256':digest(c['config']),'sha256':filehash(cp),'size':cp.stat().st_size}
  del c
 if bhash!=lastbest:issues.append('best checkpoint does not match last saved best model')
 ti=read(run/'initialization.json');template=torch.load(SUITE/'initialization'/f'{n}.pth',map_location='cpu',weights_only=False)
 full=state_hash(template['model']);common=state_hash({k:v for k,v in template['model'].items() if 'step_pattern_experts.' not in k})
 if full!=template['sha256'] or full!=init[n]['full_sha256'] or full!=ti['full_state_sha256'] or common!=init[n]['common_sha256']:issues.append('initialization mismatch')
 for k,vv in template['model'].items():
  if 'step_pattern_experts.' in k:
   origin='main_branch.pattern_experts.'+k.split('step_pattern_experts.',1)[1].split('.',1)[1]
   if not torch.equal(vv,template['model'][origin]):issues.append('independent pool initial copy mismatch')
 del template
 meta=read(run/'training_metadata.json');runtime=read(run/'runtime.json')
 if meta['total_params']!=init[n]['total_params'] or meta['trainable_params']!=init[n]['trainable_params']:issues.append('parameter mismatch')
 errors=[]
 for q in (SUITE/'launcher_logs').glob(n+'.*.log'):
  content=q.read_text(errors='replace')
  for token in ['Traceback (most recent call last)','CUDA out of memory','RuntimeError:']:
   if token in content:errors.append(q.name+': '+token)
 issues.extend(errors)
 verification['runs'][n]={'issues':issues,'checkpoints':cpmeta,'full_initialization_sha256':full,'common_initialization_sha256':common,'config_sha256':cfgsha,'validation_epochs':[x['epoch'] for x in v]}
 rows[n]={'name':n,'dataset':j['dataset'],'method':j['method'],'seed':j['config']['seed'],'run_dir':str(run),
  'best_epoch':r['best_epoch'],'best_val_mae':r['best_val_mae'],'minutes':r['total_time_sec']/60,
  'peak_gib':max(x['perf']['peak_memory_gb'] for x in h),'parameters':meta['total_params'],'trainable':meta['trainable_params'],
  'samples':meta['samples'],'steps':meta['steps'],'runtime':runtime,'amp_skips':sum(x['train'].get('train_skipped_amp_steps',0) for x in h),
  'optimizer_steps':sum(x['train'].get('train_optimizer_steps',0) for x in h),
  'empty_skips':sum(x['train'].get('train_skipped_empty_batches',0) for x in h),
  'test':{k:{m:a['metrics'][m] for m in ['mae','rmse']} for k,a in e['sets'].items()},'issues':issues}
 curves[n]=h; evaluations[n]=e
verification['failures']={q.name:read(q) for q in (SUITE/'failures').glob('*.json')}
for seed in [7,17,27]:
 hashes={v['common_initialization_sha256'] for n,v in verification['runs'].items() if n.endswith('seed'+str(seed)) and 'common_initialization_sha256' in v}
 if len(hashes)!=1:verification.setdefault('common_seed_hash_mismatch',[]).append(seed)
dest=Path('/tmp/v24_core28_audit');dest.mkdir(exist_ok=True)
for name,x in [('verification',verification),('rows',rows),('curves',curves),('evaluations',evaluations)]:write(dest/(name+'.json'),x)
print('AUDIT',len(rows),'runs',len(plan['source']),'sources',len(plan['data']),'data files','issues',{n:v['issues'] for n,v in verification['runs'].items() if v['issues']},'source mismatch',verification['source_mismatch'],'current difference',verification['current_source_difference'],'data matches',all(v['sha256_match'] and v['stat_match'] for v in verification['data'].values()))
print('COMPLETE_AT',datetime.fromtimestamp(verification['completed']['time'],ZoneInfo('Asia/Shanghai')).isoformat())
for d in ['taxibj','bikenyc']:
 for k in ['K01','K02','K03','K04']:
  group=[x for x in rows.values() if x['dataset']==d and x['method']==k]
  print('COST',d,k,'min',round(statistics.mean(x['minutes'] for x in group),3),'mem',round(statistics.mean(x['peak_gib'] for x in group),3),'latency',round(statistics.mean(evaluations[x['name']]['sets']['in_distribution']['metrics']['forward_ms_per_sample_per_rank'] for x in group),3),'best_epochs',[x['best_epoch'] for x in group],'amp',[x['amp_skips'] for x in group])
print('DIAGNOSTICS')
for n in ['taxibj_K04_seed7','taxibj_K04_seed17','taxibj_K04_seed27','taxibj_K07_seed7','bikenyc_K04_seed7','bikenyc_K04_seed17','bikenyc_K04_seed27']:
 m=evaluations[n]['sets']['in_distribution']['metrics']
 print(n,'gates',[round(m[f'coe_step{i}_memory_gate_abs_mean'],4) for i in [2,3,4]],'pairpaths',m.get('coe_pair_path_unique_count'),m.get('coe_pair_path_max_fraction'))
