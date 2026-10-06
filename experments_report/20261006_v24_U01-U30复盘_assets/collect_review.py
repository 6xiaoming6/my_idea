import json,hashlib,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'outputs/v24-COE/experiments/id_priority_exploration/taxibj/501f761e40b7e937';A=Path(__file__).resolve().parent
sys.path.insert(0,str(S/'source_snapshot/scripts/v24'))
from run_b3_c3 import digest
plan=json.loads((S/'plan.json').read_text());jobs=json.loads((S/'resolved_jobs.json').read_text());selection=json.loads((S/'selection.json').read_text())
summary={};audit={'source_mismatches':[],'data_changes':[],'groups':{},'selection_hash_ok':digest(json.loads((S/'selection_evidence.json').read_text()))==selection['evidence_sha256']}
for rel,sha in plan['source'].items():
 if hashlib.sha256((S/'source_snapshot'/rel).read_bytes()).hexdigest()!=sha:audit['source_mismatches'].append(rel)
for p,stamp in plan['data'].items():
 st=Path(p).stat()
 if stamp!={'size':st.st_size,'mtime_ns':st.st_mtime_ns}:audit['data_changes'].append(p)
for n in [f'U{i:02}' for i in range(1,31)]:
 r=json.loads((S/'results'/f'{n}.json').read_text());run=Path(r['run_dir']);cfg=json.loads((run/'config.json').read_text());meta=json.loads((run/'training_metadata.json').read_text());sha=digest(cfg)
 a={'config_match':sha==r['config_sha256']==digest(jobs[n]['config']),'epochs':[],'val_epochs':[],'nonfinite_main':[],'nonfinite_diagnostics':{},'amp_skips':0,'checkpoints':{},'protocols':{}};hist=[]
 with (run/'logs/metrics.jsonl').open() as f:
  for line in f:
   h=json.loads(line);a['epochs'].append(h['epoch']);a['amp_skips']+=h['train'].get('train_skipped_amp_steps',0)
   if h['val']:a['val_epochs'].append(h['epoch'])
   compact={'epoch':h['epoch'],'perf':h['perf'],'is_best':h['is_best']}
   for sp in ('train','val'):
    compact[sp]={} if h[sp] else None
    for k,v in (h[sp] or {}).items():
     bad=(isinstance(v,str) and v in ('nan','inf','-inf')) or (isinstance(v,float) and not math.isfinite(v))
     if bad:
      if k in ('loss','mae','rmse'):a['nonfinite_main'].append([h['epoch'],sp,k,v])
      else:a['nonfinite_diagnostics'][sp+':'+k]=a['nonfinite_diagnostics'].get(sp+':'+k,0)+1
     if k in ('loss','mae','rmse','lr','train_skipped_amp_steps') or k.startswith('four_probe') or any(t in k for t in ('memory_gate','input_delta','route_delta','state_change','triscale','scale_execution_path','expert_execution_count','expert_grid_equivalents','scale_phase','gate_fraction')):compact[sp][k]=v
   hist.append(compact)
 for cp in ('best','last'):
  p=run/'checkpoints'/f'{cp}.pth';d=torch.load(p,map_location='cpu',weights_only=False);a['checkpoints'][cp]={'exists':True,'epoch':d['epoch'],'stage_epoch':int(d['model']['main_branch.stage_epoch']),'config_match':digest(d['config'])==sha,'has_optimizer':'optimizer' in d,'has_rng':'rng_states' in d};del d
 bestrow=next(h for h in hist if h['epoch']==r['best_epoch']);a['best_matches_val']=bestrow['val']['mae']==r['best_val_mae']
 evs={}
 for folder in ('evaluations','confirmation','rate_transfer'):
  p=S/folder/f'{n}.json'
  if not p.exists():continue
  d=json.loads(p.read_text());pro=json.loads((S/'protocols'/f'{folder}.json').read_text());a['protocols'][folder]={'status':d['status'],'sets':len(d['sets']),'expected_sets':len(pro['evaluations']),'hash_match':d['protocol_sha256']==digest(pro),'config_match':d['config_sha256']==sha,'checkpoint_match':d['checkpoint']==str((run/'checkpoints/best.pth').resolve()),'best_matches':d['best_epoch']==r['best_epoch'],'samples':list(set(v['samples'] for v in d['sets'].values()))}
  evs[folder]={'sets':d['sets'],'status':d['status']}
 summary[n]={'receipt':{k:r[k] for k in ('best_epoch','best_val_mae','total_time_sec','run_dir')},'config':cfg,'metadata':meta,'history':hist,**evs};audit['groups'][n]=a
json.dump(audit,open(A/'audit.json','w'),ensure_ascii=False,indent=2)
json.dump(summary,open(A/'review_data.json','w'),ensure_ascii=False)
print(json.dumps({'groups':len(summary),'bad_source':audit['source_mismatches'],'changed_data':audit['data_changes'],'config_errors':[n for n,a in audit['groups'].items() if not a['config_match']],'bad_epochs':[n for n,a in audit['groups'].items() if a['epochs']!=list(range(1,101)) or a['val_epochs']!=list(range(5,101,5))],'nonfinite_main':[n for n,a in audit['groups'].items() if a['nonfinite_main']],'skips':{n:a['amp_skips'] for n,a in audit['groups'].items() if a['amp_skips']},'selection_hash_ok':audit['selection_hash_ok'],'final_marker':(S/'final_evaluations_complete.json').exists()},ensure_ascii=False))
