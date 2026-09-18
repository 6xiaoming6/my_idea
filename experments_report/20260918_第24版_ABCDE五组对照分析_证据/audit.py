import json,sys,math,hashlib,statistics,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/v24'))
import run_experiments as runner
S=ROOT/'outputs/v24-COE/experiments/abcde/abcde/38027eefadf2a586'
OUT=Path(__file__).resolve().parent
plan=json.loads((S/'plan.json').read_text());protocol=json.loads((S/'protocol.json').read_text())
experts=['T','S','TD','SD','TA','ST']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def route(m,prefix='coe_'):
 keys=['path_unique_count','path_max_fraction','path_entropy','routing_sample_count']
 out={k:m.get(prefix+k) for k in keys}
 out['top_paths']=sorted([(k[len(prefix+'path_'):-len('_fraction')],v) for k,v in m.items() if k.startswith(prefix+'path_') and k.endswith('_fraction') and k!=prefix+'path_max_fraction'],key=lambda x:-x[1])[:6]
 out['usage']=[[m.get(f'{prefix}step{k}_{e}_usage',m.get(f'{prefix}step{k}_{e}_weight')) for e in experts] for k in range(1,5)]
 out['probability']=[[m.get(f'{prefix}step{k}_{e}_prob') for e in experts] for k in range(1,5)]
 out['entropy']=[m.get(f'{prefix}step{k}_router_entropy') for k in range(1,5)]
 return out
result={'suite':str(S),'experts':experts,'coverage':{},'runs':{},'hashes':{},'source_mismatches':[]}
for p,expected in protocol['files_sha256'].items():
 actual=sha(p)
 if actual!=expected:result['source_mismatches'].append(p)
for job in plan['runs']:
 name=job['variant'][-1].upper(); receipt_path=next((S/'results').glob(job['name']+'.attempt*.json'))
 receipt=json.loads(receipt_path.read_text());rd=Path(receipt['run_dir']);mp=rd/'logs/metrics.jsonl';rows=[json.loads(l) for l in mp.read_text().splitlines()];history=[r for r in rows if 'epoch' in r];val=[r for r in history if r.get('val')];best=next(r for r in history if r['epoch']==receipt['best_epoch']);test=receipt['test'];last=history[-1]
 result['coverage'][name]={'complete':runner.check_complete(receipt_path,job,plan),'epochs':len(history),'validations':len(val),'test_records':sum(r.get('stage')=='test' for r in rows),'checkpoint':(rd/'checkpoints/best.pt').is_file()}
 nonfinite=[]
 for r in rows:
  for section in ['train','val','metrics']:
   for k,v in (r.get(section) or {}).items():
    if isinstance(v,(int,float)) and not math.isfinite(v):nonfinite.append([r.get('epoch',r.get('stage')),section,k])
 data={'run_dir':str(rd),'best_epoch':receipt['best_epoch'],'best_val_mae':receipt['best_val_mae'],'test':{k:test[k] for k in ['mae','rmse','mape','loss']},'time_seconds':receipt['total_time_sec'],'test_route':route(test),'last_train_route':route(last['train']),'best_val_route':route(best['val']),'nonfinite_metrics':nonfinite,'amp_skips':sum(r['train'].get('train_skipped_amp_steps',0) for r in history),'updates':sum(r['train'].get('train_optimizer_steps',0) for r in history),'mean_epoch_seconds':statistics.mean(r['perf']['epoch_time_sec'] for r in history),'trends':[], 'families':{},'zero_gradient_epochs':{},'last_gradients':{},'last_router_diagnostics':{}}
 for r in history:
  tr=r['train'];v=r.get('val') or {}
  data['trends'].append({'epoch':r['epoch'],'train_mae':tr['mae'],'val_mae':v.get('mae'),'val_rmse':v.get('rmse'),'train_pathmax':tr.get('coe_path_max_fraction'),'train_paths':tr.get('coe_path_unique_count'),'val_pathmax':v.get('coe_path_max_fraction'),'val_paths':v.get('coe_path_unique_count'),'train_route_entropy':tr.get('coe_route_entropy'),'val_route':route(v) if v else None,'main_router_grad':tr.get('coe_router_main_grad_norm'),'balance_router_grad':tr.get('coe_router_balance_grad_norm'),'z_router_grad':tr.get('coe_router_z_grad_norm'),'balance_weighted':tr.get('l_coe_balance_weighted'),'z_weighted':tr.get('l_coe_z_weighted'),'hard_fraction':tr.get('coe_hard_fraction'),'amp_skips':tr.get('train_skipped_amp_steps')})
 for e in experts:
  data['zero_gradient_epochs'][e]=[r['epoch'] for r in history if r['train'].get(f'coe_expert_{e}_finite_nonzero_gradient_batch_fraction')==0]
  data['last_gradients'][e]={k:last['train'].get(f'coe_expert_{e}_{k}') for k in ['grad_norm','finite_nonzero_gradient_batch_fraction']}
 data['last_router_diagnostics']={k:v for k,v in last['train'].items() if (k.startswith('coe_step') and any(x in k for x in ['logit_range','top12_margin','embedding_norm','router_grad'])) or k in ['l_coe_balance','l_coe_balance_weighted','l_coe_z_weighted','coe_router_main_grad_norm','coe_router_balance_grad_norm','coe_router_z_grad_norm','train_amp_scale_end']}
 for family in ['random_point','node_outage','temporal_gap','spatial_region','spatiotemporal_block','stripe','moving_region','multi_block','composite']:
  prefix='coe_condition_family_'+family+'_'
  if prefix+'sample_count' in last['train']:data['families'][family]={'sample_count':last['train'][prefix+'sample_count'],**route(last['train'],prefix)}
 result['runs'][name]=data
 for p in [receipt_path,mp,rd/'config.json',rd/'checkpoints/best.pt']:result['hashes'][str(p)]=sha(p)
for a,b in [('A','B'),('B','C'),('A','C'),('A','D'),('D','E')]:
 result.setdefault('contrasts',{})[b+'-'+a]={m:{'delta':result['runs'][b]['test'][m]-result['runs'][a]['test'][m],'percent':100*(result['runs'][b]['test'][m]/result['runs'][a]['test'][m]-1)} for m in ['mae','rmse']}
(OUT/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print('coverage',result['coverage'],'source_mismatches',result['source_mismatches'])
for name,r in result['runs'].items():
 print('\n',name,'best',r['best_epoch'],'metrics',r['test'],'skips',r['amp_skips'],'updates',r['updates'],'nonfinite',r['nonfinite_metrics'])
 print('test_route',r['test_route']); print('train_route',r['last_train_route']); print('last_gradient',r['last_gradients']); print('router_diagnostics',r['last_router_diagnostics'])
 print('trends',[x for x in r['trends'] if x['epoch'] in [2,4,6,8,10,20,30,40,50,60,70]])
 print('families',{f:{k:v for k,v in x.items() if k not in ['probability','entropy','usage']} for f,x in r['families'].items()})
print('contrasts',result['contrasts'])
