import json,math,collections,sys
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.data.diverse_masks import DiverseMaskSchedule,FAMILIES
x=json.loads((OUT/'audit.json').read_text());result={}
for name,r in x['runs'].items():
 rd=Path(r['run_dir']);hist=[json.loads(l) for l in (rd/'logs/metrics.jsonl').read_text().splitlines() if '"train"' in l]
 ck=torch.load(rd/'checkpoints/best.pt',map_location='cpu',weights_only=False)
 cfg=ck['config'];model=DualBranchSTImputer.from_config(cfg)
 rec={'checkpoint_epoch':ck['epoch'],'checkpoint_matches_best':ck['epoch']==r['best_epoch'],'parameters':sum(p.numel() for p in model.parameters()),'buffers':sum(p.numel() for p in model.buffers()),'last10_amp_skips':sum(t['train'].get('train_skipped_amp_steps',0) for t in hist[-10:]),'last_val_minus_best':hist[-1]['val']['mae']-r['best_val_mae'],'last20_val_improvement_percent':100*(hist[49]['val']['mae']-hist[-1]['val']['mae'])/hist[49]['val']['mae']}
 if name in ['D','E']:
  joint=collections.Counter()
  for row in hist[-10:]:
   m=row['train']
   for family in r['families']:
    pre='coe_condition_family_'+family+'_';n=m[pre+'sample_count']
    for k,v in m.items():
     if k.startswith(pre+'path_') and k.endswith('_fraction') and k!=pre+'path_max_fraction':joint[(family,k[len(pre+'path_'):-len('_fraction')])]+=round(n*v)
  total=sum(joint.values());fc=collections.Counter();pc=collections.Counter()
  for (f,p),n in joint.items():fc[f]+=n;pc[p]+=n
  mi=sum(n/total*math.log(n*total/(fc[f]*pc[p])) for (f,p),n in joint.items() if n);hp=-sum(n/total*math.log(n/total) for n in pc.values() if n)
  rec['last10_family_path_association']={'samples':total,'MI_nats':mi,'path_entropy_nats':hp,'MI_over_path_entropy':mi/hp,'note':'Descriptive plug-in estimate from changing checkpoints and stochastic training paths, not a paired causal or significance test.'}
  schedule=DiverseMaskSchedule(2452,cfg['data']['train_mask_diversity']);totals=collections.Counter()
  for epoch in range(1,71):
   schedule.set_epoch(epoch)
   for assignment in schedule.assignments:totals[FAMILIES[int(assignment)]]+=1
  rec['training_family_window_presentations']=dict(totals)
 if name=='E':rec['embedding_row_norms']={k:[float(v) for v in w.float().norm(dim=1)] for k,w in ck['model'].items() if k.endswith('previous_expert_embedding')}
 result[name]=rec
(OUT/'checkpoint_and_association.json').write_text(json.dumps(result,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors=dict(zip('ABCDE',['#2563eb','#0d9488','#9333ea','#ea580c','#dc2626']))
fig,axes=plt.subplots(2,2,figsize=(12,8))
for n,r in x['runs'].items():
 vals=[t for t in r['trends'] if t['val_mae'] is not None];ep=[t['epoch'] for t in vals]
 axes[0,0].plot(ep,[t['val_mae'] for t in vals],label=n,color=colors[n]);axes[0,1].plot(ep,[t['val_mae'] for t in vals],label=n,color=colors[n])
 axes[1,0].plot(ep,[t['val_pathmax'] for t in vals],label=n,color=colors[n]);axes[1,1].plot(ep,[t['val_paths'] for t in vals],label=n,color=colors[n])
axes[0,0].set_yscale('log');axes[0,0].set_title('Validation MAE (log scale; includes C warmup)')
axes[0,1].set_xlim(20,70);axes[0,1].set_ylim(7,17);axes[0,1].set_title('Validation MAE after epoch 20')
axes[1,0].set_ylim(0,1.05);axes[1,0].set_title('Dominant hard path fraction (validation)')
axes[1,1].set_title('Unique hard paths (validation)')
for ax in axes.flat:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(ncol=5)
fig.tight_layout();fig.savefig(OUT/'training_and_routing.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,5,figsize=(15,4),sharey=True)
for ax,(name,r) in zip(axes,x['runs'].items()):
 arr=r['test_route']['usage'];im=ax.imshow(arr,vmin=0,vmax=1,cmap='Blues',aspect='auto')
 for i,row in enumerate(arr):
  for j,v in enumerate(row):
   if v>.001:ax.text(j,i,f'{v*100:.0f}',ha='center',va='center',fontsize=9,color='white' if v>.6 else 'black')
 ax.set_xticks(range(6),x['experts']);ax.set_yticks(range(4),['round 1','round 2','round 3','round 4']);ax.set_title(name+f" | MAE {r['test']['mae']:.3f}")
fig.suptitle('Expert usage (%) on shared random-point test masks, best checkpoints');fig.tight_layout();fig.savefig(OUT/'test_expert_usage.png',dpi=160);plt.close(fig)
print('Saved verified checkpoint details, family counts and two figures.')
