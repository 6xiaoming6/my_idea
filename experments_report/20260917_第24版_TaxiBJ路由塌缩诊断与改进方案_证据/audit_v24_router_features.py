import sys,json
from pathlib import Path
sys.path[:0]=[str(Path.cwd()/'src'),str(Path.cwd()/'scripts/v24')]
import numpy as np
import torch
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.data.transforms import prepare_single_scale
from stmoe_imputer.engine import build_optimizer
import run_experiments as runner
torch.set_num_threads(2);torch.manual_seed(7)
plan,*_=runner.policy_plan(Path('configs/v24/experiments.json').resolve(),'chain4')
cfg=plan['runs'][0]['config'];model=DualBranchSTImputer.from_config(cfg).eval();branch=model.main_branch
optimizer=build_optimizer(model,cfg)
router_ids={id(p) for r in branch.routers for p in r.parameters()}
report={'router_count':len(branch.routers),'router_parameters_disjoint':len(router_ids)==sum(len(list(r.parameters())) for r in branch.routers),'optimizer_router_groups':[{'name':g['name'],'lr':g['lr'],'router_tensors':sum(id(p) in router_ids for p in g['params'])} for g in optimizer.param_groups if any(id(p) in router_ids for p in g['params'])],'experts':{n:sum(p.numel() for p in e.parameters()) for n,e in zip(branch.expert_names,branch.routed_experts())},'patterns':{}}
x=np.load('data/TaxiBJ/taxibj_train.npz')['x_f_gt'];indices=np.linspace(0,len(x)-1,64,dtype=int)
report['input_channel_mean_abs_difference']=float(np.abs(x[:,0]-x[:,1]).mean())
class Captured(Exception): pass
captures=[]
def capture(module,inputs):
 captures.append(inputs[0].detach().clone());raise Captured()
handle=branch.routers[0].register_forward_pre_hook(capture)
slices={'hidden':(0,128),'values':(128,132),'support':(132,212),'change':(212,216),'missing':(216,218)}
for run in plan['runs']:
 cfgmask=run['config']['data']['mask'];mask=np.loadtxt(cfgmask['train_csv'],delimiter=',',dtype=np.float32).reshape(len(x),1,12,32,32)
 captures.clear()
 for start in range(0,len(indices),4):
  idx=indices[start:start+4]
  sample=prepare_single_scale({'x_f_gt':torch.from_numpy(x[idx]),'m_f':torch.from_numpy(mask[idx])})
  try:
   with torch.no_grad(): model(sample)
  except Captured: pass
 features=torch.cat(captures);normalized=branch.routers[0][0](features).detach()
 blocks={}
 for name,(a,b) in slices.items():
  block=features[:,a:b];norm=normalized[:,a:b]
  blocks[name]={'rms':float(block.square().mean().sqrt()),'mean_across_window_std':float(block.std(0,unbiased=False).mean()),'normalized_mean_across_window_std':float(norm.std(0,unbiased=False).mean())}
 report['patterns'][Path(cfgmask['train_csv']).parents[1].name]=blocks
handle.remove()
# Record exact Gumbel temperature invariance using identical random noise.
logits=torch.tensor([[0.,1.,3.,-2.,-1.,.5]]).expand(4096,-1)
paths=[]
for tau in [.3,1.,3.]:
 torch.manual_seed(2026);paths.append(torch.nn.functional.gumbel_softmax(logits,tau=tau,hard=True).argmax(-1))
report['hard_gumbel_same_noise_same_selections_across_tau']=all(torch.equal(paths[0],p) for p in paths[1:])
Path('/tmp/v24_taxibj_router_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
