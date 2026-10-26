"""Strict checks: no tolerance for prediction, loss, gradients or optimizer state."""
import copy,sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from test_release import small_config,make_data
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.data import build_datasets,build_loader
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.engine import build_optimizer,build_scheduler,build_grad_scaler,train_one_epoch
from stmoe_imputer.performance import FEATURES,sample_batch
from stmoe_imputer.utils.deterministic import state_hash
import tempfile

class PerformanceTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name);make_data(self.path,n=4);self.cfg=small_config()
  ds,_=build_datasets(self.cfg,self.path/'taxibj_train.npz',self.path/'taxibj_val.npz');self.batch=next(iter(build_loader(ds,self.cfg,False)))
 def model(self,cfg):
  torch.manual_seed(7);return DualBranchSTImputer.from_config(cfg).train()
 def config(self,features):
  c=copy.deepcopy(self.cfg);c['train']['performance']=dict(enabled=True,**{k:k in features for k in FEATURES});return c
 def strict_step(self,cfg,compact=False):
  m=self.model(cfg);o=build_optimizer(m,cfg);state=state_hash(m.state_dict());rng=torch.get_rng_state().clone()
  out=m(self.batch,collect_diagnostics=not compact);loss,logs=compute_main_stage_loss(out,self.batch,cfg);loss.backward()
  grads=state_hash({n:p.grad for n,p in m.named_parameters()});torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['train']['grad_clip_norm']);o.step()
  return dict(init=state,pred=out['x_hat_main'].detach(),raw=out['x_hat_raw_consistency'].detach(),loss=loss.detach(),grads=grads,model=state_hash(m.state_dict()),optimizer=state_hash(o.state_dict()),rng=state_hash(torch.get_rng_state()),initial_rng=state_hash(rng))
 def test_individual_and_combined_strict_steps(self):
  a=self.strict_step(self.cfg)
  for features in [{k} for k in FEATURES]+[set(FEATURES)]:
   for compact in [False,True]:
    with self.subTest(features=features,compact=compact):
     b=self.strict_step(self.config(features),compact)
     for k in a:
      if torch.is_tensor(a[k]):torch.testing.assert_close(a[k],b[k],atol=0,rtol=0)
      else:self.assertEqual(a[k],b[k],k)
 def test_counts_and_contract(self):
  cfg=self.config(set(FEATURES));m=self.model(cfg);calls=[];hooks=[m.main_branch.decoder.register_forward_hook(lambda mod,args,out:calls.append(out.requires_grad))]
  try:out=m(self.batch)
  finally:
   for h in hooks:h.remove()
  self.assertEqual(calls,[True,False,False,False,True]);self.assertTrue((out['coe']['triscale_choices'][:,-1]==0).all())
  self.assertEqual([float(out['coe']['diagnostics'][f'x_step{i}_pool_size']) for i in range(1,5)],[0,1,2,3])
  self.assertEqual([sample_batch(m.main_branch,i,77) for i in range(77)].count(True),5)
  self.assertEqual(state_hash(self.model(cfg).state_dict()),state_hash(self.model(self.cfg).state_dict()))
 def test_engine_epoch_primary_metrics_and_state(self):
  make_data(self.path/'six',n=6)
  results=[]
  for cfg in [self.cfg,self.config(set(FEATURES))]:
   m=self.model(cfg);o=build_optimizer(m,cfg);sc=build_grad_scaler(torch.device('cpu'),cfg)
   ds,_=build_datasets(cfg,self.path/'six/taxibj_train.npz',self.path/'six/taxibj_val.npz');loader=build_loader(ds,cfg,True)
   metrics=train_one_epoch(m,loader,o,torch.device('cpu'),cfg,1,scaler=sc,show_progress=False)
   results.append((metrics,state_hash(m.state_dict()),state_hash(o.state_dict())))
  for k in ['loss','mae','rmse','mape','wape','metric_missing_count']:self.assertEqual(results[0][0][k],results[1][0][k],k)
  self.assertEqual(results[0][1:],results[1][1:])
 def test_no_hidden_leak_full_empty_and_eval(self):
  cfg=self.config(set(FEATURES));a=self.model(self.cfg).eval();b=self.model(cfg).eval()
  for mode in ['partial','empty','full']:
   batch=copy.deepcopy(self.batch)
   if mode=='empty':
    for s in 'fmc':batch['m_'+s].zero_();batch['x_'+s+'_obs'].fill_(float('nan'))
   if mode=='full':
    from stmoe_imputer.utils.deterministic import pool
    for s,f in zip('fmc',[1,2,4]):batch['m_'+s].fill_(1);batch['x_'+s+'_obs']=pool(batch['x_f_gt'],f)*f*f
   with torch.no_grad():
    x,y=a(batch),b(batch);torch.testing.assert_close(x['x_hat_main'],y['x_hat_main'],atol=0,rtol=0)
    self.assertEqual(set(x['coe']['diagnostics']),set(y['coe']['diagnostics']))
    for key in x['coe']['diagnostics']:torch.testing.assert_close(x['coe']['diagnostics'][key],y['coe']['diagnostics'][key],atol=0,rtol=0)

 def test_full_checkpoint_continuation(self):
  import subprocess,os
  cfg=self.config(set(FEATURES));path=self.path/'config.json';path.write_text(__import__('json').dumps(cfg))
  def run(target,stop=None):
   args=[sys.executable,str(ROOT/'scripts/train.py'),'--data-dir',str(self.path),'--run-dir',str(target),'--config',str(path),'--device','cpu','--threads','2']
   if stop:args+=['--stop-after',str(stop)]
   subprocess.run(args,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
  run(self.path/'all');run(self.path/'resume',1);run(self.path/'resume')
  a=torch.load(self.path/'all/checkpoints/last.pth',map_location='cpu',weights_only=False);b=torch.load(self.path/'resume/checkpoints/last.pth',map_location='cpu',weights_only=False)
  for key in ['model','optimizer','scheduler','scaler','rng_states']:self.assertEqual(state_hash(a[key]),state_hash(b[key]),key)

if __name__=='__main__':unittest.main()
