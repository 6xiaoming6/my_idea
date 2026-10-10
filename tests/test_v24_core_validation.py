import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24')]
from run_core_validation import jobs,prepare_initialization
from run_backbone_exploration import jobs as w_jobs
from run_b3_c3 import write,load
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import state_hash

class CoreTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1);cls.jobs=jobs()
 def model(self,method):
  c=copy.deepcopy(self.jobs['taxibj_'+method+'_seed7']['config']);c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
  torch.manual_seed(7);return DualBranchSTImputer.from_config(c),c
 def batch(self):
  rng=torch.Generator().manual_seed(13);x=torch.randn(2,2,3,8,8,generator=rng);mask=(torch.rand(x.shape,generator=rng)>.4).float()
  return {'x_f_obs':x*mask,'x_f_gt':x,'m_f':mask,'mask_family':torch.tensor([0,1])}
 def test_plan_and_seed_isolation(self):
  self.assertEqual(len(self.jobs),28);self.assertEqual(len(jobs(('taxibj',))),16)
  for j in self.jobs.values():
   c=j['config'];self.assertEqual(c['seed'],c['data']['loader_seed']);self.assertEqual(c['data']['mask']['missing_rate'],.4)
   self.assertEqual(c['model']['coe']['backbone_exploration'],{'enabled':True});self.assertEqual(c['train']['val_epoch'],5)
   self.assertEqual(c['model']['coe']['id_priority']['scale_policy'],'fixed')
  masks=[state_hash(j['config']['data']['train_mask_diversity']) for j in self.jobs.values() if j['dataset']=='taxibj']
  self.assertEqual(len(set(masks)),1)
 def test_old_w_exact_forward_and_gradient(self):
  new,c=self.model('K04');oldc=copy.deepcopy(c);del oldc['model']['coe']['core_validation'];torch.manual_seed(7)
  old=DualBranchSTImputer.from_config(oldc);self.assertEqual(state_hash(old.state_dict()),state_hash(new.state_dict()))
  b=self.batch();a=old(b);z=new(b);torch.testing.assert_close(a['x_hat_main'],z['x_hat_main'],rtol=0,atol=0)
  la,_=compute_coe_loss(a,b,oldc);lz,_=compute_coe_loss(z,b,c);la.backward();lz.backward()
  for (na,pa),(nb,pb) in zip(old.named_parameters(),new.named_parameters()):
   self.assertEqual(na,nb)
   if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)
  old.load_state_dict(new.state_dict(),strict=True)
 def test_all_controls_sparse_paths_common_state(self):
  base,_=self.model('K04');common=base.state_dict();b=self.batch();reference=base(b)['x_hat_main'];rng=torch.get_rng_state()
  for method in [f'K{i:02}' for i in range(1,9)]:
   m,c=self.model(method);self.assertTrue(torch.equal(rng,torch.get_rng_state()),method)
   for k,v in common.items():torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)
   unique={id(e):e for step in range(4) for e in m.main_branch.routed_experts(step)};calls=[]
   hooks=[e.register_forward_pre_hook(lambda e,args:calls.append((len(args[0]),args[0].shape[-1]))) for e in unique.values()]
   out=m(b)
   for h in hooks:h.remove()
   self.assertEqual(sum(x[0] for x in calls),16,method)
   expected=[0,0,0,0] if method in ('K01','K03') else [2,1,0,0]
   torch.testing.assert_close(out['coe']['triscale_choices'],torch.tensor([expected]*2))
   self.assertEqual(set(v for _,v in calls),{8} if expected[0]==0 else {2,4,8})
   if expected[0]==2:torch.testing.assert_close(out['x_hat_main'],reference,rtol=0,atol=0)
   loss,_=compute_coe_loss(out,b,c);loss.backward()
   grads=[p.grad for p in m.main_branch.history_gate.parameters() if p.grad is not None]
   if method not in ('K01','K02','K05'):self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads),method)
   else:self.assertFalse(grads)
   if method in ('K05','K06'):
    a=next(m.main_branch.routed_experts(0)[0].parameters());p=next(m.main_branch.routed_experts(1)[0].parameters())
    self.assertNotEqual(a.data_ptr(),p.data_ptr());torch.testing.assert_close(a,p,rtol=0,atol=0)
   m.eval();pred=m(b)['x_hat_main'];m(dict(b,x_f_obs=b['x_f_obs']*2));torch.testing.assert_close(pred,m(b)['x_hat_main'],rtol=0,atol=0)
   torch.testing.assert_close(pred,m(dict(b,x_f_gt=b['x_f_gt']*100))['x_hat_main'],rtol=0,atol=0)
 def test_memory_bound_raw_and_unconditional(self):
  b=self.batch();m,_=self.model('K04');main=m.main_branch;c=main._initialize(b['x_f_obs'],b['m_f'])
  torch.testing.assert_close(main._memory(c)[0],c['h'],rtol=0,atol=0)
  with torch.no_grad():main.history_gate[-1].bias.fill_(.7)
  c['has_prev']=True;c['prev']=c['h']*.9;result,_,gate=main._memory(c)
  change=result-c['h'];rms=lambda x:x.square().mean((2,3,4)).sqrt()
  self.assertTrue((rms(change)<=.100001*rms(c['h'])).all())
  raw,_=self.model('K08');raw.load_state_dict(m.state_dict(),strict=True)
  actual=raw.main_branch._memory(c)[0]
  torch.testing.assert_close(actual,c['h']+gate[:,:,None,None,None]*(c['h']-c['prev']),rtol=0,atol=0)
  unconditional,_=self.model('K07');unconditional.load_state_dict(m.state_dict(),strict=True)
  _,_,g1=unconditional.main_branch._memory(c);_,_,g2=unconditional.main_branch._memory(dict(c,h=c['h']*10,prev=c['prev']*2))
  torch.testing.assert_close(g1,g2,rtol=0,atol=0)
 def test_full_initialization_templates(self):
  js=jobs(('taxibj',),seeds=(7,))
  for j in js.values():j['config']['model']['main']['dim']=8;j['config']['model']['coe']['router_hidden_dim']=8
  with tempfile.TemporaryDirectory() as td:
   suite=Path(td);prepare_initialization(suite,js);a=load(suite/'initialization_audit.json')
   self.assertEqual(len({r['common_sha256'] for r in a.values()}),1)
   prepare_initialization(suite,js)
   from stmoe_imputer.utils.deterministic import common_initialization
   for j in js.values():
    torch.manual_seed(7);m=DualBranchSTImputer.from_config(j['config']);common_initialization(m,j)
 def test_training_resume_evaluation_skip_and_report(self):
  from train_four_direction import train
  from evaluate_four_direction import evaluate_sets
  from report_core_validation import export_report
  import run_four_direction_exploration as base
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);suite=root/'suite';j=copy.deepcopy(self.jobs['taxibj_K04_seed7']);_,c=self.model('K04');j['config']=c
   c['train'].update(epochs=2,val_epoch=1);c['train']['scheduler']['total_epochs']=2;c['data'].update(batch_size=2,pin_memory=False)
   protocol={'rate':.4,'evaluations':{'in_distribution':{'families':['random_point'],'seed':20260917}}};c['experiment_plan']['protocol']=protocol
   j['sources']={}
   for sp in ('train','val','test'):
    p=root/(sp+'.npz');np.savez(p,x_f_gt=np.random.RandomState(2).randn(4,2,3,8,8).astype('float32'));j['sources'][sp]=str(p)
   js={j['variant']:j};prepare_initialization(suite,js);run=root/'resumed';result=suite/'results'/(j['variant']+'.json')
   train(j,run,result,'cpu',stop_after=1);r=train(j,run,result,'cpu');self.assertEqual(train(j,run,result,'cpu'),r)
   with patch.object(base,'subprocess_run',side_effect=AssertionError('must skip')):base.launch(suite,j,0)
   train(j,root/'full',root/'full.json','cpu')
   a=torch.load(run/'checkpoints/last.pth',weights_only=False);b=torch.load(root/'full/checkpoints/last.pth',weights_only=False)
   for k in ('model','optimizer','scheduler','scaler','rng_states'):self.assertEqual(state_hash(a[k]),state_hash(b[k]),k)
   self.assertEqual(state_hash(a['training_state']['w_training']),state_hash(b['training_state']['w_training']))
   evaluate_sets(run/'checkpoints/best.pth',protocol,j['sources']['test'],suite/'evaluations'/(j['variant']+'.json'),device='cpu')
   path=export_report(root,suite,js);s=path.read_text();self.assertIn('完整批次',s);self.assertLess(s.index('具体方法'),s.index('完整结果'))
if __name__=='__main__':unittest.main()
