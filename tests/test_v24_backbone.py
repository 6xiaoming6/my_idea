import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24')]
from run_backbone_exploration import jobs,choose,resolve,replay_gate,rows,CONFIG,STATIC
from run_b3_c3 import load,write
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import pool,resize,state_hash
from stmoe_imputer.models.backbone_exploration_coe import propagate
from stmoe_imputer.backbone_training import structural_loss,nested_batch,reliability,TrainingContext

class BackboneTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1);cls.jobs=jobs()
 def model(self,n):
  c=copy.deepcopy(self.jobs[n]['config']);c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
  torch.manual_seed(7);return DualBranchSTImputer.from_config(c),c
 def batch(self):
  g=torch.Generator().manual_seed(123);x=torch.randn(2,2,3,8,8,generator=g);mask=(torch.rand(x.shape,generator=g)>.4).float()
  return {'x_f_obs':x*mask,'m_f':mask,'x_f_gt':x,'mask_family':torch.tensor([0,1])}
 def test_deterministic_operators_values_and_gradients(self):
  for factor in (2,4):
   x=torch.randn(2,3,4,8,8,requires_grad=True);a=pool(x,factor);b=F.avg_pool3d(x,(1,factor,factor))
   torch.testing.assert_close(a,b,atol=2e-7,rtol=1e-5)
   probe=torch.randn_like(a);ga=torch.autograd.grad((a*probe).sum(),x)[0];gb=torch.autograd.grad((b*probe).sum(),x)[0]
   torch.testing.assert_close(ga,gb,atol=0,rtol=0)
   x=torch.randn(2,3,4,8//factor,8//factor,requires_grad=True);a=resize(x,(8,8));b=F.interpolate(x.flatten(0,2)[:,None],size=(8,8),mode='bilinear',align_corners=False).reshape_as(a)
   torch.testing.assert_close(a,b,atol=3e-7,rtol=1e-5)
   probe=torch.randn_like(a);ga=torch.autograd.grad((a*probe).sum(),x)[0];gb=torch.autograd.grad((b*probe).sum(),x)[0]
   torch.testing.assert_close(ga,gb,atol=3e-6,rtol=1e-5)
 def test_all_variants_common_initialization_sparse_and_gradients(self):
  b=self.batch();base,c=self.model('W01');rng=torch.get_rng_state();reference=base(b)
  old=copy.deepcopy(c);del old['model']['coe']['backbone_exploration'];torch.manual_seed(7);origin=DualBranchSTImputer.from_config(old)
  origin.load_state_dict(base.state_dict(),strict=True);torch.testing.assert_close(origin(b)['x_hat_main'],reference['x_hat_main'],atol=5e-7,rtol=1e-5)
  for name in STATIC:
   m,c=self.model(name);self.assertTrue(torch.equal(rng,torch.get_rng_state()),name)
   for k,p in base.named_parameters():torch.testing.assert_close(p,dict(m.named_parameters())[k],rtol=0,atol=0)
   calls=[];hooks=[e.register_forward_pre_hook(lambda e,a:calls.append((len(a[0]),a[0].shape[-1]))) for e in m.main_branch.routed_experts()]
   out=m(b)
   for hook in hooks:hook.remove()
   self.assertEqual(sum(x[0] for x in calls),16,name);self.assertEqual(set(x[1] for x in calls),{2,4,8})
   torch.testing.assert_close(out['coe']['triscale_choices'],torch.tensor([[2,1,0,0]]*2))
   torch.testing.assert_close(out['x_hat_main'],reference['x_hat_main'],rtol=0,atol=0,msg=name)
   loss,_=compute_coe_loss(out,b,c);self.assertTrue(torch.isfinite(loss));loss.backward()
   for group in ('adapter','feedback'):
    grads=[p.grad for n,p in m.main_branch.named_parameters() if n.startswith(group) and p.grad is not None]
    if grads:self.assertTrue(all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads)>0,(name,group))
   m.eval();changed=dict(b,x_f_gt=b['x_f_gt']*100)
   torch.testing.assert_close(m(changed)['x_hat_main'],m(b)['x_hat_main'],atol=0,rtol=0)
   other,_=self.model(name);other.load_state_dict(m.state_dict(),strict=True);other.eval()
   torch.testing.assert_close(m(b)['x_hat_main'],other(b)['x_hat_main'],atol=0,rtol=0)
   self.assertFalse(hasattr(m.main_branch,'_w_stats'))
 def test_feedback_causality_and_detach(self):
  b=self.batch()
  for name in ('W12','W14','W15','W17'):
   m,_=self.model(name);main=m.main_branch;c=main._initialize(b['x_f_obs'],b['m_f']);a=main._memory(c)[0]
   torch.testing.assert_close(a,c['h'],rtol=0,atol=0)
   pred=torch.randn_like(b['x_f_obs'],requires_grad=True);c['last_prediction']=pred;c['has_prev']=True
   field,coverage,valid=main._feedback_field(c)
   self.assertEqual(field.requires_grad,name=='W15')
   if name=='W15':
    field.sum().backward();self.assertEqual(float(pred.grad[~b['m_f'].bool()].abs().sum()),0);self.assertGreater(float(pred.grad.abs().sum()),0)
   other=dict(c,x=torch.where(b['m_f'].bool(),c['x'],torch.full_like(c['x'],1e6)))
   torch.testing.assert_close(field,main._feedback_field(other)[0],atol=0,rtol=0)
   with torch.no_grad():main.feedback_input[-1].weight.fill_(.1)
   out=m(b)['x_hat_main'];m(dict(b,x_f_obs=b['x_f_obs']*2));torch.testing.assert_close(out,m(b)['x_hat_main'],atol=0,rtol=0)
   if name=='W14':
    corrected,_,_=main._memory(c);baseline=super(type(main),main)._memory(c)[0]
    fully_observed=b['m_f'].bool().all(1,keepdim=True).expand_as(corrected)
    torch.testing.assert_close(corrected[fully_observed],baseline[fully_observed],rtol=0,atol=0)
  x=torch.zeros(1,2,3,5,5);mask=x.clone();mask[:,:,0,0,0]=1;x[:,:,0,0,0]=2
  field,cov,valid=propagate(x,mask);self.assertEqual(float(field[0,0,-1,-1,-1]),0);self.assertEqual(float(field[0,0,1,1,1]),2)
  field,cov,valid=propagate(torch.zeros_like(x),torch.zeros_like(mask));self.assertFalse(valid.any());self.assertEqual(float(field.sum()),0)
 def test_nested_masks_rng_and_ema(self):
  b=self.batch();rng=np.random.default_rng(888);state=copy.deepcopy(rng.bit_generator.state);a=nested_batch(b,rng)
  self.assertTrue((a['m_f']<=b['m_f']).all());self.assertTrue(((a['m_f']==0).flatten(2).sum(2)==96).all())
  rng.bit_generator.state=state;torch.testing.assert_close(a['m_f'],nested_batch(b,rng)['m_f'],atol=0,rtol=0)
  self.assertEqual(float(a['x_f_obs'][~a['m_f'].bool()].abs().sum()),0)
  for n in ('W19','W20','W21','W22'):
   m,c=self.model(n);ctx=TrainingContext(m,c);out=m(b);loss,logs=ctx.extra_loss(m,b,out);loss.backward();self.assertTrue(torch.isfinite(loss))
   self.assertGreater(float(logs['l_w_second_view']),0)
   if ctx.teacher is not None:
    self.assertFalse(ctx.teacher.training);self.assertTrue(all(p.grad is None for p in ctx.teacher.parameters()))
    old=next(ctx.teacher.parameters()).clone()
    with torch.no_grad():next(m.parameters()).add_(1)
    torch.testing.assert_close(next(ctx.teacher.parameters()),old,rtol=0,atol=0)
    ctx.after_update(m);torch.testing.assert_close(next(ctx.teacher.parameters()),old+.01,rtol=1e-5,atol=1e-7)
   saved=copy.deepcopy(ctx.state_dict());other=TrainingContext(m,c);other.load_state_dict(saved)
   torch.testing.assert_close(nested_batch(b,ctx.rng)['m_f'],nested_batch(b,other.rng)['m_f'],atol=0,rtol=0)
   self.assertEqual(state_hash(ctx.state_dict()),state_hash(other.state_dict()))
  selected=~b['m_f'].bool();teacher=torch.ones_like(b['x_f_gt']);w=reliability(teacher,b,selected)
  torch.testing.assert_close(w.flatten(1).sum(1)/selected.flatten(1).sum(1),torch.ones(2));self.assertFalse(w.requires_grad)
 def test_structural_losses_and_empty_targets(self):
  pred=torch.zeros(1,1,2,4,4,requires_grad=True);truth=torch.ones_like(pred);mask=torch.zeros_like(pred)
  b={'x_f_gt':truth,'x_f_obs':truth*mask,'m_f':mask}
  for kind in ('region','region_count'):
   out={'x_hat_main':pred,'w_spec':{'structure':kind}};loss,_=structural_loss(out,b);self.assertAlmostEqual(float(loss),.05,places=6)
   empty=dict(b,x_f_gt=torch.full_like(truth,torch.nan));z,_=structural_loss(out,empty);self.assertEqual(float(z),0);z.backward();self.assertTrue(torch.isfinite(pred.grad).all())
  # Temporal edge with one visible endpoint contributes |(0-1)-(3-1)| = 3.
  b['m_f'][:,:,0]=1;b['x_f_obs'][:,:,0]=1;b['x_f_gt'][:,:,1]=3
  loss,_=structural_loss({'x_hat_main':pred,'w_spec':{'structure':'temporal'}},b);self.assertAlmostEqual(float(loss),.15,places=6)
  b['target_mask']=torch.zeros_like(mask).bool()
  for kind in ('temporal','spatial'):
   loss,_=structural_loss({'x_hat_main':pred,'w_spec':{'structure':kind}},b);self.assertEqual(float(loss),0)
  out={'x_hat_main':pred,'w_spec':{'observed_loss':True},'coe':{'predictions':[pred]*4}}
  loss,_=structural_loss(out,b);self.assertAlmostEqual(float(loss),.02,places=6)
 def test_demeaned_absolute_diagnostic(self):
  from stmoe_imputer.coordination import CoordinationMetrics
  pred=torch.tensor([0.,2.]*8).reshape(1,1,1,4,4);truth=torch.zeros_like(pred);mask=torch.zeros_like(pred)
  out={'x_hat_main':pred,'coordination':{'stages':{'raw':pred},'means':{},'allocations':{}}}
  metric=CoordinationMetrics();metric.update(out,{'x_f_gt':truth,'m_f':mask})
  self.assertEqual(metric.compute()['coord_c_raw_demeaned_mae'],1.)
 def test_id_selection_composition_and_replay_failure(self):
  e={n:{'split':'val','mae':10.,'trainable_params':100} for n in STATIC};e['W10']['mae']=9.;e['W18']['mae']=9.1;e['W21']['mae']=9.2
  selection=choose(e);self.assertEqual(selection['selected'],{'A':'W10','B':'W18','C':'W21'})
  j=resolve('W30',self.jobs,selection,load(CONFIG/'variants.json'));w=j['config']['model']['coe']['backbone_exploration']
  self.assertEqual(w,{'enabled':True,'adapter':'coverage_rank','feedback':'missing','observed_loss':True,'view':'ema'})
  e['W03']['split']='test'
  with self.assertRaises(ValueError):choose(e)
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);small=copy.deepcopy(self.jobs);small['W01']['config']['train']['epochs']=1
   for n in ('W01','W02'):
    write(root/n/'initialization.json',{'common':1});write(root/n/'replay_audit.json',[{'epoch':1,'metric':1}])
   with patch('run_backbone_exploration.base.receipt',side_effect=lambda s,j:{'run_dir':str(root/j['variant'])}):
    self.assertTrue(replay_gate(root,small)['passed']);write(root/'W02/replay_audit.json',[{'epoch':1,'metric':2}])
    with self.assertRaises(RuntimeError):replay_gate(root,small)
 def test_training_replay_resume_evaluation_report(self):
  from train_four_direction import train
  from evaluate_four_direction import evaluate_sets
  from report_backbone_exploration import export_report
  import run_four_direction_exploration as base
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);suite=root/'suite';js=copy.deepcopy(self.jobs)
   for n in ('W01','W02','W21'):
    j=js[n];j['common_initialization']=str(suite/'common_initialization.pth');_,c=self.model(n);j['config']=c;c['train'].update(epochs=2,val_epoch=1);c['train']['scheduler']['total_epochs']=2;c['data'].update(batch_size=2,pin_memory=False)
    j['sources']={}
    for sp in ('train','val','test'):
     p=root/(sp+'.npz');np.savez(p,x_f_gt=np.random.RandomState(2).randn(4,2,3,8,8).astype('float32'));j['sources'][sp]=str(p)
    run=root/n;rfile=suite/'results'/f'{n}.json'
    if n=='W21':train(j,run,rfile,'cpu',stop_after=1)
    result=train(j,run,rfile,'cpu');self.assertEqual(train(j,run,rfile,'cpu'),result)
    with patch.object(base,'subprocess_run',side_effect=AssertionError('must skip')):base.launch(suite,j,0)
    if n=='W21':
     train(j,root/'full',root/'full.json','cpu')
     a=torch.load(run/'checkpoints/last.pth',weights_only=False);b=torch.load(root/'full/checkpoints/last.pth',weights_only=False)
     for key in ('model','optimizer','scaler','scheduler'):self.assertEqual(state_hash(a[key]),state_hash(b[key]),key)
     self.assertEqual(state_hash(a['training_state']['w_training']),state_hash(b['training_state']['w_training']))
     self.assertEqual(state_hash(a['rng_states']),state_hash(b['rng_states']))
   self.assertTrue(replay_gate(suite,js)['passed'])
   protocol={'rate':.4,'evaluations':{'in_distribution':{'families':['random_point'],'seed':20260917}}}
   evaluate_sets(root/'W21/checkpoints/best.pth',protocol,js['W21']['sources']['test'],suite/'evaluations/W21.json',device='cpu')
   templates=load(CONFIG/'variants.json');r=rows(suite,js,templates);path=export_report(root,suite,r,js)
   text=path.read_text();self.assertIn('阶段',text);self.assertLess(text.index('具体做法'),text.index('结果'))
if __name__=='__main__':unittest.main()
