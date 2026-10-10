import copy,json,sys,tempfile,unittest,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24')]
from run_history_plane_interaction import jobs,prepare_initialization,REFERENCE_REL,REFERENCE_NAME
from run_b3_c3 import load
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import state_hash,common_initialization
from stmoe_imputer.models.history_plane_interaction_coe import HistoryReader,TriPlaneBranch

class XTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1);cls.jobs=jobs()
 def config(self,name):
  c=copy.deepcopy(self.jobs[f'taxibj_{name}_seed7']['config']);c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8;return c
 def build(self,c):torch.manual_seed(7);return DualBranchSTImputer.from_config(c)
 def batch(self):
  g=torch.Generator().manual_seed(19);x=torch.randn(2,2,3,8,8,generator=g);mask=(torch.rand(x.shape,generator=g)>.4).float()
  return dict(x_f_obs=x*mask,x_f_gt=x,m_f=mask,mask_family=torch.tensor([0,1]))
 def test_exact_plan_common_state_rng_and_zero_identity(self):
  self.assertEqual(len(self.jobs),10);b=self.batch();basec=self.config('X01');basec['model']['coe'].pop('history_plane_interaction')
  base=self.build(basec);rng=torch.get_rng_state();expected=base(b)['x_hat_main']
  for name,j in self.jobs.items():
   c=self.config(j['method']);model=self.build(c)
   self.assertTrue(torch.equal(torch.get_rng_state(),rng))
   for k,v in base.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
   out=model(b)
   if j['method']!='X02':torch.testing.assert_close(out['x_hat_main'],expected,rtol=0,atol=0)
   self.assertEqual(c['train']['epochs'],100);self.assertEqual(c['data']['batch_size'],32)
   self.assertEqual(c['data']['mask']['missing_rate'],.4);self.assertNotIn('post_fusion_ffn',c['model']['coe'])
  a=self.build(self.config('X09')).main_branch.x_interaction;b=self.build(self.config('X10')).main_branch.x_interaction
  self.assertEqual(state_hash(a.state_dict()),state_hash(b.state_dict()))
  for name in ('X09','X10'):
   full=self.build(self.jobs[f'taxibj_{name}_seed7']['config'])
   self.assertEqual(sum(p.numel() for p in full.main_branch.x_interaction.parameters()),6240)
 def test_sparse_execution_gradients_and_residual(self):
  batch=self.batch()
  for name,j in self.jobs.items():
   c=self.config(j['method']);m=self.build(c);main=m.main_branch
   # Exercise ST and pair branch in all rounds without changing dispatch budget.
   with torch.no_grad():
    for router in main.routers:router[-1].weight.zero_();router[-1].bias.zero_();router[-1].bias[4:6]=10
   calls=[];hooks=[e.register_forward_pre_hook(lambda module,args:calls.append((len(args[0]),args[0].shape[-1]))) for e in main.routed_experts(0)]
   out=m(batch)
   for h in hooks:h.remove()
   self.assertEqual(sum(n for n,w in calls),16,name);self.assertEqual({w for n,w in calls},{2,4,8})
   torch.testing.assert_close(out['coe']['triscale_choices'],torch.tensor([[2,1,0,0]]*2))
   loss,_=compute_coe_loss(out,batch,c);loss.backward()
   blocks=[getattr(main,k) for k in ('x_history','x_interaction') if hasattr(main,k)]
   if main.x_spec['plane']!='none':blocks.append(main.pattern_experts['ST'].plane)
   if main.communication!='none':blocks.append(main.history_gate)
   for block in blocks:
    grads=[p.grad for p in block.parameters() if p.grad is not None]
    self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads),name)
  c=self.config('X02');m=self.build(c).main_branch;ctx=m._initialize(batch['x_f_obs'],batch['m_f']);m._memory(ctx)
  ew=ctx['h'].new_zeros((2,8));ew[:,:2]=.5;sw=ctx['h'].new_tensor([[0,0,1]]*2)
  actual=m._execute(ctx,ctx['h'],ew,sw,0)[0]
  m.x_spec['residual']=False;plain=m._execute(ctx,ctx['h'],ew,sw,0)[0]
  torch.testing.assert_close(actual,ctx['h']+plain,rtol=0,atol=0)
 def test_history_causality_read_modes_and_isolation(self):
  b=self.batch();c=self.config('X05');model=self.build(c);main=model.main_branch;captures=[]
  original=main.x_history.read
  def read(history,mask,scale):
   captures.append([p.clone() for p in history]);self.assertTrue(all(not p.requires_grad for p in history));return original(history,mask,scale)
  with patch.object(main.x_history,'read',side_effect=read):out=model(b)
  self.assertEqual([len(x) for x in captures],[1,2,3])
  for history in captures:
   for i,p in enumerate(history):torch.testing.assert_close(p,out['coe']['completions'][i],rtol=0,atol=0)
  pred=out['x_hat_main'];model(dict(b,x_f_obs=b['x_f_obs']*5));torch.testing.assert_close(pred,model(b)['x_hat_main'],rtol=0,atol=0)
  torch.testing.assert_close(pred,model(dict(b,x_f_gt=b['x_f_gt']*100))['x_hat_main'],rtol=0,atol=0)
  # Activating injection proves history content affects subsequent rounds.
  with torch.no_grad():main.x_history.inject[-1].weight.fill_(.02)
  changed=model(b)['x_hat_main'];self.assertFalse(torch.equal(changed,pred))
  mask=torch.ones(2,2,3,8,8);scale=torch.ones(2,2,1,1,1);history=[mask*1,mask*3,mask*5]
  for mode,value in [('last',5),('mean',3),('adaptive',3)]:
   reader=HistoryReader(2,8,mode);r,w=reader.read(history,mask,scale)
   torch.testing.assert_close(r,mask*value);torch.testing.assert_close(w.sum(1),mask[:,:1])
 def test_planes_support_zero_coverage_and_active_difference(self):
  torch.manual_seed(3);uniform=TriPlaneBranch(16,'uniform');covered=copy.deepcopy(uniform);covered.mode='coverage'
  with torch.no_grad():uniform.up.weight.fill_(.02);uniform.up.bias.fill_(.4);covered.load_state_dict(uniform.state_dict())
  x=torch.randn(2,16,3,8,8);q=torch.zeros(2,2,3,8,8);q[:,:,0,:3,:]=1
  a,sa=uniform(x,q);b,sb=covered(x,q);self.assertFalse(torch.equal(a,b))
  z,stats=covered(x,torch.zeros_like(q));torch.testing.assert_close(z,torch.zeros_like(z),rtol=0,atol=0)
  full=torch.ones_like(q);a,_=uniform(x,full);b,_=covered(x,full);torch.testing.assert_close(a,b,rtol=0,atol=0)
  for name in ('X05','X08','X10'):
   model=self.build(self.config(name));batch=self.batch()
   for fill in (0.,1.):
    mask=torch.full_like(batch['m_f'],fill);out=model(dict(batch,m_f=mask,x_f_obs=batch['x_f_gt']*mask))
    self.assertTrue(torch.isfinite(out['x_hat_main']).all())
 def test_pair_representation_and_no_extra_calls(self):
  b=self.batch();models=[]
  for name in ('X09','X10'):
   m=self.build(self.config(name));
   with torch.no_grad():m.main_branch.x_interaction[-1].weight.fill_(.03)
   models.append(m)
  self.assertFalse(torch.equal(models[0](b)['x_hat_main'],models[1](b)['x_hat_main']))
 def test_full_templates_resume_evaluation_report(self):
  from train_four_direction import train
  from evaluate_four_direction import evaluate_sets
  from report_history_plane_interaction import export_report
  import run_four_direction_exploration as base
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);suite=root/'suite';js=copy.deepcopy(self.jobs)
   for j in js.values():j['config']=self.config(j['method'])
   c=self.config('X01');c['model']['coe'].pop('history_plane_interaction');state=self.build(c).state_dict()
   refinit=root/'ref.pth';torch.save({'model':state,'sha256':state_hash(state)},refinit)
   prepare_initialization(suite,js,refinit);prepare_initialization(suite,js,refinit)
   for j in js.values():common_initialization(self.build(j['config']),j)
   protocol={'rate':.4,'evaluations':{'in_distribution':{'families':['random_point'],'seed':20260917}}}
   for method in ('X05','X08','X10'):
    j=js[f'taxibj_{method}_seed7'];c=j['config'];c['train'].update(epochs=2,val_epoch=1);c['train']['scheduler']['total_epochs']=2
    c['data'].update(batch_size=2,pin_memory=False);c['experiment_plan']['protocol']=protocol;j['sources']={}
    for split in ('train','val','test'):
     p=root/(split+'.npz');np.savez(p,x_f_gt=np.random.RandomState(2).randn(4,2,3,8,8).astype('float32'));j['sources'][split]=str(p)
    run=root/method;result=suite/'results'/(j['variant']+'.json')
    train(j,run,result,'cpu',stop_after=1);r=train(j,run,result,'cpu');self.assertEqual(train(j,run,result,'cpu'),r)
    with patch.object(base,'subprocess_run',side_effect=AssertionError('must skip')):base.launch(suite,j,0)
    full=root/(method+'_full');train(j,full,root/(method+'_full.json'),'cpu')
    a=torch.load(run/'checkpoints/last.pth',weights_only=False);z=torch.load(full/'checkpoints/last.pth',weights_only=False)
    for k in ('model','optimizer','scheduler','scaler','rng_states'):self.assertEqual(state_hash(a[k]),state_hash(z[k]),method+k)
    self.assertEqual(state_hash(a['training_state']['w_training']),state_hash(z['training_state']['w_training']))
    e=evaluate_sets(run/'checkpoints/best.pth',protocol,j['sources']['test'],suite/'evaluations'/(j['variant']+'.json'),device='cpu')
    metrics=e['sets']['in_distribution']['metrics'];self.assertAlmostEqual(metrics['mae'],metrics['coe_x_step4_mae'],places=5)
   ref=suite/'reference';ref.mkdir();original=ROOT/REFERENCE_REL
   for label,source in {'config.json':original/'configs'/f'{REFERENCE_NAME}.json','result.json':original/'results'/f'{REFERENCE_NAME}.json','evaluations.json':original/'evaluations'/f'{REFERENCE_NAME}.json'}.items():shutil.copyfile(source,ref/label)
   oldrun=Path(load(ref/'result.json')['run_dir'])
   for label,source in {'metrics.jsonl':oldrun/'logs/metrics.jsonl','training_metadata.json':oldrun/'training_metadata.json'}.items():shutil.copyfile(source,ref/label)
   path=export_report(root,suite,{'jobs':js,'reference':{'name':REFERENCE_NAME}});text=path.read_text()
   self.assertIn('阶段报告',text);self.assertLess(text.index('具体方法'),text.index('完整结果'));self.assertNotIn('F01/F02',text)
   status=load(suite/'summary.json');self.assertEqual(status['taxibj_X10_seed7']['status'],'finished');self.assertEqual(status[REFERENCE_NAME]['status'],'finished')

if __name__=='__main__':unittest.main()
