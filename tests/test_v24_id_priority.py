import copy,itertools,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24')]
from run_id_priority_exploration import static_jobs,choose,resolve,templates,ORDER,SCREEN,MEMORY,SCALE,rows
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss

class UTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);cls.jobs=static_jobs()
 def config(self,n):
  c=copy.deepcopy(self.jobs[n]['config']);c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8;return c
 def model(self,n,epoch=21):
  c=self.config(n);torch.manual_seed(7);m=DualBranchSTImputer.from_config(c);m.main_branch.set_routing_epoch(epoch);return m,c
 def batch(self,n=2):
  g=torch.Generator().manual_seed(19);x=torch.randn(n,2,3,8,8,generator=g);mask=(torch.rand(x.shape,generator=g)>.4).float();return {'x_f_obs':x*mask,'m_f':mask,'x_f_gt':x}
 def test_static_forward_gradients_and_initialization(self):
  ref,c=self.model('U01');rng=torch.get_rng_state();b=self.batch();expected=ref(b)['x_hat_main']
  for n in SCREEN:
   m,c=self.model(n);self.assertTrue(torch.equal(rng,torch.get_rng_state()),n)
   for k,v in ref.named_parameters():
    if not k.startswith('main_branch.scale_heads'):torch.testing.assert_close(v,dict(m.named_parameters())[k],rtol=0,atol=0)
   calls=[];hooks=[e.register_forward_pre_hook(lambda e,a:calls.append(len(a[0]))) for e in m.main_branch.routed_experts()]
   out=m(b)
   for h in hooks:h.remove()
   self.assertEqual(sum(calls),32 if n in ('U13','U14') else 16,n)
   if n not in ('U13','U14','U18'):torch.testing.assert_close(out['x_hat_main'],expected,rtol=0,atol=0,msg=n)
   else:self.assertTrue(torch.isfinite(out['x_hat_main']).all())
   loss,_=compute_coe_loss(out,b,c);loss.backward();self.assertTrue(torch.isfinite(loss),n)
   for key in ('history_gate','innovation_router','response_head'):
    if hasattr(m.main_branch,key):
     gs=[p.grad for p in getattr(m.main_branch,key).parameters() if p.grad is not None];self.assertTrue(gs,n);self.assertTrue(all(torch.isfinite(g).all() for g in gs),n);self.assertTrue(any(g.abs().sum()>0 for g in gs),n)
   if n in ('U08','U09','U10','U13'):
    gs=[p.grad for p in m.main_branch.scale_heads.parameters() if p.grad is not None];self.assertTrue(any(g.abs().sum()>0 for g in gs),n)
   if n in ('U11','U12','U15','U16'):self.assertTrue(all(p.grad is None for p in m.main_branch.scale_heads.parameters()))
  a,_=self.model('U13');d,_=self.model('U14');torch.testing.assert_close(a(b)['x_hat_main'],d(b)['x_hat_main'],rtol=0,atol=0)
 def test_baseline_old_compatibility_and_stages(self):
  from run_four_direction_exploration import static_jobs as oldjobs
  c=copy.deepcopy(oldjobs()['S01']['config']);c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
  torch.manual_seed(7);old=DualBranchSTImputer.from_config(c);new,_=self.model('U01');b=self.batch()
  torch.testing.assert_close(old(b)['x_hat_main'],new(b)['x_hat_main'],rtol=0,atol=0)
  for n in ('U09','U10','U11','U12','U13','U14','U15','U16'):
   m,c=self.model(n,20);torch.testing.assert_close(m(b)['x_hat_main'],new(b)['x_hat_main'],rtol=0,atol=0)
   self.assertFalse(m.main_branch.scales_open());m.main_branch.set_routing_epoch(21);self.assertTrue(m.main_branch.scales_open())
   saved=copy.deepcopy(m.state_dict());other=DualBranchSTImputer.from_config(c);other.load_state_dict(saved,strict=True);other.eval();m.eval();self.assertTrue(other.main_branch.scales_open());torch.testing.assert_close(m(b)['x_hat_main'],other(b)['x_hat_main'],rtol=0,atol=0)
 def test_paths_and_sparse_execution(self):
  for n,count,legal in [('U09',81,lambda p:True),('U10',9,lambda p:p[2:]==(0,0)),('U15',10,lambda p:p[3]==0 and all(p[i]>=p[i+1] for i in range(3)))]:
   paths=[p for p in itertools.product(range(3),repeat=4) if legal(p)];self.assertEqual(len(paths),count)
   m,_=self.model(n);m.eval();b=self.batch(len(paths));calls=[]
   hooks=[e.register_forward_pre_hook(lambda e,a:calls.append(len(a[0]))) for e in m.main_branch.routed_experts()]
   with torch.no_grad():o=m.main_branch(b['x_f_obs'],b['m_f'],forced_scales=torch.tensor(paths))
   for h in hooks:h.remove()
   self.assertEqual(sum(calls),len(paths)*8);self.assertEqual(o['coe']['triscale_choices'].tolist(),[list(p) for p in paths]);self.assertTrue((o['coe']['triscale_executed'].sum(-1)==1).all())
  m,_=self.model('U15');b=self.batch(1)
  with self.assertRaises(ValueError):m.main_branch(b['x_f_obs'],b['m_f'],forced_scales=torch.tensor([[0,2,1,0]]))
 def test_message_definition_bound_detach_and_batch_isolation(self):
  b=self.batch();m,_=self.model('U05');back=m.main_branch;c=back._initialize(b['x_f_obs'],b['m_f']);seen=[]
  from stmoe_imputer.models.temporal_spatial_coe import TemporalSpatialCoE
  from stmoe_imputer.models.spatial_scale_coe import spatial_resize
  orig=TemporalSpatialCoE._dispatch_weighted
  def spy(self,z,w,step):
   e=orig(self,z,w,step);seen.append((z,e));return e
  with patch.object(TemporalSpatialCoE,'_dispatch_weighted',spy):d,row=back._round(c,0)
  torch.testing.assert_close(d['message'],spatial_resize(seen[0][1]-seen[0][0],c['h'].shape[-2:]),rtol=0,atol=0)
  self.assertTrue(torch.equal(back._memory(c)[0],c['h']))
  for n in ('U02','U03','U04','U05','U06'):
   m,_=self.model(n);a=m.main_branch;a.history_gate[-1].bias.data.fill_(.4);c=a._initialize(b['x_f_obs'],b['m_f']);d,_=a._round(c,0)
   d['message']=d['message'].detach().requires_grad_();h,_,g=a._memory(d);self.assertLessEqual(float(g.abs().max()),a.bound)
   if n in ('U05','U06'):
    grad=torch.autograd.grad(h.sum(),d['message'],allow_unused=True,retain_graph=True)[0];self.assertEqual(grad is None,n=='U06')
    if n=='U06':self.assertIsNotNone(torch.autograd.grad(h.sum(),d['prev'],allow_unused=True)[0])
   a.eval();first=a(b['x_f_obs'],b['m_f'])['x_hat_main'];a(b['x_f_obs']*2,b['m_f']);torch.testing.assert_close(first,a(b['x_f_obs'],b['m_f'])['x_hat_main'],rtol=0,atol=0)
 def test_teacher_head_only_legal_candidates_rng_and_clocks(self):
  b=self.batch(3)
  for n in ('U11','U12','U15','U16'):
   m,c=self.model(n);a=m.main_branch;a.batch_clock.fill_(20);a.prepare_training_batch(b);o=m(b);before={k:v.clone() for k,v in a.state_dict().items()};rng=torch.get_rng_state().clone();loss,d=a.candidate_loss(b,o)
   self.assertGreater(d['four_probe_candidates'],0);self.assertGreater(d['four_probe_head_grad_norm'],0);loss.backward()
   self.assertTrue(all(p.grad is None for k,p in a.named_parameters() if not k.startswith('scale_heads')))
   self.assertTrue(torch.equal(rng,torch.get_rng_state()))
   for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,before[k]),k)
   a.set_routing_epoch(20);a.prepare_training_batch(b);self.assertIsNone(a.probe_request)
   a.set_routing_epoch(21);rounds=[]
   for _ in range(6):a.batch_clock.fill_(20);a.prepare_training_batch(b);rounds.append(a.probe_request[0])
   self.assertTrue(set(rounds).issubset(set(a.learnable_steps())))
  m,c=self.model('U15');a=m.main_branch;a.probe_request=(1,torch.arange(3));o=a(b['x_f_obs'],b['m_f'],forced_scales=torch.zeros(3,4,dtype=torch.long));loss,d=a.candidate_loss(b,o);self.assertEqual(d['four_probe_candidates'],0)
 def test_selection_resolution_and_budget(self):
  evidence={n:{'split':'val','val_mae':10+i*.01,'area':4,'trainable_params':100} for i,n in enumerate(('U01',)+MEMORY+SCALE)}
  evidence['U02']['val_mae']=9.;evidence['U03']['val_mae']=9.1;evidence['U05']['val_mae']=9.2;evidence['U12']['val_mae']=9.;evidence['U11']['val_mae']=9.1
  s=choose(evidence);self.assertEqual(s['memory'],['U02','U05']);self.assertEqual(s['scale'],['U12','U11']);self.assertEqual(len(set(ORDER)),30)
  js=copy.deepcopy(self.jobs)
  for n in ORDER:
   if n not in js:js[n]=resolve(n,js,None if n in ('U29','U30') else s)
  self.assertEqual(js['U17']['config']['model']['coe']['id_priority']['communication'],'delta')
  self.assertEqual(js['U17']['config']['model']['coe']['id_priority']['scale_policy'],'teacher')
  for n,j in js.items():self.assertEqual(j['config']['seed'],j['config']['data']['loader_seed'])
  import run_id_priority_exploration as runner
  with tempfile.TemporaryDirectory() as td:
   with patch.object(runner,'collect_evidence',return_value=evidence):
    saved=runner.selection_record(Path(td),js)
    with patch.object(runner,'choose',side_effect=AssertionError('must reuse frozen selection')):self.assertEqual(runner.selection_record(Path(td),js),saved)
  evidence['U02']['split']='test'
  with self.assertRaises(ValueError):choose(evidence)
  for d in ('taxibj','bikenyc'):
   for j in static_jobs(d,50,16).values():self.assertEqual(j['config']['train']['epochs'],50);self.assertEqual(j['config']['data']['batch_size'],16)
 def test_tiny_training_resume_evaluation_and_report(self):
  from train_four_direction import train
  from evaluate_four_direction import evaluate_sets
  from report_id_priority_exploration import export_report
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);j=copy.deepcopy(self.jobs['U12']);c=j['config'];c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8;c['model']['coe']['id_priority']['fixed_epochs']=1;c['train']['epochs']=3;c['train']['scheduler']['total_epochs']=3;c['train']['val_epoch']=1;c['data']['batch_size']=2;c['data']['pin_memory']=False;j['sources']={}
   for sp in ('train','val','test'):
    p=root/(sp+'.npz');np.savez(p,x_f_gt=np.random.RandomState(5).randn(40 if sp=='train' else 4,2,3,8,8).astype('float32'));j['sources'][sp]=str(p)
   suite=root/'suite';run=root/'run';result=suite/'results/U12.json'
   train(j,run,result,'cpu',stop_after=1);r=train(j,run,result,'cpu');self.assertEqual(train(j,run,result,'cpu'),r)
   import run_four_direction_exploration as base
   with patch.object(base,'subprocess_run',side_effect=AssertionError('completed jobs must skip')):base.launch(suite,j,0)
   train(j,root/'full',root/'full_result.json','cpu')
   a=torch.load(run/'checkpoints/last.pth',map_location='cpu',weights_only=False);b=torch.load(root/'full/checkpoints/last.pth',map_location='cpu',weights_only=False)
   for k,v in a['model'].items():torch.testing.assert_close(v,b['model'][k],rtol=0,atol=0,msg=k)
   self.assertGreater(int(a['model']['main_branch.teacher_clock']),0)
   def same(x,y):
    if isinstance(x,torch.Tensor):torch.testing.assert_close(x,y,rtol=0,atol=0)
    elif isinstance(x,np.ndarray):np.testing.assert_array_equal(x,y)
    elif isinstance(x,dict):
     self.assertEqual(x.keys(),y.keys())
     for k in x:same(x[k],y[k])
    elif isinstance(x,(tuple,list)):
     self.assertEqual(len(x),len(y))
     for xx,yy in zip(x,y):same(xx,yy)
    else:self.assertEqual(x,y)
   for key in ('optimizer','scheduler','scaler','rng_states'):same(a[key],b[key])
   protocol={'rate':.4,'split':'test','evaluations':{'in_distribution':{'families':['random_point'],'seed':20260917}}}
   evaluate_sets(run/'checkpoints/best.pth',protocol,j['sources']['test'],suite/'evaluations/U12.json',device='cpu')
   rs=rows(suite,{'U12':j},['U12']);self.assertEqual(rs[0]['status'],'finished');report=export_report(root,suite,rs,{'U12':j},'test');t=report.read_text();self.assertLess(t.index('具体做法'),t.index('结果、曲线'));self.assertIn('尚未整体完成',t)
if __name__=='__main__':unittest.main()
