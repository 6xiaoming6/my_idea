import copy,json,sys,tempfile,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from test_release import small_config,make_data
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.spatial_size_coe import spatial_map,area_weights,LinearSpatialBridge
from stmoe_imputer.data import build_datasets,build_loader
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.utils.deterministic import pool,configure,state_hash
from stmoe_imputer.engine import build_optimizer
class SizeContract(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);make_data(self.root)
  self.base=small_config();self.base['train']['performance']={'enabled':True};configure(self.base)
  ds,_=build_datasets(self.base,self.root/'taxibj_train.npz',self.root/'taxibj_val.npz');self.batch=next(iter(build_loader(ds,self.base,False)))
 def cfg(self,h,w,projection="fixed"):
  c=copy.deepcopy(self.base);c['model']['architecture']='v25_spatial_size_coe';c['model']['coe']['latent_size']={'lh':h,'lw':w,'projection':projection};return c
 def model(self,c):torch.manual_seed(7);return DualBranchSTImputer.from_config(c)
 def test_area_values_and_rectangular_integer_pool(self):
  x=torch.arange(16).float().reshape(1,1,1,4,4)
  torch.testing.assert_close(spatial_map(x,(2,2)),pool(x,2),atol=0,rtol=0)
  torch.testing.assert_close(spatial_map(x,(4,2)),x.reshape(1,1,1,4,2,2).mean(-1),atol=0,rtol=0)
  w=area_weights(4,3);torch.testing.assert_close(w,torch.tensor([[.75,.25,0,0],[0,.5,.5,0],[0,0,.25,.75]]),atol=0,rtol=0)
  for h,w in [(5,7),(3,3),(6,2),(2,6)]:
   y=spatial_map(torch.ones(2,2,3,8,8),(h,w));torch.testing.assert_close(y,torch.ones_like(y),atol=1e-6,rtol=1e-6)
 def test_original_and_old16_grid_step_exact(self):
  old=copy.deepcopy(self.base);old['model']['architecture']='v25_latent_space_coe';old['model']['coe']['latent_space']=dict(factor=2,mapping='linear',return_mode='replace',delta=True,history=True,path='free_last_f')
  for a,b in [(self.base,self.cfg(8,8)),(old,self.cfg(4,4))]:
   results=[]
   for c in (a,b):
    m=self.model(c);op=build_optimizer(m,c);out=m(self.batch);loss,_=compute_main_stage_loss(out,self.batch,c);loss.backward()
    grad=state_hash({k:p.grad for k,p in m.named_parameters()});op.step()
    results.append((out['x_hat_main'],loss.detach(),grad,state_hash(m.state_dict()),state_hash(op.state_dict()),state_hash(torch.get_rng_state())))
   for x,y in zip(*results):
    if torch.is_tensor(x):torch.testing.assert_close(x,y,atol=0,rtol=0)
    else:self.assertEqual(x,y)
 def test_only_size_changes_and_ten_shapes_calls_gradients(self):
  ref=self.model(self.base).state_dict();configs=[]
  for row in json.loads((ROOT/'configs/latent_size/suite.json').read_text()):
   spec=row['spec'];c=self.cfg(spec['lh'],spec['lw'],spec['projection']);m=self.model(c);calls=[];shape=[]
   for name,value in ref.items():self.assertTrue(torch.equal(value,m.state_dict()[name]),name)
   if spec['projection']=='fixed':self.assertEqual(state_hash(m.state_dict()),state_hash(ref))
   hooks=[e.register_forward_hook(lambda mod,inp,out:calls.append(len(inp[0]))) for e in m.main_branch.pattern_experts.values()]
   original=m.main_branch._round
   def step(ctx,n,*a,**kw):shape.append(tuple(ctx['h'].shape[-2:]));return original(ctx,n,*a,**kw)
   m.main_branch._round=step
   out=m(self.batch);loss,_=compute_main_stage_loss(out,self.batch,c);loss.backward()
   for h in hooks:h.remove()
   self.assertEqual(shape,[(spec['lh'],spec['lw'])]*4);self.assertEqual(sum(calls),16)
   self.assertEqual(out['x_hat_main'].shape,self.batch['x_f_gt'].shape)
   self.assertTrue(torch.isfinite(loss));self.assertGreater(m.main_branch.observation_fusion.weight.grad.abs().sum(),0)
   self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
   if spec['projection']!='fixed':self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for name,p in m.named_parameters() if '.to_latent.' in name or '.to_native.' in name))
   actual=json.loads((ROOT/row['config']).read_text());actual['model']['coe'].pop('latent_size');configs.append(actual)
  for c in configs:self.assertEqual(c,configs[0])
 def test_hidden_pollution_empty_full_projection(self):
  for dims in [(5,7),(12,12),(16,32)]:
   m=self.model(self.cfg(*dims)).eval()
   for mode in ('partial','empty','full'):
    b=copy.deepcopy(self.batch)
    for s,f in [('f',1),('m',2),('c',4)]:
     if mode=='empty':b['m_'+s].zero_()
     if mode=='full':b['m_'+s].fill_(1);b['x_'+s+'_obs']=pool(b['x_f_gt'],f)*f*f
    with torch.no_grad():x=m(b)['x_comp']
    if mode!='empty':
     for s,f in [('m',2),('c',4)]:
      mask=b['m_'+s].bool();torch.testing.assert_close((pool(x,f)*f*f)[mask],b['x_'+s+'_obs'][mask],atol=2e-4,rtol=1e-5)
    b['x_f_gt'].fill_(float('nan'))
    for s in 'fmc':b['x_'+s+'_obs'][~b['m_'+s].bool()]=float('nan')
    with torch.no_grad():y=m(b)['x_comp']
    torch.testing.assert_close(x,y,atol=0,rtol=0);self.assertTrue(torch.isfinite(y).all())
 def test_linear_mapping_initialization_and_zero(self):
  x=torch.randn(2,3,4,8,8,requires_grad=True)
  for size in [(4,4),(5,7),(12,12),(8,8)]:
   m=LinearSpatialBridge((8,8),size);y=m(x)
   torch.testing.assert_close(y,spatial_map(x,size),atol=3e-7,rtol=3e-6)
   self.assertEqual(torch.count_nonzero(m(torch.zeros_like(x))),0)
   y.square().mean().backward();self.assertGreater(m.height.weight.grad.abs().sum(),0);self.assertGreater(m.width.weight.grad.abs().sum(),0)
 def test_full_cli_resume(self):
  from test_cli import CommandLine
  from unittest.mock import patch
  c=self.cfg(5,7,"linear_both")
  with patch('test_cli.small_config',lambda:copy.deepcopy(c)):CommandLine().test_train_resume_evaluate_infer()
if __name__=='__main__':unittest.main()
