import copy,json,sys,tempfile,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from test_release import small_config,make_data
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.latent_space_coe import pack,unpack,grid_map,GridBridge
from stmoe_imputer.data import build_datasets,build_loader
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.utils.deterministic import configure,pool,state_hash
from stmoe_imputer.engine import build_optimizer

class LatentContract(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.root=Path(self.temp.name);make_data(self.root)
  self.base=small_config();self.base['train']['performance']={'enabled':True};configure(self.base)
  ds,_=build_datasets(self.base,self.root/'taxibj_train.npz',self.root/'taxibj_val.npz')
  self.batch=next(iter(build_loader(ds,self.base,False)))
 def config(self,name='L01'):
  c=copy.deepcopy(self.base);c['model']['architecture']='v25_latent_space_coe'
  c['model']['coe']['latent_space']=json.loads((ROOT/f'configs/latent_space/{name}.json').read_text())['model']['coe']['latent_space']
  return c
 def model(self,c):
  torch.manual_seed(7);return DualBranchSTImputer.from_config(c)
 def test_f_grid_exact_reference_forward_gradient_update_rng(self):
  cfg=self.config();cfg['model']['coe']['latent_space']['factor']=1
  a=self.model(self.base);b=self.model(cfg)
  self.assertEqual(state_hash(a.state_dict()),state_hash(b.state_dict()))
  records=[]
  for m,c in ((a,self.base),(b,cfg)):
   op=build_optimizer(m,c);o=m(self.batch);loss,_=compute_main_stage_loss(o,self.batch,c);loss.backward()
   grads=state_hash({k:p.grad for k,p in m.named_parameters()});op.step()
   records.append((o['x_hat_main'],loss.detach(),grads,state_hash(m.state_dict()),state_hash(op.state_dict()),state_hash(torch.get_rng_state())))
  for x,y in zip(*records):
   if torch.is_tensor(x):torch.testing.assert_close(x,y,atol=0,rtol=0)
   else:self.assertEqual(x,y)
 def test_identical_data_adapter(self):
  for a,b in zip(build_datasets(self.base,self.root/'taxibj_train.npz',self.root/'taxibj_val.npz'),build_datasets(self.config(),self.root/'taxibj_train.npz',self.root/'taxibj_val.npz')):
   for i in range(len(a)):
    aa,bb=a[i],b[i]
    self.assertEqual(set(aa),set(bb))
    for key in aa:torch.testing.assert_close(aa[key],bb[key],atol=0,rtol=0)
 def test_complete_training_resume_evaluate_cli(self):
  from test_cli import CommandLine
  from unittest.mock import patch
  cfg=self.config('L04')
  with patch('test_cli.small_config',lambda:copy.deepcopy(cfg)):
   CommandLine().test_train_resume_evaluate_infer()
 def test_subcell_reordering_and_identity_initialization(self):
  x=torch.arange(2*4*3*8*8).float().reshape(2,4,3,8,8)
  self.assertTrue(torch.equal(unpack(pack(x,2),2),x))
  b=GridBridge(4,(8,8),(4,4),'subcell');torch.testing.assert_close(b(x),pool(x,2),atol=0,rtol=0)
  u=GridBridge(4,(4,4),(8,8),'subcell')
  torch.testing.assert_close(u(b(x)),pool(x,2).repeat_interleave(2,-2).repeat_interleave(2,-1),atol=0,rtol=0)
 def test_ten_variants_shapes_gradients_calls_history_and_projection(self):
  common=self.model(self.base).state_dict()
  for k in range(1,11):
   cfg=self.config(f'L{k:02}');m=self.model(cfg);calls=[];shapes=[]
   for name,p in common.items():self.assertTrue(torch.equal(p,m.state_dict()[name]),name)
   hooks=[e.register_forward_hook(lambda mod,inp,out:calls.append(len(inp[0]))) for e in m.main_branch.pattern_experts.values()]
   for bridge in m.main_branch.to_latent:hooks.append(bridge.register_forward_hook(lambda mod,inp,out:shapes.append(tuple(out.shape[-2:]))))
   out=m(self.batch);loss,_=compute_main_stage_loss(out,self.batch,cfg);loss.backward()
   for hook in hooks:hook.remove()
   self.assertEqual(sum(calls),8*len(self.batch['x_f_obs']))
   self.assertEqual(set(shapes),{m.main_branch.latent_size})
   self.assertEqual(out['x_hat_main'].shape,self.batch['x_f_obs'].shape)
   self.assertTrue(torch.isfinite(loss));self.assertGreater(m.main_branch.observation_fusion.weight.grad.abs().sum(),0)
   self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
   self.assertTrue((out['coe']['triscale_choices'][:,-1]==0).all())
   if k==9:self.assertEqual(out['coe']['triscale_choices'][0].tolist(),[2,1,0,0])
   mf=self.batch['m_f'].bool();self.assertTrue(torch.equal(out['x_comp'][mf],self.batch['x_f_obs'][mf]))
   for s,f in [('m',2),('c',4)]:
    visible=self.batch['m_'+s].bool()
    torch.testing.assert_close((pool(out['x_comp'],f)*f*f)[visible],self.batch['x_'+s+'_obs'][visible],atol=2e-4,rtol=1e-5)
   self.assertEqual([int(out['coe']['diagnostics'][f'x_step{i}_pool_size']) for i in range(1,5)],[0,1,2,3] if cfg['model']['coe']['latent_space']['history'] else [0,0,0,0])
 def test_hidden_pollution_empty_full_and_causal_history(self):
  for name in ('L01','L02','L03','L04','L05'):
   m=self.model(self.config(name)).eval()
   for mode in ('partial','empty','full'):
    b=copy.deepcopy(self.batch)
    for s,f in [('f',1),('m',2),('c',4)]:
     if mode=='empty':b['m_'+s].zero_()
     if mode=='full':
      b['m_'+s].fill_(1);b['x_'+s+'_obs']=pool(b['x_f_gt'],f)*f*f
    with torch.no_grad():o=m(b)['x_comp']
    b['x_f_gt'].fill_(float('nan'))
    for s in 'fmc':b['x_'+s+'_obs'][~b['m_'+s].bool()]=float('nan')
    with torch.no_grad():p=m(b)['x_comp']
    torch.testing.assert_close(o,p,atol=0,rtol=0);self.assertTrue(torch.isfinite(o).all())

if __name__=='__main__':unittest.main()
