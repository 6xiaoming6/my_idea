import copy,json,sys,tempfile,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from test_release import small_config,make_data
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.data import build_datasets,build_loader
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.utils.deterministic import configure,pool,state_hash
from stmoe_imputer.models.latent_token_coe import CrossMap
class TokenContract(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);make_data(self.root)
  self.base=small_config();self.base['train']['performance']={'enabled':True};configure(self.base)
  ds,_=build_datasets(self.base,self.root/'taxibj_train.npz',self.root/'taxibj_val.npz');self.batch=next(iter(build_loader(ds,self.base,False)))
 def cfg(self,name='LT01'):
  c=copy.deepcopy(self.base);c['model']['architecture']='v25_latent_token_coe'
  c['model']['coe']['latent_tokens']=json.loads((ROOT/f'configs/latent_tokens/{name}.json').read_text())['model']['coe']['latent_tokens'];return c
 def model(self,c):torch.manual_seed(7);return DualBranchSTImputer.from_config(c)
 def test_all_variants_common_initialization_shapes_calls_and_gradients(self):
  old=self.model(self.base).state_dict();first=None
  for i in range(1,11):
   cfg=self.cfg(f'LT{i:02}');m=self.model(cfg);calls=[];shapes=[]
   for k,v in old.items():self.assertTrue(torch.equal(v,m.state_dict()[k]),k)
   hooks=[e.register_forward_hook(lambda mod,inp,out:calls.append(len(inp[0]))) for e in m.main_branch.pattern_experts.values()]
   original=m.main_branch._round
   def checked(c,step,*args,**kwargs):
    shapes.append(c['h'].shape);return original(c,step,*args,**kwargs)
   m.main_branch._round=checked
   out=m(self.batch);loss,_=compute_main_stage_loss(out,self.batch,cfg);loss.backward()
   for hook in hooks:hook.remove()
   self.assertEqual(len(shapes),4)
   self.assertTrue(all(tuple(s)==(2,64,4,cfg['model']['coe']['latent_tokens']['count'],1) for s in shapes))
   self.assertEqual(sum(calls),16);self.assertEqual(out['x_hat_main'].shape,(2,2,4,8,8))
   self.assertTrue(torch.isfinite(loss));self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
   for p in [m.main_branch.token_slots,m.main_branch.grid_to_tokens[0].query.weight,m.main_branch.tokens_to_grid[0].key.weight,m.main_branch.decoder[-1].weight]:
    self.assertIsNotNone(p.grad);self.assertGreater(p.grad.abs().sum(),0)
   self.assertEqual([int(out['coe']['diagnostics'][f'x_step{s}_pool_size']) for s in range(1,5)],[0,1,2,3] if cfg['model']['coe']['latent_tokens']['history'] else [0,0,0,0])
   if i==9:self.assertEqual(out['coe']['triscale_choices'][0].tolist(),[2,1,0,0])
   self.assertTrue((out['coe']['triscale_choices'][:,-1]==0).all())
   mf=self.batch['m_f'].bool();self.assertTrue(torch.equal(out['x_comp'][mf],self.batch['x_f_obs'][mf]))
   for s,f in [('m',2),('c',4)]:
    visible=self.batch['m_'+s].bool();torch.testing.assert_close((pool(out['x_comp'],f)*f*f)[visible],self.batch['x_'+s+'_obs'][visible],atol=3e-4,rtol=1e-5)
   self.assertFalse(hasattr(m.main_branch,'_token_positions'))
 def test_non_square_prime_slots_and_permutation(self):
  c=self.cfg();c['model']['coe']['latent_tokens']['count']=97
  a=self.model(c).eval();b=self.model(c).eval()
  order=torch.randperm(97)
  with torch.no_grad():b.main_branch.token_slots.copy_(a.main_branch.token_slots[order])
  with torch.no_grad():x=a(self.batch);y=b(self.batch)
  torch.testing.assert_close(x['x_hat_raw_consistency'],y['x_hat_raw_consistency'],atol=2e-5,rtol=2e-5)
  torch.testing.assert_close(x['x_comp'],y['x_comp'],atol=2e-4,rtol=2e-5)
 def test_chunking_and_zero_message(self):
  torch.manual_seed(9);a=CrossMap(64,chunk=2);b=copy.deepcopy(a);b.chunk=100
  q=torch.randn(7,13,64);k=torch.randn(7,17,64);v=torch.randn_like(k)
  torch.testing.assert_close(a(q,k,v),b(q,k,v),atol=1e-6,rtol=1e-6)
  with torch.no_grad():
   for p in a.parameters():p.normal_()
  self.assertEqual(torch.count_nonzero(a(q,k,torch.zeros_like(v))),0)
 def test_hidden_pollution_empty_full_and_decoder_cache(self):
  for name in ('LT01','LT03','LT04','LT05','LT10'):
   m=self.model(self.cfg(name)).eval()
   for mode in ('partial','empty','full'):
    b=copy.deepcopy(self.batch)
    for s,f in [('f',1),('m',2),('c',4)]:
     if mode=='empty':b['m_'+s].zero_()
     if mode=='full':b['m_'+s].fill_(1);b['x_'+s+'_obs']=pool(b['x_f_gt'],f)*f*f
    with torch.no_grad():x=m(b)['x_comp']
    b['x_f_gt'].fill_(float('nan'))
    for s in 'fmc':b['x_'+s+'_obs'][~b['m_'+s].bool()]=float('nan')
    with torch.no_grad():y=m(b)['x_comp']
    torch.testing.assert_close(x,y,atol=0,rtol=0);self.assertTrue(torch.isfinite(x).all())
 def test_complete_cli_restore(self):
  from test_cli import CommandLine
  from unittest.mock import patch
  cfg=self.cfg('LT10');cfg['model']['coe']['latent_tokens']['count']=31
  with patch('test_cli.small_config',lambda:copy.deepcopy(cfg)):CommandLine().test_train_resume_evaluate_infer()
 def test_same_data_masks_and_mapping_initialization(self):
  for a,b in zip(build_datasets(self.base,self.root/'taxibj_train.npz',self.root/'taxibj_val.npz'),build_datasets(self.cfg(),self.root/'taxibj_train.npz',self.root/'taxibj_val.npz')):
   for i in range(len(a)):
    aa,bb=a[i],b[i]
    for k in aa:torch.testing.assert_close(aa[k],bb[k],atol=0,rtol=0)
  a=self.model(self.cfg('LT01'));b=self.model(self.cfg('LT04'));c=self.model(self.cfg('LT02'))
  self.assertTrue(torch.equal(a.main_branch.token_slots[:64],c.main_branch.token_slots))
  for prefix in ('grid_to_tokens','tokens_to_grid'):
   for index in range(3):self.assertEqual(state_hash(getattr(a.main_branch,prefix)[0].state_dict()),state_hash(getattr(b.main_branch,prefix)[index].state_dict()))
  with torch.no_grad():torch.testing.assert_close(a(self.batch)['x_comp'],b(self.batch)['x_comp'],atol=0,rtol=0)
if __name__=='__main__':unittest.main()
