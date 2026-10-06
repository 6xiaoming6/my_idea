import copy,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24')]
from run_scale_memory_followup import jobs,ORDER,collect
from run_four_direction_exploration import static_jobs
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss

class FollowupTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def test_jobs_initial_outputs_sparse_execution_and_gradient(self):
  js=jobs();self.assertEqual(tuple(js),ORDER)
  g=torch.Generator().manual_seed(19);x=torch.randn(2,2,3,8,8,generator=g);mask=(torch.rand(x.shape,generator=g)>.4).float();b={'x_f_obs':x*mask,'m_f':mask,'x_f_gt':x}
  for name,j in js.items():
   c=copy.deepcopy(j['config']);c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
   self.assertEqual(c['seed'],c['data']['loader_seed']);self.assertEqual(c['train']['epochs'],100)
   torch.manual_seed(c['seed']);m=DualBranchSTImputer.from_config(c)
   ref=copy.deepcopy(c);ref['model']['coe']['four_direction']['memory']='none';torch.manual_seed(c['seed']);z=DualBranchSTImputer.from_config(ref)
   calls=[];hooks=[e.register_forward_pre_hook(lambda e,a:calls.append(len(a[0]))) for e in m.main_branch.routed_experts()]
   out=m(b)
   for h in hooks:h.remove()
   self.assertEqual(sum(calls),16)
   torch.testing.assert_close(out['x_hat_main'],z(b)['x_hat_main'],rtol=0,atol=0)
   path=c['model']['coe']['four_direction']['fixed_path'];self.assertEqual(out['coe']['triscale_choices'].tolist(),[path]*2)
   loss,_=compute_coe_loss(out,b,c);loss.backward();self.assertTrue(torch.isfinite(loss))
   if c['model']['coe']['four_direction']['memory']=='delta':
    grads=[p.grad for p in m.main_branch.history_gate.parameters() if p.grad is not None]
    self.assertTrue(all(torch.isfinite(v).all() for v in grads));self.assertTrue(any(v.abs().sum()>0 for v in grads))
   self.assertTrue(all(not p.requires_grad for p in m.main_branch.scale_heads.parameters()))
 def test_fixed_baseline_equivalence(self):
  c=copy.deepcopy(jobs()['D1']['config']);c['model']['coe']['four_direction']['memory']='none'
  old=static_jobs()['S01']['config']
  for cfg in (c,old):cfg['model']['main']['dim']=8;cfg['model']['coe']['router_hidden_dim']=8
  torch.manual_seed(7);a=DualBranchSTImputer.from_config(c);torch.manual_seed(7);b=DualBranchSTImputer.from_config(old)
  for key,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[key],rtol=0,atol=0)
 def test_training_resume_evaluation_skip_report(self):
  from train_four_direction import train
  from evaluate_four_direction import evaluate_sets
  from run_b3_c3 import write
  from report_scale_memory_followup import export_report
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);j=copy.deepcopy(jobs()['D3']);c=j['config'];c['train']['epochs']=2;c['train']['scheduler']['total_epochs']=2;c['train']['val_epoch']=1;c['data']['batch_size']=2;c['data']['pin_memory']=False;c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
   j['sources']={}
   for split in ('train','val','test'):
    p=root/(split+'.npz');np.savez(p,x_f_gt=np.random.RandomState(5).randn(4,2,3,8,8).astype('float32'));j['sources'][split]=str(p)
   suite=root/'suite';result=suite/'results/D3.json';run=root/'run'
   train(j,run,result,'cpu',stop_after=1);r=train(j,run,result,'cpu');self.assertEqual(train(j,run,result,'cpu'),r)
   protocol={'rate':.4,'split':'test','evaluations':{'in_distribution':{'families':['random_point'],'seed':20260917},'unseen_triple':{'families':['node_plus_time_plus_space'],'seed':20261027}}}
   for folder in ('evaluations','confirmation','rate_transfer'):evaluate_sets(run/'checkpoints/best.pth',protocol,j['sources']['test'],suite/folder/'D3.json',device='cpu')
   rows=collect(suite,{'D3':j},['D3']);self.assertEqual(rows[0]['status'],'finished')
   report=export_report(root,suite,rows,{'D3':j});t=report.read_text();self.assertLess(t.index('具体做法'),t.index('结果、训练曲线'));self.assertIn('未齐',t)
if __name__=='__main__':unittest.main()
