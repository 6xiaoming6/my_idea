import copy,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24')]
from run_coordination_exploration import jobs,rows,report
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.coordination import blocks,unblocks,coordinate,auxiliary_loss,CoordinationMetrics

class CoordinationTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1);cls.jobs=jobs()
 def model(self,n):
  c=copy.deepcopy(self.jobs[n]['config']);c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
  torch.manual_seed(7);return DualBranchSTImputer.from_config(c),c
 def batch(self):
  g=torch.Generator().manual_seed(123);x=torch.randn(2,2,3,8,8,generator=g);mask=(torch.rand(x.shape,generator=g)>.4).float()
  return {'x_f_obs':x*mask,'m_f':mask,'x_f_gt':x,'mask_family':torch.tensor([0,1])}
 def test_initialization_sparse_execution_and_gradients(self):
  b=self.batch();base,c=self.model('V1');rng=torch.get_rng_state();ref=base(b)
  old=copy.deepcopy(c);old['experiment_plan']={};torch.manual_seed(7);origin=DualBranchSTImputer.from_config(old)
  origin.load_state_dict(base.state_dict(),strict=True);torch.testing.assert_close(origin(b)['x_hat_main'],ref['x_hat_main'],rtol=0,atol=0)
  outputs={}
  for n in ('V2','V3','V4'):
   m,c=self.model(n);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
   for k,v in base.named_parameters():torch.testing.assert_close(v,dict(m.named_parameters())[k],rtol=0,atol=0)
   calls=[];hooks=[e.register_forward_pre_hook(lambda e,a:calls.append(len(a[0]))) for e in m.main_branch.routed_experts()]
   out=m(b);outputs[n]=out['x_hat_main'].detach()
   for h in hooks:h.remove()
   self.assertEqual(sum(calls),16);self.assertFalse(hasattr(m.main_branch,'_coordination_output'))
   self.assertEqual(out['coordination']['means']['c'].shape[-2:],(2,2));self.assertEqual(out['coordination']['means']['m'].shape[-2:],(4,4))
   torch.testing.assert_close(out['x_comp'][b['m_f'].bool()],b['x_f_obs'][b['m_f'].bool()],rtol=0,atol=0)
   loss,logs=compute_coe_loss(out,b,c);loss.backward();self.assertTrue(torch.isfinite(loss));self.assertGreater(float(logs['l_coord_weighted']),0)
   for head in m.main_branch.region_heads.values():self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters()))
   if n=='V4':self.assertGreater(float(m.main_branch.allocation_head[-1].weight.grad.abs().sum()),0)
   saved=copy.deepcopy(m.state_dict());other,_=self.model(n);other.load_state_dict(saved,strict=True)
   m.eval();other.eval();torch.testing.assert_close(m(b)['x_hat_main'],other(b)['x_hat_main'],rtol=0,atol=0)
   changed=dict(b,x_f_gt=b['x_f_gt']*100);torch.testing.assert_close(m(b)['x_hat_main'],m(changed)['x_hat_main'],rtol=0,atol=0)
  torch.testing.assert_close(outputs['V2'],ref['x_hat_main'],rtol=0,atol=0)
  torch.testing.assert_close(outputs['V3'],outputs['V4'],rtol=0,atol=0)
 def test_projection_weights_no_observations_and_empty(self):
  b=self.batch();p=b['x_f_gt'];obs=b['m_f'];f=4;means=torch.randn(2,2,3,2,2)
  for scores in (None,torch.randn_like(p)):
   out,r=coordinate(p,obs,means,f,.1,scores);q=blocks(~obs.bool(),f);n=q.sum(-1)
   before=torch.where(q,blocks(p,f),0.).sum(-1);after=torch.where(q,blocks(out,f),0.).sum(-1)
   torch.testing.assert_close(after,before+.1*(n*means-before),rtol=1e-5,atol=1e-5)
   torch.testing.assert_close(r['weights'].sum(-1),(n>0).float())
   torch.testing.assert_close(out[obs.bool()],p[obs.bool()],rtol=0,atol=0)
  for mask in (torch.zeros_like(obs),torch.ones_like(obs)):
   out,r=coordinate(p,mask,means,f,scores=torch.zeros_like(p));self.assertTrue(torch.isfinite(out).all());self.assertTrue(torch.isfinite(r['entropy']).all())
  torch.testing.assert_close(unblocks(blocks(p,4),4),p,rtol=0,atol=0)
 def test_natural_missing_is_not_partial_region_label(self):
  b=self.batch();m,c=self.model('V2');out=m(b);q=~b['m_f'].bool();b['target_mask']=torch.ones_like(q);b['target_mask'][q]=False
  b['x_f_gt'][q]=torch.nan
  loss,logs=auxiliary_loss(out,b);self.assertEqual(float(loss),0);self.assertTrue(torch.isfinite(loss));loss.backward()
  metrics=CoordinationMetrics();metrics.update(out,b);self.assertTrue(all(np.isfinite(v) for v in metrics.compute().values()))
  # One invalid missing target suppresses exactly its C and M bins.
  b=self.batch();b['m_f'].zero_();b['x_f_obs'].zero_();o=m(b);_,a=auxiliary_loss(o,b);b['x_f_gt'][0,0,0,0,0]=torch.nan;_,d=auxiliary_loss(o,b)
  for label in ('c','m'):self.assertEqual(int(a['coord_'+label+'_supervised_regions']-d['coord_'+label+'_supervised_regions']),1)
 def test_metrics_exact_split_merge_and_batch_isolation(self):
  b=self.batch();m,c=self.model('V4');m.eval();o=m(b);whole=CoordinationMetrics();whole.update(o,b)
  total=CoordinationMetrics()
  for i in range(2):
   part={k:v[i:i+1] for k,v in b.items()};a=CoordinationMetrics();a.update(m(part),part);total.merge(a)
  for k,v in whole.compute().items():self.assertAlmostEqual(v,total.compute()[k],places=5)
  m(dict(b,x_f_obs=b['x_f_obs']*3));torch.testing.assert_close(o['x_hat_main'],m(b)['x_hat_main'],rtol=0,atol=0)
 def test_tiny_training_resume_evaluation_skip_and_report(self):
  from train_four_direction import train
  from evaluate_four_direction import evaluate_sets
  from report_coordination_exploration import export_report
  from unittest.mock import patch
  import run_four_direction_exploration as base
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);j=copy.deepcopy(self.jobs['V4']);_,c=self.model('V4');j['config']=c
   c['train']['epochs']=2;c['train']['val_epoch']=1;c['train']['scheduler']['total_epochs']=2;c['data']['batch_size']=2;c['data']['pin_memory']=False;j['sources']={}
   for sp in ('train','val','test'):
    p=root/(sp+'.npz');np.savez(p,x_f_gt=np.random.RandomState(2).randn(4,2,3,8,8).astype('float32'));j['sources'][sp]=str(p)
   run=root/'run';suite=root/'suite';result=suite/'results/V4.json'
   train(j,run,result,'cpu',stop_after=1);r=train(j,run,result,'cpu');self.assertEqual(train(j,run,result,'cpu'),r)
   with patch.object(base,'subprocess_run',side_effect=AssertionError('must skip')):base.launch(suite,j,0)
   train(j,root/'full',root/'full_result.json','cpu')
   a=torch.load(run/'checkpoints/last.pth',weights_only=False);b=torch.load(root/'full/checkpoints/last.pth',weights_only=False)
   for k,v in a['model'].items():torch.testing.assert_close(v,b['model'][k],rtol=0,atol=0)
   protocol={'rate':.4,'evaluations':{'in_distribution':{'families':['random_point'],'seed':20260917}}}
   ev=evaluate_sets(run/'checkpoints/best.pth',protocol,j['sources']['test'],suite/'evaluations/V4.json',device='cpu')
   self.assertIn('coord_raw_mae',ev['sets']['in_distribution']['metrics'])
   js=copy.deepcopy(self.jobs);js['V4']=j;rs=rows(suite,js);report=export_report(root,suite,rs,js);text=report.read_text();self.assertIn('尚未完成',text);self.assertLess(text.index('具体做法'),text.index('结果、训练曲线'))
if __name__=='__main__':unittest.main()
