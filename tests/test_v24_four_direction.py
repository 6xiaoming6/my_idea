import copy,itertools,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v24')]
from run_four_direction_exploration import static_jobs,choose,resolve,ORDER,STATIC,receipt
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.routing_metrics import CoERoutingMetricAccumulator


def cfg(name):
 c=copy.deepcopy(static_jobs()[name]['config']);c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
 return c

def model(name):
 torch.manual_seed(7);return DualBranchSTImputer.from_config(cfg(name))

def batch(n=2):
 g=torch.Generator().manual_seed(91);x=torch.randn(n,2,3,8,8,generator=g);mask=(torch.rand(x.shape,generator=g)>.4).float()
 return {'x_f_obs':x*mask,'m_f':mask,'x_f_gt':x}


class FourTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)

 def test_all_static_forward_backward(self):
  for n in STATIC:
   m=model(n);b=batch();o=m(b);loss,_=compute_coe_loss(o,b,cfg(n));loss.backward();self.assertTrue(torch.isfinite(loss),n)
   self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None),n)

 def test_native_compatibility_and_initial_pairs(self):
  b=batch();c=cfg('M01');del c['model']['coe']['four_direction'];torch.manual_seed(7);old=DualBranchSTImputer.from_config(c)
  new=model('M01');a=old(b);z=new(b)
  torch.testing.assert_close(a['x_hat_main'],z['x_hat_main'],rtol=0,atol=0)
  for field in ('route_logits','route_weights','route_importance'):
   torch.testing.assert_close(a['coe'][field],z['coe'][field],rtol=0,atol=0)
  for n in ['M04','M07','M08','M09','M10','M11','M12','P02','P03','P04','P05','P06','P07','P08']:
   out=model(n)(b)
   torch.testing.assert_close(out['x_hat_main'],z['x_hat_main'],rtol=0,atol=0,msg=n)
   torch.testing.assert_close(out['coe']['route_importance'],z['coe']['route_importance'],rtol=0,atol=0)
  torch.testing.assert_close(model('M05')(b)['x_hat_main'],model('M06')(b)['x_hat_main'],rtol=1e-6,atol=1e-6)
  for n in ['P02','P03','P05']:
   m=model(n)
   for h in m.main_branch.routers:h[-1].weight.data.zero_();h[-1].bias.data.zero_()
   out=m(b);native=out['coe']['route_logits'].topk(2,-1).indices.sort(-1).values
   torch.testing.assert_close(out['coe']['selected_experts'],native)

 def test_scale_initial_outputs_and_81_sparse_paths(self):
  b=batch();fixed=model('S01')(b)
  for n in ['S02','S04','S05','S06']:
   out=model(n)(b);torch.testing.assert_close(out['x_hat_main'],fixed['x_hat_main'],rtol=0,atol=0,msg=n)
  for a,z in [('S07','S08'),('S09','S10')]:torch.testing.assert_close(model(a)(b)['x_hat_main'],model(z)(b)['x_hat_main'],rtol=0,atol=0)
  m=model('S02');back=m.main_branch
  for h in back.routers:h[-1].weight.data.zero_();h[-1].bias.data.copy_(torch.arange(8,0,-1))
  for path in itertools.product(range(3),repeat=4):
   calls=[];hooks=[e.register_forward_pre_hook(lambda e,a,i=i:calls.append((i,len(a[0]),a[0].shape[-1]))) for i,e in enumerate(back.routed_experts())]
   out=back(b['x_f_obs'],b['m_f'],forced_scales=torch.tensor([path]*2))
   for h in hooks:h.remove()
   self.assertEqual(sum(v[1] for v in calls),16);self.assertEqual(set(v[0] for v in calls),{0,1})
   for sid,size in enumerate((8,4,2)):self.assertEqual(sum(n for _,n,w in calls if w==size),path.count(sid)*4)
  for n,count in [('S07',16),('S08',16),('S09',24),('S10',24)]:
   m=model(n);calls=[];hooks=[e.register_forward_pre_hook(lambda e,a:calls.append(len(a[0]))) for e in m.main_branch.routed_experts()]
   out=m(b)
   for h in hooks:h.remove()
   self.assertEqual(sum(calls),count*2)
   acc=CoERoutingMetricAccumulator();acc.update(out['coe']);r=acc.compute();self.assertEqual(r['coe_expert_execution_count'],count)
   torch.testing.assert_close(out['coe']['triscale_weights'].sum(2),torch.ones(2,4))

 def test_memory_isolation_and_gradients(self):
  b=batch()
  for n in ['M04','M05','M06','M07','M08','M09','M10','M11','M12','S05','S06']:
   m=model(n);first=m(b)['x_hat_main'];m(batch(3));second=m(b)['x_hat_main'];torch.testing.assert_close(first,second,rtol=0,atol=0,msg=n)
   first.square().mean().backward()
   extra=[p.grad for k,p in m.named_parameters() if any(x in k for x in ('history_','message_','gru_','bank_','scale_history')) and p.grad is not None]
   self.assertTrue(any(g.abs().sum()>0 for g in extra),n);self.assertTrue(all(torch.isfinite(g).all() for g in extra))
  a=model('M07').main_branch;z=model('M08').main_branch;z.load_state_dict(a.state_dict())
  a.history_gate[-1].bias.data.fill_(.3);z.history_gate[-1].bias.data.fill_(.3)
  def grad(m):
   c=m._initialize(b['x_f_obs'],b['m_f']);prev=c['h'].detach().clone().requires_grad_();c['h']=c['h'].detach()+.7;c['prev']=prev;c['has_prev']=True
   # Gate inputs also carry history; freeze gate as a constant for this causal test.
   for p in m.history_gate.parameters():p.requires_grad_(False)
   y=m._memory(c)[0];return torch.autograd.grad(y.sum(),prev,allow_unused=True)[0] if y.requires_grad else None
  self.assertGreater(grad(a).abs().sum(),0);gz=grad(z);self.assertTrue(gz is None or gz.abs().sum()==0)

 def test_probe_head_only_and_context_rng_isolation(self):
  for n in ['S04','P03','P04','P05','P06','P08']:
   m=model(n);b=batch();back=m.main_branch;back.prepare_training_batch(b);out=m(b)
   state=torch.get_rng_state().clone();route_rng=back.route_rng.clone();expected=m(b)['x_hat_main'].detach()
   loss,logs=back.candidate_loss(b,out);loss.backward()
   self.assertGreater(logs['four_probe_head_grad_norm'],0)
   for key,p in m.named_parameters():
    if p.grad is not None and p.grad.abs().sum()>0:self.assertTrue('pair_heads' in key or 'scale_heads' in key,key)
   torch.testing.assert_close(route_rng,back.route_rng);torch.testing.assert_close(state,torch.get_rng_state())
   torch.testing.assert_close(expected,m(b)['x_hat_main'],rtol=0,atol=0)
   changed={**b,'x_f_gt':b['x_f_gt']+10000,'x_f_obs':torch.where(b['m_f'].bool(),b['x_f_obs'],float('nan'))}
   torch.testing.assert_close(expected,m(changed)['x_hat_main'],rtol=0,atol=0)
   self.assertEqual(logs['four_probe_candidates'],6 if n=='S04' else 10)

 def test_exploration_resume_and_windows(self):
  b=batch(64)
  for n,active_epoch,inactive in [('S03',1,21),('G07',1,9),('G08',100,101),('G09',93,92)]:
   m=model(n);back=m.main_branch;back.set_routing_epoch(active_epoch);global_state=torch.get_rng_state().clone()
   out=m(b);torch.testing.assert_close(global_state,torch.get_rng_state())
   if n.startswith('G'):self.assertEqual(len(set(map(tuple,out['coe']['triscale_choices'].tolist()))),12)
   state=copy.deepcopy(m.state_dict());expected=m(b)['coe']['triscale_choices'];r=model(n);r.load_state_dict(state);r.main_branch.set_routing_epoch(active_epoch)
   torch.testing.assert_close(expected,r(b)['coe']['triscale_choices'])
   m.eval();before=back.route_rng.clone();m(b);torch.testing.assert_close(before,back.route_rng)
   m.train();back.set_routing_epoch(inactive);m(b);torch.testing.assert_close(before,back.route_rng)

 def test_selection_and_dynamic_counts(self):
  jobs=static_jobs();self.assertEqual(len(jobs),39);self.assertEqual(len(ORDER),60)
  evidence={n:{'split':'val','val_mae':10.,'ood':{k:10. for k in ['unseen_combinations','unseen_geometry','unseen_triple']},'area':8.,'params':100} for n in STATIC}
  result=choose(evidence);self.assertEqual(result['memory'],['M03','M04'])
  for n in ORDER:
   j=jobs[n] if n in jobs else resolve(n,jobs,result)
   self.assertEqual(j['variant'],n);self.assertEqual(j['config']['train']['epochs'],100)
   DualBranchSTImputer.from_config(j['config'])
  evidence['M01']['split']='test'
  with self.assertRaises(ValueError):choose(evidence)

 def test_training_resume_exact_and_eval_report(self):
  from train_four_direction import train
  from evaluate_four_direction import evaluate_sets
  from run_b3_c3 import load
  from report_four_direction import export_report
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);job=copy.deepcopy(static_jobs(epochs=2,batch_size=2)['S03']);c=job['config'];c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8;c['train']['val_epoch']=1;c['data']['pin_memory']=False
   paths={}
   for split in ['train','val','test']:
    path=root/(split+'.npz');np.savez(path,x_f_gt=np.random.RandomState(1).randn(4,2,3,8,8).astype('float32'));paths[split]=str(path)
   job['sources']=paths
   a=train(job,root/'full',root/'full.json','cpu')
   train(job,root/'resume',root/'resume.json','cpu',stop_after=1);z=train(job,root/'resume',root/'resume.json','cpu')
   aa=torch.load(root/'full/checkpoints/last.pth',weights_only=False);zz=torch.load(root/'resume/checkpoints/last.pth',weights_only=False)
   for k,v in aa['model'].items():torch.testing.assert_close(v,zz['model'][k],rtol=0,atol=0,msg=k)
   self.assertEqual(a['test']['mae'],z['test']['mae']);self.assertEqual(train(job,root/'resume',root/'resume.json','cpu'),z)
   protocol={'rate':.4,'split':'val','evaluations':{'unseen_triple':{'families':['node_plus_time_plus_space'],'seed':20261013}}}
   e=evaluate_sets(root/'resume/checkpoints/best.pth',protocol,paths['val'],root/'eval.json',device='cpu')
   self.assertEqual(e['sets']['unseen_triple']['effective_mask_seed'],20281013);self.assertEqual(e['split'],'val')
   suite=root/'suite';(suite/'results').mkdir(parents=True);(suite/'results/S03.json').write_text(json.dumps(z));self.assertEqual(receipt(suite,job)['status'],'finished')
   r={'variant':'S03','status':'trained','method':'sample','reference':'S01','seed':7,'run_dir':z['run_dir'],'val_mae':z['best_val_mae'],'best_epoch':z['best_epoch'],'training_seconds':z['total_time_sec'],'evaluations':{}}
   report=export_report(root,suite,[r],{'S03':job},'smoke');self.assertTrue(report.exists());self.assertLess(report.read_text().index('具体做法'),report.read_text().index('实验结果'))

if __name__=='__main__':unittest.main()
