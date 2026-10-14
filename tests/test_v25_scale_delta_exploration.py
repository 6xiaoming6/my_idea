import copy,json,shutil,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v25'),str(ROOT/'scripts/v24')]
from run_scale_delta_exploration import jobs,prepare_initialization
from stmoe_imputer.data.observed_multiscale import aggregate_sum
from stmoe_imputer.data import build_datasets
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import state_hash
from stmoe_imputer.routing_metrics import CoERoutingMetricAccumulator


class ExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setup_jobs(self,root):
        x=np.arange(4*2*3*8*8,dtype=np.float32).reshape(4,2,3,8,8)%51
        for split in ('train','val','test'):
            np.savez(root/f'taxibj_{split}.npz',x_f_gt=x,x_m_gt=aggregate_sum(x,2),x_c_gt=aggregate_sum(x,4))
        js=jobs(root)
        for j in js.values():
            c=j['config'];c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
            c['data'].update(batch_size=2,pin_memory=False);c['train'].update(epochs=2,val_epoch=1)
            c['train']['scheduler']['total_epochs']=2
        return js

    def model(self,j):
        torch.manual_seed(7);return DualBranchSTImputer.from_config(j['config'])

    def batch(self,j):
        ds,_=build_datasets(j['config'],j['sources']['train'],j['sources']['val'])
        return {k:torch.stack([ds[i][k] for i in (0,1)]) for k in ds[0]}

    def test_baseline_and_complete_delta_forward_gradient_compatibility(self):
        with tempfile.TemporaryDirectory() as td:
            js=self.setup_jobs(Path(td));j=js['taxibj_S04_seed7']
            for communication,steps in [('none',[]),('conditional',[2,3,4])]:
                new=copy.deepcopy(j);new['config']['model']['coe']['core_validation']['communication']=communication
                new['config']['model']['coe']['scale_delta_exploration']['delta_steps']=steps
                old=copy.deepcopy(new);old['config']['model']['coe'].pop('scale_delta_exploration')
                a=self.model(old);b=self.model(new)
                self.assertEqual(state_hash(a.state_dict()),state_hash(b.state_dict()))
                # Test a nonzero gate as well as the zero-init forward.
                for m in (a,b):
                    with torch.no_grad():m.main_branch.history_gate[-1].bias.fill_(.2)
                batch=self.batch(new);oa=a(batch);ob=b(batch)
                torch.testing.assert_close(oa['x_hat_main'],ob['x_hat_main'],rtol=0,atol=0)
                la,_=compute_coe_loss(oa,batch,old['config']);lb,_=compute_coe_loss(ob,batch,new['config'])
                la.backward();lb.backward()
                for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
                    self.assertEqual(na,nb)
                    self.assertEqual(pa.grad is None,pb.grad is None,na)
                    if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)

    def test_all_nine_sparse_sets_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as td:
            js=self.setup_jobs(Path(td));hashes=[];rngs=[]
            for j in js.values():
                m=self.model(j);hashes.append(state_hash(m.state_dict()));rngs.append(state_hash(torch.get_rng_state()))
                calls=[];hooks=[e.register_forward_pre_hook(lambda e,a:calls.append(len(a[0]))) for e in m.main_branch.routed_experts()]
                batch=self.batch(j)
                with torch.no_grad():m.main_branch.history_gate[-1].bias.fill_(.2)
                try:out=m(batch)
                finally:
                    for h in hooks:h.remove()
                spec=j['config']['model']['coe']['scale_delta_exploration'];ks=spec['top_k_per_step']
                self.assertEqual(sum(calls),len(batch['m_f'])*sum(ks))
                self.assertTrue(bool((out['coe']['triscale_executed'].sum(2)==1).all()))
                for step,k in enumerate(ks):
                    selected=out['coe']['selected_experts'][:,step,:k]
                    positive=out['coe']['route_weights'][:,step]>0
                    self.assertTrue(bool((positive.sum(1)==k).all()))
                    self.assertTrue(bool(positive.gather(1,selected).all()))
                    enabled=(step+1 in spec['delta_steps'])
                    self.assertEqual(float(out['coe']['diagnostics'][f'step{step+1}_input_delta'])>0,enabled)
                metrics=CoERoutingMetricAccumulator();metrics.update(out['coe']);values=metrics.compute()
                self.assertEqual(values['coe_expert_execution_count'],sum(ks))
                choices=out['coe']['triscale_choices'];area=sum(ks[i]*sum((1.,.25,.0625)[s] for s in choices[:,i].tolist())/len(choices) for i in range(4))
                self.assertAlmostEqual(values['coe_expert_grid_equivalents'],area)
                self.assertAlmostEqual(sum(values[f'coe_step1_{n}_selection_rate'] for n in m.main_branch.expert_names),ks[0])
                if j['method']=='S09':
                    self.assertEqual(float(out['coe']['diagnostics']['step4_delta_current_source_scale']),0)
                    self.assertEqual(float(out['coe']['diagnostics']['step4_delta_previous_source_scale']),0)
                loss,_=compute_coe_loss(out,batch,j['config']);loss.backward()
                self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None))
                self.assertFalse(hasattr(m.main_branch,'_observations'));self.assertFalse(hasattr(m.main_branch,'_exploration_rows'))
            self.assertEqual(len(set(hashes)),1);self.assertEqual(len(set(rngs)),1)
            prepare_initialization(Path(td)/'suite',js)

    def test_free_scale_and_top1_receive_task_gradients_without_balance(self):
        with tempfile.TemporaryDirectory() as td:
            js=self.setup_jobs(Path(td))
            for method in ('S01','S03'):
                j=js[f'taxibj_{method}_seed7'];m=self.model(j);batch=self.batch(j)
                c=copy.deepcopy(j['config']);c['loss']['lambda_coe_balance']=0.
                out=m(batch);loss,_=compute_coe_loss(out,batch,c);loss.backward()
                parameters=m.main_branch.scale_heads.parameters() if method=='S01' else m.main_branch.routers[0].parameters()
                grads=[p.grad for p in parameters if p.grad is not None]
                self.assertTrue(grads and any(bool(g.abs().sum()>1e-10) for g in grads),method)
                # Explicitly force each scale, verify raw corresponding observations matter.
                if method=='S01':
                    for sid in range(3):
                        with torch.no_grad():
                            for h in m.main_branch.scale_heads:h[-1].weight.zero_();h[-1].bias.zero_();h[-1].bias[sid]=10
                        calls=[];hooks=[e.register_forward_pre_hook(lambda e,a:calls.append(a[0].shape[-1])) for e in m.main_branch.routed_experts()]
                        try:out=m(batch)
                        finally:
                            for h in hooks:h.remove()
                        self.assertEqual(set(calls),{(8,4,2)[sid]})
                        self.assertTrue(bool((out['coe']['triscale_choices']==sid).all()))

    def test_hidden_values_and_labels_never_enter_any_variant(self):
        with tempfile.TemporaryDirectory() as td:
            for j in self.setup_jobs(Path(td)).values():
                m=self.model(j).eval();batch=self.batch(j)
                with torch.no_grad():
                    a=m(batch)['x_hat_main'];bad=copy.deepcopy(batch);bad['x_f_gt'].fill_(float('nan'))
                    for name in ('f','m','c'):bad['x_'+name+'_obs'][~bad['m_'+name].bool()]=float('nan')
                    torch.testing.assert_close(m(bad)['x_hat_main'],a,rtol=0,atol=0)

    def test_dynamic_and_variable_topk_resume_and_report(self):
        from train_four_direction import train
        from evaluate_four_direction import evaluate_sets
        from report_scale_delta_exploration import export_report
        from run_first_validation import protocol
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);js=self.setup_jobs(root);suite=root/'suite';suite.mkdir();prepare_initialization(suite,js)
            small={'rate':.4,'evaluations':{'in_distribution':{'families':['random_point'],'seed':20260917}}}
            for method in ('S01','S03','S09'):
                n=f'taxibj_{method}_seed7';j=js[n];run=root/n;receipt=suite/'results'/f'{n}.json'
                train(j,run,receipt,'cpu',stop_after=1);train(j,run,receipt,'cpu')
                train(j,root/(n+'_full'),root/(n+'_full.json'),'cpu')
                a=torch.load(run/'checkpoints/last.pth',weights_only=False);b=torch.load(root/(n+'_full')/'checkpoints/last.pth',weights_only=False)
                for k in ('model','optimizer','scheduler','scaler','rng_states'):self.assertEqual(state_hash(a[k]),state_hash(b[k]),method+' '+k)
                evaluate_sets(run/'checkpoints/best.pth',small,j['sources']['test'],suite/'evaluations'/f'{n}.json',device='cpu')
            p=export_report(root,suite,js);text=p.read_text()
            self.assertIn('阶段',p.name);self.assertIn('S09',text);self.assertLess(text.index('具体做法'),text.index('实验结果'))
            # Missing runs / incomplete evaluation must not be called a final report.
            self.assertFalse(json.loads((suite/'report_path.json').read_text())['complete'])
            # Exercise the real final-report gate, all nine jobs and both references.
            for n,j in js.items():
                train(j,root/n,suite/'results'/f'{n}.json','cpu')
                evaluate_sets(root/n/'checkpoints/best.pth',protocol(),j['sources']['test'],suite/'evaluations'/f'{n}.json',device='cpu')
            for method,source in [('B1','S08'),('B2','S04')]:
                n=f'taxibj_{source}_seed7';directory=suite/'references'/method;directory.mkdir(parents=True)
                for filename,path in [('receipt.json',suite/'results'/f'{n}.json'),('evaluation.json',suite/'evaluations'/f'{n}.json'),
                    ('config.json',root/n/'config.json'),('history.json',root/n/'logs/metrics.jsonl')]:shutil.copyfile(path,directory/filename)
            # Synthetic references here test report coverage, not scientific comparability.
            p=export_report(root,suite,js)
            self.assertIn('完整',p.name);self.assertTrue(json.loads((suite/'report_path.json').read_text())['complete'])
            corrupt=suite/'evaluations/taxibj_S09_seed7.json';saved=json.loads(corrupt.read_text());saved['best_epoch']+=1
            corrupt.write_text(json.dumps(saved));export_report(root,suite,js)
            self.assertFalse(json.loads((suite/'report_path.json').read_text())['complete'])


if __name__=='__main__':unittest.main()
