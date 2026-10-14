"""I09 wiring, sparse execution, causal history, projection and strict resume."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'src'),str(ROOT/'scripts/v25'),str(ROOT/'scripts/v24')]
import test_v25_observed_multiscale as fixtures
from run_taxibj_input_communication import jobs
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import state_hash
from preflight_taxibj_input_communication import check_consistency

class FullCombinationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.helper=fixtures.V25Tests();self.base=self.helper.setup_jobs(self.root)['taxibj_B2_seed7'];self.batch=self.helper.batch(self.base)
    def job(self,method):
        j=copy.deepcopy(self.base);full=jobs(methods=[method])[f'taxibj_{method}_seed7'];j.update(method=method,variant=full['variant'])
        for key in ('scale_delta_exploration','observed_history','observation_consistency'):
            if key in full['config']['model']['coe']:j['config']['model']['coe'][key]=copy.deepcopy(full['config']['model']['coe'][key])
        return j
    def test_configuration_and_initial_forward_match_i08(self):
        current=jobs(methods=['I08','I09']);a,b=[current[f'taxibj_{m}_seed7']['config'] for m in ('I08','I09')]
        for k in ('data','train','loss','seed'):self.assertEqual(a[k],b[k])
        self.assertEqual(a['experiment_plan']['protocol'],b['experiment_plan']['protocol'])
        models=[self.helper.model(self.job(m)).eval() for m in ('I08','I09')]
        self.assertEqual(state_hash(models[0].state_dict()),state_hash(models[1].state_dict()))
        with torch.no_grad():torch.testing.assert_close(models[0](self.batch)['x_hat_main'],models[1](self.batch)['x_hat_main'],rtol=0,atol=0)
        self.assertTrue(all(p.requires_grad for p in models[1].main_branch.scale_heads.parameters()))
    def test_gradients_causality_projection_and_hidden_values(self):
        j=self.job('I09');m=self.helper.model(j).train();calls=[];pools=[]
        hooks=[x.register_forward_pre_hook(lambda mod,args:calls.append(len(args[0]))) for x in m.main_branch.routed_experts()]
        hook=m.main_branch.x_history.register_forward_pre_hook(lambda mod,args:pools.append([(x.requires_grad,x.grad_fn) for x in args[1]]))
        try:out=m(self.batch)
        finally:
            for h in hooks+[hook]:h.remove()
        self.assertEqual(sum(calls),8*len(self.batch['m_f']))
        self.assertEqual([len(p) for p in pools],[1,2,3]);self.assertTrue(all(not req and fn is None for p in pools for req,fn in p))
        self.assertEqual([float(out['coe']['diagnostics'][f'x_step{i}_pool_size']) for i in range(1,5)],[0,1,2,3])
        check_consistency(out,self.batch,j['config']);loss,_=compute_coe_loss(out,self.batch,j['config']);loss.backward()
        for module in (m.main_branch.scale_heads,m.main_branch.history_gate,m.main_branch.x_history,m.main_branch.x_consistency):
            grads=[p.grad for p in module.parameters() if p.grad is not None];self.assertTrue(grads);self.assertTrue(all(torch.isfinite(g).all() for g in grads));self.assertTrue(any(g.abs().sum()>0 for g in grads))
        m.eval();poison=copy.deepcopy(self.batch)
        for scale in ('f','m','c'):
            poison['x_'+scale+'_gt']=torch.full_like(poison['x_'+scale+'_obs'],float('nan'))
            poison['x_'+scale+'_obs'][~poison['m_'+scale].bool()]=float('nan')
        with torch.no_grad():torch.testing.assert_close(m(poison)['x_hat_main'],m(self.batch)['x_hat_main'],rtol=0,atol=0)
        # Check a genuinely different scale path, with the same history/projection.
        with torch.no_grad():
            for head in m.main_branch.scale_heads:
                head[-1].weight.zero_();head[-1].bias.zero_();head[-1].bias[0]=20
            out=m(self.batch)
        check_consistency(out,self.batch,j['config'])
        self.assertTrue(all(float(out['coe']['diagnostics'][f'x_step{i}_pool_size'])==i-1 for i in range(1,5)))
        self.assertEqual(float(out['coe']['diagnostics']['step2_delta_current_source_scale']),0.)
    def test_strict_resume(self):
        from train_four_direction import train
        j=self.job('I09');j['config']['train'].update(epochs=2,val_epoch=1);j['config']['train']['scheduler']['total_epochs']=2
        run=self.root/'resumed';train(j,run,self.root/'r.json','cpu',stop_after=1);train(j,run,self.root/'r.json','cpu');train(j,self.root/'full',self.root/'f.json','cpu')
        a=torch.load(run/'checkpoints/last.pth',weights_only=False);b=torch.load(self.root/'full/checkpoints/last.pth',weights_only=False)
        for k in ('model','optimizer','scheduler','scaler','rng_states'):self.assertEqual(state_hash(a[k]),state_hash(b[k]),k)
        from evaluate_four_direction import evaluate_sets
        from run_b3_c3 import digest
        ev=self.root/'evaluation.json'
        evaluate_sets(run/'checkpoints/best.pth',j['config']['experiment_plan']['protocol'],j['sources']['test'],ev,digest(j['config']),'cpu')
        record=json.loads(ev.read_text());self.assertEqual(len(record['sets']),6)
        for row in record['sets'].values():
            self.assertEqual(row['metrics']['coe_expert_execution_count'],8.)
            self.assertLess(row['metrics']['coe_v25_observed_sum_residual_C'],2e-3)

if __name__=='__main__':unittest.main()
