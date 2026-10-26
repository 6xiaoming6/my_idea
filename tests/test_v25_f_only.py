"""I29 strict F observations, retaining I25's latent scale routing."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'src'),str(ROOT/'scripts/v25')]
from run_taxibj_input_communication import jobs
import test_v25_observed_multiscale as helpers
from stmoe_imputer.data import build_datasets,build_test_dataset
from stmoe_imputer.losses import compute_coe_loss,supervision_mask

class FOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.helper=helpers.V25Tests();self.base=self.helper.setup_jobs(Path(self.tmp.name))['taxibj_B2_seed7']

    def job(self,method):
        j=copy.deepcopy(jobs(methods=[method])[f'taxibj_{method}_seed7'])
        j['sources']=self.base['sources'];j.pop('common_initialization',None)
        j['config']['model']['main']['dim']=8
        j['config']['model']['coe']['router_hidden_dim']=8
        j['config']['data'].update(batch_size=2,pin_memory=False)
        return j

    def test_original_unchanged_and_config_delta(self):
        frozen=json.loads((ROOT/'outputs/v25/experiments/taxibj_input_communication/2c07a1e3f7f177b7/plan.json').read_text())['jobs']
        self.assertEqual(jobs(methods=['I25']),frozen)
        a=copy.deepcopy(jobs(methods=['I25'])['taxibj_I25_seed7']['config'])
        b=copy.deepcopy(jobs(methods=['I29'])['taxibj_I29_seed7']['config'])
        a.pop('experiment_plan');b.pop('experiment_plan')
        a['data'].update(input_scales='F',return_multiscale_targets=False)
        a['model']['coe']['initial_observation_scales']='F'
        a['model']['coe'].pop('observation_consistency')
        for k in ['visible_consistency_aux','learnable_f_mix']:a['loss'].pop(k)
        self.assertEqual(a,b)
        self.assertEqual(len(jobs()),8)

    def test_data_mask_parity_and_coarse_values_cannot_enter(self):
        a,b=self.job('I29'),self.job('I25')
        for split in ['train','val','test']:
            from stmoe_imputer.data.observed_multiscale import build_observed_dataset
            da=build_observed_dataset(a['config'],a['sources'][split],split)
            db=build_observed_dataset(b['config'],b['sources'][split],split)
            for epoch in [1,2]:
                da.set_epoch(epoch);db.set_epoch(epoch)
                for i in range(len(da)):
                    x,y=da[i],db[i]
                    for k in ['m_f','x_f_obs','x_f_gt','target_mask']:
                        torch.testing.assert_close(x[k],y[k],rtol=0,atol=0)
                    for n in ['m','c']:
                        self.assertFalse(bool(x['m_'+n].any()))
                        self.assertFalse(bool(x['x_'+n+'_obs'].any()))
                        self.assertNotIn('x_'+n+'_gt',x)
            before=da[0]
            for arr in da.values[1:]:arr.fill(float('nan'))
            after=da[0]
            for k in before:torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)

    def test_sparse_forward_loss_gradients_and_parent_parity(self):
        j=self.job('I29');model=self.helper.model(j).train();parent=self.helper.model(self.job('I25')).eval()
        for k,v in model.state_dict().items():torch.testing.assert_close(v,parent.state_dict()[k],rtol=0,atol=0)
        batch=self.helper.batch(j);calls=[]
        hooks=[e.register_forward_pre_hook(lambda e,args:calls.append(len(args[0]))) for e in model.main_branch.routed_experts()]
        try:out=model(batch)
        finally:
            for h in hooks:h.remove()
        self.assertEqual(sum(calls),len(batch['m_f'])*8)
        self.assertTrue(bool((out['coe']['triscale_choices'][:,-1]==0).all()))
        self.assertEqual([float(out['coe']['diagnostics'][f'x_step{i}_pool_size']) for i in range(1,5)],[0,1,2,3])
        self.assertFalse(hasattr(model.main_branch,'x_consistency'))
        self.assertFalse(hasattr(model.main_branch,'x_loss_mix_logit'))
        loss,logs=compute_coe_loss(out,batch,j['config'])
        mask=supervision_mask(batch['x_f_gt'],batch['m_f'],batch['target_mask'])
        main=(out['x_hat_main'][mask]-batch['x_f_gt'][mask]).abs().mean()
        torch.testing.assert_close(loss,main+logs['l_coe_balance_weighted'])
        loss.backward()
        for name in ['history_gate','x_history','scale_heads']:
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in getattr(model.main_branch,name).parameters()),name)
        model.eval()
        with torch.no_grad():
            expected=model(batch)['x_hat_main']
            torch.testing.assert_close(expected,parent(batch)['x_hat_raw_consistency'],rtol=0,atol=0)
            poison=copy.deepcopy(batch)
            for n in ['m','c']:
                poison['x_'+n+'_obs'].fill_(float('nan'));poison['x_'+n+'_gt']=torch.full_like(poison['x_'+n+'_obs'],float('nan'))
            actual=model(poison)
            torch.testing.assert_close(expected,actual['x_hat_main'],rtol=0,atol=0)
            torch.testing.assert_close(compute_coe_loss(model(batch),batch,j['config'])[0],compute_coe_loss(actual,poison,j['config'])[0],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
