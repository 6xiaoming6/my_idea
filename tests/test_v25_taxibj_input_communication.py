import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'src'),str(ROOT/'scripts/v25'),str(ROOT/'scripts/v24')]
from test_v25_observed_multiscale import V25Tests
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import state_hash


class InputCommunicationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.helper=V25Tests();self.jobs=self.helper.setup_jobs(Path(self.temp.name))
        self.base=self.jobs['taxibj_B2_seed7']
        self.batch=self.helper.batch(self.base)

    def capture(self, model, batch):
        states=[];native=model.main_branch._round
        def record(c,step,*args,**kwargs):
            if step==0:states.append(c['h'].detach().clone())
            new,row=native(c,step,*args,**kwargs);states.append(new['h'].detach().clone());return new,row
        with patch.object(model.main_branch,'_round',side_effect=record),torch.no_grad():out=model(batch)
        return states,out

    def test_initial_f_only_keeps_later_native_observations_and_blocks_leakage(self):
        j=copy.deepcopy(self.base);j['config']['model']['coe']['initial_observation_scales']='F'
        m=self.helper.model(j).eval();b=self.helper.model(self.base).eval()
        self.assertEqual(state_hash(m.state_dict()),state_hash(b.state_dict()))
        states,out=self.capture(m,self.batch)
        for name,first in [('c',1),('m',2)]:
            changed=copy.deepcopy(self.batch);changed['x_'+name+'_obs']+=10*changed['m_'+name]
            altered,_=self.capture(m,changed)
            for i in range(first):torch.testing.assert_close(states[i],altered[i],rtol=0,atol=0)
            self.assertGreater(float((states[first]-altered[first]).abs().max()),1e-7)
        # F-only H0 equals old A2 at initialization, while later M/C are retained.
        a2=copy.deepcopy(self.batch)
        for name in ['m','c']:a2['m_'+name].zero_();a2['x_'+name+'_obs'].zero_()
        a2states,_=self.capture(b,a2);torch.testing.assert_close(states[0],a2states[0],rtol=0,atol=0)
        poison=copy.deepcopy(self.batch)
        for name in ['f','m','c']:poison['x_'+name+'_obs'][~poison['m_'+name].bool()]=float('nan')
        poison['x_f_gt'].fill_(99999)
        _,poisonout=self.capture(m,poison);torch.testing.assert_close(out['x_hat_main'],poisonout['x_hat_main'],rtol=0,atol=0)
        m.train();batch=copy.deepcopy(self.batch)
        for name in ['m','c']:batch['x_'+name+'_obs'].requires_grad_()
        loss,_=compute_coe_loss(m(batch),batch,j['config']);loss.backward()
        d=m.main_branch.dim;grad=m.main_branch.observation_fusion.weight.grad
        self.assertTrue(bool(grad[:,:d].abs().sum()>0));self.assertFalse(bool(grad[:,d:].any()))
        for name in ['m','c']:
            g=batch['x_'+name+'_obs'].grad;self.assertTrue(bool(g[batch['m_'+name].bool()].abs().sum()>0));self.assertFalse(bool(g[~batch['m_'+name].bool()].any()))

    def test_default_fmc_is_unchanged_and_history_starts_empty_detached(self):
        baseline=self.helper.model(self.base).eval();explicit=copy.deepcopy(self.base)
        explicit['config']['model']['coe']['initial_observation_scales']='FMC'
        with torch.no_grad():torch.testing.assert_close(baseline(self.batch)['x_hat_main'],self.helper.model(explicit).eval()(self.batch)['x_hat_main'],rtol=0,atol=0)
        j=copy.deepcopy(self.base);j['config']['model']['coe']['observed_history']={'enabled':True,'mode':'adaptive'}
        m=self.helper.model(j).eval();common={k:m.state_dict()[k] for k in baseline.state_dict()}
        self.assertEqual(state_hash(common),state_hash(baseline.state_dict()))
        original=m.main_branch._memory;seen=[]
        def check(c):
            seen.append(len(c['x_pool']))
            for v in c['x_pool']:
                self.assertFalse(v.requires_grad)
                torch.testing.assert_close(v[c['mask'].bool()],c['x'][c['mask'].bool()],rtol=0,atol=0)
            return original(c)
        with patch.object(m.main_branch,'_memory',side_effect=check),torch.no_grad():out=m(self.batch)
        self.assertEqual(seen,[0,1,2,3])
        with torch.no_grad():torch.testing.assert_close(out['x_hat_main'],baseline(self.batch)['x_hat_main'],rtol=0,atol=0)
        self.assertFalse(hasattr(m.main_branch,'_x_stats'))
        with torch.no_grad():torch.testing.assert_close(m(self.batch)['x_hat_main'],out['x_hat_main'],rtol=0,atol=0)
        m.train();loss,_=compute_coe_loss(m(self.batch),self.batch,j['config']);loss.backward()
        self.assertGreater(float(m.main_branch.x_history.inject[-1].weight.grad.abs().sum()),0.)

    def test_free_scale_keeps_controlled_delta_and_sparse_top2(self):
        j=copy.deepcopy(self.base);j['config']['model']['coe']['scale_delta_exploration']={'enabled':True,'path':'free','top_k_per_step':[2]*4,'delta_steps':[2,3,4],'same_scale_only':False,'scale_gradient':'selected_branch_st','top1_gradient':'selected_probability_st'}
        m=self.helper.model(j);calls=[]
        hooks=[e.register_forward_pre_hook(lambda e,a:calls.append(len(a[0]))) for e in m.main_branch.routed_experts()]
        try:out=m(self.batch)
        finally:
            for h in hooks:h.remove()
        self.assertEqual(sum(calls),8*len(self.batch['m_f']))
        d=out['coe']['diagnostics'];self.assertEqual([float(d[f'step{i}_delta_enabled']) for i in range(1,5)],[0.,1.,1.,1.])
        loss,_=compute_coe_loss(out,self.batch,j['config']);loss.backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in m.main_branch.history_gate.parameters()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in m.main_branch.scale_heads.parameters()))

    def test_three_job_budget_and_history_resume(self):
        from run_taxibj_input_communication import jobs
        from run_history_plane_interaction import prepare_initialization
        from train_four_direction import train
        import json
        reference=ROOT/'outputs/v25/experiments/first_validation/659d3ba9678e1dca'
        old=json.loads((reference/'jobs/taxibj_B2_seed7.json').read_text())['config']
        actual=jobs(methods=['I01','I02','I03'])
        self.assertEqual(len(actual),3)
        for j in actual.values():
            self.assertEqual(j['config']['train'],old['train'])
            self.assertEqual(j['config']['data'],old['data'])
            self.assertEqual(j['config']['experiment_plan']['protocol'],old['experiment_plan']['protocol'])
        self.assertEqual(actual['taxibj_I02_seed7']['config']['model']['coe']['initial_observation_scales'],'FMC')
        self.assertEqual(actual['taxibj_I03_seed7']['config']['model']['coe']['initial_observation_scales'],'FMC')
        root=Path(self.temp.name);state=self.helper.model(self.base).state_dict()
        torch.save({'model':state,'sha256':state_hash(state),'seed':7},root/'original.pth')
        j=copy.deepcopy(self.base);j.update(variant='taxibj_I03_seed7',method='I03')
        j['config']['model']['coe']['observed_history']={'enabled':True,'mode':'adaptive'}
        j['config']['train'].update(epochs=2,val_epoch=1);j['config']['train']['scheduler']['total_epochs']=2
        prepare_initialization(root/'suite',{j['variant']:j},root/'original.pth')
        train(j,root/'resumed',root/'resumed.json','cpu',stop_after=1)
        train(j,root/'resumed',root/'resumed.json','cpu')
        train(j,root/'full',root/'full.json','cpu')
        a=torch.load(root/'resumed/checkpoints/last.pth',weights_only=False)
        b=torch.load(root/'full/checkpoints/last.pth',weights_only=False)
        for name in ['model','optimizer','scheduler','scaler','rng_states']:
            self.assertEqual(state_hash(a[name]),state_hash(b[name]),name)


if __name__=='__main__':unittest.main()
