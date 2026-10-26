"""I28 data isolation, identical masks, scalar losses and boundary guards."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v25'),str(ROOT/'scripts/v24')]
import run_inflow_priority as runner
from stmoe_imputer.data.observed_multiscale import aggregate_sum, ObservedMultiScaleDataset
from stmoe_imputer.data import build_datasets, build_loader
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss, supervision_mask
from stmoe_imputer.utils.deterministic import common_initialization, state_hash
from preflight_taxibj_input_communication import check_consistency


class InflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.original=json.loads((ROOT/'outputs/v25/experiments/taxibj_input_communication/2c07a1e3f7f177b7/jobs/taxibj_I25_seed7.json').read_text())
        fine=np.random.default_rng(7).uniform(1,15,(8,2,3,8,8)).astype(np.float32)
        self.arrays={'x_f_gt':fine,'x_m_gt':aggregate_sum(fine,2),'x_c_gt':aggregate_sum(fine,4)}
        self.selected=runner.select_inflow(self.arrays)
        self.path=self.root/'inflow.npz';np.savez(self.path,**self.selected)
        self.job=runner.make_job(self.original,{s:str(self.path) for s in ('train','val','test')})

    def small(self):
        j=copy.deepcopy(self.job);cfg=j['config']
        cfg['model']['main'].update(dim=8,max_t=3,h=8,w=8)
        cfg['data']['batch_size']=4
        cfg['model']['coe']['router_hidden_dim']=8
        return j

    def test_only_authorized_config_and_independent_data(self):
        a=copy.deepcopy(self.original['config']);b=copy.deepcopy(self.job['config'])
        a.pop('experiment_plan');b.pop('experiment_plan')
        b['model']['c_in']=2;b['data'].pop('channel_selection')
        self.assertEqual(a,b)
        changed={k:v.copy() for k,v in self.arrays.items()}
        for v in changed.values():v[:,1]+=12345
        selected=runner.select_inflow(changed)
        for k,v in self.selected.items():
            np.testing.assert_array_equal(v,self.arrays[k][:,0:1])
            np.testing.assert_array_equal(v,selected[k])
        invalid={k:v.copy() for k,v in self.arrays.items()};invalid['x_c_gt'][:,0]+=1
        with self.assertRaises(ValueError):runner.select_inflow(invalid)

    def test_masks_and_observations_preserve_channel_zero_at_every_scale(self):
        original=self.root/'dual.npz';np.savez(original,**self.arrays)
        schedule=self.job['config']['data']['train_mask_diversity']
        dual=ObservedMultiScaleDataset(original,schedule)
        single=ObservedMultiScaleDataset(self.path,schedule)
        for epoch in (1,2,9):
            dual.set_epoch(epoch);single.set_epoch(epoch)
            for index in range(len(single)):
                a,b=dual[index],single[index]
                for key in ('x_f_gt','x_f_obs','x_m_obs','x_c_obs','m_f','m_m','m_c'):
                    torch.testing.assert_close(a[key][0:1],b[key],rtol=0,atol=0)
                self.assertNotIn('x_m_gt',b);self.assertNotIn('x_c_gt',b)

    def test_single_channel_forward_gradient_constraints_and_loss(self):
        j=self.small();cfg=j['config'];torch.manual_seed(7)
        model=DualBranchSTImputer.from_config(cfg).train()
        ds,_=build_datasets(cfg,str(self.path),str(self.path))
        batch=next(iter(build_loader(ds,cfg,shuffle=False)))
        calls=[];hooks=[e.register_forward_pre_hook(lambda e,a:calls.append(len(a[0]))) for e in model.main_branch.routed_experts(0)]
        try:out=model(batch)
        finally:
            for h in hooks:h.remove()
        self.assertEqual(sum(calls),8*len(batch['m_f']))
        self.assertEqual(out['x_hat_main'].shape,(4,1,3,8,8))
        self.assertTrue(bool((out['coe']['triscale_choices'][:,3]==0).all()))
        check_consistency(out,batch,cfg)
        loss,logs=compute_coe_loss(out,batch,cfg)
        q=supervision_mask(batch['x_f_gt'],batch['m_f'])
        raw=(out['x_hat_raw_consistency'][q]-batch['x_f_gt'][q]).abs().mean()
        post=(out['x_hat_main'][q]-batch['x_f_gt'][q]).abs().mean()
        completed=torch.where(batch['m_f'].bool(),batch['x_f_obs'],out['x_hat_raw_consistency'])
        aux=[]
        for scale,factor in [('m',2),('c',4)]:
            agg=completed.reshape(4,1,3,8//factor,factor,8//factor,factor).sum((4,6))
            mask=batch['m_'+scale].bool()
            aux.append((agg[mask]-batch['x_'+scale+'_obs'][mask]).abs().mean()/factor**2)
        beta=out['f_loss_raw_weight']
        expected=(1-beta)*post+beta*raw+.1*torch.stack(aux).mean()+.01*logs['l_coe_balance']
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.main_branch.encoder.parameters()))
        self.assertTrue(bool(torch.isfinite(model.main_branch.x_loss_mix_logit.grad)))
        poison=copy.deepcopy(batch)
        for s in ('f','m','c'):poison['x_'+s+'_obs'][~poison['m_'+s].bool()]=float('nan')
        poison['x_f_gt'].fill_(float('nan'))
        model.eval()
        with torch.no_grad():torch.testing.assert_close(model(batch)['x_hat_main'],model(poison)['x_hat_main'],rtol=0,atol=0)

    def test_fresh_initialization_reproducible_and_loadable(self):
        j=self.small();plan={'jobs':{runner.NAME:j}}
        js=runner.prepare_initialization(self.root,plan)
        again=runner.prepare_initialization(self.root,plan)
        self.assertEqual(js,again)
        torch.manual_seed(7);model=DualBranchSTImputer.from_config(j['config'])
        before=state_hash(model.state_dict());common_initialization(model,js[runner.NAME])
        self.assertEqual(before,state_hash(model.state_dict()))

    def test_waiter_rejects_active_i27_and_reboot(self):
        i27=self.root/'i27';i27.mkdir()
        runner.write(i27/'plan.json',{})
        hold=i27/'hold.json'
        identity={'pid':100,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
        runner.write(hold,identity)
        plan={'held_next':{'path':str(i27),'plan_sha256':runner.file_sha(i27/'plan.json'),
                          'hold':str(hold),'hold_sha256':runner.file_sha(hold)}}
        with patch.object(runner,'same_process',return_value=True),patch.object(runner,'process',return_value={'state':'S'}):
            with self.assertRaisesRegex(RuntimeError,'remain held'):runner.held_next(plan)
        with patch.object(runner,'same_process',return_value=True),patch.object(runner,'process',return_value={'state':'T'}),patch.object(runner,'children',return_value=[]),patch.object(runner,'check_ready',return_value=(False,'I26 still training',{})):
            self.assertFalse(runner.held_next(plan)[1])
        identity['boot_id']='different_boot';runner.write(hold,identity)
        plan['held_next']['hold_sha256']=runner.file_sha(hold)
        with self.assertRaisesRegex(RuntimeError,'rebooted'):runner.held_next(plan)

    def test_training_resume_six_protocols_and_formal_report(self):
        from train_four_direction import train
        from evaluate_four_direction import evaluate_sets
        from report_inflow_priority import export_report
        job=self.small();job['config']['train'].update(epochs=2,val_epoch=1)
        job['config']['train']['scheduler']['total_epochs']=2
        plan={'jobs':{runner.NAME:job},'order':[runner.NAME]}
        js=runner.prepare_initialization(self.root,plan);job=js[runner.NAME]
        run=self.root/'run';receipt=self.root/'results'/f'{runner.NAME}.json'
        train(job,run,receipt,'cpu',stop_after=1)
        train(job,run,receipt,'cpu')
        train(job,self.root/'full',self.root/'full_receipt.json','cpu')
        a=torch.load(run/'checkpoints/last.pth',weights_only=False)
        b=torch.load(self.root/'full/checkpoints/last.pth',weights_only=False)
        for key in ('model','optimizer','scheduler','scaler','rng_states'):
            self.assertEqual(state_hash(a[key]),state_hash(b[key]),key)
        ev=self.root/'evaluations'/f'{runner.NAME}.json'
        evaluate_sets(run/'checkpoints/best.pth',job['config']['experiment_plan']['protocol'],str(self.path),ev,runner.digest(job['config']),'cpu')
        result=runner.load(ev)
        self.assertEqual(len(result['sets']),6)
        for item in result['sets'].values():
            self.assertEqual(item['metrics']['coe_expert_execution_count'],8.)
            self.assertIn('combination_raw_all_mae',item['metrics'])
        report=export_report(self.root,self.root,js)
        self.assertTrue(runner.load(self.root/'report_path.json')['complete'])
        self.assertIn('不直接比较',report.read_text())
        runner.write(self.root/'completed.json',{'count':1,'report':str(report)})
        runner.write(self.root/'state.json',{'phase':'completed','report':str(report)})
        self.assertEqual(runner._completion_status(self.root,plan)[0],'ready')


if __name__=='__main__':unittest.main()
