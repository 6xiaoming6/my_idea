"""I26 removes only projection-related computation from the I25 backbone."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'src'),str(ROOT/'scripts/v25')]
import test_v25_combination_aux as fixtures
from run_taxibj_input_communication import jobs
from stmoe_imputer.losses import compute_coe_loss,visible_observation_consistency_loss,supervision_mask
from stmoe_imputer.utils.deterministic import state_hash
import run_priority_resume as priority


class NoProjectionTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.CombinationAuxTests();self.f.setUp();self.addCleanup(self.f.doCleanups)

    def test_config_single_factor_and_old_i25_unchanged(self):
        frozen=json.loads((ROOT/'outputs/v25/experiments/taxibj_input_communication/2c07a1e3f7f177b7/plan.json').read_text())['jobs']
        self.assertEqual(jobs(methods=['I25']),frozen)
        a=copy.deepcopy(jobs(methods=['I25'])['taxibj_I25_seed7']['config'])
        b=copy.deepcopy(jobs(methods=['I26'])['taxibj_I26_seed7']['config'])
        a.pop('experiment_plan');b.pop('experiment_plan')
        a['model']['coe'].pop('observation_consistency');a['loss'].pop('learnable_f_mix')
        a['loss']['visible_consistency_aux']['source']='prediction'
        self.assertEqual(a,b)
        c=copy.deepcopy(jobs(methods=['I16'])['taxibj_I16_seed7']['config']);c.pop('experiment_plan')
        c['model']['coe']['scale_delta_exploration']=copy.deepcopy(b['model']['coe']['scale_delta_exploration'])
        self.assertEqual(c,b)

    def test_raw_forward_parity_sparse_history_gradients_and_loss(self):
        f=self.f;j=f.job('I26');old=f.helper.model(f.job('I25')).eval();new=f.helper.model(j).eval()
        expected={k:v for k,v in old.state_dict().items() if not k.startswith('main_branch.x_consistency.') and k!='main_branch.x_loss_mix_logit'}
        self.assertEqual(state_hash(expected),state_hash(new.state_dict()))
        calls=[];hooks=[e.register_forward_pre_hook(lambda e,a:calls.append(len(a[0]))) for e in new.main_branch.routed_experts()]
        try:out=new(f.batch)
        finally:
            for h in hooks:h.remove()
        parent=old(f.batch)
        torch.testing.assert_close(out['x_hat_main'],parent['x_hat_raw_consistency'],rtol=0,atol=0)
        self.assertNotIn('x_hat_raw_consistency',out);self.assertNotIn('f_loss_raw_weight',out)
        self.assertFalse(hasattr(new.main_branch,'x_consistency'))
        self.assertEqual(sum(calls),8*len(f.batch['m_f']))
        self.assertTrue(bool((out['coe']['triscale_choices'][:,3]==0).all()))
        self.assertEqual([float(out['coe']['diagnostics'][f'x_step{i}_pool_size']) for i in range(1,5)],[0,1,2,3])
        # Scale straight-through gradients are enabled in training mode only.
        out=new.train()(f.batch)
        loss,logs=compute_coe_loss(out,f.batch,j['config'])
        selected=supervision_mask(f.batch['x_f_gt'],f.batch['m_f'],f.batch.get('target_mask'))
        main=(out['x_hat_main'][selected]-f.batch['x_f_gt'][selected]).abs().mean()
        aux,_=visible_observation_consistency_loss(out,f.batch,j['config'])
        torch.testing.assert_close(loss,main+.1*aux+logs['l_coe_balance_weighted'])
        # Retaining beta with identical raw/post predictions has zero beta gradient.
        beta=torch.tensor(.3,requires_grad=True);equivalent=(1-beta)*main+beta*main+.1*aux+logs['l_coe_balance_weighted']
        torch.testing.assert_close(equivalent,loss)
        self.assertEqual(float(torch.autograd.grad(equivalent,beta,retain_graph=True)[0]),0.)
        loss.backward()
        for name in ('observation_fusion','history_gate','x_history','scale_heads'):
            module=getattr(new.main_branch,name)
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in module.parameters()),name)
        # Hidden F/M/C values and full coarse labels cannot affect prediction/loss.
        poisoned=copy.deepcopy(f.batch)
        for scale in ('f','m','c'):
            poisoned['x_'+scale+'_obs'][~poisoned['m_'+scale].bool()]=float('nan')
        for scale in ('m','c'):poisoned['x_'+scale+'_gt']=torch.full_like(poisoned['x_'+scale+'_obs'],float('nan'))
        po=new(poisoned);torch.testing.assert_close(po['x_hat_main'],out['x_hat_main'],rtol=0,atol=0)
        torch.testing.assert_close(compute_coe_loss(po,poisoned,j['config'])[0],loss,rtol=0,atol=0)

    def test_checkpoint_replay_and_six_evaluations(self):
        f=self.f;original=f.job;f.job=lambda method:original('I26' if method=='I13' else method)
        f.test_strict_resume_and_six_evaluations()
        e=json.loads((f.root/'evaluation.json').read_text())
        for row in e['sets'].values():
            m=row['metrics'];self.assertEqual(m['combination_correction_mae'],0.)
            self.assertEqual(m['combination_raw_all_mae'],m['combination_post_all_mae'])

    def test_nonzero_history_and_changed_scale_paths_match_raw_backbone(self):
        f=self.f;new=f.helper.model(f.job('I26')).eval();old=f.helper.model(f.job('I25')).eval()
        for path in ((0,1,2),(2,2,2),(1,0,1)):
            with torch.no_grad():
                new.main_branch.x_history.inject[-1].weight.fill_(.01)
                for sid,head in zip(path,new.main_branch.scale_heads[:3]):
                    head[-1].weight.zero_();head[-1].bias.zero_();head[-1].bias[sid]=20
                old.load_state_dict(new.state_dict(),strict=False)
                a,b=new(f.batch),old(f.batch)
                torch.testing.assert_close(a['x_hat_main'],b['x_hat_raw_consistency'],rtol=0,atol=0)
                self.assertTrue(bool((a['coe']['triscale_choices']==torch.tensor([*path,0])).all()))
                self.assertGreater(float(a['coe']['diagnostics']['x_step2_pool_delta_abs']),0.)

    def test_priority_failure_does_not_resume_original(self):
        suite=self.f.root/'priority';suite.mkdir();runner=suite/'source_snapshot'/priority.RUNNER
        runner.parent.mkdir(parents=True);runner.write_text('# fixture')
        plan={'resume_after_priority':{},'project_root':str(self.f.root),'source':{priority.RUNNER:priority.file_sha(runner)}}
        priority.write(suite/'plan.json',plan)
        with patch.object(priority,'verify_pause') as verify,patch.object(priority.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['fixture'])) as launch:
            with self.assertRaises(subprocess.CalledProcessError):priority.run(suite)
            self.assertEqual(launch.call_count,1);self.assertEqual(verify.call_count,1)
            self.assertEqual(priority.load(suite/'priority_status.json')['phase'],'blocked')

    def test_priority_success_and_pause_evidence_guards(self):
        root=self.f.root;old=root/'old';old.mkdir();runner=old/'source_snapshot'/priority.RUNNER
        runner.parent.mkdir(parents=True);runner.write_text('# old frozen')
        priority.write(old/'plan.json',{'source':{priority.RUNNER:priority.file_sha(runner)}})
        priority.write(old/'state.json',{'phase':'paused_by_user'})
        cp=old/'last.pth';cp.write_bytes(b'complete checkpoint')
        pause={'status':'paused_by_user','checkpoint':str(cp),'checkpoint_sha256':priority.file_sha(cp)}
        pp=old/'paused_for_i26.json';priority.write(pp,pause)
        spec={'path':str(old),'pause_path':str(pp),'pause_sha256':priority.file_sha(pp),'plan_sha256':priority.file_sha(old/'plan.json')}
        priority.verify_pause(spec)
        suite=root/'priority';suite.mkdir();newrunner=suite/'source_snapshot'/priority.RUNNER
        newrunner.parent.mkdir(parents=True);newrunner.write_text('# new frozen')
        priority.write(suite/'plan.json',{'resume_after_priority':spec,'project_root':str(root),'source':{priority.RUNNER:priority.file_sha(newrunner)}})
        with patch.object(priority.subprocess,'run') as launch,patch.object(priority,'_completion_status',return_value=('ready','ok')):
            priority.run(suite);self.assertEqual(launch.call_count,2)
            self.assertEqual(launch.call_args_list[1].args[0][-1],str(old))
            self.assertEqual(priority.load(suite/'priority_status.json')['phase'],'completed')
        cp.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'checkpoint changed'):priority.verify_pause(spec)


if __name__=='__main__':unittest.main()
