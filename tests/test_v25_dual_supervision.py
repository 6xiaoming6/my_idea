"""Normalized raw/final F losses and safely deferred I18-boundary insertion."""
import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'src'),str(ROOT/'scripts/v25'),str(ROOT/'scripts/v24')]
from test_v25_combination_aux import CombinationAuxTests
from test_v25_queue_dependency import QueueDependencyTests
from run_taxibj_input_communication import jobs
from stmoe_imputer.losses import compute_coe_loss, supervision_mask
from stmoe_imputer.utils.deterministic import state_hash
import run_interleaved as interleave


class DualSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=CombinationAuxTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)

    def test_old_ten_configs_unchanged_and_new_is_i14(self):
        previous=json.loads((ROOT/'outputs/v25/experiments/taxibj_input_communication/22d4ae99d3e86986/plan.json').read_text())['jobs']
        self.assertEqual(jobs(methods=[f'I{i:02}' for i in range(13,23)]),previous)
        parent=jobs(methods=['I14'])['taxibj_I14_seed7']['config']
        for method,beta in [('I23',.1),('I24',.3)]:
            cfg=jobs(methods=[method])[f'taxibj_{method}_seed7']['config']
            self.assertEqual(cfg['loss']['lambda_final_f']+cfg['loss']['lambda_raw_f'],1.)
            self.assertEqual(cfg['loss']['lambda_raw_f'],beta)
            a,b=copy.deepcopy(parent),copy.deepcopy(cfg)
            a.pop('experiment_plan');b.pop('experiment_plan')
            b['loss'].pop('lambda_final_f');b['loss'].pop('lambda_raw_f')
            self.assertEqual(a,b)

    def test_exact_loss_formula_gradients_and_same_forward(self):
        f=self.fixture;old=f.job('I14');base=f.helper.model(old).eval();baseout=base(f.batch)
        parentloss,_=compute_coe_loss(baseout,f.batch,old['config'])
        selected=supervision_mask(f.batch['x_f_gt'],f.batch['m_f'],f.batch.get('target_mask'))
        for method,beta in [('I23',.1),('I24',.3)]:
            j=f.job(method);model=f.helper.model(j).eval();out=model(f.batch)
            self.assertEqual(state_hash(model.state_dict()),state_hash(base.state_dict()))
            torch.testing.assert_close(out['x_hat_main'],baseout['x_hat_main'],rtol=0,atol=0)
            loss,logs=compute_coe_loss(out,f.batch,j['config'])
            final=(out['x_hat_main'][selected]-f.batch['x_f_gt'][selected]).abs().mean()
            raw=(out['x_hat_raw_consistency'][selected]-f.batch['x_f_gt'][selected]).abs().mean()
            torch.testing.assert_close(loss,parentloss-beta*final+beta*raw)
            torch.testing.assert_close(logs['l_final_weighted'],(1-beta)*final)
            torch.testing.assert_close(logs['l_raw_f_weighted'],beta*raw)
            loss.backward()
            for module in (model.main_branch.observation_fusion,model.main_branch.x_consistency):
                self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in module.parameters()))
        cfg=copy.deepcopy(old['config']);cfg['loss']['lambda_final_f']=1.
        self.assertEqual(state_hash(compute_coe_loss(baseout,f.batch,cfg)),state_hash(compute_coe_loss(baseout,f.batch,old['config'])))
        for bad in (-1.,float('nan'),float('inf')):
            cfg['loss']['lambda_final_f']=bad
            with self.assertRaisesRegex(ValueError,'lambda_final_f'):compute_coe_loss(baseout,f.batch,cfg)

    def test_new_complete_checkpoint_replay_and_six_evaluations(self):
        fixture=self.fixture;original=fixture.job
        fixture.job=lambda method:original('I23' if method=='I13' else method)
        fixture.test_strict_resume_and_six_evaluations()


class InterleaveTests(unittest.TestCase):
    def test_launch_requires_stopped_scheduler_and_exited_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite=Path(tmp);identity={'pid':102,'start':'2','args':['worker']}
            lease={'status':'leased','controller':{'pid':os.getppid()},'parent':{'pid':101},'worker':identity}
            interleave.write(suite/'interleave_lease.json',lease)
            plan={'interleave_after':{}}
            with patch.object(interleave,'same_process',return_value=True), \
                 patch.object(interleave,'locks_owned'), \
                 patch.object(interleave,'boundary_status',return_value=True), \
                 patch.object(interleave,'children',return_value=[]), \
                 patch.object(interleave,'process',return_value={'state':'S'}):
                with self.assertRaisesRegex(RuntimeError,'not suspended'):interleave.validate_launch(suite,plan)
            with patch.object(interleave,'same_process',return_value=True), \
                 patch.object(interleave,'locks_owned'), \
                 patch.object(interleave,'boundary_status',return_value=True), \
                 patch.object(interleave,'children',return_value=[103]), \
                 patch.object(interleave,'process',side_effect=lambda pid:{'state':'T' if pid==102 else 'R'}):
                with self.assertRaisesRegex(RuntimeError,'has not exited'):interleave.validate_launch(suite,plan)

    def test_boundary_receipts_reject_incomplete_and_changed(self):
        f=QueueDependencyTests();f.setUp();self.addCleanup(f.doCleanups)
        spec={**f.dependency,'job':'job'}
        self.assertFalse(interleave.boundary_status(spec))
        f.finish();self.assertTrue(interleave.boundary_status(spec))
        e=copy.deepcopy(f.evaluation);e['sets'].pop(next(iter(e['sets'])))
        interleave.write(f.dep/'evaluations/job.json',e)
        with self.assertRaisesRegex(RuntimeError,'receipts mismatch'):interleave.boundary_status(spec)

    def test_scheduler_only_pause_resume_and_fail_stop(self):
        for returncode in (0,1):
            with self.subTest(returncode=returncode), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);dep=root/'old';suite=root/'new';dep.mkdir();suite.mkdir()
                runner=suite/'source_snapshot/scripts/v25/run_taxibj_input_communication.py'
                runner.parent.mkdir(parents=True);runner.write_text('# fixture')
                plan={'project_root':str(root),'interleave_after':{'path':str(dep),'job':'I18'},
                      'source':{'scripts/v25/run_taxibj_input_communication.py':interleave.file_sha(runner)}}
                interleave.write(suite/'plan.json',plan)
                interleave.write(dep/'state.json',{'phase':'evaluating','variant':'I18'})
                interleave.write(dep/'plan.json',{'order':['I18','I19']})
                parent={'pid':101,'ppid':1,'state':'S','start':'1','args':[str(dep)]}
                worker={'pid':102,'ppid':101,'state':'S','start':'2','args':['--worker',str(dep)]}
                child={'pid':103,'ppid':102,'state':'R','start':'3','args':[str(dep/'evaluations/I18.json')]}
                controller={'pid':os.getpid(),'ppid':1,'state':'R','start':'4','args':['controller']}
                processes={101:parent,102:worker,103:child,os.getpid():controller};signals=[]
                def kill(pid,sig):
                    signals.append((pid,sig))
                    worker['state']='T' if sig==signal.SIGSTOP else 'S'
                    child['state']='Z'
                with patch.object(interleave,'process',side_effect=lambda pid:processes.get(pid)), \
                     patch.object(interleave,'same_process',return_value=True), \
                     patch.object(interleave,'children',return_value=[103]), \
                     patch.object(interleave,'locks_owned'), \
                     patch.object(interleave,'boundary_status',return_value=True), \
                     patch.object(interleave,'_completion_status',return_value=('ready','ok')), \
                     patch.object(interleave.os,'kill',side_effect=kill), \
                     patch.object(interleave.subprocess,'call',return_value=returncode) as launch:
                    if returncode:
                        with self.assertRaisesRegex(RuntimeError,'Inserted queue failed'):interleave.run(suite,101,102)
                        self.assertEqual(signals,[(102,signal.SIGSTOP)])
                    else:
                        interleave.run(suite,101,102)
                        self.assertEqual(signals,[(102,signal.SIGSTOP),(102,signal.SIGCONT)])
                    self.assertEqual(launch.call_count,1)
                    self.assertTrue(all(pid!=103 for pid,_ in signals))

if __name__=='__main__':unittest.main()
