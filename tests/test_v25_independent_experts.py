"""I27: independent expert storage, common values, sparse execution and queue order."""
import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'src'),str(ROOT/'scripts/v25')]
import test_v25_combination_aux as fixtures
from run_taxibj_input_communication import jobs
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import state_hash
import run_before_priority_resume as queue


class IndependentExpertTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.CombinationAuxTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        original=self.f.job
        def job(method):
            j=original(method)
            j['config']['model']['coe']['expert_sharing']=jobs(methods=[method])[f'taxibj_{method}_seed7']['config']['model']['coe']['expert_sharing']
            return j
        self.f.job=job

    def test_single_config_change_and_old_jobs(self):
        js=jobs(methods=['I25','I27']);a,b=[copy.deepcopy(js[f'taxibj_{m}_seed7']['config']) for m in ('I25','I27')]
        a.pop('experiment_plan');b.pop('experiment_plan');b['model']['coe']['expert_sharing']='shared'
        self.assertEqual(a,b)
        for method,d in [('I25','2c07a1e3f7f177b7'),('I26','8f67d8cc2cbf072e')]:
            old=json.loads((ROOT/f'outputs/v25/experiments/taxibj_input_communication/{d}/plan.json').read_text())['jobs']
            self.assertEqual(jobs(methods=[method]),old)

    def test_independent_storage_equal_initial_forward_and_gradient_sums(self):
        f=self.f;ja,jb=f.job('I25'),f.job('I27');a,b=f.helper.model(ja).train(),f.helper.model(jb).train()
        shared={k:v for k,v in b.state_dict().items() if not k.startswith('main_branch.step_pattern_experts.')}
        self.assertEqual(state_hash(shared),state_hash(a.state_dict()))
        pools=[b.main_branch.pattern_experts,*b.main_branch.step_pattern_experts]
        self.assertEqual(len(pools),4)
        for pool in pools:self.assertEqual(state_hash(pool.state_dict()),state_hash(a.main_branch.pattern_experts.state_dict()))
        pointers=[{p.data_ptr() for p in pool.parameters()} for pool in pools]
        self.assertTrue(all(not pointers[i]&pointers[j] for i in range(4) for j in range(i)))
        calls=[0]*4;hooks=[]
        for step in range(4):
            def record(module,args,step=step):calls[step]+=len(args[0])
            hooks.extend(e.register_forward_pre_hook(record) for e in b.main_branch.routed_experts(step))
        try:oa,ob=a(f.batch),b(f.batch)
        finally:
            for h in hooks:h.remove()
        self.assertEqual(calls,[2*len(f.batch['m_f'])]*4)
        torch.testing.assert_close(oa['x_hat_main'],ob['x_hat_main'],rtol=0,atol=0)
        la,_=compute_coe_loss(oa,f.batch,ja['config']);lb,_=compute_coe_loss(ob,f.batch,jb['config'])
        torch.testing.assert_close(la,lb,rtol=0,atol=0);la.backward();lb.backward()
        for pool in pools:self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in pool.parameters()))
        for name,p in a.main_branch.pattern_experts.named_parameters():
            gs=[dict(pool.named_parameters())[name].grad for pool in pools]
            total=sum((g for g in gs if g is not None),torch.zeros_like(p))
            torch.testing.assert_close(total,p.grad if p.grad is not None else torch.zeros_like(p),rtol=1e-4,atol=1e-6)
        before=[state_hash(pool.state_dict()) for pool in pools]
        with torch.no_grad():next(pools[1].parameters()).add_(1)
        self.assertNotEqual(state_hash(pools[1].state_dict()),before[1])
        for i in (0,2,3):self.assertEqual(state_hash(pools[i].state_dict()),before[i])

    def test_checkpoint_replay_and_six_evaluations(self):
        f=self.f;original=f.job;f.job=lambda method:original('I27' if method=='I13' else method)
        f.test_strict_resume_and_six_evaluations()

    def test_completed_dependency_still_waits_for_queue_process_exit(self):
        plan={'insert_before_resume':{}}
        with patch.object(queue,'held_controller',return_value={'pid':1}),patch.object(queue,'dependency_status',return_value=('ready','ok')),patch.object(queue,'children',return_value=[2]),patch.object(queue,'process',return_value={'state':'R'}):
            self.assertFalse(queue.check_ready(plan)[0])
            with self.assertRaisesRegex(RuntimeError,'release'):queue.validate_launch(plan)

    def test_queue_wait_then_run_then_resume_only_on_success(self):
        for fail in (False,True):
            with self.subTest(fail=fail):
                suite=self.f.root/str(fail);suite.mkdir();runner=suite/'source_snapshot'/queue.RUNNER
                runner.parent.mkdir(parents=True);runner.write_text('# frozen')
                plan={'insert_before_resume':{},'source':{queue.RUNNER:queue.file_sha(runner)},'project_root':str(self.f.root)}
                queue.write(suite/'plan.json',plan)
                identity={'pid':123}
                with patch.object(queue,'check_ready',side_effect=[(False,'waiting',identity),(True,'ok',identity)]),patch.object(queue.time,'sleep'),patch.object(queue,'held_controller',return_value=identity),patch.object(queue,'_completion_status',return_value=('ready','ok')),patch.object(queue.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['test']) if fail else None) as launch,patch.object(queue.os,'kill') as kill:
                    if fail:
                        with self.assertRaises(subprocess.CalledProcessError):queue.run(suite)
                        kill.assert_not_called()
                    else:
                        queue.run(suite);kill.assert_called_once_with(123,signal.SIGCONT)
                    self.assertEqual(launch.call_count,1)

    def test_stale_controller_rejected(self):
        path=self.f.root/'hold.json';queue.write(path,{'pid':123,'start':'old','args':['controller']})
        spec={'controller_hold':str(path),'controller_hold_sha256':queue.file_sha(path)}
        with patch.object(queue,'same_process',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'identity changed'):queue.held_controller(spec)


if __name__=='__main__':unittest.main()
