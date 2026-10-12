import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/v24')]
from run_history_followup import jobs, reserve_run, wait_for_upstream, CORE, UPSTREAM
from run_history_plane_interaction import prepare_initialization
from run_b3_c3 import load
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss, supervision_mask
from stmoe_imputer.utils.deterministic import state_hash


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.jobs = jobs()

    def config(self, name):
        c = copy.deepcopy(self.jobs[f'taxibj_{name}_seed7']['config'])
        c['model']['main']['dim'] = 8; c['model']['coe']['router_hidden_dim'] = 8
        return c

    def build(self, c):
        torch.manual_seed(7)
        return DualBranchSTImputer.from_config(c)

    def batch(self):
        g = torch.Generator().manual_seed(43)
        x = 10*torch.randn(2,2,3,8,8,generator=g)
        mask = (torch.rand(x.shape,generator=g)>.4).float()
        return dict(x_f_gt=x, x_f_obs=x*mask, m_f=mask, mask_family=torch.tensor([0,1]))

    def test_plan_rates_budget_initialization_and_zero_equivalence(self):
        self.assertEqual(len(self.jobs),12)
        b = self.batch(); original = self.config('Y01')
        original['model']['coe'].pop('history_plane_interaction'); original['loss']['lambda_coe_mid']=0
        baseline = self.build(original); expected_rng = torch.get_rng_state(); expected = baseline(b)['x_hat_main']
        for name,j in self.jobs.items():
            c = self.config(j['method']); m = self.build(c)
            self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng),name)
            for k,v in baseline.state_dict().items():
                torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)
            torch.testing.assert_close(m(b)['x_hat_main'],expected,rtol=0,atol=0)
            rate = .6 if j['method'] in ('Y05','Y06','Y07','Y08') else .4
            self.assertEqual(c['data']['mask']['missing_rate'],rate)
            self.assertEqual(c['data']['train_mask_diversity']['rates'],[rate])
            self.assertEqual(c['data']['eval_mask_diversity']['rates'],[rate])
            self.assertEqual(c['experiment_plan']['protocol']['rate'],rate)
            self.assertEqual(c['train']['epochs'],100); self.assertEqual(c['data']['batch_size'],32)
            self.assertEqual(len(c['experiment_plan']['protocol']['evaluations']),6)
        models = [self.build(self.config(n)).main_branch for n in ('Y09','Y10','Y11')]
        self.assertEqual(len({state_hash(m.x_global_plane.state_dict()) for m in models}),1)
        full = self.build(self.jobs['taxibj_Y10_seed7']['config'])
        self.assertEqual(sum(p.numel() for p in full.main_branch.x_global_plane.parameters()),3424)
        # Combined reader and interaction retain precisely their respective X initial values.
        y = self.build(self.config('Y12')).main_branch
        xjobs = load(ROOT/UPSTREAM/'plan.json')['jobs']
        for label, attr in [('X05','x_history'),('X10','x_interaction')]:
            c=copy.deepcopy(xjobs[f'taxibj_{label}_seed7']['config'])
            c['model']['main']['dim']=8; c['model']['coe']['router_hidden_dim']=8
            x=self.build(c).main_branch
            self.assertEqual(state_hash(getattr(x,attr).state_dict()),state_hash(getattr(y,attr).state_dict()))

    def test_intermediate_loss_formula_and_direct_gradient(self):
        b = self.batch(); c = self.config('Y04'); model = self.build(c)
        out = model(b); preds = out['coe']['predictions']; selected = supervision_mask(b['x_f_gt'],b['m_f'])
        for p in preds:
            p.retain_grad()
        loss,logs = compute_coe_loss(out,b,c)
        expected_mid = torch.stack([(p[selected]-b['x_f_gt'][selected]).abs().mean() for p in preds[:-1]]).mean()
        torch.testing.assert_close(logs['l_coe_mid'],expected_mid)
        torch.testing.assert_close(loss,logs['l_main']+.1*expected_mid+.01*logs['l_coe_balance'])
        loss.backward()
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum()>0 for p in preds))
        c['loss']['lambda_coe_mid']=0; model=self.build(c); out=model(b)
        for p in out['coe']['predictions']:
            p.retain_grad()
        loss,_=compute_coe_loss(out,b,c); loss.backward()
        self.assertTrue(all(p.grad is None or not p.grad.abs().sum() for p in out['coe']['predictions'][:-1]))

    def test_all_selected_experts_read_planes_and_native_top2(self):
        b=self.batch()
        for name in ('Y09','Y10','Y11','Y12'):
            c=self.config(name); model=self.build(c); main=model.main_branch
            with torch.no_grad():
                # No ST selected: all-expert planes must still execute and receive gradients.
                for router in main.routers:
                    router[-1].weight.zero_(); router[-1].bias.zero_(); router[-1].bias[:2]=10
            calls=[]; handles=[e.register_forward_pre_hook(lambda m,a:calls.append((len(a[0]),a[0].shape[-1]))) for e in main.routed_experts(0)]
            out=model(b)
            for h in handles:h.remove()
            self.assertEqual(sum(n for n,w in calls),16)
            self.assertEqual({w for n,w in calls},{2,4,8})
            torch.testing.assert_close(out['coe']['triscale_choices'],torch.tensor([[2,1,0,0]]*2))
            loss,_=compute_coe_loss(out,b,c); loss.backward()
            for attr in ('x_global_plane','x_history','x_interaction'):
                if hasattr(main,attr):
                    grads=[p.grad for p in getattr(main,attr).parameters() if p.grad is not None]
                    self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads),name+attr)
            if hasattr(main,'x_global_plane'):
                self.assertTrue(all(float(out['coe']['diagnostics'][f'x_step{i}_global_plane_executions'])==1 for i in range(1,5)))
                inputs=[]
                with torch.no_grad():main.x_global_plane.up.weight.fill_(.05)
                handle=main.pattern_experts['T'].register_forward_pre_hook(lambda m,a:inputs.append(a[0].detach().clone()))
                active=model(b)['x_hat_main']; handle.remove()
                self.assertFalse(torch.equal(active,out['x_hat_main']))
                self.assertEqual(len(inputs),4)
            torch.testing.assert_close(model(b)['x_hat_main'],model(dict(b,x_f_gt=b['x_f_gt']*100))['x_hat_main'],rtol=0,atol=0)
            for value in (0.,1.):
                mask=torch.full_like(b['m_f'],value)
                pred=model(dict(b,m_f=mask,x_f_obs=b['x_f_gt']*mask))['x_hat_main']
                self.assertTrue(torch.isfinite(pred).all())

    def test_active_uniform_coverage_and_unsupported_regions(self):
        model=self.build(self.config('Y10')); branch=model.main_branch.x_global_plane
        uniform=copy.deepcopy(branch); uniform.mode='uniform'
        with torch.no_grad():
            branch.up.weight.fill_(.03);branch.up.bias.fill_(.4);uniform.load_state_dict(branch.state_dict())
        x=torch.randn(2,8,3,8,8);mask=torch.zeros(2,2,3,8,8);mask[:,:,0,:3,:]=1
        a,_=branch(x,mask);z,_=uniform(x,mask)
        self.assertFalse(torch.equal(a,z))
        z,_=branch(x,torch.zeros_like(mask));torch.testing.assert_close(z,torch.zeros_like(z),rtol=0,atol=0)

    def test_rate_aware_resume_loss_and_evaluation(self):
        from train_four_direction import train
        from evaluate_four_direction import evaluate_sets
        import run_four_direction_exploration as base
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);suite=root/'suite'; js={}
            for name in ('Y04','Y08','Y11','Y12'):
                job=copy.deepcopy(self.jobs[f'taxibj_{name}_seed7']);c=self.config(name);job['config']=c
                c['output_dir']=str(root/'outputs/v24-COE');c['train'].update(epochs=2,val_epoch=1)
                c['train']['scheduler']['total_epochs']=2;c['data'].update(batch_size=2,pin_memory=False)
                c['experiment_plan']['protocol']={'rate':c['data']['mask']['missing_rate'],'evaluations':{'in_distribution':{'families':['random_point'],'seed':20260917}}}
                for split in ('train','val','test'):
                    p=root/(split+'.npz');np.savez(p,x_f_gt=np.random.RandomState(4).randn(4,2,3,8,8).astype('float32'));job['sources'][split]=str(p)
                js[job['variant']]=job
            c=self.config('Y01');c['model']['coe'].pop('history_plane_interaction');model=self.build(c)
            ref=root/'init.pth';torch.save({'model':model.state_dict(),'sha256':state_hash(model.state_dict())},ref)
            prepare_initialization(suite,js,ref)
            for name,job in js.items():
                run=reserve_run(suite,job);self.assertIn(f'rate{job["config"]["data"]["mask"]["missing_rate"]:g}',str(run))
                self.assertEqual(run,reserve_run(suite,job))
                result=suite/'results'/f'{name}.json'
                train(job,run,result,'cpu',stop_after=1);train(job,run,result,'cpu')
                with patch.object(base,'subprocess_run',side_effect=AssertionError('must skip finished')):
                    base.launch(suite,job,0)
                full=root/(name+'_full');train(job,full,root/(name+'_result.json'),'cpu')
                a=torch.load(run/'checkpoints/last.pth',weights_only=False);b=torch.load(full/'checkpoints/last.pth',weights_only=False)
                for key in ('model','optimizer','scheduler','scaler','rng_states'):
                    self.assertEqual(state_hash(a[key]),state_hash(b[key]),name+key)
                e=evaluate_sets(run/'checkpoints/best.pth',job['config']['experiment_plan']['protocol'],job['sources']['test'],suite/'evaluations'/f'{name}.json',device='cpu')
                self.assertEqual(e['sets']['in_distribution']['mask_spec']['rates'],[job['config']['data']['mask']['missing_rate']])
                metrics=e['sets']['in_distribution']['metrics'];self.assertAlmostEqual(metrics['mae'],metrics['coe_x_step4_mae'],places=5)
            from report_history_followup import collect
            job=js['taxibj_Y08_seed7'];r,_=collect(suite,'Y08',job)
            self.assertEqual(r['status'],'finished')
            ep=suite/'evaluations/taxibj_Y08_seed7.json';e=load(ep);e['sets']['in_distribution']['mask_spec']['rates']=[.4];ep.write_text(json.dumps(e))
            self.assertEqual(collect(suite,'Y08',job)[0]['status'],'failed')

    def test_queue_lock_wait_and_stage_report(self):
        import run_history_followup as runner
        from report_history_followup import export_report
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);suite=root/'suite';upstream=root/'upstream';upstream.mkdir();suite.mkdir()
            # No launch is made while another queue owns the upstream lock.
            with patch.object(runner.fcntl,'flock',side_effect=[BlockingIOError(),None]),patch.object(runner.time,'sleep') as sleep:
                wait_for_upstream(suite,upstream)
            sleep.assert_called_once_with(10)
            self.assertEqual(load(suite/'state.json')['phase'],'waiting_for_existing_X_queue')
            core=load(ROOT/CORE/'plan.json')['jobs']['taxibj_K04_seed7']
            refs={'K04':core};refs.update({f'X{i:02}':load(ROOT/UPSTREAM/'plan.json')['jobs'][f'taxibj_X{i:02}_seed7'] for i in range(1,11)})
            payload={'jobs':self.jobs,'references':refs,'upstream':str(ROOT/UPSTREAM)}
            dest=export_report(root,suite,payload);text=dest.read_text()
            self.assertIn('阶段报告',text);self.assertLess(text.index('具体方法'),text.index('完整结果'))
            self.assertEqual(len(load(suite/'summary.json')),23)
            self.assertIn('Y12',text);self.assertIn('0.6',text)

    def test_upstream_tmux_cleanup_protects_running_children(self):
        import run_history_followup as runner
        listed=SimpleNamespace(returncode=0,stdout='123|0|bash|/dev/pts/5\n')
        with patch.object(runner.subprocess,'run',return_value=listed) as run,patch.object(
                runner.subprocess,'check_output',return_value='123 99 pts/5 bash\n456 123 pts/5 python\n'):
            self.assertEqual(runner.close_idle_upstream_session(),'kept_busy')
            self.assertEqual(run.call_count,1)
        with patch.object(runner.subprocess,'run',return_value=listed) as run,patch.object(
                runner.subprocess,'check_output',return_value='123 99 pts/5 bash\n'):
            self.assertEqual(runner.close_idle_upstream_session(),'closed_idle')
            self.assertEqual(run.call_args.args[0],['tmux','kill-session','-t','=v24-history-plane-10'])


if __name__=='__main__':unittest.main()
