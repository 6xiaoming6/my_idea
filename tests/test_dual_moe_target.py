"""Observation-preserving routing: mechanism contracts, not accuracy claims."""
import copy
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

from test_dual_moe import ROOT, batch, config
from stmoe_imputer.config import deep_update, load_config
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.dual_moe import TargetReadoutRouter
from stmoe_imputer.losses import compute_main_stage_loss, token_topk_balance_loss
from stmoe_imputer.routing_metrics import DualMoEMetricAccumulator
from stmoe_imputer.utils.train_logger import TrainLogger
from stmoe_imputer.utils.checkpoint import snapshot_model_state
from stmoe_imputer.engine import train_one_epoch, evaluate, build_optimizer


def target_config(channels=2):
    cfg = config(channels)
    cfg['model']['dual_moe'].update(design='target_readout_v1', aggregation_experts=8,
                                   aggregation_mode='topk', aggregation_top_k=4,
                                   completion_mode='topk', completion_top_k=2)
    cfg['loss'].update(dual_moe_expert_weight=.01, dual_moe_aggregation_balance_weight=.001,
                       dual_moe_completion_balance_weight=.01)
    cfg['train']['save_best_checkpoint'] = False
    return cfg


class TargetMoETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_backend_blend_identity_bounds_gradients_and_no_leakage(self):
        cfg = target_config()
        cfg['model']['dual_moe'].update(completion_top_k=3, backend_diagnostics=True)
        cfg['loss']['dual_moe_completion_balance_weight'] = 0.
        data = batch(h=4,w=5)
        models, outputs = {}, {}
        for mode in ('none','fixed','learned'):
            local = copy.deepcopy(cfg)
            local['model']['dual_moe'].update(completion_blend=mode,completion_alpha=.5)
            torch.manual_seed(42)
            models[mode] = DualBranchSTImputer.from_config(local)
            outputs[mode] = models[mode](data)
        shared = models['none'].state_dict()
        for mode in ('fixed','learned'):
            for key,value in shared.items():
                torch.testing.assert_close(value,models[mode].state_dict()[key],rtol=0,atol=0)
            gates = outputs[mode]['completion_gates']
            torch.testing.assert_close(gates,1/6+.5*outputs['none']['completion_gates'])
            torch.testing.assert_close(gates.sum(1),torch.ones_like(gates[:,0]))
            self.assertGreaterEqual(float(gates.min()),1/6-1e-7)
            self.assertLessEqual(float(gates.max()),2/3+1e-7)
        loss,_ = compute_main_stage_loss(outputs['learned'],data,cfg,epoch=1)
        loss.backward()
        for p in (models['learned'].main_branch.completion_alpha_logit,
                  models['learned'].main_branch.completion_router.head.weight):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()),0.)
        altered = {**data,'x_f_gt':torch.full_like(data['x_f_gt'],float('nan')),
                   'x_f_obs':torch.where(data['m_f'].bool(),data['x_f_obs'],float('nan'))}
        torch.testing.assert_close(models['learned'](altered)['x_hat_main'],outputs['learned']['x_hat_main'])
        # Legacy default preserves keys, initialization and exact prediction.
        plain=copy.deepcopy(cfg);plain['model']['dual_moe'].pop('backend_diagnostics')
        torch.manual_seed(42); legacy=DualBranchSTImputer.from_config(plain)
        self.assertEqual(set(legacy.state_dict()),set(shared))
        torch.testing.assert_close(legacy(data)['x_hat_main'],outputs['none']['x_hat_main'],rtol=0,atol=0)

    def test_backend_invalid_alpha_and_sparse_blend_rejected(self):
        for patch in ({'completion_alpha':float('nan')},{'completion_alpha':True},
                      {'completion_alpha':1.1},{'completion_blend':'learned','completion_alpha':0.},
                      {'completion_top_k':2},{'completion_mode':'uniform'},
                      {'backend_diagnostics':'yes'}):
            cfg=target_config()
            cfg['model']['dual_moe'].update(completion_blend='fixed',completion_alpha=.5,completion_top_k=3)
            cfg['model']['dual_moe'].update(patch)
            with self.assertRaises(ValueError): DualBranchSTImputer.from_config(cfg)

    def test_backend_diagnostics_exact_bins_ties_and_batch_partition(self):
        from stmoe_imputer.routing_metrics import BackendDiagnosticAccumulator
        target=torch.zeros(3,2,1,3,3)
        mask=torch.ones(3,1,1,3,3);mask[0]=0;mask[1,:,:,1,1]=0
        preds={'fine':target+1,'mid':target+2,'coarse':target+3}
        out={'scale_predictions':preds,'x_hat_main':target+2,
             'completion_gates':torch.ones(3,3,1,3,3)/3,'completion_alpha':torch.tensor(.5)}
        data={'m_f':mask,'x_f_gt':target}
        whole=BackendDiagnosticAccumulator();whole.update(out,data)
        split=BackendDiagnosticAccumulator()
        for i in range(3):
            o={k:({s:v[i:i+1] for s,v in value.items()} if isinstance(value,dict)
                  else value[i:i+1] if value.ndim else value) for k,value in out.items()}
            split.update(o,{k:v[i:i+1] for k,v in data.items()})
        a,b=whole.compute(),split.compute()
        self.assertEqual(a,b)
        self.assertEqual(a['backend_diag_all_count'],20)
        self.assertEqual(a['backend_diag_low_count'],18)
        self.assertEqual(a['backend_diag_high_count'],2)
        self.assertEqual(a['backend_diag_medium_count'],0)
        self.assertNotIn('backend_diag_medium_mae',a)
        self.assertEqual(a['backend_diag_all_mae'],2)
        self.assertEqual(a['backend_diag_all_rmse'],2)
        self.assertEqual(a['backend_diag_all_fine_win_fraction'],1)
        self.assertEqual(a['backend_diag_fine_mid_corr_defined'],0)
        self.assertNotIn('backend_diag_fine_mid_abs_error_corr',a)
        self.assertEqual(a['backend_diag_alpha_mean'],.5)
        out['scale_predictions']={s:target+1 for s in preds}
        tie=BackendDiagnosticAccumulator();tie.update(out,data)
        self.assertAlmostEqual(tie.compute()['backend_diag_all_fine_win_fraction'],1/3,places=6)

    def test_backend_four_controls_train_validate_restore_test_and_log(self):
        for name,mode,blend in [('U','uniform','none'),('D','topk','none'),('H','topk','fixed'),('L','topk','learned')]:
            cfg=target_config();cfg['train']['amp']=False
            cfg['model']['dual_moe'].update(completion_mode=mode,completion_top_k=3,
                completion_blend=blend,completion_alpha=.5,backend_diagnostics=True)
            cfg['loss']['dual_moe_completion_balance_weight']=0.
            model=DualBranchSTImputer.from_config(cfg);optimizer=build_optimizer(model,cfg)
            splits=[batch(h=4,w=5) for _ in range(3)]
            train=train_one_epoch(model,[splits[0]],optimizer,torch.device('cpu'),cfg,1)
            valid=evaluate(model,[splits[1]],torch.device('cpu'),cfg)
            state=snapshot_model_state(model)
            with torch.no_grad(): next(model.parameters()).add_(1.)
            model.load_state_dict(state)
            test=evaluate(model,[splits[2]],torch.device('cpu'),cfg)
            self.assertIn('backend_diag_grad_fine_expert',train)
            for result in (train,valid,test):
                self.assertTrue(all(math.isfinite(v) for v in result.values()))
                self.assertIn('backend_diag_low_count',result)
            if name=='L': self.assertIn('backend_diag_grad_completion_alpha_logit',train)
            stream=io.StringIO();TrainLogger._log_dual_moe(stream,valid)
            self.assertIn('backend_diag_alpha_mean',stream.getvalue())

    @unittest.skipUnless(os.environ.get('STABILITY_GPU_SMOKE')=='1' and torch.cuda.is_available(),
                         'Opt-in single-GPU three-point real-data AMP smoke, not formal results')
    def test_backend_stability_real_points_amp(self):
        import numpy as np
        from contextlib import redirect_stderr, redirect_stdout
        sys.path.insert(0,str(ROOT/'scripts'))
        import run_scale_completion_experiments as runner
        policy=load_config(ROOT/'configs/presets/dual_moe_backend_stability.json')
        for point in policy['points']:
            dataset,rate=point['dataset'],point['rate']
            folder,prefix,base=runner.common.SPECS[dataset]
            splits={}
            for split in ('train','val','test'):
                x,_,_=runner.common.read_windows(ROOT/f'data/{folder}/{prefix}_{split}.npz',1)
                x=torch.from_numpy(x).float()
                m=np.loadtxt(ROOT/f'data/{folder}/random_mask/{rate:g}/{split}.csv',delimiter=',',max_rows=1,ndmin=2)
                t,h,w=x.shape[2:]
                mask=torch.from_numpy(m.copy()).float().reshape(1,1,1 if m.size==h*w else t,h,w).expand(1,1,t,h,w)
                splits[split]={'x_f_gt':x,'x_f_obs':x*mask,'m_f':mask}
            for name,spec in policy['variants'].items():
                cfg=deep_update(load_config(ROOT/f'configs/datasets/{base}.json'),load_config(ROOT/'configs/presets/dual_moe_target.json'))
                cfg=deep_update(cfg,spec['patch']);cfg['train']['amp']=True
                torch.manual_seed(42)
                model=DualBranchSTImputer.from_config(cfg).cuda();optimizer=build_optimizer(model,cfg)
                best=float('inf');state=None
                with redirect_stderr(io.StringIO()),redirect_stdout(io.StringIO()):
                    for epoch in (1,2):
                        train=train_one_epoch(model,[splits['train']],optimizer,torch.device('cuda'),cfg,epoch)
                        val=evaluate(model,[splits['val']],torch.device('cuda'),cfg)
                        if val['mae']<best: best=val['mae'];state=snapshot_model_state(model)
                    model.load_state_dict(state)
                    test=evaluate(model,[splits['test']],torch.device('cuda'),cfg)
                for logs in (train,val,test):
                    self.assertTrue(all(math.isfinite(v) for v in logs.values()))
                    self.assertIn('backend_diag_alpha_mean',logs)
                    self.assertEqual(logs['backend_diag_all_count'],logs['metric_missing_count'])
                print(f'[stability AMP smoke] {dataset} random@{rate} {name}: train/val/best/test finite',flush=True)
                del model,optimizer,state
                torch.cuda.empty_cache()

    def test_uniform_target_exactly_matches_existing_e8_uniform(self):
        cfg = target_config(); cfg['model']['dual_moe']['aggregation_mode'] = 'uniform'
        cfg['loss']['dual_moe_aggregation_balance_weight'] = 0.
        legacy = copy.deepcopy(cfg); legacy['model']['dual_moe']['design'] = 'learned_regions_v2'
        torch.manual_seed(42); old = DualBranchSTImputer.from_config(legacy).eval()
        torch.manual_seed(42); new = DualBranchSTImputer.from_config(cfg).eval()
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
        data = batch(n=1); a, b = old(data), new(data)
        for key in ('x_hat_final', 'completion_gates'):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        for key in ('aggregation_support', 'aggregation_mass', 'scale_predictions'):
            for scale in a[key]: torch.testing.assert_close(a[key][scale], b[key][scale], rtol=0, atol=0)

    def test_target_gates_cannot_truncate_observation_pooling(self):
        cfg = target_config(); model = DualBranchSTImputer.from_config(cfg).eval(); data = batch(n=1)
        a = model(data)
        for router in model.main_branch.readout_routers.values():
            with torch.no_grad(): router.score.weight.mul_(-300)
            router.top_k = 2
        b = model(data)
        for scale in ('mid', 'coarse'):
            for key in ('region_assignments', 'aggregation_mass', 'aggregation_effective_count', 'aggregation_support'):
                torch.testing.assert_close(a[key][scale], b[key][scale], rtol=0, atol=0)
            self.assertGreater(float((a['aggregation_gates'][scale]-b['aggregation_gates'][scale]).abs().sum()), 0)
            # E expert masses together conserve observed count (each uses 1/E).
            torch.testing.assert_close(a['aggregation_mass'][scale].sum(), data['m_f'].sum())

    def test_router_is_target_local_and_expert_permutation_equivariant(self):
        torch.manual_seed(7)
        router = TargetReadoutRouter(8, 4, 8, 'topk', 4)
        fine = torch.randn(1, 8, 2, 3, 4)
        candidates = torch.randn(1, 8, 2, 12, 8)
        support = torch.rand(1, 8, 2, 12, 3)
        original = router(fine, candidates, support)
        perm = torch.tensor([7, 2, 4, 0, 1, 5, 3, 6])
        torch.testing.assert_close(router(fine, candidates[:, perm], support[:, perm]), original[:, perm])
        changed = fine.clone(); changed[:, :, 0, 1, 2] += 10*torch.arange(8)
        revised = router(changed, candidates, support)
        self.assertGreater(float((original[:, :, 0, 1, 2]-revised[:, :, 0, 1, 2]).abs().sum()), 0)
        revised[:, :, 0, 1, 2] = original[:, :, 0, 1, 2]
        torch.testing.assert_close(revised, original, rtol=0, atol=0)

    def test_shapes_gradients_no_hidden_target_leak_and_finite_masks(self):
        for c, t, h, w in [(2,12,32,32), (2,12,24,12), (1,7,32,32), (1,2,7,11), (1,1,1,1)]:
            cfg = target_config(c); model = DualBranchSTImputer.from_config(cfg)
            data = batch(c,t,h,w,n=1); out = model(data)
            self.assertEqual(out['x_hat_final'].shape, data['x_f_gt'].shape)
            loss, _ = compute_main_stage_loss(out, data, cfg); loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
            for scale in ('mid','coarse'):
                self.assertTrue(((out['aggregation_gates'][scale]>0).sum(1) == 4).all())
            altered = copy.deepcopy(data); altered['x_f_gt'].fill_(float('nan'))
            altered['x_f_obs'] = torch.where(data['m_f'].bool(), data['x_f_obs'], torch.full_like(data['x_f_obs'],float('nan')))
            torch.testing.assert_close(out['x_hat_final'], model(altered)['x_hat_final'], rtol=0, atol=0)
        for observed in (0,1):
            data = batch(n=1); data['m_f'].fill_(observed); data['x_f_obs'] = data['x_f_gt']*observed
            cfg = target_config(); model = DualBranchSTImputer.from_config(cfg)
            loss, logs = compute_main_stage_loss(model(data), data, cfg)
            self.assertTrue(torch.isfinite(loss)); loss.backward()
            if observed: self.assertEqual(float(loss), 0.)

    def test_query_and_backend_receive_task_gradients_without_balance(self):
        cfg = target_config(); cfg['loss'].update(dual_moe_aggregation_balance_weight=0., dual_moe_completion_balance_weight=0.)
        model = DualBranchSTImputer.from_config(cfg); data = batch(n=1)
        loss, _ = compute_main_stage_loss(model(data), data, cfg); loss.backward()
        for router in model.main_branch.readout_routers.values():
            for layer in (router.query, router.key, router.score):
                self.assertGreater(float(layer.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.main_branch.completion_router.head.weight.grad.abs().sum()), 0)

    def test_balance_and_logging_count_missing_targets_not_sources(self):
        cfg = target_config(); model = DualBranchSTImputer.from_config(cfg)
        data = batch(n=1); out = model(data)
        _, logs = compute_main_stage_loss(out, data, cfg)
        expected = torch.stack([token_topk_balance_loss(out['routing_details'][f'aggregation_{s}'], 1-data['m_f']) for s in ('mid','coarse')]).mean()
        torch.testing.assert_close(logs['l_balance_aggregation'], expected)
        acc = DualMoEMetricAccumulator(); acc.update(out, data); metrics = acc.compute()
        for name in ('aggregation_mid','aggregation_coarse'):
            self.assertEqual(metrics[f'topk_{name}_token_count'], float((1-data['m_f']).sum()))
            self.assertAlmostEqual(metrics[f'topk_{name}_selected_per_token'], 4.)
            self.assertNotIn(f'{name}_observed_e0_mean', metrics)
        buffer = io.StringIO(); TrainLogger._log_dual_moe(buffer, metrics)
        self.assertIn('aggregation(missing,components)', buffer.getvalue())

    def test_invalid_router_options(self):
        for mode, k in [('invalid',4),('topk',0),('topk',9),('topk',True)]:
            with self.assertRaises(ValueError): TargetReadoutRouter(8,4,8,mode,k)

    def test_launcher_four_controls_keep_budgets_and_disable_uniform_balance(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        import train_scale_completion as trainer
        for front in ('uniform','topk'):
            for back in ('uniform','topk'):
                cfg = target_config()
                # Read temporary resolved config at subprocess launch, but do
                # not start training or depend on local data in this unit test.
                captured = []
                def launch(command, **kwargs):
                    captured.append((command,json.loads(Path(command[command.index('-c')+1]).read_text())))
                argv = ['train_scale_completion.py','--preset','dual_moe_target','--dataset','TaxiBJ',
                        '--mask','random','--front-mode',front,'--back-mode',back]
                with mock.patch.object(sys,'argv',argv), mock.patch.object(trainer,'build_config',return_value=(cfg,{})), mock.patch.object(trainer.subprocess,'run',side_effect=launch), mock.patch('sys.stdout',new_callable=io.StringIO):
                    trainer.main()
                command, actual = captured[0]
                self.assertEqual(command[command.index('--name')+1],f'ablation_Q{int(front=="topk")}{int(back=="topk")}')
                self.assertEqual(actual['train'],cfg['train'])
                self.assertFalse(actual['train']['save_best_checkpoint'])
                self.assertEqual(actual['loss']['dual_moe_aggregation_balance_weight'],.001 if front=='topk' else 0.)
                self.assertEqual(actual['loss']['dual_moe_completion_balance_weight'],.01 if back=='topk' else 0.)

    def test_real_training_entry_memory_best_and_logs(self):
        cfg = target_config()
        with tempfile.TemporaryDirectory() as tmp:
            cfg['output_dir'] = tmp; path = Path(tmp)/'config.json'; path.write_text(json.dumps(cfg))
            result = subprocess.run([sys.executable,str(ROOT/'scripts/train.py'),'-c',str(path),'--synthetic','--no_plot','--quiet','--name','ablation_query_smoke'],cwd=ROOT,capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout[-2000:]+result.stderr[-2000:])
            self.assertFalse(list(Path(tmp).rglob('*.pt')))
            log = next(Path(tmp).rglob('metrics.jsonl'))
            records = [json.loads(line) for line in log.read_text().splitlines()]
            tests = [r for r in records if r.get('stage')=='test']; self.assertEqual(len(tests),1)
            self.assertTrue(tests[0]['extra']['best_weights_restored'])
            self.assertEqual(tests[0]['extra']['best_model_source'],'memory')
            for name in ('train.log','val.log','test.log'): self.assertTrue((log.parent/name).is_file())
            self.assertIn('aggregation(missing,components)', (log.parent/'val.log').read_text())

    @unittest.skipUnless(os.environ.get('TARGET_REAL_TEST')=='1' and torch.cuda.is_available(), 'Opt-in real three-dataset CUDA AMP train/val/best/test')
    def test_real_three_dataset_fixed_random(self):
        import numpy as np
        for folder, prefix, c in [('TaxiBJ','taxibj',2),('BikeNYC','bikenyc',2),('CHAP/beijing','chap_beijing',1)]:
            values = {}
            for split in ('train','val','test'):
                with np.load(ROOT/f'data/{folder}/{prefix}_{split}.npz',allow_pickle=False) as f:
                    values[split] = torch.from_numpy(f['x_f_gt'][:1].copy()).float()
            for pattern in ('fixed','random'):
                cfg = deep_update(config(c),load_config(ROOT/'configs/presets/dual_moe_target.json'))
                model = DualBranchSTImputer.from_config(cfg).cuda(); optimizer = build_optimizer(model,cfg); splits = {}
                for split,x in values.items():
                    a = np.loadtxt(ROOT/f'data/{folder}/{pattern}_mask/0.4/{split}.csv',delimiter=',',max_rows=1,ndmin=2)
                    _,_,t,h,w = x.shape
                    mask = torch.from_numpy(a.copy()).float().reshape(1,1,1 if a.size==h*w else t,h,w).expand(1,1,t,h,w)
                    splits[split] = {k:v.cuda() for k,v in {'x_f_gt':x,'x_f_obs':x*mask,'m_f':mask}.items()}
                best = float('inf'); state = None
                for epoch in (1,2):
                    train = train_one_epoch(model,[splits['train']],optimizer,torch.device('cuda'),cfg,epoch)
                    val = evaluate(model,[splits['val']],torch.device('cuda'),cfg)
                    if val['mae'] < best: best = val['mae']; state = snapshot_model_state(model)
                model.load_state_dict(state)
                test = evaluate(model,[splits['test']],torch.device('cuda'),cfg)
                for logs in (train,val,test):
                    self.assertTrue(all(math.isfinite(v) for v in logs.values()))
                    self.assertAlmostEqual(logs['topk_aggregation_mid_selected_per_token'],4)
                    self.assertAlmostEqual(logs['topk_completion_selected_per_token'],2)
                print(f'[target smoke] {folder} {pattern}: finite train/val/best-memory/test',flush=True)


class SharedTopKCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def cfg(self, channels=2):
        return deep_update(config(channels),load_config(ROOT/'configs/presets/dual_moe_shared_topk.json'))

    def test_eight_routed_plus_shared_and_exact_top_three(self):
        cfg=self.cfg();model=DualBranchSTImputer.from_config(cfg)
        self.assertEqual(len(model.main_branch.completion_experts),8)
        data=batch(n=1,h=4,w=5);out=model(data);g=out['completion_gates']
        self.assertEqual(g.shape,(1,8,3,4,5))
        self.assertTrue(((g>0).sum(1)==3).all())
        torch.testing.assert_close(g.sum(1),torch.ones_like(g[:,0]))
        # Each routed candidate is shared+r_i; normalized routed weights sum1,
        # so the shared base participates exactly once, not once per expert.
        expected=sum(g[:,i:i+1]*out['completion_routed_predictions'][f'e{i}'] for i in range(8))
        torch.testing.assert_close(out['x_hat_main'],expected)
        self.assertEqual(out['routing_details']['completion']['probabilities'].shape[1],8)
        self.assertEqual(set(out['scale_predictions']),{'fine','mid','coarse'})
        loss,logs=compute_main_stage_loss(out,data,cfg,1)
        expected_balance=token_topk_balance_loss(out['routing_details']['completion'],1-data['m_f'])
        torch.testing.assert_close(logs['l_balance_completion'],expected_balance.detach())
        torch.testing.assert_close(logs['l_balance_completion_weighted'],.001*expected_balance.detach())
        self.assertTrue(torch.isfinite(loss))

    def test_only_selected_experts_receive_main_gradient_shared_always_does(self):
        model=DualBranchSTImputer.from_config(self.cfg())
        router=model.main_branch.completion_router
        with torch.no_grad():
            router.head.weight.zero_();router.head.bias.copy_(torch.arange(8,dtype=torch.float))
        out=model(batch(n=1,h=4,w=5));out['x_hat_main'].square().mean().backward()
        for i,expert in enumerate(model.main_branch.completion_experts):
            grad=sum(float(p.grad.abs().sum())for p in expert.parameters())
            self.assertGreater(grad,0) if i>=5 else self.assertEqual(grad,0)
        self.assertGreater(sum(float(p.grad.abs().sum())for p in model.main_branch.completion_shared.parameters()),0)
        self.assertGreater(float(router.head.bias.grad.abs().sum()),0)

    def test_shared_frontend_initialization_and_no_target_leakage(self):
        cfg=self.cfg();old=copy.deepcopy(cfg)
        old['model']['dual_moe'].update(completion_layout='scale',completion_experts=3)
        torch.manual_seed(42);reference=DualBranchSTImputer.from_config(old)
        torch.manual_seed(42);model=DualBranchSTImputer.from_config(cfg)
        for name,value in reference.state_dict().items():
            if 'completion_router.' not in name:
                torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
        data=batch(n=1,h=4,w=5);expected=model(data)['x_hat_main']
        data['x_f_gt'].fill_(float('nan'))
        data['x_f_obs']=torch.where(data['m_f'].bool(),data['x_f_obs'],float('nan'))
        torch.testing.assert_close(model(data)['x_hat_main'],expected,rtol=0,atol=0)

    def test_shared_metrics_labels_and_train_val_memory_restore(self):
        cfg=self.cfg();cfg['train']['amp']=False
        model=DualBranchSTImputer.from_config(cfg);opt=build_optimizer(model,cfg)
        data=batch(n=1,h=4,w=5)
        train=train_one_epoch(model,[data],opt,torch.device('cpu'),cfg,1)
        state=snapshot_model_state(model)
        with torch.no_grad():model.main_branch.completion_shared.net[-1].bias.add_(10)
        model.load_state_dict(state)
        val=evaluate(model,[data],torch.device('cpu'),cfg)
        self.assertIn('backend_diag_grad_completion_shared',train)
        self.assertIn('backend_diag_grad_completion_experts',train)
        for result in (train,val):
            self.assertTrue(all(math.isfinite(v) for v in result.values()))
            self.assertAlmostEqual(result['topk_completion_selected_per_token'],3)
            self.assertEqual(result['completion_shared_always_active'],1)
            self.assertNotIn('completion_missing_fine_mean',result)
            self.assertNotIn('backend_diag_alpha_mean',result)
            for i in range(8):
                self.assertIn(f'completion_missing_e{i}_mean',result)
                self.assertIn(f'mae_completion_e{i}_with_shared',result)
                self.assertIn(f'backend_diag_all_e{i}_win_fraction',result)
            self.assertAlmostEqual(sum(result[f'backend_diag_all_e{i}_win_fraction']for i in range(8)),1,places=5)
        stream=io.StringIO();TrainLogger._log_dual_moe(stream,val)
        self.assertIn('routed experts',stream.getvalue())
        self.assertIn('always on',stream.getvalue())
        self.assertIn('e7=',stream.getvalue())

    def test_shared_shapes_masks_and_invalid_options(self):
        for c,t,h,w in [(2,12,32,32),(2,12,24,12),(1,7,32,32),(1,1,1,1)]:
            cfg=self.cfg(c);model=DualBranchSTImputer.from_config(cfg)
            data=batch(c,t,h,w,n=1);out=model(data)
            self.assertEqual(out['x_hat_main'].shape,data['x_f_gt'].shape)
            loss,_=compute_main_stage_loss(out,data,cfg,1);loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
        for observed in (0.,1.):
            cfg=self.cfg();model=DualBranchSTImputer.from_config(cfg)
            data=batch(n=1,h=3,w=3);data['m_f'].fill_(observed);data['x_f_obs']=data['x_f_gt']*observed
            out=model(data);loss,_=compute_main_stage_loss(out,data,cfg,1)
            self.assertTrue(torch.isfinite(loss))
            if observed:self.assertEqual(float(loss),0)
        for patch in [{'completion_experts':True},{'completion_top_k':9},{'completion_top_k':0},
                      {'completion_blend':'learned'},{'completion_mode':'uniform'},
                      {'design':'learned_regions_v2'},{'completion_layout':'scale'}]:
            cfg=self.cfg();cfg['model']['dual_moe'].update(patch)
            with self.assertRaises(ValueError):DualBranchSTImputer.from_config(cfg)


if __name__ == '__main__': unittest.main()
