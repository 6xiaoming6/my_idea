import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

from test_dual_moe import ROOT
from test_recoverability import recovery_config
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.engine import train_one_epoch, evaluate, build_optimizer
from stmoe_imputer.utils.checkpoint import snapshot_model_state

sys.path.insert(0, str(ROOT/'scripts'))
import run_recoverability_experiments as study


class RecoveryExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def policy(self):
        return study.common.load(ROOT/'configs/presets/dual_moe_recoverability_experiments.json')

    def test_policy_full_crossing_no_budget_or_mask_confounds(self):
        p = self.policy(); study.validate(p)
        jobs = list(study.jobs(p, Path('/tmp/recovery_test')))
        self.assertEqual(len(jobs), 60)
        self.assertEqual(len({j['key'] for j in jobs}), 60)
        study.infrastructure.data_manifest(jobs)
        for j in jobs:
            cfg = j['cfg']; options = cfg['model']['dual_moe']
            self.assertFalse(cfg['train']['save_best_checkpoint'])
            self.assertFalse(cfg['train']['early_stopping']['enabled'])
            self.assertEqual(cfg['train']['epochs'], p['dataset_epochs'][j['dataset']])
            self.assertEqual(j['expected_train_samples'], {'TaxiBJ':2491, 'BikeNYC':511, 'CHAP':2626}[j['dataset']])
            self.assertEqual((options['completion_experts'],options['completion_top_k']), (8,3))
            self.assertEqual(options['completion_expert_type'], 'st_dilated')
            self.assertNotIn('training_selection', cfg['data'])
            self.assertEqual(cfg['loss']['dual_moe_recoverability_weight'], 0. if j['mode']=='off' else .01)
        bad = copy.deepcopy(p); bad['points'].pop()
        with self.assertRaises(ValueError): study.validate(bad)

    def test_new_single_run_preset_matches_D(self):
        p = self.policy(); j = next(j for j in study.jobs(p, Path('/tmp/recovery_test')) if j['variant']=='D_RECOVERY')
        cfg, paths = study.trainer.build_config('TaxiBJ','fixed',.4,'learned_regions',preset='dual_moe_recoverability')
        self.assertEqual(cfg['model'], j['cfg']['model'])
        self.assertEqual(cfg['loss'], j['cfg']['loss'])
        self.assertEqual(cfg['train'], j['cfg']['train'])

    def test_tonight_budget_keeps_five_controls_and_full_sources(self):
        raw = self.policy(); saved = copy.deepcopy(raw)
        full = study.resolve_profile(raw)
        p = study.resolve_profile(raw, 'tonight'); study.validate(p)
        self.assertEqual(raw, saved)
        self.assertEqual(full['dataset_epochs'], {'TaxiBJ':140,'BikeNYC':100,'CHAP':150})
        self.assertEqual(p['dataset_epochs'], {'TaxiBJ':40,'BikeNYC':60,'CHAP':50})
        self.assertEqual(p['datasets'], ['BikeNYC','CHAP','TaxiBJ'])
        jobs = list(study.jobs(p, Path('/tmp/recovery_tonight_test')))
        self.assertEqual(len(jobs), 30)
        self.assertEqual({j['variant'] for j in jobs}, set(study.VARIANTS))
        self.assertEqual({(j['pattern'],j['rate']) for j in jobs}, {('fixed',.8),('random',.8)})
        study.infrastructure.data_manifest(jobs)
        for j in jobs:
            self.assertEqual(j['expected_train_samples'], {'TaxiBJ':2491,'BikeNYC':511,'CHAP':2626}[j['dataset']])
            self.assertEqual(j['cfg']['train']['epochs'], p['dataset_epochs'][j['dataset']])
            self.assertEqual(j['cfg']['train']['val_epoch'], 2)
            self.assertNotIn('training_selection', j['cfg']['data'])

    def test_advisory_target_buffer_and_full_profile_without_target(self):
        target = study.deadline_time('2026-09-15 22:00')
        self.assertTrue(study.meets_deadline(target-timedelta(minutes=30), target, 30))
        self.assertFalse(study.meets_deadline(target-timedelta(minutes=29), target, 30))
        self.assertTrue(study.meets_deadline(target, None, 30))
        self.assertIsNone(study.deadline_time(None))

    def test_expired_target_and_late_ETA_still_launch_formal_training(self):
        # Mock GPU work, exercise the real queue control flow: even a target
        # already in the past and an over-budget estimate must not block launch.
        with tempfile.TemporaryDirectory() as tmp:
            p = study.resolve_profile(self.policy(), 'tonight')
            p['output_dir'] = tmp
            p['target_time'] = '2000-01-01 22:00'
            original_jobs = study.jobs
            finished = set()
            def launch(job, suite): finished.add(job['key'])
            def calibrated(policy, suite, jobs):
                return {f'{j["dataset"]}/{j["variant"]}': {'seconds_per_run':100000} for j in jobs}
            with mock.patch.dict(os.environ), \
                 mock.patch.object(sys,'argv',['runner','--gpu','0']), \
                 mock.patch.object(study.common,'load',return_value=p), \
                 mock.patch.object(study,'identity',return_value={'fixture':'advisory'}), \
                 mock.patch.object(study,'jobs',side_effect=lambda policy,suite: [{**next(original_jobs(policy,suite)), 'expected_train_samples':2}]), \
                 mock.patch.object(study.infrastructure,'data_manifest',return_value={}), \
                 mock.patch.object(study,'geometry_check',return_value={'note':'fixture','rows':[]}), \
                 mock.patch.object(study,'completed',side_effect=lambda j: True if j['key'] in finished else None), \
                 mock.patch.object(study,'summarize',return_value=[]), \
                 mock.patch.object(study.runner,'calibrate',side_effect=calibrated), \
                 mock.patch.object(study.runner,'launch',side_effect=launch) as called:
                # Mock loading only for the outer config; the job builder needs
                # a minimal resolved configuration rather than disk preset reads.
                cfg = {'train':{'epochs':60},'data':{},'model':{},'loss':{}}
                with mock.patch.object(study.trainer,'build_config',return_value=(cfg,{})):
                    study.main()
                called.assert_called_once()
            schedule = json.loads(next(Path(tmp).rglob('schedule.json')).read_text())
            self.assertEqual(schedule['deadline_mode'],'advisory')
            self.assertFalse(schedule['within_target'])
            self.assertTrue(schedule['accepted'])

    def test_geometry_equal_counts_and_common_queries(self):
        report = study.geometry_check(.05)
        rows = {r['pattern']:r for r in report['rows']}
        self.assertEqual({r['observed_count'] for r in rows.values()}, {12})
        self.assertEqual(len({r['common_missing_count'] for r in rows.values()}), 1)
        self.assertGreater(rows['spread']['common_missing_count'], 0)
        self.assertEqual(rows['spread']['gram_rank'], 4)
        self.assertEqual(rows['spatial_line']['gram_rank'], 3)
        self.assertEqual(rows['single_time']['gram_rank'], 3)
        self.assertGreater(rows['spatial_line']['common_missing_rmse'], rows['spread']['common_missing_rmse'])

    def test_summary_pairs_and_missing_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp); jobs = list(study.jobs(self.policy(),suite))
            def result(j):
                if (j['dataset'],j['pattern'],j['rate']) != ('BikeNYC','random',.8): return None
                value = 1. if j['variant']=='D_RECOVERY' else 2.
                return dict(val_mae=value,test_mae=value,test_rmse=value,best_epoch=2,run_dir='fixture',
                            best_val_metrics={'rmse':value},test_metrics={})
            with mock.patch.object(study,'completed',side_effect=result):
                rows = study.summarize(jobs,suite)
            paired = study.common.load(suite/'comparison.json')['paired']
            self.assertEqual(sum(r['status']=='complete' for r in rows), 5)
            self.assertEqual(len(paired), len(study.PAIRS))
            self.assertTrue(all((x['dataset'],x['pattern'],x['rate'])==('BikeNYC','random',.8) for x in paired))
            self.assertEqual(paired[0]['test_mae_pct'], -50)
            j = jobs[0]
            with mock.patch.object(study.runner,'completed') as underlying:
                self.assertIsNone(study.completed(j)); underlying.assert_not_called()
            audit = suite/'logs'/f'{j["key"]}.status.json'
            study.common.write_json(audit, {'status':'running'})
            with mock.patch.object(study.runner,'completed') as underlying:
                self.assertIsNone(study.completed(j)); underlying.assert_not_called()

    def test_real_three_dataset_CPU_one_window_train_val_restore_test(self):
        """Real split/mask compatibility only, never a formal accuracy result."""
        import numpy as np
        for dataset in ('TaxiBJ','BikeNYC','CHAP'):
            full, paths = study.trainer.build_config(dataset,'random',.8,'learned_regions',preset='dual_moe_recoverability')
            data = []
            for split in ('train','val','test'):
                x, _, _ = study.common.read_windows(paths[split], 1)
                x = torch.from_numpy(x).float()
                raw = np.loadtxt(full['data']['mask'][f'{split}_csv'], delimiter=',', max_rows=1, ndmin=2)
                t,h,w = x.shape[2:]
                mask = torch.from_numpy(raw.copy()).float().reshape(1,1,1 if raw.size==h*w else t,h,w).expand(1,1,t,h,w)
                data.append({'x_f_gt':x, 'x_f_obs':torch.where(mask.bool(),x,0.), 'm_f':mask})
            # Exact main model sizes; only batch=1 and epoch=1 for CPU smoke.
            cfg = full; cfg['train']['amp']=False; cfg['model']['dual_moe']['backend_diagnostics']=False
            model = DualBranchSTImputer.from_config(cfg)
            optimizer = build_optimizer(model,cfg)
            train = train_one_epoch(model,[data[0]],optimizer,torch.device('cpu'),cfg,1)
            val = evaluate(model,[data[1]],torch.device('cpu'),cfg)
            state = snapshot_model_state(model)
            model.load_state_dict(state)
            test = evaluate(model,[data[2]],torch.device('cpu'),cfg)
            for metrics in (train,val,test):
                for key in ('loss','mae','rmse','recovery_mid_all_branch_mae'):
                    self.assertTrue(np.isfinite(metrics[key]), (dataset,key))

    def test_training_CLI_logs_and_verified_resume(self):
        # Exercise the real entry point and completion audit, in a disposable
        # synthetic run; no checkpoint files, 3 epochs, VAL at 2 and final 3.
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)
            cfg = recovery_config()
            cfg.update(device='cpu', output_dir=str(suite/'runs'))
            cfg['data'].update(batch_size=2, num_workers=0)
            cfg['train'].update(epochs=3, val_epoch=2, save_best_checkpoint=False)
            path = suite/'input.json'; study.common.write_json(path,cfg)
            env = {**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'CUDA_VISIBLE_DEVICES':''}
            result = subprocess.run([sys.executable,str(ROOT/'scripts/train.py'),'-c',str(path),
                                     '--synthetic','--no_plot','--quiet','--name','ablation_D_RECOVERY_seed42'],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode,0,result.stdout[-2000:]+result.stderr[-2000:])
            run = next((suite/'runs').rglob('metrics.jsonl')).parent.parent
            records = [json.loads(line) for line in (run/'logs/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['epoch'] for r in records if r.get('val') is not None],[2,3])
            tests = [r for r in records if r.get('stage')=='test']
            self.assertEqual(len(tests),1)
            self.assertEqual(tests[0]['extra']['best_model_source'],'memory')
            self.assertTrue(tests[0]['extra']['best_weights_restored'])
            self.assertFalse(list((suite/'runs').rglob('*.pt')))
            for name in ('train','val','test'):
                self.assertIn('recovery_mid_all_branch_mae',(run/f'logs/{name}.log').read_text())
            # The actual saved config includes explicit synthetic overrides.
            job = dict(key='fixture', name='D_RECOVERY_seed42', mode='constrained',
                       cfg=study.common.load(run/'config.json'))
            self.assertIsNone(study.completed(job))
            audit = suite/'logs/fixture.status.json'
            study.common.write_json(audit,{'status':'verified'})
            self.assertIsNotNone(study.completed(job))


if __name__ == '__main__': unittest.main()
