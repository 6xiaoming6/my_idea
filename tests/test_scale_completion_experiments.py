import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_scale_completion_experiments as runner


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.p = runner.common.load(ROOT/'configs/presets/scale_completion_experiments.json')

    def test_backend_stability_36_matched_fresh_jobs(self):
        p=runner.common.load(ROOT/'configs/presets/dual_moe_backend_stability.json')
        runner.validate(p)
        jobs=list(runner.jobs(p,Path('/tmp/stability')))
        self.assertEqual(len(jobs),36)
        self.assertEqual(len({j['key'] for j in jobs}),36)
        self.assertEqual(runner.bind_reuse(p,jobs),[])
        self.assertEqual({j['seed'] for j in jobs},{42,2026,3407})
        for j in jobs:
            self.assertTrue(j['stability_study'])
            self.assertNotIn('confirmation_study',j)
            self.assertEqual(j['epochs'],100 if j['dataset']=='BikeNYC' else 140)
            self.assertEqual(j['cfg']['train']['val_epoch'],2)
            self.assertFalse(j['cfg']['train']['save_best_checkpoint'])
            self.assertEqual(j['expected_train_samples'],511 if j['dataset']=='BikeNYC' else 2491)
            self.assertIn(f'rate{j["rate"]:g}',j['cfg']['data']['mask']['train_csv'])
        reference=jobs[0]['cfg']
        for j in jobs[1:4]:
            self.assertEqual(j['cfg'],runner.common.merge(reference,p['variants'][j['variant']]['patch']))
        for key,value in [('seeds',[42]),('reuse_enabled',True),('points',p['points'][:1])]:
            q=copy.deepcopy(p);q[key]=value
            with self.assertRaises(ValueError): runner.validate(q)
        q=copy.deepcopy(p);q['variants']['H']['patch']['train']={'epochs':10}
        with self.assertRaises(ValueError): runner.validate(q)

    def test_backend_stability_summary_keeps_rates_and_missing_pairs_separate(self):
        p=runner.common.load(ROOT/'configs/presets/dual_moe_backend_stability.json')
        with tempfile.TemporaryDirectory() as tmp:
            suite=Path(tmp);jobs=list(runner.jobs(p,suite))
            with patch.object(runner,'completed',return_value=None):
                rows=runner.summarize(jobs,suite)
            self.assertEqual(len(rows),36)
            summary=runner.common.load(suite/'comparison.json')
            self.assertIn('H=0.5',summary['note'])
            self.assertEqual(summary['interaction'],[])
            self.assertEqual(summary['coverage_effects'],[])
            self.assertEqual(len(summary['groups']),12)
            self.assertEqual(len(summary['paired_groups']),18)
            self.assertTrue(all(g['status']=='incomplete' for g in summary['paired_groups']))
            self.assertEqual(len(runner.common.load(suite/'mechanism_summary.json')),36)

    @unittest.skipUnless(os.environ.get('STABILITY_E2E')=='1','Four real BikeNYC CPU train/VAL/best TEST jobs')
    def test_backend_stability_real_entrypoint_and_resume(self):
        # Deliberately tiny TRAIN for integration only; formal validation rejects
        # this reduced policy. Full real VAL/TEST exercise logging and selection.
        p=runner.common.load(ROOT/'configs/presets/dual_moe_backend_stability.json')
        p.update(datasets=['BikeNYC'],points=[{'dataset':'BikeNYC','rate':.4}],rates=[.4],
                 seeds=[42],train_windows=8,expected_train_samples={'BikeNYC':8},
                 dataset_epochs={'BikeNYC':2})
        with tempfile.TemporaryDirectory() as tmp:
            suite=Path(tmp);runner.common.prepare(p,suite)
            jobs=list(runner.jobs(p,suite))
            for job in jobs:
                job['cfg']['device']='cpu';job['cfg']['train']['amp']=False
                with patch('sys.stdout',new_callable=__import__('io').StringIO):
                    runner.launch(job,suite)
                result=runner.completed(job)
                self.assertIsNotNone(result)
                self.assertIn('backend_diag_alpha_mean',result['test_metrics'])
                log=Path(result['run_dir'])/'logs'
                for file in ('train.log','val.log','test.log'):
                    self.assertIn('backend_diag_', (log/file).read_text())
                self.assertEqual(result['test_metrics']['backend_diag_all_count'],
                                 result['test_metrics']['metric_missing_count'])
            self.assertEqual(len([j for j in jobs if runner.completed(j) is None]),0)
            self.assertFalse(list(suite.rglob('*.pt')))
            rows=runner.summarize(jobs,suite)
            self.assertEqual(sum(r['status']=='complete' for r in rows),4)
            self.assertEqual(len(runner.common.load(suite/'comparison.json')['paired']),6)

    def test_coverage_matched_budgets_and_hypotheses(self):
        runner.validate(self.p)
        jobs = list(runner.jobs(self.p, Path('/tmp/suite')))
        self.assertEqual(len(jobs), 78)
        self.assertEqual(len({j['key'] for j in jobs}), 78)
        for j in jobs:
            self.assertEqual(j['cfg']['train']['epochs'], 120)
            self.assertEqual(j['cfg']['train']['val_epoch'], 2)
            self.assertEqual(j['cfg']['data']['batch_size'], 4)
            self.assertEqual(j['cfg']['seed'], j['seed'])
            self.assertFalse(j['cfg']['train']['early_stopping']['enabled'])
            self.assertEqual(j['cfg']['model']['dual_moe']['aggregation_mode'], 'uniform')
            self.assertNotIn('/tmp/suite', str(j['paths']['test']))
        one = {j['variant']: j['cfg'] for j in jobs if j['seed'] == 42 and j['dataset'] == 'TaxiBJ' and j['pattern'] == 'fixed'}
        reference = copy.deepcopy(one['B00']); reference['loss']['dual_moe_expert_weight'] = .01
        self.assertEqual(reference, one['B01'])
        reference['model']['dual_moe']['design'] = 'anchored_scale_moe'
        self.assertEqual(reference, one['N01'])
        for name, patch in [('N00', {'loss': {'dual_moe_expert_weight': 0}}),
                            ('NU', {'model': {'dual_moe': {'completion_mode': 'uniform'}}}),
                            ('NB2', {'model': {'dual_moe': {'residual_bound': 2}}})]:
            self.assertEqual(runner.common.merge(one['N01'], patch), one[name])

    def test_invalid_budgets_and_unmatched_training_rejected(self):
        for key, value in [('epochs', 0), ('seeds', [42, 42]), ('ablation_seed', 123), ('rate', .7)]:
            p = copy.deepcopy(self.p); p[key] = value
            with self.assertRaises(ValueError): runner.validate(p)
        p = copy.deepcopy(self.p); p['variants']['N01']['patch']['train'] = {'epochs': 1}
        with self.assertRaises(ValueError): runner.validate(p)

    def test_target_study_three_datasets_four_controls_and_dataset_epochs(self):
        p = runner.common.load(ROOT/'configs/presets/dual_moe_target_comparison.json')
        runner.validate(p)
        jobs = list(runner.jobs(p,Path('/tmp/target_suite')))
        self.assertEqual(len(jobs),12)
        self.assertEqual(len({j['key'] for j in jobs}),12)
        for dataset, epochs in [('TaxiBJ',140),('BikeNYC',100),('CHAP',150)]:
            selected = [j for j in jobs if j['dataset']==dataset]
            self.assertEqual({j['variant'] for j in selected},{'Q00','Q01','Q10','Q11'})
            for job in selected:
                cfg = job['cfg']; options = cfg['model']['dual_moe']
                self.assertEqual(cfg['train']['epochs'],epochs)
                self.assertEqual(job['epochs'],epochs)
                self.assertEqual(options['design'],'target_readout_v1')
                self.assertEqual(options['aggregation_experts'],8)
                self.assertEqual(options['aggregation_top_k'],4)
                self.assertEqual(options['completion_top_k'],2)
                self.assertEqual(cfg['train']['val_epoch'],2)
                self.assertFalse(cfg['train']['save_best_checkpoint'])
                self.assertFalse(cfg['train']['early_stopping']['enabled'])
                self.assertEqual(job['expected_train_samples'],511 if dataset=='BikeNYC' else 2048)
                self.assertNotIn('target_suite',str(job['paths']['test']))
                self.assertNotIn('target_suite',str(job['paths']['val']))
            # No group-specific budget or non-router structural changes.
            reference = selected[0]['cfg']
            for job in selected:
                self.assertEqual(job['cfg'],runner.common.merge(reference,p['variants'][job['variant']]['patch']))
        self.assertIn('configs/presets/dual_moe_target.json',runner.identity(p)['code'])

    def test_target_study_rejects_ambiguous_epochs_and_unmatched_factors(self):
        ref = runner.common.load(ROOT/'configs/presets/dual_moe_target_comparison.json')
        for key,value in [('epochs',120),('dataset_epochs',{'TaxiBJ':140}),
                          ('dataset_epochs',{'TaxiBJ':True,'BikeNYC':100,'CHAP':150})]:
            p = copy.deepcopy(ref); p[key]=value
            with self.assertRaises(ValueError): runner.validate(p)
        for section,value in [('train',{'epochs':3}),('data',{'batch_size':2})]:
            p = copy.deepcopy(ref); p['variants']['Q11']['patch'][section]=value
            with self.assertRaises(ValueError): runner.validate(p)
        p=copy.deepcopy(ref); del p['variants']['Q00']
        with self.assertRaises(ValueError): runner.validate(p)

    def test_eta_uses_resolved_epochs_and_final_validation(self):
        counts = [10,3,4]
        # Five training epochs, VAL at 2,4,5, then one TEST.
        cfg = {'train':{'epochs':5,'val_epoch':2}}
        self.assertAlmostEqual(runner.estimate_run_seconds(cfg,counts,2.,1.,1.2),(5*10*2+(3*3+4))*1.2+30)
        cfg['train']['epochs']=7
        self.assertAlmostEqual(runner.estimate_run_seconds(cfg,counts,2.,1.,1.2),(7*10*2+(4*3+4))*1.2+30)

    def test_target_deadline_refuses_formal_launch(self):
        import io
        with tempfile.TemporaryDirectory() as tmp:
            p=runner.common.load(ROOT/'configs/presets/dual_moe_target_comparison.json')
            p['output_dir']=str(Path(tmp)/'suite')
            config=Path(tmp)/'policy.json'; runner.common.write_json(config,p)
            timing={f'{d}/{v}':{'seconds_per_run':3600} for d in p['datasets'] for v in p['variants']}
            argv=['runner','--config',str(config),'--deadline','2000-01-01 00:00']
            with patch.object(sys,'argv',argv), patch.object(runner.common,'prepare'), patch.object(runner,'completed',return_value=None), patch.object(runner,'calibrate',return_value=timing), patch.object(runner,'launch') as launch, patch('sys.stdout',new_callable=io.StringIO):
                with self.assertRaisesRegex(SystemExit,'exceeds deadline'): runner.main()
                launch.assert_not_called()

    def test_replication_only_adds_one_seed_with_identical_training_budgets(self):
        p=runner.common.load(ROOT/'configs/presets/dual_moe_target_replication.json')
        ref=runner.common.load(ROOT/'configs/presets/dual_moe_target_comparison.json')
        runner.validate(p)
        jobs=list(runner.jobs(p,Path('/tmp/shared_suite')))
        originals={(j['dataset'],j['variant']):j for j in runner.jobs(ref,Path('/tmp/shared_suite'))}
        self.assertEqual(len(jobs),8)
        self.assertEqual({j['seed'] for j in jobs},{2026})
        self.assertEqual({j['dataset'] for j in jobs},{'TaxiBJ','BikeNYC'})
        self.assertEqual(p['deadline_mode'],'advisory')
        for job in jobs:
            old=copy.deepcopy(originals[job['dataset'],job['variant']]['cfg'])
            old['seed']=2026
            self.assertEqual(job['cfg'],old)
            self.assertEqual(job['paths'],originals[job['dataset'],job['variant']]['paths'])
        invalid=copy.deepcopy(p);invalid['deadline_mode']='silently_truncate'
        with self.assertRaises(ValueError):runner.validate(invalid)

    def test_backend_study_preserves_frontend_and_matches_each_dataset_seed(self):
        p=runner.common.load(ROOT/'configs/presets/dual_moe_backend_study.json')
        runner.validate(p);jobs=list(runner.jobs(p,Path('/tmp/backend_suite')))
        self.assertEqual(len(jobs),12)
        self.assertTrue(all(j['dataset']=='BikeNYC' for j in jobs[:8]))
        self.assertTrue(all(j['dataset']=='TaxiBJ' and j['seed']==42 for j in jobs[8:]))
        for dataset,seeds in p['dataset_seeds'].items():
            for seed in seeds:
                group=[j for j in jobs if j['dataset']==dataset and j['seed']==seed]
                self.assertEqual({j['variant'] for j in group},{'BU','BD','BK0','BK1'})
                for job in group:
                    cfg=job['cfg'];opt=cfg['model']['dual_moe']
                    self.assertEqual(cfg,runner.common.merge(group[0]['cfg'],p['variants'][job['variant']]['patch']))
                    self.assertEqual(opt['aggregation_mode'],'topk')
                    self.assertEqual(opt['aggregation_experts'],8)
                    self.assertEqual(opt['aggregation_top_k'],4)
                    self.assertEqual(cfg['loss']['dual_moe_aggregation_balance_weight'],.001)
                    self.assertEqual(cfg['train']['epochs'],100 if dataset=='BikeNYC' else 140)
                    self.assertFalse(cfg['train']['save_best_checkpoint'])
                    self.assertEqual(job['expected_train_samples'],511 if dataset=='BikeNYC' else 1280)
                    self.assertNotIn('backend_suite',str(job['paths']['val']))
                    self.assertNotIn('backend_suite',str(job['paths']['test']))

    def test_backend_study_rejects_frontend_or_unmatched_budget_changes(self):
        ref=runner.common.load(ROOT/'configs/presets/dual_moe_backend_study.json')
        for field,value in [('dataset_seeds',{'BikeNYC':[42]}),('dataset_seeds',{'BikeNYC':[42,42],'TaxiBJ':[42]}),
                            ('dataset_seeds',{'BikeNYC':[42],'TaxiBJ':[42]})]:
            p=copy.deepcopy(ref);p[field]=value
            with self.assertRaises(ValueError):runner.validate(p)
        for section,value in [('model',{'dual_moe':{'aggregation_experts':4}}),('train',{'epochs':5})]:
            p=copy.deepcopy(ref);p['variants']['BK0']['patch'][section]=value
            with self.assertRaises(ValueError):runner.validate(p)
        p=copy.deepcopy(ref);p['variants']['BD']['patch']['loss']['dual_moe_completion_balance_weight']=.01
        with self.assertRaises(ValueError):runner.validate(p)

    def test_backend_summary_is_mechanism_pairs_not_two_by_two_interaction(self):
        p=runner.common.load(ROOT/'configs/presets/dual_moe_backend_study.json')
        with tempfile.TemporaryDirectory() as tmp:
            suite=Path(tmp);jobs=list(runner.jobs(p,suite))
            def fake(job):
                v={'BU':10.,'BD':9.,'BK0':9.5,'BK1':9.8}[job['variant']]
                return {'val_mae':v,'test_mae':v,'test_rmse':v*2,'best_epoch':2,'run_dir':'/fake'}
            with patch.object(runner,'completed',side_effect=fake):runner.summarize(jobs,suite)
            report=runner.common.load(suite/'comparison.json')
            self.assertEqual(len(report['paired']),18)
            self.assertEqual(len(report['paired_groups']),12)
            self.assertEqual(report['interaction'],[])
            for group in report['paired_groups']:
                self.assertEqual(group['expected_seeds'],[42,2026] if group['dataset']=='BikeNYC' else [42])
            with patch.object(runner,'completed',side_effect=lambda j:None if j['seed']==2026 and j['variant']=='BD' else fake(j)):
                runner.summarize(jobs,suite)
            report=runner.common.load(suite/'comparison.json')
            self.assertTrue(all(g['status']=='incomplete' for g in report['paired_groups'] if g['dataset']=='BikeNYC' and 'BD' in (g['candidate'],g['control'])))

    def test_backend_softmax_initialization_loss_and_forward_gradients(self):
        import torch
        from test_dual_moe import batch
        from stmoe_imputer.models import DualBranchSTImputer
        from stmoe_imputer.losses import compute_main_stage_loss
        torch.set_num_threads(1)
        p=runner.common.load(ROOT/'configs/presets/dual_moe_backend_study.json')
        jobs=[j for j in runner.jobs(p,Path('/tmp/backend_suite')) if j['dataset']=='BikeNYC' and j['seed']==42]
        states={};losses={};data=batch(n=1)
        for job in jobs:
            torch.manual_seed(42);model=DualBranchSTImputer.from_config(job['cfg'])
            states[job['variant']]={k:v.clone() for k,v in model.state_dict().items()}
            output=model(data);loss,logs=compute_main_stage_loss(output,data,job['cfg']);loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad))
            losses[job['variant']]=float(loss)
            for s in ('mid','coarse'):
                self.assertTrue((output['routing_details'][f'aggregation_{s}']['selected'].sum(1)==4).all())
            if job['variant']=='BU':
                self.assertNotIn('completion',output['routing_details'])
            else:
                route=output['routing_details']['completion']
                self.assertTrue((route['selected'].sum(1)==(3 if job['variant']=='BD' else 2)).all())
                if job['variant']=='BD':
                    torch.testing.assert_close(output['completion_gates'],route['probabilities'])
                    self.assertEqual(float(logs['l_balance_completion_weighted']),0.)
                if job['variant']=='BK1':self.assertGreater(float(logs['l_balance_completion_weighted']),0.)
        for key,value in states['BD'].items():
            for name in ('BK0','BK1'):torch.testing.assert_close(value,states[name][key],rtol=0,atol=0)
            if key!='main_branch.completion_router.head.weight':
                torch.testing.assert_close(value,states['BU'][key],rtol=0,atol=0)
        self.assertGreater(losses['BK1'],losses['BK0'])

    @unittest.skipUnless(os.environ.get('BACKEND_STUDY_E2E')=='1','Opt-in four BikeNYC short real backend jobs and resume')
    def test_backend_study_real_short_queue_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=runner.common.load(ROOT/'configs/presets/dual_moe_backend_study.json')
            p.update(datasets=['BikeNYC'],dataset_epochs={'BikeNYC':2},expected_train_samples={'BikeNYC':8},
                     dataset_seeds={'BikeNYC':[42]},seeds=[42],train_windows=8,output_dir=str(Path(tmp)/'suite'))
            p.pop('deadline');config=Path(tmp)/'policy.json';runner.common.write_json(config,p)
            command=[sys.executable,'-B',str(ROOT/'scripts/run_scale_completion_experiments.py'),'--config',str(config)]
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=240)
            self.assertEqual(result.returncode,0,result.stdout[-3000:]+result.stderr[-2000:])
            summary=next((Path(tmp)/'suite').glob('*/summary.json'));rows=runner.common.load(summary)
            self.assertEqual(len(rows),4);self.assertTrue(all(r['status']=='complete' for r in rows))
            self.assertFalse(list(Path(tmp).rglob('*.pt')))
            for row in rows:
                if row['variant']=='BD':self.assertEqual(row['test_metrics']['topk_completion_selected_per_token'],3.)
                if row['variant']=='BK0':self.assertEqual(row['test_metrics']['l_balance_completion_weighted'],0.)
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout[-2000:]+result.stderr[-1000:])
            self.assertEqual(result.stdout.count('SKIP complete'),4)

    def test_advisory_deadline_runs_all_jobs_without_changing_epochs(self):
        import io
        with tempfile.TemporaryDirectory() as tmp:
            p=runner.common.load(ROOT/'configs/presets/dual_moe_target_replication.json')
            p['output_dir']=str(Path(tmp)/'suite')
            config=Path(tmp)/'policy.json'; runner.common.write_json(config,p)
            timing={f'{d}/{v}':{'seconds_per_run':3600} for d in p['datasets'] for v in p['variants']}
            done=set(); budgets=[]
            def launch(job,suite):
                budgets.append((job['dataset'],job['cfg']['train']['epochs']))
                done.add(job['key'])
            argv=['runner','--config',str(config),'--deadline','2000-01-01 00:00']
            with patch.object(sys,'argv',argv), patch.object(runner.common,'prepare'), patch.object(runner,'completed',side_effect=lambda j:True if j['key'] in done else None), patch.object(runner,'calibrate',return_value=timing), patch.object(runner,'launch',side_effect=launch), patch.object(runner,'summarize'), patch('sys.stdout',new_callable=io.StringIO) as output:
                runner.main()
                self.assertIn('advisory target',output.getvalue())
            self.assertEqual(len(done),8)
            self.assertEqual(budgets,[('TaxiBJ',140)]*4+[('BikeNYC',100)]*4)

    def test_target_summary_paired_effects_and_partial_groups(self):
        p=runner.common.load(ROOT/'configs/presets/dual_moe_target_comparison.json')
        with tempfile.TemporaryDirectory() as tmp:
            suite=Path(tmp); jobs=list(runner.jobs(p,suite))
            def fake(job):
                value={'Q00':10.,'Q01':9.,'Q10':9.5,'Q11':8.}[job['variant']]
                return {'val_mae':value,'test_mae':value,'test_rmse':2*value,'best_epoch':2,'run_dir':'/fake'}
            with patch.object(runner,'completed',side_effect=fake): runner.summarize(jobs,suite)
            report=runner.common.load(suite/'comparison.json')
            self.assertEqual(len(report['paired']),15)
            self.assertEqual(len(report['interaction']),3)
            self.assertTrue(all(r['val_mae']==-.5 for r in report['interaction']))
            self.assertIn('epochs',(suite/'summary.csv').read_text().splitlines()[0])
            self.assertIn('Q11-Q10-Q01+Q00',report['note'])
            with patch.object(runner,'completed',side_effect=lambda j:None if j['variant']=='Q11' else fake(j)):
                runner.summarize(jobs,suite)
            report=runner.common.load(suite/'comparison.json')
            self.assertEqual(report['interaction'],[])
            self.assertTrue(all(g['status']=='incomplete' for g in report['paired_groups'] if g['candidate']=='Q11'))

    @unittest.skipUnless(os.environ.get('TARGET_STUDY_E2E')=='1','Opt-in four real BikeNYC 2-epoch jobs and completed-run skipping')
    def test_target_study_real_queue_validation_test_and_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=runner.common.load(ROOT/'configs/presets/dual_moe_target_comparison.json')
            p.update(datasets=['BikeNYC'],dataset_epochs={'BikeNYC':2},expected_train_samples={'BikeNYC':8},
                     train_windows=8,output_dir=str(Path(tmp)/'suite'))
            p.pop('deadline')
            config=Path(tmp)/'policy.json';runner.common.write_json(config,p)
            command=[sys.executable,'-B',str(ROOT/'scripts/run_scale_completion_experiments.py'),'--config',str(config)]
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=240)
            self.assertEqual(result.returncode,0,result.stdout[-4000:]+result.stderr[-2000:])
            summary=next((Path(tmp)/'suite').glob('*/summary.json'))
            rows=runner.common.load(summary)
            self.assertEqual(len(rows),4); self.assertTrue(all(r['status']=='complete' for r in rows))
            self.assertFalse(list(Path(tmp).rglob('*.pt')))
            for row in rows:
                self.assertEqual(row['epochs'],2)
                self.assertEqual(row['test_metrics']['topk_completion_selected_per_token'] if row['variant'] in ('Q01','Q11') else 2,2)
            # Even with an expired admission deadline complete jobs must skip.
            result=subprocess.run(command+['--deadline','2000-01-01 00:00'],cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout[-2000:]+result.stderr[-1000:])
            self.assertEqual(result.stdout.count('SKIP complete'),4)

    def test_routing_study_two_datasets_two_seeds_and_matched_factors(self):
        p = runner.common.load(ROOT/'configs/presets/dual_moe_routing_study.json')
        runner.validate(p)
        jobs = list(runner.jobs(p, Path('/tmp/routing_suite')))
        self.assertEqual(len(jobs), 20)
        self.assertEqual(sum(j['role'] == 'main' for j in jobs), 16)
        self.assertEqual(sum(j['role'] == 'ablation' for j in jobs), 4)
        for job in jobs:
            cfg = job['cfg']; options = cfg['model']['dual_moe']
            self.assertEqual(options['aggregation_experts'], 8)
            self.assertEqual(options['coarse_nodes'], [32, 8])
            self.assertEqual(cfg['train']['epochs'], 120)
            self.assertEqual(cfg['train']['val_epoch'], 2)
            self.assertFalse(cfg['train']['save_best_checkpoint'])
            self.assertEqual(job['expected_train_samples'], 2048 if job['dataset']=='TaxiBJ' else 511)
            self.assertIn('/tmp/routing_suite/data/', str(job['paths']['train']))
            self.assertNotIn('/tmp/', str(job['paths']['val']))
            self.assertNotIn('/tmp/', str(job['paths']['test']))
            self.assertEqual(cfg['data']['training_selection']['limit'], 2048)
            if job['role'] == 'ablation': self.assertEqual(job['seed'], 42)
            else: self.assertEqual(options['completion_mode'], 'topk')
        lookup = {j['variant']:j['cfg'] for j in jobs if j['dataset']=='TaxiBJ' and j['seed']==42}
        ref = copy.deepcopy(lookup['K2']); ref['model']['dual_moe']['aggregation_top_k'] = 4
        self.assertEqual(ref, lookup['K4'])
        ref['loss']['dual_moe_aggregation_balance_weight'] = 0
        self.assertEqual(ref, lookup['K4_NB'])
        self.assertEqual(lookup['D8']['model']['dual_moe']['aggregation_top_k'], 8)
        self.assertEqual(lookup['D8']['loss']['dual_moe_aggregation_balance_weight'], 0)
        for variant, section, change in [('D8', 'train', {'epochs': 60}), ('K4', 'model', {'dual_moe': {'aggregation_top_k': 1}})]:
            bad = copy.deepcopy(p); bad['variants'][variant]['patch'][section] = change
            with self.assertRaises(ValueError): runner.validate(bad)

    def test_routing_study_summary_counts_paired_seeds_without_partial_claims(self):
        p = runner.common.load(ROOT/'configs/presets/dual_moe_routing_study.json')
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp); jobs = list(runner.jobs(p, suite))
            def fake(job):
                v = 10. if job['variant']=='U8' else 9.
                return {'val_mae': v, 'test_mae': v, 'test_rmse': v*2, 'best_epoch': 120, 'run_dir': '/fake'}
            with patch.object(runner, 'completed', side_effect=fake): runner.summarize(jobs, suite)
            report = runner.common.load(suite/'comparison.json')
            self.assertEqual(len(report['paired']), 28)
            self.assertEqual(len(report['paired_groups']), 16)
            for group in report['paired_groups']:
                expected = [42] if group['control'] in ('K4_NB', 'D8_BU') else [42, 2026]
                self.assertEqual(group['expected_seeds'], expected)
                self.assertEqual(group['complete_seeds'], expected)
                self.assertEqual(group['status'], 'complete')
            with patch.object(runner, 'completed', side_effect=lambda j: fake(j) if j['seed']==42 else None): runner.summarize(jobs, suite)
            report = runner.common.load(suite/'comparison.json')
            self.assertTrue(all(g['status']=='incomplete' for g in report['paired_groups'] if len(g['expected_seeds'])==2))

    @unittest.skipUnless(os.environ.get('ROUTING_STUDY_E2E') == '1', 'Opt-in six BikeNYC 2-epoch diagnostic jobs, no formal scores')
    def test_routing_study_real_six_variants_memory_test_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = runner.common.load(ROOT/'configs/presets/dual_moe_routing_study.json')
            p.update(datasets=['BikeNYC'], expected_train_samples={'BikeNYC':8}, seeds=[42],
                     epochs=2, train_windows=8, output_dir=str(Path(tmp)/'suite'))
            p.pop('deadline')
            config = Path(tmp)/'policy.json'; runner.common.write_json(config, p)
            command = [sys.executable, '-B', str(ROOT/'scripts/run_scale_completion_experiments.py'), '--config', str(config)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=240)
            self.assertEqual(result.returncode, 0, result.stdout[-4000:]+result.stderr[-2000:])
            summary = next((Path(tmp)/'suite').glob('*/summary.json'))
            rows = runner.common.load(summary)
            self.assertEqual(len(rows), 6); self.assertTrue(all(r['status']=='complete' for r in rows))
            self.assertFalse(list(Path(tmp).rglob('best.pt')))
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout[-2000:]+result.stderr[-1000:])
            self.assertEqual(result.stdout.count('SKIP complete'), 6)

    def test_complete_requires_validation_schedule_and_finite_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)
            p = copy.deepcopy(self.p); p.update(epochs=3, val_epoch=2)
            job = next(runner.jobs(p, suite))
            run = suite/'runs/TaxiBJ/ablation/B00_seed42/fixed/rate0.4/stamp'
            (run/'logs').mkdir(parents=True); (run/'checkpoints').mkdir()
            (run/'checkpoints/best.pt').touch()
            runner.common.write_json(run/'config.json', job['cfg'])
            (run/'logs/train.log').write_text('Training finished normally: today')
            (run/'logs/test.log').write_text('Testing finished: today')
            metric = {'mae': 1., 'rmse': 2., 'loss': .1}
            records = [{'epoch': e, 'train': metric, 'val': metric if e in (2, 3) else None} for e in (1, 2, 3)]
            records += [{'stage': 'test', 'metrics': metric, 'extra': {'best_epoch': 2}}]
            def save(): (run/'logs/metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            save(); self.assertIsNotNone(runner.completed(job))
            records[0]['val'] = metric; save(); self.assertIsNone(runner.completed(job))
            records[0]['val'] = None; records[0]['train'] = dict(metric, loss=float('nan'))
            save(); self.assertIsNone(runner.completed(job))

    def test_incomplete_summary_does_not_manufacture_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp); jobs = list(runner.jobs(self.p, suite))
            rows = runner.summarize(jobs, suite)
            self.assertEqual(len(rows), 78)
            self.assertTrue(all(r['status'] == 'incomplete' and 'test_mae' not in r for r in rows))
            report = runner.common.load(suite/'comparison.json')
            self.assertEqual(report['paired'], [])
            self.assertTrue(all(g['complete'] == 0 for g in report['groups']))

    def test_topk_five_full_data_groups_and_matched_budgets(self):
        p = runner.common.load(ROOT/'configs/presets/dual_moe_topk_comparison.json')
        runner.validate(p)
        jobs = list(runner.jobs(p, Path('/tmp/topk_suite')))
        self.assertEqual([j['variant'] for j in jobs], ['T00', 'T10', 'T01', 'T11', 'T11_NB'])
        ref = jobs[0]['cfg']
        for j in jobs:
            self.assertEqual(j['cfg']['train'], ref['train'])
            self.assertEqual(j['cfg']['data'], ref['data'])
            self.assertEqual(j['paths'], jobs[0]['paths'])
            self.assertEqual(j['paths']['train'], ROOT/'data/TaxiBJ/taxibj_train.npz')
            self.assertEqual(j['expected_train_samples'], 2491)
            self.assertEqual(j['cfg']['train']['epochs'], 120)
            self.assertEqual(j['cfg']['train']['val_epoch'], 2)
            self.assertEqual(j['cfg']['data']['batch_size'], 4)
            model = j['cfg']['model']['dual_moe']
            self.assertEqual((model['aggregation_experts'], model['aggregation_top_k'], model['completion_top_k']), (3, 2, 2))
            self.assertEqual(model['design'], 'learned_regions_v2')
        for key, value in [('train_windows', 512), ('datasets', ['BikeNYC'])]:
            invalid = copy.deepcopy(p); invalid[key] = value
            with self.assertRaises(ValueError): runner.validate(invalid)
        invalid = copy.deepcopy(p); invalid['variants']['T00']['patch']['train'] = {'epochs': 1}
        with self.assertRaises(ValueError): runner.validate(invalid)

    def test_expert_count_matched_pairs_full_sources_and_strict_factors(self):
        p = runner.common.load(ROOT/'configs/presets/dual_moe_expert_count.json')
        runner.validate(p)
        jobs = list(runner.jobs(p, Path('/tmp/expert_count_suite')))
        self.assertEqual(len(jobs), 8)
        for e in (3, 4, 6, 8):
            u, k = [next(j for j in jobs if j['variant'] == f'E{e}_{mode}') for mode in ('U', 'K')]
            expected = copy.deepcopy(u['cfg'])
            expected['model']['dual_moe']['aggregation_mode'] = 'topk'
            expected['loss']['dual_moe_aggregation_balance_weight'] = .01
            self.assertEqual(expected, k['cfg'])
            for job in (u, k):
                self.assertEqual(job['expert_count'], e)
                self.assertEqual(job['expected_train_samples'], 2491)
                self.assertEqual(job['cfg']['train'], jobs[0]['cfg']['train'])
                self.assertIs(job['cfg']['train']['save_best_checkpoint'], False)
                self.assertEqual(job['cfg']['data'], jobs[0]['cfg']['data'])
                self.assertEqual(job['paths'], jobs[0]['paths'])
                options = job['cfg']['model']['dual_moe']
                self.assertEqual(options['aggregation_experts'], e)
                self.assertEqual(options['aggregation_top_k'], 2)
                self.assertEqual(options['completion_top_k'], 2)
                self.assertEqual(options['completion_mode'], 'topk')
        for key, value in [('expert_counts', [3, 3, 8]), ('expert_counts', [4, 8]), ('aggregation_top_k', 3), ('save_best_checkpoint', 'false')]:
            bad = copy.deepcopy(p); bad[key] = value
            with self.assertRaises(ValueError): runner.validate(bad)
        bad = copy.deepcopy(p); del bad['variants']['E8_U']
        with self.assertRaises(ValueError): runner.validate(bad)
        bad = copy.deepcopy(p); bad['variants']['E8_K']['patch']['train'] = {'epochs': 60}
        with self.assertRaises(ValueError): runner.validate(bad)

    def test_expert_count_summary_separates_capacity_from_routing_gain(self):
        p = runner.common.load(ROOT/'configs/presets/dual_moe_expert_count.json')
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp); jobs = list(runner.jobs(p, suite))
            def fake(job):
                e = job['expert_count']; v = 10.+(e-3)
                if job['front_mode'] == 'topk': v -= .1*e
                return {'val_mae': v, 'test_mae': v, 'test_rmse': v*2, 'best_epoch': 120, 'run_dir': '/fake'}
            with patch.object(runner, 'completed', side_effect=fake): runner.summarize(jobs, suite)
            report = runner.common.load(suite/'comparison.json')
            self.assertEqual(len(report['paired']), 10)
            self.assertEqual(len(report['front_routing_gain']), 4)
            e8 = next(r for r in report['front_routing_gain'] if r['expert_count'] == 8)
            self.assertAlmostEqual(e8['uniform_minus_topk']['val_mae'], .8)
            self.assertAlmostEqual(e8['gain_change_vs_E3']['val_mae'], .5)
            with patch.object(runner, 'completed', return_value=None): runner.summarize(jobs, suite)
            report = runner.common.load(suite/'comparison.json')
            self.assertEqual(report['paired'], [])
            self.assertEqual(report['front_routing_gain'], [])

    def test_topk_full_count_guard_does_not_create_subsets(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'train.npz'
            np.savez(path, x_f_gt=np.zeros((2491, 1, 1, 1, 1), dtype=np.float32))
            jobs = [{'paths': {'train': path}}, {'paths': {'train': path}}]
            self.assertEqual(runner.verify_full_data({'train_windows': 2491}, jobs), {str(path): 2491})
            with self.assertRaises(ValueError): runner.verify_full_data({'train_windows': 512}, jobs)

    def test_topk_pairs_interaction_and_incomplete_are_not_scores(self):
        p = runner.common.load(ROOT/'configs/presets/dual_moe_topk_comparison.json')
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp); jobs = list(runner.jobs(p, suite))
            def fake(job):
                v = {'T00': 10., 'T10': 9., 'T01': 8., 'T11': 6., 'T11_NB': 7.}[job['variant']]
                return {'val_mae': v, 'test_mae': v, 'test_rmse': v*2, 'best_epoch': 120, 'run_dir': '/fake'}
            with patch.object(runner, 'completed', side_effect=fake): runner.summarize(jobs, suite)
            report = runner.common.load(suite/'comparison.json')
            self.assertEqual(len(report['paired']), 6)
            self.assertEqual(report['interaction'][0]['val_mae'], -1.)
            with patch.object(runner, 'completed', return_value=None): runner.summarize(jobs, suite)
            report = runner.common.load(suite/'comparison.json')
            self.assertEqual(report['paired'], [])
            self.assertEqual(report['interaction'], [])

    def test_topk_all_five_variants_forward_backward_shared_initialization(self):
        import torch
        from test_dual_moe import batch
        from stmoe_imputer.models import DualBranchSTImputer
        from stmoe_imputer.losses import compute_main_stage_loss
        torch.set_num_threads(1)
        p = runner.common.load(ROOT/'configs/presets/dual_moe_topk_comparison.json')
        reference = None
        for job in runner.jobs(p, Path('/tmp/topk_suite')):
            torch.manual_seed(42)
            model = DualBranchSTImputer.from_config(job['cfg'])
            state = model.state_dict()
            if reference is not None:
                for name in state:
                    if not name.endswith('router.head.weight'):
                        torch.testing.assert_close(state[name], reference[name], rtol=0, atol=0)
            else:
                reference = {k: v.clone() for k, v in state.items()}
            data = batch(n=1); output = model(data)
            loss, logs = compute_main_stage_loss(output, data, job['cfg'])
            self.assertTrue(torch.isfinite(loss)); loss.backward()
            self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters() if x.requires_grad))
            for stage in ('aggregation', 'completion'):
                key = f'dual_moe_{stage}_balance_weight'
                value = float(logs.get(f'l_balance_{stage}_weighted', 0))
                self.assertEqual(value == 0, job['cfg']['loss'][key] == 0)

    @unittest.skipUnless(os.environ.get('TOPK_SUITE_E2E') == '1', 'Opt-in five real-data 2-epoch diagnostic jobs, not formal scores')
    def test_topk_five_real_training_validation_test_and_resume_detection(self):
        p = runner.common.load(ROOT/'configs/presets/dual_moe_topk_comparison.json')
        p['epochs'] = 2
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)
            # Diagnostic-only subset, isolated from the 2491-sample formal suite.
            data = runner.common.prepare(dict(p, train_windows=8), suite)['TaxiBJ']
            jobs = list(runner.jobs(p, suite))
            for job in jobs:
                job['paths']['train'] = data/'train.npz'
                job['cfg']['data']['mask']['train_csv'] = str(data/'random_train.csv')
                job['expected_train_samples'] = 8
            env = {k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
            env['CUDA_VISIBLE_DEVICES'] = '0'
            with patch.dict(os.environ, env):
                for job in jobs:
                    runner.launch(job, suite)
                    self.assertIsNotNone(runner.completed(job))
            rows = runner.summarize(jobs, suite)
            self.assertEqual(sum(r['status'] == 'complete' for r in rows), 5)
            self.assertEqual(len(runner.common.load(suite/'comparison.json')['paired']), 6)
            for job in jobs:
                job['expected_train_samples'] = 2491
                self.assertIsNone(runner.completed(job))

    @unittest.skipUnless(os.environ.get('SCALE_SUITE_E2E') == '1', 'Opt-in seven real-data 2-epoch jobs')
    def test_seven_variants_real_training_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = copy.deepcopy(self.p)
            p.update(datasets=['BikeNYC'], patterns=['random'], seeds=[42], epochs=2, train_windows=8,
                     output_dir=str(Path(tmp)/'suite'))
            config = Path(tmp)/'policy.json'; runner.common.write_json(config, p)
            command = [sys.executable, '-B', str(ROOT/'scripts/run_scale_completion_experiments.py'), '--config', str(config)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=240)
            self.assertEqual(result.returncode, 0, result.stdout[-5000:]+result.stderr[-3000:])
            summaries = list((Path(tmp)/'suite').glob('*/summary.json'))
            self.assertEqual(len(summaries), 1)
            rows = runner.common.load(summaries[0])
            self.assertEqual(len(rows), 7)
            self.assertTrue(all(r['status'] == 'complete' for r in rows))
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(result.stdout.count('SKIP complete'), 7)


class CoverageTests(unittest.TestCase):
    def policy(self, budget=None):
        return runner.resolve_policy(runner.common.load(ROOT/'configs/presets/dual_moe_coverage.json'), budget)

    def test_all_rates_full_data_117_unique_jobs_and_budget_matching(self):
        p = self.policy(); runner.validate(p)
        jobs = list(runner.jobs(p, Path('/tmp/coverage')))
        self.assertEqual(len(jobs), 117)
        self.assertEqual(len({j['key'] for j in jobs}), 117)
        self.assertEqual([sum(j['stage']==s for j in jobs) for s in ('coverage','replication','mechanism')], [96,12,9])
        counts = runner.verify_routing_sources(p)
        for d in p['datasets']:
            self.assertEqual(counts[d]['original_train'], counts[d]['selected_train'])
        for j in jobs:
            self.assertEqual(j['cfg']['data']['batch_size'],16)
            self.assertEqual(j['cfg']['train']['epochs'], {'TaxiBJ':70,'BikeNYC':100,'CHAP':80}[j['dataset']])
            self.assertEqual(j['cfg']['loss']['dual_moe_completion_balance_weight'],0)
            self.assertEqual(j['cfg']['data']['mask']['missing_rate'],j['rate'])
            self.assertIn(f'rate{j["rate"]:g}_train.csv',j['cfg']['data']['mask']['train_csv'])
            for split in ('val','test'):
                self.assertIn(f'/{j["rate"]:g}/', j['cfg']['data']['mask'][split+'_csv'])
                self.assertNotIn('/tmp/coverage',str(j['paths'][split]))
            self.assertFalse(j['cfg']['train']['save_best_checkpoint'])
        for d in p['datasets']:
            for mask in p['patterns']:
                for rate in p['rates']:
                    block = [j for j in jobs if (j['dataset'],j['pattern'],j['rate'],j['seed']) == (d,mask,rate,42) and j['variant'].startswith('Q')]
                    self.assertEqual(len(block),4)
                    ref = block[0]['cfg']
                    for j in block:
                        self.assertEqual(j['cfg'],runner.common.merge(ref,p['variants'][j['variant']]['patch']))
        extended = self.policy('extended'); runner.validate(extended)
        self.assertIsNone(extended['deadline'])
        self.assertEqual(extended['dataset_epochs'],{'TaxiBJ':140,'BikeNYC':100,'CHAP':150})
        self.assertNotEqual(runner.common.digest(runner.identity(p)),runner.common.digest(runner.identity(extended)))
        self.assertEqual(sum('/fixed_mask/' in s['path'] or '/random_mask/' in s['path'] for s in runner.identity(p)['sources']),72)

    def test_coverage_rejects_missing_rates_and_unmatched_controls(self):
        for key,value in [('rates',[.2,.4]),('patterns',['random']),('seeds',[42]),('rate',.4)]:
            p=self.policy();p[key]=value
            with self.assertRaises(ValueError):runner.validate(p)
        p=self.policy();p['variants']['SF']['patch']['train']={'epochs':10}
        with self.assertRaises(ValueError):runner.validate(p)

    def test_multirate_mask_preparation_keeps_npz_csv_alignment(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'data/TaxiBJ';base.mkdir(parents=True)
            np.savez(base/'taxibj_train.npz', x_f_gt=np.arange(5,dtype=np.float32).reshape(5,1,1,1,1))
            for rate in (.2,.8):
                for pattern in ('fixed','random'):
                    dest=base/f'{pattern}_mask/{rate:g}';dest.mkdir(parents=True)
                    (dest/'train.csv').write_text('fixed\n' if pattern=='fixed' else ''.join(f'{rate}:{i}\n' for i in range(5)))
            p={'datasets':['TaxiBJ'],'patterns':['fixed','random'],'rates':[.2,.8],'train_windows':3}
            with patch.object(runner.common,'ROOT',root):runner.common.prepare(p,root/'suite')
            dest=root/'suite/data/TaxiBJ'
            with np.load(dest/'train.npz') as data:self.assertEqual(data['x_f_gt'].flatten().tolist(),[0,2,4])
            for rate in (.2,.8):
                self.assertEqual((dest/f'random_rate{rate:g}_train.csv').read_text(), ''.join(f'{rate}:{i}\n' for i in (0,2,4)))
                self.assertEqual((dest/f'fixed_rate{rate:g}_train.csv').read_text(),'fixed\n')
            self.assertEqual(len(list(dest.glob('*.npz'))),1)

    def test_seven_controls_forward_gradient_and_shared_initialization(self):
        import torch
        from test_dual_moe import batch
        from stmoe_imputer.models import DualBranchSTImputer
        from stmoe_imputer.losses import compute_main_stage_loss
        torch.set_num_threads(1)
        p=self.policy();data=batch(n=1);states={}
        subset=[j for j in runner.jobs(p,Path('/tmp/coverage')) if j['dataset']=='BikeNYC' and j['pattern']=='random' and j['rate']==.4 and j['seed']==42]
        for j in subset:
            torch.manual_seed(42);model=DualBranchSTImputer.from_config(j['cfg'])
            states[j['variant']]={k:v.clone() for k,v in model.state_dict().items()}
            output=model(data);loss,_=compute_main_stage_loss(output,data,j['cfg']);loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad),j['variant'])
            if j['variant']=='SF':
                for router in model.main_branch.readout_routers.values():
                    self.assertIsNotNone(router.logits.grad)
                    self.assertFalse(router.query.weight.requires_grad)
        for name,state in states.items():
            for key,value in state.items():
                if 'router' not in key:torch.testing.assert_close(value,states['Q11'][key],rtol=0,atol=0)

    def test_summary_never_pairs_different_rates_and_marks_partial_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite=Path(tmp);jobs=list(runner.jobs(self.policy(),suite))
            def fake(job):
                if job['variant']=='Q11' and job['rate']==.4 and job['seed']==2026:return None
                value=100*job['rate']+{'Q00':4,'Q10':3,'Q01':2,'Q11':1,'SF':3,'FD':2,'SB':3}[job['variant']]
                run=suite/'fake'/job['key'];(run/'logs').mkdir(parents=True,exist_ok=True)
                runner.common.write_json(run/'config.json',job['cfg'])
                (run/'logs/metrics.jsonl').write_text(json.dumps({'epoch':2,'val':{'mae':value},'perf':{}})+'\n')
                return {'val_mae':value,'test_mae':value,'test_rmse':value*2,'best_epoch':2,'run_dir':str(run)}
            with patch.object(runner,'completed',side_effect=fake):runner.summarize(jobs,suite)
            report=runner.common.load(suite/'comparison.json')
            self.assertEqual(len(runner.common.load(suite/'summary.json')),117)
            self.assertEqual({r['rate'] for r in report['paired']},{.2,.4,.6,.8})
            for r in report['paired']:
                if (r['candidate'],r['control'])==('Q11','Q01'):
                    self.assertAlmostEqual(r['test_mae_change_percent'],100*((100*r['rate']+1)/(100*r['rate']+2)-1))
            self.assertTrue(all(r['status']=='incomplete' for r in report['paired_groups'] if r['pattern']=='random' and r['rate']==.4 and r['candidate']=='Q11' and r['control'] in ('Q00','Q01','Q10')))
            self.assertTrue(all(r['complete_points']==8 for r in report['coverage_effects']))
            self.assertEqual(sum(r['candidate']=='FD' and r['control']=='SF' for r in report['paired']),3)

    @unittest.skipUnless(os.environ.get('COVERAGE_E2E')=='1','Opt-in seven real 2-epoch BikeNYC jobs, full VAL/TEST, not formal results')
    def test_real_seven_controls_best_memory_test_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite=Path(tmp);p=self.policy()
            p.update(datasets=['BikeNYC'],patterns=['random'],rates=[.4],train_windows=8,
                     expected_train_samples={'BikeNYC':8},dataset_epochs={'BikeNYC':2})
            runner.common.prepare(p,suite)
            jobs=[j for j in runner.jobs(p,suite) if j['seed']==42]
            self.assertEqual(len(jobs),7)
            for j in jobs:
                runner.launch(j,suite)
                self.assertIsNotNone(runner.completed(j))
            self.assertFalse(list(suite.rglob('*.pt')))
            runner.summarize(jobs,suite)
            self.assertTrue(all(r['status']=='complete' for r in runner.common.load(suite/'summary.json')))
            self.assertEqual(len([j for j in jobs if runner.completed(j) is None]),0)


class BackendConfirmationTests(unittest.TestCase):
    def setUp(self):
        # Historical reuse tests exercise the old-code eligibility branch.
        # Never require current production code to match an archived hash.
        # Only the TEST identity is matched; production remains fail-closed.
        self.real_identity = runner.identity
        reference = runner.common.resolve(self.policy()['reuse_suite'])/'protocol.json'
        if not reference.exists():
            self.skipTest('Archived reference fixture unavailable')
        archived_code = runner.common.load(reference)['code']
        def fixture_identity(p):
            record=self.real_identity(p)
            record['code']=copy.deepcopy(archived_code)
            return record
        patcher=patch.object(runner,'identity',side_effect=fixture_identity)
        patcher.start();self.addCleanup(patcher.stop)

    def test_actual_changed_scientific_code_disables_old_reuse(self):
        p=self.policy();record=self.real_identity(p)
        record['code']['src/stmoe_imputer/models/dual_moe.py']='new-scientific-implementation'
        with patch.object(runner,'identity',return_value=record):
            audit=runner.bind_reuse(p,list(runner.jobs(p,Path('/tmp/confirmation'))))
        self.assertEqual(audit[0]['status'],'disabled')

    def policy(self):
        return runner.common.load(ROOT/'configs/presets/dual_moe_backend_confirmation.json')

    def test_18_jobs_full_matched_budget_and_three_backend_controls(self):
        p=self.policy();runner.validate(p)
        jobs=list(runner.jobs(p,Path('/tmp/confirmation')))
        self.assertEqual(len(jobs),18);self.assertEqual(len({j['key'] for j in jobs}),18)
        self.assertEqual(sum(j['dataset']=='TaxiBJ' for j in jobs),12)
        counts=runner.verify_routing_sources(p)
        self.assertTrue(all(c['original_train']==c['selected_train'] for c in counts.values()))
        for point in p['points']:
            for seed in p['seeds']:
                block=[j for j in jobs if j['dataset']==point['dataset'] and j['rate']==point['rate'] and j['seed']==seed]
                self.assertEqual({j['variant'] for j in block},{'Q10','SB','Q11'})
                for j in block:
                    self.assertEqual(j['cfg'],runner.common.merge(block[0]['cfg'],p['variants'][j['variant']]['patch']))
                    self.assertEqual(j['cfg']['train']['epochs'],140 if j['dataset']=='TaxiBJ' else 100)
                    self.assertEqual(j['cfg']['data']['batch_size'],16)
                    self.assertEqual(j['cfg']['train']['val_epoch'],2)
                    self.assertFalse(j['cfg']['train']['save_best_checkpoint'])
                    self.assertEqual(j['cfg']['model']['dual_moe']['aggregation_top_k'],4)
                    self.assertIn(f'rate{j["rate"]:g}_train.csv',j['cfg']['data']['mask']['train_csv'])
        self.assertIsNone(p['deadline'])

    def test_rejects_unplanned_points_and_variant_budget_changes(self):
        for key,value in [('points',[{'dataset':'BikeNYC','rate':.4}]),('rates',[.4]),('rate',.4),('reuse_enabled','yes')]:
            p=self.policy();p[key]=value
            with self.assertRaises(ValueError):runner.validate(p)
        p=self.policy();p['variants']['Q11']['patch']['train']={'epochs':150}
        with self.assertRaises(ValueError):runner.validate(p)

    def test_reuses_only_five_verified_bikenyc_results(self):
        p=self.policy();jobs=list(runner.jobs(p,Path('/tmp/confirmation')))
        audit=runner.bind_reuse(p,jobs)
        self.assertEqual(sum(x['status']=='eligible' for x in audit),5,audit)
        reused=[j for j in jobs if j.get('reuse_job')]
        self.assertEqual(len(reused),5)
        self.assertTrue(all(j['dataset']=='BikeNYC' for j in reused))
        for j in reused:
            result=runner.completed(j)
            self.assertEqual(result['result_origin'],'reused')
            self.assertIn('coverage/19437d53520efacc',result['run_dir'])
        p['reuse_enabled']=False
        self.assertEqual(runner.bind_reuse(p,list(runner.jobs(p,Path('/tmp/confirmation')))),[])
        self.assertEqual(runner.reference_files(p),[])

    def test_reuse_fails_closed_on_scientific_changes_or_incomplete_runs(self):
        p=self.policy();now=runner.identity(p);now['code']['src/stmoe_imputer/models/dual_moe.py']='different'
        jobs=list(runner.jobs(p,Path('/tmp/confirmation')))
        with patch.object(runner,'identity',return_value=now):audit=runner.bind_reuse(p,jobs)
        self.assertEqual(audit[0]['status'],'disabled');self.assertFalse(any(j.get('reuse_job') for j in jobs))
        p['dataset_epochs']['BikeNYC']=102;jobs=list(runner.jobs(p,Path('/tmp/confirmation')))
        audit=runner.bind_reuse(p,jobs)
        self.assertFalse(any(j.get('reuse_job') for j in jobs))
        self.assertEqual(sum(x['status']=='fresh' for x in audit),6)
        p=self.policy();jobs=list(runner.jobs(p,Path('/tmp/confirmation')))
        with patch.object(runner.common,'completed_run',return_value=None):runner.bind_reuse(p,jobs)
        self.assertFalse(any(j.get('reuse_job') for j in jobs))

    def test_reference_data_manifest_tampering_disables_reuse(self):
        p=self.policy();original=runner.common.load
        def load(path):
            value=original(path)
            if str(path).endswith('data/BikeNYC/selection.json'):
                value={**value,'indices':[0]}
            return value
        jobs=list(runner.jobs(p,Path('/tmp/confirmation')))
        with patch.object(runner.common,'load',side_effect=load):audit=runner.bind_reuse(p,jobs)
        self.assertEqual(audit[0]['status'],'disabled');self.assertFalse(any(j.get('reuse_job') for j in jobs))

    def test_summary_retains_provenance_rates_and_incomplete_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite=Path(tmp);p=self.policy();jobs=list(runner.jobs(p,suite));runner.bind_reuse(p,jobs)
            rows=runner.summarize(jobs,suite)
            self.assertEqual(len(rows),18);self.assertEqual(sum(r['status']=='complete' for r in rows),5)
            self.assertTrue(all(r['result_origin']=='reused' for r in rows if r['status']=='complete'))
            report=runner.common.load(suite/'comparison.json')
            self.assertEqual(report['interaction'],[]);self.assertEqual(report['coverage_effects'],[])
            self.assertEqual({r['rate'] for r in report['groups']},{.4,.8})
            complete=[r for r in report['paired_groups'] if r['status']=='complete']
            self.assertEqual(len(complete),1)
            self.assertEqual((complete[0]['dataset'],complete[0]['candidate'],complete[0]['control']),('BikeNYC','Q11','Q10'))
            self.assertIn('result_origin',(suite/'summary.csv').read_text().splitlines()[0])

    def test_queue_launches_only_thirteen_missing_jobs(self):
        import io
        with tempfile.TemporaryDirectory() as tmp:
            p=self.policy();p['output_dir']=str(Path(tmp)/'suite')
            config=Path(tmp)/'policy.json';runner.common.write_json(config,p)
            original=runner.completed;done=set();launched=[]
            def complete(job):
                return {'result_origin':'trained'} if job['key'] in done else original(job)
            def launch(job,suite):
                launched.append(job);done.add(job['key'])
            timing={f'{d}/{v}':{'seconds_per_run':1} for d in p['datasets'] for v in p['variants']}
            with patch.object(sys,'argv',['runner','--config',str(config)]), patch.object(runner.common,'prepare'), patch.object(runner,'completed',side_effect=complete), patch.object(runner,'launch',side_effect=launch), patch.object(runner,'summarize'), patch.object(runner,'calibrate',return_value=timing), patch('sys.stdout',new_callable=io.StringIO) as out:
                runner.main()
                self.assertIn('13 remaining',out.getvalue())
                self.assertEqual(out.getvalue().count('REUSE verified'),5)
            self.assertEqual(len(launched),13)
            self.assertEqual([(j['variant'],j['seed']) for j in launched if j['dataset']=='BikeNYC'],[('SB',2026)])
            self.assertEqual(sum(j['dataset']=='TaxiBJ' for j in launched),12)

    @unittest.skipUnless(os.environ.get('CONFIRMATION_E2E')=='1','Three short real BikeNYC jobs, no formal experiment')
    def test_three_real_controls_train_val_best_test_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.policy();p.update(datasets=['BikeNYC'],points=[{'dataset':'BikeNYC','rate':.4}],rates=[.4],
                                     seeds=[42],train_windows=8,expected_train_samples={'BikeNYC':8},
                                     dataset_epochs={'BikeNYC':2},reuse_enabled=False)
            suite=Path(tmp);runner.common.prepare(p,suite);jobs=list(runner.jobs(p,suite))
            self.assertEqual(len(jobs),3)
            for job in jobs:
                runner.launch(job,suite)
                self.assertEqual(runner.completed(job)['result_origin'],'trained')
            self.assertFalse(list(suite.rglob('*.pt')))
            self.assertEqual(len([j for j in jobs if runner.completed(j) is None]),0)
            runner.summarize(jobs,suite)
            report=runner.common.load(suite/'comparison.json')
            self.assertEqual(len(report['paired']),3)
            self.assertEqual(report['interaction'],[])


if __name__ == '__main__':
    unittest.main()
