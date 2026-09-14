import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('dual_comparison', ROOT/'scripts/run_dual_moe_comparison.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.policy = runner.load(ROOT/'configs/presets/dual_moe_comparison.json')

    def test_matched_four_groups_and_complete_six_points(self):
        runner.validate(self.policy)
        self.assertEqual(len(self.policy['variants'])*len(self.policy['datasets'])*len(self.policy['patterns']),24)
        configs = [runner.job_config(self.policy,Path('/tmp/testsuite'),Path('/tmp/testdata'),'TaxiBJ','fixed',v)[0] for v in self.policy['variants']]
        normalized = []
        for c in configs:
            c = copy.deepcopy(c)
            c['model']['dual_moe'].pop('aggregation_mode')
            c['model']['dual_moe'].pop('completion_mode')
            normalized.append(c)
        self.assertTrue(all(c == normalized[0] for c in normalized))
        self.assertFalse(normalized[0]['train']['early_stopping']['enabled'])

    def test_unmatched_changes_are_rejected(self):
        p=copy.deepcopy(self.policy);p['variants']['A11']['dim']=64
        with self.assertRaises(ValueError):runner.validate(p)
        p=copy.deepcopy(self.policy);p['epochs']=0
        with self.assertRaises(ValueError):runner.validate(p)

    def test_streamed_npz_and_random_csv_indices_stay_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=runner.ROOT;runner.ROOT=root
            try:
                folder=root/'data/BikeNYC';folder.mkdir(parents=True)
                x=np.arange(9*2*2*3*4,dtype=np.float32).reshape(9,2,2,3,4)
                np.savez_compressed(folder/'bikenyc_train.npz',x_f_gt=x)
                masks=np.arange(9*12).reshape(9,12)
                for pattern in ['fixed','random']:
                    dest=folder/f'{pattern}_mask/0.4';dest.mkdir(parents=True)
                    np.savetxt(dest/'train.csv',masks[:1] if pattern=='fixed' else masks,delimiter=',')
                p=copy.deepcopy(self.policy);p.update(datasets=['BikeNYC'],train_windows=4)
                data=runner.prepare(p,root/'suite')['BikeNYC']
                ids=runner.load(data/'selection.json')['indices']
                self.assertEqual(ids,[0,2,5,8])
                with np.load(data/'train.npz') as z:np.testing.assert_array_equal(z['x_f_gt'],x[ids])
                np.testing.assert_array_equal(np.loadtxt(data/'random_train.csv',delimiter=','),masks[ids])
                self.assertEqual(np.loadtxt(data/'fixed_train.csv',delimiter=',',ndmin=2).shape,(1,12))
                self.assertFalse((data/'val.npz').exists())
                # Cached preparations preserve selection IDs.
                runner.prepare(p,root/'suite')
                self.assertEqual(runner.load(data/'selection.json')['indices'],ids)
            finally:runner.ROOT=old

    def test_skip_requires_successful_complete_matching_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg={'output_dir':tmp,'data':{'dataset_name':'BikeNYC','mask':{'pattern':'fixed','missing_rate':.4}},'train':{'epochs':2}}
            run=Path(tmp)/'BikeNYC/ablation/A11/fixed/rate0.4/stamp'
            (run/'logs').mkdir(parents=True);(run/'checkpoints').mkdir()
            runner.write_json(run/'config.json',cfg)
            (run/'checkpoints/best.pt').touch()
            (run/'logs/train.log').write_text('Training finished normally: today')
            records=[{'epoch':1,'val':{'mae':2.}},{'epoch':2,'val':{'mae':1.}}]
            records.append({'stage':'test','metrics':{'mae':1.2,'rmse':2.3},'extra':{'best_epoch':2}})
            metrics=run/'logs/metrics.jsonl'
            metrics.write_text('\n'.join(json.dumps(r) for r in records))
            test=run/'logs/test.log';test.write_text('Testing finished: today')
            self.assertEqual(runner.completed_run(cfg,'A11')['test_mae'],1.2)
            changed=copy.deepcopy(cfg);changed['train']['epochs']=3
            self.assertIsNone(runner.completed_run(changed,'A11'))
            records[-1]['metrics']['mae']=float('nan')
            metrics.write_text('\n'.join(json.dumps(r) for r in records))
            self.assertIsNone(runner.completed_run(cfg,'A11'))
            records[-1]['metrics']['mae']=1.2
            metrics.write_text('\n'.join(json.dumps(r) for r in records))
            (run/'logs/train.log').write_text('Training failed')
            self.assertIsNone(runner.completed_run(cfg,'A11'))

    @unittest.skipUnless(os.environ.get('DUAL_COMPARISON_E2E')=='1','Opt-in real-data CUDA launcher test')
    def test_real_four_jobs_finish_and_rerun_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=copy.deepcopy(self.policy)
            p.update(datasets=['BikeNYC'],patterns=['random'],epochs=2,train_windows=8,output_dir=str(Path(tmp)/'runs'))
            cfg=Path(tmp)/'policy.json';runner.write_json(cfg,p)
            command=[sys.executable,str(ROOT/'scripts/run_dual_moe_comparison.py'),'--config',str(cfg),'--gpu','0']
            first=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=180)
            self.assertEqual(first.returncode,0,first.stdout[-3000:]+first.stderr[-1000:])
            summaries=list((Path(tmp)/'runs').glob('*/summary.json'))
            self.assertEqual(len(summaries),1)
            rows=runner.load(summaries[0]);self.assertEqual(len(rows),4)
            self.assertTrue(all(r['status']=='complete' for r in rows))
            second=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(second.returncode,0,second.stdout+second.stderr)
            self.assertEqual(second.stdout.count('SKIP complete'),4)


if __name__=='__main__':
    unittest.main()
