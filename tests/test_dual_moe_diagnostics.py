import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_dual_moe_diagnostics as diag
import run_dual_moe_comparison as common
from test_dual_moe import config,batch
from stmoe_imputer.models import DualBranchSTImputer


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_default_stage_counts_and_single_variable_patches(self):
        policy=common.load(ROOT/'configs/presets/dual_moe_diagnostics.json');diag.validate(policy)
        p=common.load(ROOT/'configs/presets/dual_moe_comparison.json');refs={}
        for dataset in p['datasets']:
            for pattern in p['patterns']:
                for variant in ['A01','A11']:
                    cfg,paths=common.job_config(p,Path('/tmp/source'),Path('/tmp/data'),dataset,pattern,variant)
                    refs[f'{dataset}_{pattern}_{variant}']={'dataset':dataset,'pattern':pattern,'variant':variant,'cfg':cfg}
        jobs=diag.training_jobs(policy,refs,Path('/tmp/diag'),'cuda:0')
        self.assertEqual((len(jobs[2]),len(jobs[3])),(6,24))
        self.assertEqual(len(diag.intervention_specs(policy)),9)
        for stage in [2,3]:
            for name,ref,cfg in jobs[stage]:
                original=copy.deepcopy(ref['cfg']);candidate=copy.deepcopy(cfg)
                candidate['output_dir']=original['output_dir']
                if stage==2:
                    self.assertEqual(candidate['model']['dual_moe']['aggregation_experts'],1)
                    candidate['model']['dual_moe']['aggregation_experts']=3
                else:
                    self.assertIn(candidate['loss']['dual_moe_expert_weight'],[.01,.05])
                    candidate['loss']['dual_moe_expert_weight']=0
                self.assertEqual(candidate,original)

    def test_interventions_touch_only_declared_condition_channels(self):
        model=DualBranchSTImputer.from_config(config()).eval();data=batch()
        def capture(mode):
            values={};hooks=[]
            with diag.intervention(model,mode,17):
                for scale,branch in model.main_branch.scale_experts.items():
                    def hook(module,args,s=scale):values[s]=args[0].detach().clone()
                    hooks.append(branch.condition.register_forward_pre_hook(hook))
                try:out=model(data)
                finally:
                    for h in hooks:h.remove()
            return values,out
        original,base=capture('baseline')
        for mode in ['zero_mid','zero_coarse','zero_both','zero_fine_context','shuffle_both']:
            values,out=capture(mode)
            torch.testing.assert_close(base['scale_predictions']['fine'],out['scale_predictions']['fine'])
            for scale,z in values.items():
                d=z.shape[1]//2
                for key in ('region_assignments','aggregation_gates','aggregation_mass'):
                    torch.testing.assert_close(out[key][scale],base[key][scale])
                if mode=='zero_fine_context':
                    self.assertEqual(float(z[:,d:].abs().max()),0)
                    torch.testing.assert_close(z[:,:d],original[scale][:,:d])
                else:
                    torch.testing.assert_close(z[:,d:],original[scale][:,d:])
                    if mode in ('zero_both',f'zero_{scale}'):
                        self.assertEqual(float(z[:,:d].abs().max()),0)
                    elif mode=='shuffle_both':
                        torch.testing.assert_close(z[:,:d].flatten(-2).sort(-1).values,original[scale][:,:d].flatten(-2).sort(-1).values)
                    else:torch.testing.assert_close(z,original[scale])
        torch.testing.assert_close(model(data)['x_hat_final'],base['x_hat_final'])

    def test_permutation_control_and_exception_restore_parameters(self):
        model=DualBranchSTImputer.from_config(config()).eval();data=batch()
        state={k:v.clone() for k,v in model.state_dict().items()};base=model(data)['x_hat_final']
        with diag.intervention(model,'slot_permutation_control',19):
            torch.testing.assert_close(model(data)['x_hat_final'],base,rtol=1e-5,atol=1e-5)
        with self.assertRaises(RuntimeError):
            with diag.intervention(model,'slot_permutation_control',19):raise RuntimeError('test')
        for k,v in model.state_dict().items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)

    def test_metric_diagnostics_finite_and_use_missing_positions(self):
        cfg=config();model=DualBranchSTImputer.from_config(cfg).eval();data=batch()
        values=diag.measure(model,[data],torch.device('cpu'))
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in values.values()))
        self.assertGreaterEqual(values['error_cancellation_fraction'],-1e-5)
        self.assertLessEqual(values['error_cancellation_fraction'],1)
        self.assertEqual(values['metric_missing_count'],float((1-data['m_f']).expand_as(data['x_f_gt']).sum()))

    @unittest.skipUnless(os.environ.get('DUAL_DIAGNOSTICS_E2E')=='1','Opt-in CUDA full launcher integration')
    def test_three_stages_real_data_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            p=common.load(ROOT/'configs/presets/dual_moe_comparison.json')
            p.update(datasets=['BikeNYC'],patterns=['random'],epochs=2,train_windows=8,output_dir=str(tmp/'source'))
            config_path=tmp/'source.json';common.write_json(config_path,p)
            cmd=[sys.executable,str(ROOT/'scripts/run_dual_moe_comparison.py'),'--config',str(config_path),'--gpu','0']
            source=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=180)
            self.assertEqual(source.returncode,0,source.stdout[-2000:]+source.stderr[-1000:])
            suite=next((tmp/'source').glob('*/protocol.json')).parent
            p=common.load(ROOT/'configs/presets/dual_moe_diagnostics.json')
            p.update(source_suite=str(suite),output_dir=str(tmp/'diag'),datasets=['BikeNYC'],patterns=['random'],shuffle_seeds=[17])
            common.write_json(tmp/'diag.json',p)
            cmd=[sys.executable,str(ROOT/'scripts/run_dual_moe_diagnostics.py'),'--config',str(tmp/'diag.json'),'--gpu','0']
            first=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=240)
            self.assertEqual(first.returncode,0,first.stdout[-2500:]+first.stderr[-1500:])
            summary=common.load(next((tmp/'diag').glob('*/summary.json')))
            self.assertEqual([len(summary[k]) for k in ('stage1','stage2','stage3','missing')],[2,1,4,0])
            second=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(second.returncode,0,second.stdout[-2000:]+second.stderr[-1000:])
            self.assertEqual(second.stdout.count(' SKIP]'),7)
            # Changing a source checkpoint must never silently reuse its scores.
            run=next(suite.glob('runs/BikeNYC/ablation/A01/random/rate0.4/*/config.json'))
            old=common.load(run);old['seed']=123;common.write_json(run,old)
            bad=subprocess.run(cmd+['--dry-run'],cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(bad.returncode,0)


if __name__=='__main__':unittest.main()
