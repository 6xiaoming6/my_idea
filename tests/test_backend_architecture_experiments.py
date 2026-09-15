from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

from test_dual_moe import ROOT, batch, config
from stmoe_imputer.config import deep_update, load_config
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.dual_moe import MultiScaleCompletionHead, ContextCompletionHead
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.engine import train_one_epoch, evaluate, build_optimizer
from stmoe_imputer.utils.checkpoint import snapshot_model_state

sys.path.insert(0,str(ROOT/'scripts'))
import run_backend_architecture_experiments as study


class BackendArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def cfg(self, kind='mlp'):
        cfg=deep_update(config(), load_config(ROOT/'configs/presets/dual_moe_shared_topk.json'))
        cfg['model']['dual_moe']['completion_expert_type']=kind
        return cfg

    def test_context_heads_spatial_and_temporal_receptive_field(self):
        for dilation in (1,2):
            head=ContextCompletionHead(3,4,1,dilation)
            x=torch.randn(1,3,7,7,7,requires_grad=True)
            head(x)[0,0,3,3,3].backward()
            self.assertGreater(x.grad[0,:,3+dilation,3,3].abs().sum().item(),0)
            self.assertGreater(x.grad[0,:,3,3+dilation,3].abs().sum().item(),0)
        plain=MultiScaleCompletionHead(3,4,1)
        x=torch.randn(1,3,7,7,7,requires_grad=True)
        plain(x)[0,0,3,3,3].backward()
        self.assertEqual(x.grad[0,:,4,3,3].abs().sum().item(),0)
        self.assertEqual(x.grad[0,:,3,4,3].abs().sum().item(),0)

    def test_context_preserves_common_initialization_and_topk(self):
        models={}
        for kind in ('mlp','st_local','st_dilated'):
            torch.manual_seed(42)
            models[kind]=DualBranchSTImputer.from_config(self.cfg(kind))
        reference=models['mlp'].state_dict()
        for kind in ('st_local','st_dilated'):
            for key,value in reference.items():
                torch.testing.assert_close(value,models[kind].state_dict()[key],rtol=0,atol=0)
        for kind, model in models.items():
            data=batch(t=3,h=4,w=5)
            output=model(data)
            self.assertTrue(torch.isfinite(output['x_hat_main']).all())
            self.assertTrue(((output['completion_gates']>0).sum(1)==3).all())
            loss,_=compute_main_stage_loss(output,data,self.cfg(kind),epoch=1)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for name,param in model.named_parameters():
                if param.grad is not None: self.assertTrue(torch.isfinite(param.grad).all(),name)
            if kind!='mlp':
                total=sum(p.grad.abs().sum().item() for n,p in model.named_parameters() if '.context.' in n and p.grad is not None)
                self.assertGreater(total,0)
            altered={**data,'x_f_gt':torch.full_like(data['x_f_gt'],float('nan')),
                     'x_f_obs':torch.where(data['m_f'].bool(),data['x_f_obs'],float('nan'))}
            with torch.no_grad():
                torch.testing.assert_close(output['x_hat_main'],model(altered)['x_hat_main'])

    def test_parameter_control_and_degenerate_shapes(self):
        a=ContextCompletionHead(103,32,2,1)
        b=ContextCompletionHead(103,32,2,2)
        c=MultiScaleCompletionHead(103,46,2)
        count=lambda m:sum(p.numel() for p in m.parameters())
        self.assertEqual(count(a),count(b))
        self.assertLess(abs(count(a)-count(c))/count(a),.01)
        for head in (a,b,c):
            self.assertEqual(head(torch.randn(1,103,1,1,1)).shape,(1,2,1,1,1))

    def test_cpu_train_val_restore_test_for_context_experts(self):
        for kind in ('st_local','st_dilated'):
            cfg=self.cfg(kind); cfg['train']['amp']=False
            model=DualBranchSTImputer.from_config(cfg)
            optimizer=build_optimizer(model,cfg)
            splits=[batch(n=1,t=3,h=4,w=5) for _ in range(3)]
            train=train_one_epoch(model,[splits[0]],optimizer,torch.device('cpu'),cfg,1)
            val=evaluate(model,[splits[1]],torch.device('cpu'),cfg)
            state=snapshot_model_state(model)
            with torch.no_grad(): next(model.parameters()).add_(10)
            model.load_state_dict(state)
            test=evaluate(model,[splits[2]],torch.device('cpu'),cfg)
            for metrics in (train,val,test):
                for name in ('loss','mae','rmse'):
                    self.assertTrue(torch.isfinite(torch.tensor(metrics[name])))

    def test_policy_jobs_and_full_sources(self):
        p=study.common.load(ROOT/'configs/presets/dual_moe_backend_architecture.json')
        study.validate(p)
        jobs=list(study.jobs(p,Path('/tmp/backend_architecture_test')))
        self.assertEqual(len(jobs),31)
        self.assertEqual(len({j['key'] for j in jobs}),31)
        self.assertEqual([j['stage'] for j in jobs],['upgrade']*13+['structure']*18)
        study.data_manifest(jobs)
        for j in jobs:
            self.assertEqual(j['expected_train_samples'],{'TaxiBJ':2491,'BikeNYC':511,'CHAP':2626}[j['dataset']])
            self.assertNotIn('training_selection',j['cfg']['data'])
            self.assertFalse(j['cfg']['train']['save_best_checkpoint'])
            self.assertEqual(j['cfg']['train']['epochs'],p['dataset_epochs'][j['dataset']])
            o=j['cfg']['model']['dual_moe']
            self.assertEqual((o['aggregation_experts'],o['aggregation_top_k']),(8,4))
            if not j['variant'].startswith('OLD'):
                self.assertEqual((o['completion_experts'],o['completion_top_k']),(8,3))
            self.assertIn(f'{j["pattern"]}_mask/{j["rate"]:g}',j['cfg']['data']['mask']['train_csv'])

    def test_summary_pairs_never_cross_conditions_and_tolerate_missing(self):
        p=study.common.load(ROOT/'configs/presets/dual_moe_backend_architecture.json')
        with tempfile.TemporaryDirectory() as tmp:
            suite=Path(tmp);jobs=list(study.jobs(p,suite))
            def result(job):
                if job['dataset']!='BikeNYC' or job['pattern']!='fixed':return None
                value=2 if job['variant']=='OLD_D' else 1
                return dict(val_mae=value,test_mae=value,test_rmse=value,best_epoch=2,
                            run_dir='fixture',best_val_metrics={'rmse':value})
            with mock.patch.object(study.runner,'completed',side_effect=result):
                rows=study.summarize(jobs,suite)
            report=study.common.load(suite/'comparison.json')
            self.assertEqual(sum(r['status']=='complete' for r in rows),5)
            self.assertTrue(all(r['dataset']=='BikeNYC' and r['pattern']=='fixed' and r['rate']==.4 for r in report['paired']))
            change=next(r for r in report['paired'] if r['candidate']=='NEW_MLP' and r['baseline']=='OLD_D')
            self.assertEqual(change['val_mae_pct'],-50)

    def test_unverified_runs_are_not_resumed_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            job={'key':'fixture','cfg':{'output_dir':str(Path(tmp)/'runs')}}
            path=Path(tmp)/'logs/fixture.status.json'
            for status in ('running','invalid'):
                study.common.write_json(path,{'status':status})
                with mock.patch.object(study.runner,'completed',return_value={'test_mae':1}) as check:
                    self.assertIsNone(study.completed(job)); check.assert_not_called()
            study.common.write_json(path,{'status':'verified'})
            with mock.patch.object(study.runner,'completed',return_value={'test_mae':1}):
                self.assertEqual(study.completed(job),{'test_mae':1})


if __name__=='__main__':unittest.main()
