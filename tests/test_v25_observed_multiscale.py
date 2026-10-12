import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v25'),str(ROOT/'scripts/v24')]
from run_first_validation import jobs,prepare_initialization,protocol
from stmoe_imputer.data.observed_multiscale import aggregate_sum,ObservedMultiScaleDataset
from stmoe_imputer.data import build_datasets,build_test_dataset
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import state_hash
from stmoe_imputer.config import uses_multiscale


class V25Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setup_jobs(self,root):
        x=np.arange(4*2*3*8*8,dtype=np.float32).reshape(4,2,3,8,8)%51
        for split in ('train','val','test'):
            np.savez(root/f'taxibj_{split}.npz',x_f_gt=x,x_m_gt=aggregate_sum(x,2),x_c_gt=aggregate_sum(x,4))
        js=jobs(root)
        for j in js.values():
            c=j['config'];c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
            c['data'].update(batch_size=2,pin_memory=False)
        return js

    def model(self,j):
        torch.manual_seed(7);return DualBranchSTImputer.from_config(j['config'])

    def batch(self,j):
        ds,_=build_datasets(j['config'],j['sources']['train'],j['sources']['val'])
        return {k:torch.stack([ds[i][k] for i in (0,1)]) for k in ds[0]}

    def test_v24_mixed_id_ood_protocol(self):
        expected=json.loads((ROOT/'configs/v24/core_validation/test_protocol.json').read_text())
        actual=protocol()
        for key,value in expected.items():self.assertEqual(actual[key],value)
        self.assertEqual(len(actual['evaluations']),6)
        with tempfile.TemporaryDirectory() as td:
            js=self.setup_jobs(Path(td))
            for j in js.values():
                c=j['config']
                for key in ('train_mask_diversity','eval_mask_diversity'):
                    self.assertEqual(c['data'][key]['families'],expected['train_families'])
                    self.assertEqual(c['data'][key]['rates'],[expected['rate']])
                    self.assertEqual(c['data'][key]['seed'],expected['mask_seed'])
                ds,val=build_datasets(c,j['sources']['train'],j['sources']['val'])
                for dataset in (ds,val):
                    self.assertEqual(set(dataset.diverse_masks.assignments.tolist()),set(range(4)))
                self.assertEqual(c['experiment_plan']['protocol'],actual)

    def test_physical_aggregation_masks_and_replay(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);js=self.setup_jobs(root);a=js['taxibj_A1_seed7'];b=js['taxibj_B1_seed7']
            da,va=build_datasets(a['config'],a['sources']['train'],a['sources']['val'])
            db,vb=build_datasets(b['config'],b['sources']['train'],b['sources']['val'])
            before=torch.get_rng_state();first=state_hash(db[0]);self.assertTrue(torch.equal(before,torch.get_rng_state()))
            for index in range(len(da)):
                left,right=da[index],db[index]
                self.assertTrue(torch.equal(left['m_f'],right['m_f']))
                self.assertFalse(left['m_m'].any());self.assertFalse(left['m_c'].any())
                for sid,name in enumerate(('m','c'),1):
                    m=right['m_'+name];x=right['x_'+name+'_obs']
                    expected=torch.from_numpy(db.values[sid][index]);torch.testing.assert_close(x[m.bool()],expected[m.bool()],rtol=0,atol=0)
                    self.assertEqual(int((m[0]==0).sum()),round(m[0].numel()*.4))
                    self.assertFalse(x[~m.bool()].any())
                # An observed region can contain hidden fine targets.
                block_hidden=aggregate_sum((1-right['m_f']).numpy()[None],2)[0]>0
                self.assertTrue(bool((torch.from_numpy(block_hidden)&right['m_m'].bool()).any()))
            db.set_epoch(2);self.assertNotEqual(first,state_hash(db[0]));db.set_epoch(1);self.assertEqual(first,state_hash(db[0]))
            old=state_hash(vb[0]);vb.set_epoch(5);self.assertEqual(old,state_hash(vb[0]))
            # No masked-F fallback accepted when precomputed scales are missing.
            np.savez(root/'bad.npz',x_f_gt=db.values[0])
            with self.assertRaises(KeyError):ObservedMultiScaleDataset(root/'bad.npz',b['config']['data']['train_mask_diversity'])
            self.assertTrue(uses_multiscale(b['config']))
            oldcfg=copy.deepcopy(b['config']);oldcfg['model']['architecture']='v24_ts_coe';self.assertFalse(uses_multiscale(oldcfg))

    def test_real_observations_influence_predictions_and_no_hidden_leak(self):
        with tempfile.TemporaryDirectory() as td:
            js=self.setup_jobs(Path(td));j=js['taxibj_B2_seed7'];m=self.model(j).eval();batch=self.batch(j)
            with torch.no_grad():
                baseline=m(batch)['x_hat_main']
                for name in ('m','c'):
                    modified=copy.deepcopy(batch);modified['x_'+name+'_obs']+=modified['m_'+name]*10
                    self.assertGreater(float((m(modified)['x_hat_main']-baseline).abs().max()),1e-7,name)
                modified=copy.deepcopy(batch)
                for name in ('f','m','c'):
                    modified['x_'+name+'_obs'][~modified['m_'+name].bool()]=float('nan')
                modified['x_f_gt'].fill_(99999)
                torch.testing.assert_close(m(modified)['x_hat_main'],baseline,rtol=0,atol=0)
                m(copy.deepcopy(modified));torch.testing.assert_close(m(batch)['x_hat_main'],baseline,rtol=0,atol=0)
                self.assertFalse(hasattr(m.main_branch,'_observations'))
                for value in (0.,1.):
                    edge=copy.deepcopy(batch)
                    for name in ('f','m','c'):edge['m_'+name].fill_(value)
                    out=m(edge);self.assertTrue(bool(torch.isfinite(out['x_hat_main']).all()))
                    if value==1.:torch.testing.assert_close(out['x_comp'],edge['x_f_obs'],rtol=0,atol=0)
            for name in ('m','c'):batch['x_'+name+'_obs'].requires_grad_()
            out=m(batch);loss,_=compute_coe_loss(out,batch,j['config']);loss.backward()
            for name in ('m','c'):
                grad=batch['x_'+name+'_obs'].grad
                self.assertTrue(bool(grad[batch['m_'+name].bool()].abs().sum()>0))
                self.assertFalse(bool(grad[~batch['m_'+name].bool()].any()))

    def test_four_cells_initialization_sparse_execution_and_delta(self):
        with tempfile.TemporaryDirectory() as td:
            js=self.setup_jobs(Path(td));hashes=[];rngs=[]
            for j in js.values():
                m=self.model(j);hashes.append(state_hash(m.state_dict()));rngs.append(state_hash(torch.get_rng_state()))
                calls=[];hooks=[e.register_forward_pre_hook(lambda e,a:calls.append((len(a[0]),a[0].shape[-1]))) for e in m.main_branch.routed_experts()]
                batch=self.batch(j)
                if j['config']['model']['coe']['core_validation']['communication']=='none':
                    with patch.object(m.main_branch,'rms_match',side_effect=AssertionError('disabled delta called')):out=m(batch)
                else:out=m(batch)
                for h in hooks:h.remove()
                self.assertEqual(sum(n for n,w in calls),16);self.assertEqual(set(w for n,w in calls),{2,4,8})
                torch.testing.assert_close(out['coe']['triscale_choices'],torch.tensor([[2,1,0,0]]*2))
                loss,_=compute_coe_loss(out,batch,j['config']);loss.backward()
                grads=[p.grad for p in m.main_branch.history_gate.parameters() if p.grad is not None]
                if j['config']['model']['coe']['core_validation']['communication']=='none':self.assertFalse(grads)
                else:self.assertTrue(grads and any(bool(g.abs().sum()>0) for g in grads))
                self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None))
            self.assertEqual(len(set(hashes)),1);self.assertEqual(len(set(rngs)),1)
            prepare_initialization(Path(td)/'suite',js)

    def test_train_resume_eval_and_report(self):
        from train_four_direction import train
        from evaluate_four_direction import evaluate_sets
        from report_first_validation import export_report
        import run_four_direction_exploration as base
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);js=self.setup_jobs(root);suite=root/'suite';suite.mkdir()
            for j in js.values():
                c=j['config'];c['train'].update(epochs=2,val_epoch=1);c['train']['scheduler']['total_epochs']=2
            prepare_initialization(suite,js)
            for n,j in js.items():
                run=root/n;receipt=suite/'results'/f'{n}.json'
                train(j,run,receipt,'cpu',stop_after=1);r=train(j,run,receipt,'cpu')
                self.assertEqual(train(j,run,receipt,'cpu'),r)
                with patch.object(base,'subprocess_run',side_effect=AssertionError('completed run restarted')):base.launch(suite,j,0)
                train(j,root/(n+'_full'),root/(n+'_full.json'),'cpu')
                a=torch.load(run/'checkpoints/last.pth',weights_only=False);b=torch.load(root/(n+'_full')/'checkpoints/last.pth',weights_only=False)
                for k in ('model','optimizer','scheduler','scaler','rng_states'):
                    self.assertEqual(state_hash(a[k]),state_hash(b[k]),n+' '+k)
                self.assertEqual(state_hash(a['training_state']['w_training']),state_hash(b['training_state']['w_training']))
                small={'rate':.4,'evaluations':{'in_distribution':{'families':['random_point'],'seed':20260917}}}
                evaluate_sets(run/'checkpoints/best.pth',small,j['sources']['test'],suite/'evaluations'/f'{n}.json',device='cpu')
            # Only ID evaluated so the report must remain explicitly staged.
            p=export_report(root,suite,js);text=p.read_text();self.assertIn('阶段',text)
            self.assertLess(text.index('具体做法'),text.index('实验结果'))
            for n,j in js.items():
                evaluate_sets(root/n/'checkpoints/best.pth',protocol(),j['sources']['test'],suite/'evaluations'/f'{n}.json',device='cpu')
            p=export_report(root,suite,js)
            self.assertIn('完整',p.name)
            self.assertTrue(json.loads((suite/'report_path.json').read_text())['complete'])
            self.assertIn('差中之差',p.read_text())


if __name__=='__main__':unittest.main()
