import copy,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v25'),str(ROOT/'scripts/v24')]
from prepare_cross_dataset import SPECS,prepare
from run_cross_dataset_validation import jobs,initialize
from stmoe_imputer.data import build_datasets
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import state_hash

class CrossDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setup_jobs(self,root):
        roots={}
        for ds,spec in SPECS.items():
            directory=root/'data'/spec['directory'];directory.mkdir(parents=True)
            # Distinct frames, two windows per split; one cross-split duplicate
            # window is deliberately supplied to validate purging and indexing.
            channels=2 if ds=='bikenyc' else 1
            rng=np.random.default_rng(718)
            original=rng.uniform(1,50,size=(6,channels,3,8,8)).astype(np.float32)
            for split,inds in [('train',[0,1]),('val',[1,2,3]),('test',[3,4,5])]:
                np.savez(directory/f'{spec["prefix"]}_{split}.npz',x_f_gt=original[inds])
            roots[ds]=prepare(ds,root)
            manifest=json.loads((roots[ds]/'manifest.json').read_text())
            self.assertEqual(manifest['outputs']['val']['rejected_indices'],[0])
            self.assertEqual(manifest['outputs']['test']['rejected_indices'],[0])
            self.assertEqual(prepare(ds,root),roots[ds])
        js=jobs(roots,root)
        for j in js.values():
            c=j['config'];c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
            c['data'].update(batch_size=2,pin_memory=False)
            c['train'].update(epochs=2,val_epoch=1);c['train']['scheduler']['total_epochs']=2
        return js

    def test_mean_units_masks_and_hidden_exclusion(self):
        with tempfile.TemporaryDirectory() as td:
            js=self.setup_jobs(Path(td));j=js['chap_B2_seed7']
            ds,_=build_datasets(j['config'],j['sources']['train'],j['sources']['val'])
            b={k:torch.stack([ds[i][k] for i in (0,1)]) for k in ds[0]}
            torch.manual_seed(7);mean=DualBranchSTImputer.from_config(j['config']).eval()
            c=copy.deepcopy(j['config']);c['data']['scales']['pooling_mode']='sum'
            torch.manual_seed(7);summed=DualBranchSTImputer.from_config(c).eval()
            self.assertEqual(state_hash(mean.state_dict()),state_hash(summed.state_dict()))
            sums=copy.deepcopy(b);sums['x_m_obs']*=4;sums['x_c_obs']*=16
            with torch.no_grad():
                a=mean(b);z=summed(sums)
                torch.testing.assert_close(a['x_hat_main'],z['x_hat_main'],rtol=0,atol=0)
                self.assertIn('v25_observed_mean_residual_M',a['coe']['diagnostics'])
                corrupt=copy.deepcopy(b);corrupt['x_f_gt'].fill_(99999)
                for name in ('f','m','c'):corrupt['x_'+name+'_obs'][~corrupt['m_'+name].bool()]=float('nan')
                torch.testing.assert_close(mean(corrupt)['x_hat_main'],a['x_hat_main'],rtol=0,atol=0)
            for name in ('m','c'):b['x_'+name+'_obs'].requires_grad_()
            out=mean(b);loss,_=compute_coe_loss(out,b,j['config']);loss.backward()
            for name in ('m','c'):
                g=b['x_'+name+'_obs'].grad
                self.assertTrue(bool(g[b['m_'+name].bool()].abs().sum()>0))
                self.assertFalse(bool(g[~b['m_'+name].bool()].any()))

    def test_eight_cell_train_resume_and_report(self):
        from train_four_direction import train
        from evaluate_four_direction import evaluate_sets
        from report_cross_dataset_validation import export_report
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);js=self.setup_jobs(root);suite=root/'suite';suite.mkdir();initialize(suite,js)
            audit=json.loads((suite/'initialization_audit.json').read_text())
            for ds in SPECS:
                self.assertEqual(len({v['full_sha256'] for k,v in audit.items() if k.startswith(ds+'_')}),1)
            self.assertNotEqual(audit['chap_A1_seed7']['full_sha256'],audit['bikenyc_A1_seed7']['full_sha256'])
            self.assertIn('阶段',export_report(root,suite,js).name)
            for n,j in js.items():
                run=root/n;result=suite/'results'/f'{n}.json'
                if n=='chap_B2_seed7':train(j,run,result,'cpu',stop_after=1)
                train(j,run,result,'cpu')
                if n=='chap_B2_seed7':
                    train(j,root/'uninterrupted',root/'uninterrupted.json','cpu')
                    a=torch.load(run/'checkpoints/last.pth',weights_only=False);b=torch.load(root/'uninterrupted/checkpoints/last.pth',weights_only=False)
                    for key in ('model','optimizer','scheduler','scaler','rng_states'):self.assertEqual(state_hash(a[key]),state_hash(b[key]))
                evaluate_sets(run/'checkpoints/best.pth',j['config']['experiment_plan']['protocol'],j['sources']['test'],suite/'evaluations'/f'{n}.json',device='cpu')
            path=export_report(root,suite,js)
            self.assertTrue(json.loads((suite/'report_path.json').read_text())['complete'])
            text=path.read_text()
            for n in js:self.assertIn(n,text)
            self.assertIn('chap_B2/chap_B1',text);self.assertIn('bikenyc_B2/bikenyc_B1',text)
            ev=suite/'evaluations/chap_B2_seed7.json';obj=json.loads(ev.read_text());obj['npz']='wrong';ev.write_text(json.dumps(obj))
            self.assertIn('阶段',export_report(root,suite,js).name)

if __name__=='__main__':unittest.main()
