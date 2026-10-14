import copy,json,sys,tempfile,unittest,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v25'),str(ROOT/'scripts/v24')]
from run_bikenyc_scale_delta_rates import jobs,ensure_run_pointer
from run_cross_dataset_validation import jobs as reference_jobs
from run_core_validation import prepare_initialization
from stmoe_imputer.data.observed_multiscale import aggregate_sum
from stmoe_imputer.data import build_datasets
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils.deterministic import state_hash
from report_bikenyc_scale_delta_rates import export_report

class BikeRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setup_jobs(self,root):
        outputs={};rng=np.random.default_rng(57)
        for split in ('train','val','test'):
            x=rng.uniform(0,51,size=(4,2,3,8,8)).astype(np.float32);p=root/f'bikenyc_{split}.npz'
            np.savez(p,x_f_gt=x,x_m_gt=aggregate_sum(x,2),x_c_gt=aggregate_sum(x,4))
            outputs[split]={'path':str(p),'shapes':{'F':list(x.shape)}}
        (root/'manifest.json').write_text(json.dumps({'outputs':outputs}))
        js=jobs(root,root)
        for j in js.values():
            c=j['config'];c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
            c['data'].update(batch_size=2,pin_memory=False)
            c['train'].update(epochs=2,val_epoch=1);c['train']['scheduler']['total_epochs']=2
        return js

    def test_rates_masks_paths_initialization_and_reference_compatibility(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);js=self.setup_jobs(root);suite=root/'suite';suite.mkdir()
            self.assertEqual(len(js),19);prepare_initialization(suite,js)
            audit=json.loads((suite/'initialization_audit.json').read_text());self.assertEqual(len({r['full_sha256'] for r in audit.values()}),1)
            masks={}
            for n,j in js.items():
                c=j['config'];rate=j['rate'];self.assertEqual(c['experiment_plan']['protocol']['rate'],rate)
                ds,vs=build_datasets(c,j['sources']['train'],j['sources']['val'])
                for data in (ds,vs):
                    for i in range(len(data)):
                        b=data[i]
                        for scale in ('f','m','c'):
                            m=b['m_'+scale][0];self.assertEqual(int((m==0).sum()),round(m.numel()*rate))
                if rate in masks:self.assertEqual(state_hash(ds[0]),masks[rate])
                masks[rate]=state_hash(ds[0])
                ensure_run_pointer(suite,j);pointer=suite/'runs'/f'{n}.json';saved=pointer.read_text();self.assertIn(f'rate{rate}',saved)
                ensure_run_pointer(suite,j);self.assertEqual(saved,pointer.read_text())
            self.assertNotEqual(masks[.4],masks[.6])
            for n in ('bikenyc_r04_S04_seed7','bikenyc_r06_B1_seed7'):
                j=js[n];old=copy.deepcopy(j['config']);old['model']['coe'].pop('scale_delta_exploration')
                torch.manual_seed(7);a=DualBranchSTImputer.from_config(old)
                torch.manual_seed(7);b=DualBranchSTImputer.from_config(j['config'])
                self.assertEqual(state_hash(a.state_dict()),state_hash(b.state_dict()))
                for m in (a,b):
                    with torch.no_grad():m.main_branch.history_gate[-1].bias.fill_(.3)
                ds,_=build_datasets(j['config'],j['sources']['train'],j['sources']['val']);batch={k:torch.stack([ds[i][k] for i in (0,1)]) for k in ds[0]}
                with torch.no_grad():torch.testing.assert_close(a(batch)['x_hat_main'],b(batch)['x_hat_main'],rtol=0,atol=0)
            default=jobs(root,root)
            for j in default.values():
                self.assertEqual(j['config']['train']['epochs'],200);self.assertEqual(j['config']['train']['scheduler']['total_epochs'],200)

    def test_train_evaluate_report_keeps_rates_separate(self):
        from train_four_direction import train
        from evaluate_four_direction import evaluate_sets
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);js=self.setup_jobs(root);suite=root/'suite';suite.mkdir();prepare_initialization(suite,js)
            refs={}
            for method in ('B1','B2'):
                j=copy.deepcopy(js['bikenyc_r04_S04_seed7']);j['method']=method;j['variant']='ref_'+method
                j['config']['model']['coe'].pop('scale_delta_exploration')
                j['config']['model']['coe']['core_validation']['communication']='none' if method=='B1' else 'conditional'
                d=suite/'references'/method;run=root/('ref_'+method)
                train(j,run,d/'receipt.json','cpu')
                evaluate_sets(run/'checkpoints/best.pth',j['config']['experiment_plan']['protocol'],j['sources']['test'],d/'evaluation.json',device='cpu')
                shutil.copyfile(run/'config.json',d/'config.json');shutil.copyfile(run/'logs/metrics.jsonl',d/'history.json')
            self.assertIn('阶段',export_report(root,suite,js).name)
            for n,j in js.items():
                run=root/n;train(j,run,suite/'results'/f'{n}.json','cpu')
                evaluate_sets(run/'checkpoints/best.pth',j['config']['experiment_plan']['protocol'],j['sources']['test'],suite/'evaluations'/f'{n}.json',device='cpu')
            path=export_report(root,suite,js);text=path.read_text()
            self.assertTrue(json.loads((suite/'report_path.json').read_text())['complete'])
            self.assertIn('|0.6|S04/B1|',text);self.assertNotIn('|0.6|S04/B2|',text)
            for n in js:self.assertIn(n,text)
            p=suite/'evaluations/bikenyc_r06_S04_seed7.json';e=json.loads(p.read_text());e['sets']['in_distribution']['mask_spec']['rates']=[.4];p.write_text(json.dumps(e))
            self.assertIn('阶段',export_report(root,suite,js).name)

if __name__=='__main__':unittest.main()
