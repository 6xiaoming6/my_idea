import copy,json,sys,tempfile,unittest,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/v25'),str(ROOT/'scripts/v24'),str(ROOT/'tests')]
import torch
from run_cross_dataset_validation import jobs,initialize
from test_v25_cross_dataset import CrossDatasetTests
from report_cross_dataset_validation import export_report

class Bike200Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_only_budget_changes_and_complete_four_cell_report(self):
        from train_four_direction import train
        from evaluate_four_direction import evaluate_sets
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);helper=CrossDatasetTests();small=helper.setup_jobs(root)
            data_root=Path(small['bikenyc_A1_seed7']['config']['experiment_plan']['data_manifest']).parent
            short=jobs({'bikenyc':data_root},root,100);long=jobs({'bikenyc':data_root},root,200)
            self.assertEqual(len(long),4)
            for n in long:
                a,b=copy.deepcopy(short[n]['config']),copy.deepcopy(long[n]['config'])
                self.assertEqual(b['train']['epochs'],200);self.assertEqual(b['train']['scheduler']['total_epochs'],200)
                b['train']['epochs']=100;b['train']['scheduler']['total_epochs']=100
                self.assertEqual(a,b)
            suite=root/'suite';suite.mkdir();(suite/'plan.json').write_text(json.dumps({'datasets':['bikenyc']}))
            js={n:j for n,j in small.items() if j['dataset']=='bikenyc'};initialize(suite,js)
            self.assertIn('阶段',export_report(root,suite,js).name)
            for n,j in js.items():
                run=root/n;train(j,run,suite/'results'/f'{n}.json','cpu')
                path=suite/'evaluations'/f'{n}.json'
                evaluate_sets(run/'checkpoints/best.pth',j['config']['experiment_plan']['protocol'],j['sources']['test'],path,device='cpu')
                ref=suite/'references/bikenyc100'/j['method'];ref.mkdir(parents=True);shutil.copyfile(path,ref/'evaluation.json')
            path=export_report(root,suite,js);text=path.read_text()
            self.assertTrue(json.loads((suite/'report_path.json').read_text())['complete'])
            self.assertIn('与原100 epoch的配对比较',text)
            self.assertNotIn('|chap_A1|',text)
            incomplete=dict(js);incomplete.pop('bikenyc_A2_seed7')
            self.assertIn('阶段',export_report(root,suite,incomplete).name)

if __name__=='__main__':unittest.main()
