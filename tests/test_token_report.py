"""Validate six-protocol grouping and complete/partial report handling offline."""
import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from release_utils import load,write
from report_tokens import report,scores
class ReportContract(unittest.TestCase):
 def test_missing_runs_and_equal_category_weights(self):
  with tempfile.TemporaryDirectory() as temp:
   base=Path(temp);source=base/'source';(source/'configs').mkdir(parents=True)
   protocol=load(ROOT/'configs/test_protocol.json');write(source/'configs/test_protocol.json',protocol)
   ref=base/'ref';ref.mkdir();ev={'status':'finished','sets':{k:{'metrics':{'mae':10.,'rmse':20.}} for k in protocol['evaluations']}}
   write(ref/'evaluation.json',ev)
   plan=dict(source=str(source),reference=str(ref),source_commit='TEST_ONLY',base_commit='TEST_ONLY',python=sys.executable,data_dir=str(base),report_dir=str(base/'reports'),jobs=load(ROOT/'configs/latent_tokens/suite.json'),initializations={})
   write(base/'plan.json',plan)
   path=report(base);text=path.read_text()
   self.assertIn('阶段报告',text)
   for i in range(1,11):self.assertIn(f'LT{i:02}',text)
   ev['sets']['unseen_triple']['metrics']['mae']=40.
   self.assertEqual(scores(ev,ev)['mae']['mean'],20.)
   self.assertFalse(load(base/'report_receipt.json')['complete'])
   for job in plan['jobs']:
    run=base/'runs'/job['name'];(run/'checkpoints').mkdir(parents=True)
    for name in ('best','last'):(run/'checkpoints'/f'{name}.pth').write_text('TEST ONLY')
    write(run/'result.json',dict(status='finished',completed_epochs=100,best_epoch=95,best_val_mae=8.,total_time_sec=3600))
    write(run/'evaluation.json',ev)
   path=report(base)
   self.assertTrue(load(base/'report_receipt.json')['complete'])
   self.assertEqual(len(load(base/'comparison.json')),10)
if __name__=='__main__':unittest.main()
