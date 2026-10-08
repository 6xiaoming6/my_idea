import gzip,json,math,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from stmoe_imputer.utils.metric_logging import compact_metrics,bounded_diagnostics,MetricLogPolicy,read_epoch_history
from stmoe_imputer.utils.train_logger import TrainLogger

class MetricLoggingTests(unittest.TestCase):
 def metrics(self):
  m={'loss':2.,'mae':1.9,'rmse':3.,'lr':.001,'coe_expert_grid_equivalents':4.625,'coe_step2_memory_gate_abs_mean':.08,'train_skipped_amp_steps':0.,'coe_condition_family_random_point_step1_T_prob':.3}
  for prefix in ('coe_pair_path','coe_condition_family_random_point_pair_path','coe_scale_execution_path'):
   for i in range(1000):m[f'{prefix}_route{i}_fraction']=1/1000
   m[prefix+'_max_fraction']=.001;m[prefix+'_entropy']=math.log(1000);m[prefix+'_unique_count']=1000.
  return m
 def test_compaction_retains_quality_cost_and_gate_without_path_tables(self):
  m=self.metrics();c=compact_metrics(m)
  for k in ('loss','mae','rmse','lr','coe_expert_grid_equivalents','coe_step2_memory_gate_abs_mean','train_skipped_amp_steps'):self.assertEqual(c[k],m[k])
  self.assertFalse(any('_route0_' in k or k.startswith('coe_condition_') for k in c))
  self.assertEqual(c['coe_pair_path_unique_count'],1000);self.assertEqual(compact_metrics(c),c)
  self.assertLess(len(json.dumps(c)),len(json.dumps(m))*.02)
 def test_topk_bounds_each_distribution_preserves_tail_mass_and_entropy(self):
  m=self.metrics();d=bounded_diagnostics(m,10)
  for prefix in ('coe_pair_path','coe_condition_family_random_point_pair_path','coe_scale_execution_path'):
   self.assertEqual(sum(k.startswith(prefix+'_route') for k in d),10)
   self.assertAlmostEqual(d[prefix+'_other_fraction'],.99)
   self.assertEqual(d[prefix+'_unique_count'],1000)
   self.assertAlmostEqual(d[prefix+'_entropy'],math.log(1000))
  self.assertIn('coe_condition_family_random_point_step1_T_prob',d)
 def test_schedule_restore_prunes_future_and_reads_old_or_new(self):
  with tempfile.TemporaryDirectory() as td:
   log=Path(td)/'logs';policy=MetricLogPolicy(total_epochs=7);m=self.metrics();rows=[{'epoch':i,'train':m,'val':m if i%5==0 else None,'perf':{'epoch_time_sec':1},'is_best':i==5} for i in range(1,8)]
   for row in rows:policy.write_diagnostics(log,row);policy.append(log,row)
   self.assertEqual([p.name for p in sorted((Path(td)/'diagnostics').glob('*.gz'))],['epoch_00001.json.gz','epoch_00005.json.gz','epoch_00007.json.gz'])
   compact=policy.restore(log,rows[:5]);self.assertFalse(policy.diagnostic_path(log,7).exists())
   self.assertEqual(len(list(read_epoch_history(log/'metrics.jsonl'))),5)
   rich=list(read_epoch_history(log/'metrics.jsonl',True));self.assertIn('coe_pair_path_route0_fraction',rich[0]['train']);self.assertNotIn('coe_pair_path_route0_fraction',rich[1]['train'])
   simple=[json.loads(l) for l in (log/'train.log').read_text().splitlines()];self.assertEqual(set(simple[0]['metrics']),{'loss','mae','rmse','lr'})
   policy.diagnostic_path(log,1).unlink();policy.restore(log,compact)
   with gzip.open(policy.diagnostic_path(log,1),'rt') as f:self.assertEqual(json.load(f)['detail_source'],'checkpoint_summary_only')
   old=log/'legacy.jsonl';old.write_text(json.dumps(rows[0])+'\n');self.assertEqual(next(read_epoch_history(old)),rows[0])
 def test_general_logger_full_test_only_and_inputs_unchanged(self):
  with tempfile.TemporaryDirectory() as td:
   m=self.metrics();before=json.dumps(m,sort_keys=True);log=Path(td)/'logs';logger=TrainLogger(log,total_epochs=2)
   logger.log_header({"large_config_marker": "do not duplicate config"});logger.log_epoch(1,m,m);logger.log_epoch(2,m,m);logger.log_test(m);logger.close()
   self.assertEqual(before,json.dumps(m,sort_keys=True))
   self.assertNotIn('large_config_marker', (log/'train.log').read_text());self.assertNotIn('coe_pair_path', (log/'train.log').read_text());self.assertNotIn('coe_pair_path', (log/'test.log').read_text())
   rows=[json.loads(l) for l in (log/'metrics.jsonl').read_text().splitlines()];self.assertNotIn('coe_pair_path_route0_fraction',rows[2]['metrics'])
   with gzip.open(Path(td)/'diagnostics/test.json.gz','rt') as f:full=json.load(f)
   self.assertEqual(full['metrics'],m)
   self.assertEqual(list(read_epoch_history(log/'metrics.jsonl',True))[-1]['metrics'],m)
 def test_nonfinite_diagnostics_and_invalid_policy(self):
  m=self.metrics();m['coe_pair_path_invalid_fraction']=float('nan');m['coe_step1_router_grad_norm']=float('inf')
  c=compact_metrics(m);self.assertEqual(c['coe_step1_router_grad_norm'],'inf');json.dumps(c,allow_nan=False)
  d=bounded_diagnostics(m);self.assertEqual(d['coe_pair_path_nonfinite_entry_count'],1);json.dumps(d,allow_nan=False)
  for cfg in ({'diagnostic_every':0},{'path_top_k':False},{'path_top_k':-2}):
   with self.assertRaises(ValueError):MetricLogPolicy(cfg)
if __name__=='__main__':unittest.main()
