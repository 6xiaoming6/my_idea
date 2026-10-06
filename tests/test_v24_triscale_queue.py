from datetime import datetime,timedelta,timezone
from pathlib import Path
import sys
import copy
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts/v24'))
import run_triscale_exploration as r


def evidence():
    return {k:{'val_mae':10.,'evaluations':{s:{'mae':20.} for s in r.DEV_SETS}} for k in r.VARIANTS[:5]}


class QueueTests(unittest.TestCase):
    def test_selection_gate_and_fallback(self):
        e=evidence();e['T3']['val_mae']=9.;e['T2']['val_mae']=9.1;e['T4']['val_mae']=12.
        for s in r.DEV_SETS:e['T4']['evaluations'][s]['mae']=1.
        a=r.select_candidates(e);self.assertEqual(a['B'],'T3');self.assertEqual(a['A'],'T2');self.assertTrue(a['id_gate_passed'])
        e['T2']['val_mae']=11.;a=r.select_candidates(e);self.assertEqual(a['A'],'T4');self.assertFalse(a['id_gate_passed'])
        with self.assertRaises(ValueError):r.select_candidates({'T1':e['T1']})
    def test_freeze_and_resolved_configs(self):
        plan=r.jobs();selection=r.select_candidates(evidence());templates={v:r.baseline.load(r.CONFIG_DIR/f'{v}.json') for v in r.VARIANTS[5:]}
        for variant in r.VARIANTS[5:]:
            job=r.resolve_job(variant,plan,selection,templates);cfg=job['config']
            self.assertEqual(cfg['train']['epochs'],100);self.assertEqual(cfg['data']['batch_size'],32)
            if variant in ('T6','T7'):self.assertEqual(cfg['seed'],17);self.assertEqual(cfg['data']['loader_seed'],17)
            if variant=='T8':self.assertTrue(cfg['model']['coe']['triscale']['recent_memory'])
            if variant=='T9':self.assertEqual(cfg['model']['coe']['expert_sharing'],'per_step')
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'selection.json';r.frozen_json(p,selection);stamp=p.stat().st_mtime_ns;r.frozen_json(p,selection);self.assertEqual(stamp,p.stat().st_mtime_ns)
            with self.assertRaises(RuntimeError):r.frozen_json(p,{**selection,'A':'T5'})
    def test_order_dataset_and_budget(self):
        self.assertEqual(r.required_variants(['T9']),['T1','T2','T3','T4','T5','T9'])
        self.assertEqual(r.required_variants(['T3','T1']),['T1','T3'])
        for job in r.jobs('bikenyc',42,13):
            self.assertEqual(job['config']['train']['epochs'],42);self.assertEqual(job['config']['data']['batch_size'],13)
        now=datetime.now(timezone.utc);self.assertTrue(r.can_start_optional(now,now+timedelta(seconds=100),99));self.assertFalse(r.can_start_optional(now,now+timedelta(seconds=100),101))

if __name__=='__main__':unittest.main()
