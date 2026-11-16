"""Budget isolation, learning-rate endpoints/resumption, and report coverage."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from release_utils import load, write, digest
from epoch_budget_common import validate_configs, INITIAL_HASH, completed, scores
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.engine import build_scheduler
from stmoe_imputer.utils.deterministic import state_hash
from report_epoch_budget import report


class BudgetContract(unittest.TestCase):
    def test_only_budget_and_eta_change_same_initialization(self):
        torch.set_num_threads(2)
        jobs = validate_configs()
        for job in jobs:
            torch.manual_seed(7)
            model = DualBranchSTImputer.from_config(load(ROOT / job['config']))
            self.assertEqual(state_hash(model.state_dict()), INITIAL_HASH)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 347930)

    def test_schedule_endpoints_monotonicity_and_recovery(self):
        for job in validate_configs():
            cfg = load(ROOT / job['config'])
            weight = torch.nn.Parameter(torch.ones(1))
            optimizer = torch.optim.AdamW([weight], lr=cfg['train']['lr_main'])
            scheduler = build_scheduler(optimizer, cfg); lrs = []; snapshot = None
            for epoch in range(job['epochs']):
                if epoch == 50:
                    snapshot = copy.deepcopy((optimizer.state_dict(), scheduler.state_dict()))
                lrs.append(optimizer.param_groups[0]['lr']); optimizer.step(); scheduler.step()
            self.assertEqual(lrs[0], .001)
            self.assertAlmostEqual(lrs[-1], job['eta_min'], places=14)
            self.assertTrue(all(a >= b for a, b in zip(lrs, lrs[1:])))
            optimizer.load_state_dict(snapshot[0]); scheduler.load_state_dict(snapshot[1])
            recovered = []
            for _ in range(50, job['epochs']):
                recovered.append(optimizer.param_groups[0]['lr']); optimizer.step(); scheduler.step()
            self.assertEqual(recovered, lrs[50:])

    def test_report_incomplete_complete_and_bad_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory); source = suite / 'source'
            (source / 'configs/epoch_budget').mkdir(parents=True)
            jobs = validate_configs(); protocol = load(ROOT / 'configs/test_protocol.json')
            write(source / 'configs/test_protocol.json', protocol)
            ref = suite / 'reference'; ref.mkdir()
            ev = dict(status='finished', selection='ID validation only', split='test', best_epoch=5,
                      protocol_sha256=digest(protocol), sets={k: dict(metrics=dict(mae=10., rmse=20.)) for k in protocol['evaluations']})
            write(ref / 'evaluation.json', ev)
            plan = dict(jobs=jobs, source=str(source), reference=str(ref), base_commit='TEST', source_commit='TEST',
                        python=sys.executable, report_dir=str(suite / 'reports'))
            write(suite / 'plan.json', plan)
            for job in jobs: write(source / job['config'], load(ROOT / job['config']))
            path = report(suite)
            self.assertIn('阶段报告', path.read_text())
            self.assertFalse(load(suite / 'report_receipt.json')['complete'])
            for job in jobs:
                cfg = load(source / job['config']); run = suite / 'runs' / job['name']
                (run / 'checkpoints').mkdir(parents=True)
                for name in ('best', 'last'): (run / 'checkpoints' / f'{name}.pth').write_text('TEST FIXTURE')
                record = copy.deepcopy(ev); record['config_sha256'] = digest(cfg)
                write(run / 'evaluation.json', record)
                write(run / 'initialization.json', dict(full_state_sha256=INITIAL_HASH))
                write(run / 'result.json', dict(status='finished', completed_epochs=job['epochs'],
                      config_sha256=digest(cfg), best_epoch=5, best_val_mae=10., total_time_sec=1))
                self.assertTrue(completed(run, job, cfg, protocol))
            path = report(suite)
            self.assertTrue(load(suite / 'report_receipt.json')['complete'])
            for job in jobs: self.assertIn(job['name'], path.read_text())
            self.assertEqual(scores(ev, ev)['mae']['ratio'], 1.)
            altered = copy.deepcopy(ev); altered['sets']['unseen_triple']['metrics']['mae'] = 40.
            self.assertEqual(scores(altered, ev)['mae']['mean'], 20.)
            job = jobs[0]; run = suite / 'runs' / job['name']; r = load(run / 'result.json')
            r['completed_epochs'] -= 1; write(run / 'result.json', r)
            with self.assertRaises(AssertionError): completed(run, job, load(source / job['config']), protocol)


if __name__ == '__main__': unittest.main()
