import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/v24'))
import run_four_direction_exploration as base
from resume_history_followup_protocol_fix import scoped_evaluate_record


class ProtocolScopingTests(unittest.TestCase):
    def exercise(self, fn):
        with tempfile.TemporaryDirectory() as td:
            suite=Path(td); cp=suite/'best.pth'; npz=str(suite/'test.npz'); paths=[]
            legacy=suite/'protocols/evaluations.json';base.frozen(legacy,{'rate':.4})
            original=legacy.read_bytes()
            def evaluate(command,*args,**kwargs):
                get=lambda flag:command[command.index(flag)+1]
                pp=Path(get('--protocol')); protocol=base.load(pp); paths.append(pp)
                base.write(get('--output'),dict(status='finished',checkpoint=str(cp.resolve()),npz=npz,
                    config_sha256=get('--expected-sha'),protocol_sha256=base.digest(protocol),sets={}))
            with patch.object(base,'subprocess_run',side_effect=evaluate) as subprocess_run:
                for name,rate in [('Y04',.4),('Y05',.6),('Y06',.6)]:
                    protocol={'rate':rate,'evaluations':{'in_distribution':{'seed':7}}}
                    fn(suite,name,cp,name,protocol,npz,'evaluations',0)
                    fn(suite,name,cp,name,protocol,npz,'evaluations',0)
                self.assertEqual(subprocess_run.call_count,3)
            self.assertNotEqual(paths[0],paths[1]);self.assertEqual(paths[1],paths[2])
            self.assertEqual(legacy.read_bytes(),original)
            self.assertEqual(base.load(paths[0])['rate'],.4);self.assertEqual(base.load(paths[1])['rate'],.6)
            # If an existing protocol is corrupted, do not overwrite it or launch evaluation.
            paths[1].write_text(json.dumps({'rate':.8}))
            with patch.object(base,'subprocess_run',side_effect=AssertionError('must not launch')):
                with self.assertRaises(RuntimeError):
                    fn(suite,'Y07',cp,'Y07',{'rate':.6,'evaluations':{'in_distribution':{'seed':7}}},npz,'evaluations',0)

    def test_future_queue_uses_protocol_identity(self):
        self.exercise(base.evaluate_record)

    def test_frozen_queue_adapter_uses_protocol_identity(self):
        self.exercise(lambda *args,**kwargs:scoped_evaluate_record(base,*args,**kwargs))


if __name__=='__main__':unittest.main()
