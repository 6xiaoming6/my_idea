from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts/v21.1-single"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v21_1_protocol import CORE6, VARIANTS, run_root


class V21OneProtocolTest(unittest.TestCase):
    def test_core6_and_variant_contract(self) -> None:
        self.assertEqual(len(CORE6), 6)
        self.assertEqual(len(set(CORE6)), 6)
        self.assertEqual(set(VARIANTS), {"P0", "P1", "P2", "P3", "P4"})
        self.assertIsNone(VARIANTS["P0"].train_variant)
        self.assertEqual(VARIANTS["P3"].train_variant, "distortion_calibrated")

    def test_new_configs_match_protocol_metadata(self) -> None:
        paths = {
            "P3": ROOT / "configs/v21.1-single/distortion_calibrated.json",
            "P4": ROOT / "configs/v21.1-single/ablations/evidence_gate_no_distortion.json",
        }
        for key, path in paths.items():
            with self.subTest(variant=key):
                cfg = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(cfg["experiment_policy"]["name"], VARIANTS[key].policy_name)
                self.assertEqual(cfg["data"]["scales"]["pyramid_mode"], "dual_observation_moment")
                self.assertEqual(cfg["model"]["architecture"], "v21_distortion_calibrated_moe")

    def test_old_results_are_reused_and_new_results_are_isolated(self) -> None:
        old = run_root(VARIANTS["P0"], CORE6[0])
        p3 = run_root(VARIANTS["P3"], CORE6[0])
        p4 = run_root(VARIANTS["P4"], CORE6[0])
        self.assertIn("/outputs/v21-single/", str(old))
        self.assertIn("/outputs/v21.1-single/", str(p3))
        self.assertIn("/outputs/v21.1-single/", str(p4))
        self.assertNotEqual(p3, p4)


if __name__ == "__main__":
    unittest.main()
