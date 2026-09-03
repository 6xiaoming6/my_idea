from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v20-single/run_compare_v14_v20.py"
SPEC = importlib.util.spec_from_file_location("v20_comparison_protocol", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {SCRIPT}")
PROTOCOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROTOCOL
SPEC.loader.exec_module(PROTOCOL)


class V20ComparisonProtocolTest(unittest.TestCase):
    def test_protocol_profiles_have_the_intended_paired_coverage(self) -> None:
        seeds = [42, 2026, 3407]
        self.assertEqual(len(PROTOCOL._profile_cases("screening", seeds)), 8)
        self.assertEqual(len(PROTOCOL._profile_cases("robust", seeds)), 12)
        self.assertEqual(len(PROTOCOL._profile_cases("full", seeds)), 24)
        comprehensive = PROTOCOL._profile_cases("comprehensive", seeds)
        self.assertEqual(len(comprehensive), 32)
        self.assertEqual(len(set(comprehensive)), 32)

    def test_comprehensive_main_protocol_has_64_paired_training_jobs(self) -> None:
        cases = PROTOCOL._profile_cases("comprehensive", [42, 2026, 3407])
        jobs = [
            PROTOCOL.Job(model, case, "comparison")
            for case in cases
            for model in ("v14", "v20")
        ]
        self.assertEqual(len(jobs), 64)
        self.assertEqual(sum(job.model == "v14" for job in jobs), 32)
        self.assertEqual(sum(job.model == "v20" for job in jobs), 32)


if __name__ == "__main__":
    unittest.main()
