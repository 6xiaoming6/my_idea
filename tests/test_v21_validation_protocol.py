from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts/v21-single"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _protocol import ALL24, CORE6, VARIANTS, filtered_points, run_root


class V21ValidationProtocolTest(unittest.TestCase):
    def test_protocol_matrices_have_expected_unique_points(self) -> None:
        self.assertEqual(len(CORE6), 6)
        self.assertEqual(len(set(CORE6)), 6)
        self.assertEqual(len(ALL24), 24)
        self.assertEqual(len(set(ALL24)), 24)
        self.assertEqual(
            len(filtered_points("all24", ("CHAP",), ("random",), ("0.2", "0.4"))),
            2,
        )

    def test_variant_metadata_matches_controlled_configs(self) -> None:
        config_paths = {
            "P0": ROOT / "configs/v21-single/ablations/legacy_pyramid.json",
            "P1": ROOT / "configs/v21-single/ablations/content_only.json",
            "P2": ROOT / "configs/v21-single/observation_moment_dual_state.json",
        }
        for key, path in config_paths.items():
            with self.subTest(variant=key):
                cfg = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    cfg["experiment_policy"]["name"], VARIANTS[key].policy_name
                )

    def test_formal_runs_are_separated_from_debug_outputs(self) -> None:
        for variant in VARIANTS.values():
            root = run_root(variant, CORE6[0])
            self.assertIn("/ablation/v21_", str(root))
            self.assertNotIn("/debug/", str(root))


if __name__ == "__main__":
    unittest.main()
