from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from stmoe_imputer.data import build_loader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "v18-single"))

from pipeline_common import (  # noqa: E402
    _formal_signature,
    build_resolved_config,
    comparison_protocol_signature,
)


class V18PipelineTest(unittest.TestCase):
    def test_resolved_config_records_mask_hash_and_mask_specific_policy(self) -> None:
        fixed = build_resolved_config("TaxiBJ", "fixed", "0.4", 42)
        random = build_resolved_config("TaxiBJ", "random", "0.4", 42)
        for config in (fixed, random):
            self.assertEqual(
                set(config["data"]["mask"]["sha256"]),
                {"train", "val", "test"},
            )
            self.assertEqual(
                set(config["data"]["sha256"]),
                {"train", "val", "test"},
            )
            self.assertEqual(
                len(config["reproducibility"]["source_sha256"]),
                64,
            )
            self.assertEqual(
                config["evaluation"]["masked_metric_aggregation"],
                "global_missing_elements",
            )
        self.assertFalse(fixed["train"]["early_stopping"]["enabled"])
        self.assertTrue(random["train"]["early_stopping"]["enabled"])

    def test_short_or_changed_protocol_cannot_match_formal_signature(self) -> None:
        formal = build_resolved_config("BikeNYC", "random", "0.4", 42)
        short = copy.deepcopy(formal)
        short["train"]["epochs"] = 2
        changed_main = copy.deepcopy(formal)
        changed_main["model"]["main"]["dim"] = 16
        self.assertNotEqual(
            _formal_signature(formal), _formal_signature(short)
        )
        self.assertNotEqual(
            _formal_signature(formal), _formal_signature(changed_main)
        )

    def test_pairing_rejects_old_metric_or_early_stopping_protocol(self) -> None:
        formal = build_resolved_config("BikeNYC", "random", "0.4", 42)
        old_metrics = copy.deepcopy(formal)
        old_metrics.pop("evaluation")
        no_early_stop = copy.deepcopy(formal)
        no_early_stop["train"]["early_stopping"]["enabled"] = False
        expected = comparison_protocol_signature(formal)
        self.assertNotEqual(
            expected, comparison_protocol_signature(old_metrics)
        )
        self.assertNotEqual(
            expected, comparison_protocol_signature(no_early_stop)
        )

    def test_shuffle_order_does_not_depend_on_model_rng_consumption(self) -> None:
        config = {
            "seed": 42,
            "data": {
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
                "drop_last": False,
            },
        }
        dataset = TensorDataset(torch.arange(20))
        first_loader = build_loader(dataset, config, shuffle=True)
        _ = torch.randn(1000)
        second_loader = build_loader(dataset, config, shuffle=True)
        first_order = torch.cat([batch[0] for batch in first_loader])
        _ = torch.randn(2000)
        second_order = torch.cat([batch[0] for batch in second_loader])
        self.assertTrue(torch.equal(first_order, second_order))


if __name__ == "__main__":
    unittest.main()
