from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from stmoe_imputer.engine import build_optimizer
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils.checkpoint import load_checkpoint, save_checkpoint

from _v18_utils import compact_v18_config, make_batch


class V18CheckpointTest(unittest.TestCase):
    def test_v18_checkpoint_round_trip_preserves_predictions(self) -> None:
        cfg = compact_v18_config()
        batch = make_batch(cfg)
        model = DualBranchSTImputer.from_config(cfg).eval()
        optimizer = build_optimizer(model, cfg)
        with torch.no_grad():
            expected = model(batch)["x_hat_final"].clone()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(path, model, optimizer, 3, {"val_mae": 1.0}, cfg)
            restored = DualBranchSTImputer.from_config(cfg).eval()
            restored_optimizer = build_optimizer(restored, cfg)
            checkpoint = load_checkpoint(
                path, restored, restored_optimizer, map_location="cpu"
            )
            with torch.no_grad():
                actual = restored(batch)["x_hat_final"]

        self.assertEqual(checkpoint["epoch"], 3)
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
