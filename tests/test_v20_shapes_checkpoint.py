from __future__ import annotations

import io
import torch
import unittest

from stmoe_imputer.models import DualBranchSTImputer

from _v14_utils import make_batch
from _v20_utils import compact_v20_config


class V20ShapeCheckpointTest(unittest.TestCase):
    def test_three_dataset_geometries_have_finite_outputs(self) -> None:
        for channels, time_steps, height, width in (
            (2, 3, 32, 32),
            (2, 3, 24, 12),
            (1, 3, 40, 40),
        ):
            with self.subTest(shape=(channels, time_steps, height, width)):
                cfg = compact_v20_config(channels, time_steps, height, width)
                batch = make_batch(cfg)
                model = DualBranchSTImputer.from_config(cfg).eval()
                with torch.no_grad():
                    outputs = model(batch)
                self.assertEqual(outputs["x_hat_final"].shape, batch["x_f_gt"].shape)
                self.assertTrue(torch.isfinite(outputs["x_hat_final"]).all())
                for scale in outputs["v20_probe"]["routing_evidence"]:
                    competence = outputs["v20_probe"][scale]["competence"]
                    self.assertTrue(torch.isfinite(competence).all())
                    torch.testing.assert_close(
                        competence.sum(dim=-1), torch.ones(competence.shape[0]), atol=1e-6, rtol=0.0
                    )

    def test_state_dict_round_trip_keeps_probe_parameters(self) -> None:
        cfg = compact_v20_config()
        first = DualBranchSTImputer.from_config(cfg).eval()
        buffer = io.BytesIO()
        torch.save(first.state_dict(), buffer)
        buffer.seek(0)
        second = DualBranchSTImputer.from_config(cfg).eval()
        second.load_state_dict(torch.load(buffer, weights_only=True))
        batch = make_batch(cfg)
        with torch.no_grad():
            expected = first(batch)["x_hat_final"]
            actual = second(batch)["x_hat_final"]
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
