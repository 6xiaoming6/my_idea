from __future__ import annotations

import unittest
import json
from pathlib import Path

import torch

from stmoe_imputer.models.fusion import ReliabilityAwareScaleGate
from stmoe_imputer.config import deep_update
from stmoe_imputer.models.v_single import V14SafeC2FMoE

from _v14_utils import backbone_kwargs, compact_v14_config, make_batch


ROOT = Path(__file__).resolve().parents[1]


class ESAPScaleGateTest(unittest.TestCase):
    @staticmethod
    def _inputs(batch_size: int = 2, dim: int = 4):
        generator = torch.Generator().manual_seed(19)
        features = [
            torch.randn(batch_size, dim, 2, 2, 2, generator=generator)
            for _ in range(3)
        ]
        q_f = torch.tensor(
            [[0.8, 0.2, 0.0, 0.8, 0.2], [0.4, 0.6, 0.0, 0.4, 0.6]]
        )
        q_m = torch.tensor(
            [[0.4, 0.6, 0.0, 0.4, 0.6], [0.1, 0.9, 0.0, 0.1, 0.9]]
        )
        q_c = torch.tensor(
            [[0.1, 0.9, 0.0, 0.1, 0.9], [0.0, 1.0, 0.0, 0.0, 1.0]]
        )
        reliability = torch.ones(batch_size, 1, 2, 2, 2)
        active = torch.ones(batch_size, 3, dtype=torch.bool)
        return (*features, q_f, q_m, q_c, reliability, reliability, active)

    def test_legacy_mode_exactly_matches_original_softmax(self) -> None:
        gate = ReliabilityAwareScaleGate(dim=4, dropout=0.0).eval()
        h_f, h_m, h_c, q_f, q_m, q_c, r_m, r_c, active = self._inputs()
        with torch.no_grad():
            actual, evidence = gate(
                h_f, h_m, h_c, q_f, q_m, q_c, r_m, r_c, active
            )
            pooled = [value.mean(dim=(2, 3, 4)) for value in (h_f, h_m, h_c)]
            gate_input = torch.cat(
                (
                    *pooled,
                    q_f,
                    q_m,
                    q_c,
                    r_m.mean(dim=(1, 2, 3, 4)).view(-1, 1),
                    r_c.mean(dim=(1, 2, 3, 4)).view(-1, 1),
                ),
                dim=-1,
            )
            expected = torch.softmax(gate.mlp(gate_input), dim=-1)
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
        torch.testing.assert_close(evidence, torch.ones_like(evidence))

    def test_availability_purity_and_hybrid_are_target_free_and_exact(self) -> None:
        inputs = self._inputs(batch_size=2)
        q_f, q_m, q_c, active = inputs[3], inputs[4], inputs[5], inputs[-1]
        expected_availability = torch.stack(
            (q_f[:, 1], q_m[:, 1], q_c[:, 1]), dim=-1
        )
        expected_purity = torch.stack(
            (
                torch.ones_like(q_f[:, 1]),
                q_f[:, 1] / q_m[:, 1],
                q_f[:, 1] / q_c[:, 1],
            ),
            dim=-1,
        )
        evidence = {}
        for mode in ("availability", "purity", "hybrid"):
            gate = ReliabilityAwareScaleGate(
                dim=4, dropout=0.0, evidence_mode=mode, evidence_gain=0.5
            ).eval()
            with torch.no_grad():
                _, evidence[mode] = gate(*inputs)
        torch.testing.assert_close(evidence["availability"], expected_availability)
        torch.testing.assert_close(evidence["purity"], expected_purity)
        torch.testing.assert_close(
            evidence["hybrid"], 0.5 * (expected_availability + expected_purity)
        )
        self.assertEqual(active.dtype, torch.bool)

    def test_uniform_floor_prevents_scale_collapse(self) -> None:
        gate = ReliabilityAwareScaleGate(
            dim=4,
            dropout=0.0,
            evidence_mode="availability",
            evidence_gain=20.0,
            uniform_floor=0.12,
        ).eval()
        with torch.no_grad():
            weights, _ = gate(*self._inputs())
        self.assertTrue(torch.all(weights >= 0.12 / 3.0 - 1e-7))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))

    def test_invalid_esap_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ReliabilityAwareScaleGate(dim=4, evidence_mode="unknown")
        with self.assertRaises(ValueError):
            ReliabilityAwareScaleGate(dim=4, temperature=0.0)
        with self.assertRaises(ValueError):
            ReliabilityAwareScaleGate(dim=4, uniform_floor=1.0)

    def test_all_18_configs_build_and_forward(self) -> None:
        paths = sorted((ROOT / "configs/v14-exploration/esap").glob("E*.json"))
        self.assertEqual(len(paths), 18)
        for path in paths:
            with self.subTest(candidate=path.stem):
                patch = json.loads(path.read_text(encoding="utf-8"))
                cfg = deep_update(compact_v14_config(), patch)
                model = V14SafeC2FMoE.from_config(cfg).eval()
                batch = make_batch(cfg)
                with torch.no_grad():
                    outputs = model(**backbone_kwargs(batch))
                self.assertTrue(torch.isfinite(outputs["x_hat_main"]).all())
                self.assertTrue(torch.isfinite(outputs["gates"]["scale_gate"]).all())
                self.assertTrue(torch.isfinite(outputs["gates"]["scale_evidence"]).all())


if __name__ == "__main__":
    unittest.main()
