from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.main_branch import MultiScaleMoEBackbone

from _v14_utils import make_batch
from _v20_utils import compact_v20_config


class V20RoutingTest(unittest.TestCase):
    def test_zero_initialized_decoder_exactly_falls_back_to_v14_prior(self) -> None:
        cfg = compact_v20_config()
        model = DualBranchSTImputer.from_config(cfg).eval()
        with torch.no_grad():
            outputs = model(make_batch(cfg))
        for scale in ("fine", "mid", "coarse"):
            probe = outputs["v20_probe"][scale]
            self.assertTrue(torch.allclose(probe["confidence"], torch.zeros_like(probe["confidence"])))
            self.assertTrue(torch.allclose(probe["eta"], torch.zeros_like(probe["eta"])))
            torch.testing.assert_close(
                outputs["gates"][scale], outputs["prior_gates"][scale], atol=1e-7, rtol=0.0
            )

    def test_log_probability_calibration_uses_measured_competence(self) -> None:
        prior = torch.tensor([[0.7, 0.2, 0.1]])
        competence = torch.tensor([[0.1, 0.2, 0.7]])
        final = MultiScaleMoEBackbone._apply_routing_evidence(
            prior, {"competence": competence, "eta": torch.ones(1, 1)}
        )
        torch.testing.assert_close(final, competence, atol=1e-7, rtol=1e-6)

    def test_neutral_fusion_keeps_prior_exact_for_uniform_competence(self) -> None:
        prior = torch.tensor([[0.7, 0.2, 0.1]])
        uniform = torch.full_like(prior, 1.0 / prior.shape[-1])
        final = MultiScaleMoEBackbone._apply_routing_evidence(
            prior,
            {
                "competence": uniform,
                "eta": torch.full((1, 1), 0.9),
                "fusion_mode": "neutral_multiplicative",
            },
        )
        torch.testing.assert_close(final, prior, atol=1e-7, rtol=0.0)

    def test_training_protects_v14_route_and_only_fine_can_supply_evidence(self) -> None:
        cfg = compact_v20_config()
        model = DualBranchSTImputer.from_config(cfg).train()
        with torch.no_grad():
            model.main_branch.probe_evaluator.probe_decoder.out_proj.weight.normal_()
        outputs = model(make_batch(cfg))
        self.assertEqual(set(outputs["v20_probe"]["routing_evidence"]), {"fine"})
        for scale in ("fine", "mid", "coarse"):
            torch.testing.assert_close(
                outputs["gates"][scale],
                outputs["prior_gates"][scale],
                atol=1e-7,
                rtol=0.0,
            )
            self.assertTrue(
                torch.allclose(
                    outputs["v20_probe"][scale]["eta"],
                    torch.zeros_like(outputs["v20_probe"][scale]["eta"]),
                )
            )

    def test_hidden_ground_truth_cannot_change_probe_or_prediction(self) -> None:
        cfg = compact_v20_config()
        batch = make_batch(cfg)
        changed = {key: value.clone() for key, value in batch.items()}
        hidden = (1.0 - changed["m_f"]).expand_as(changed["x_f_gt"])
        changed["x_f_gt"] = changed["x_f_gt"] + hidden * 100000.0
        model = DualBranchSTImputer.from_config(cfg).eval()
        with torch.no_grad():
            first = model(batch)
            second = model(changed)
        torch.testing.assert_close(first["x_hat_final"], second["x_hat_final"], atol=0.0, rtol=0.0)
        for scale in ("fine", "mid", "coarse"):
            for key in ("probe_mask", "raw_error", "competence", "eta", "final_gate"):
                torch.testing.assert_close(
                    first["v20_probe"][scale][key],
                    second["v20_probe"][scale][key],
                    atol=0.0,
                    rtol=0.0,
                )

    def test_inactive_coarse_scale_does_not_run_probe(self) -> None:
        cfg = compact_v20_config()
        cfg["model"]["main"]["scale_mode"] = "fine_mid"
        outputs = DualBranchSTImputer.from_config(cfg).eval()(make_batch(cfg))
        self.assertIn("fine", outputs["v20_probe"])
        self.assertIn("mid", outputs["v20_probe"])
        self.assertNotIn("coarse", outputs["v20_probe"])
        self.assertNotIn("coarse", outputs["v20_probe"]["routing_evidence"])
