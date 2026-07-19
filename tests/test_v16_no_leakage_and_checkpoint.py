from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import torch
import unittest

from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.teacher_utils import prepare_v16_teacher
from stmoe_imputer.utils.checkpoint import load_checkpoint, save_checkpoint

from _v16_utils import compact_v16_config, make_batch


class V16NoLeakageAndCheckpointTest(unittest.TestCase):
    def test_hidden_target_cannot_change_forward_alpha(self) -> None:
        cfg = compact_v16_config()
        batch = make_batch(cfg)
        changed = {key: value.clone() for key, value in batch.items()}
        hidden = (1.0 - changed["m_f"]).expand_as(changed["x_f_gt"])
        changed["x_f_gt"] = changed["x_f_gt"] + 10000.0 * hidden
        model = DualBranchSTImputer.from_config(cfg).eval()
        with torch.no_grad():
            first = model(batch)
            second = model(changed)
        for key in ("x_hat_final", "x_hat_candidate", "residual_alpha", "calibration_condition"):
            torch.testing.assert_close(first[key], second[key], atol=0.0, rtol=0.0)

    def test_student_checkpoint_has_metadata_and_loads_without_teacher(self) -> None:
        cfg = compact_v16_config()
        model = DualBranchSTImputer.from_config(cfg).eval()
        batch = make_batch(cfg)
        metadata = {
            "teacher_branch": "v14-single",
            "teacher_commit": "abc123",
            "teacher_checkpoint_sha256": "deadbeef",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(path, model, None, 2, {"val_mae": 1.0}, cfg, metadata=metadata)
            restored = DualBranchSTImputer.from_config(cfg).eval()
            payload = load_checkpoint(path, restored)
            self.assertEqual(payload["metadata"], metadata)
            self.assertFalse(any("teacher" in key for key in payload["model"]))
            with torch.no_grad():
                output = restored(batch)
            self.assertEqual(output["x_hat_final"].shape, batch["x_f_gt"].shape)

    def test_teacher_is_frozen_and_initializes_the_student_backbone(self) -> None:
        cfg = compact_v16_config()
        teacher_cfg = copy.deepcopy(cfg)
        teacher_cfg["model"]["version"] = "v14-single"
        teacher_cfg["model"]["architecture"] = "v14_safe_c2f_moe"
        teacher_cfg["model"]["v14"] = {"enabled": True}
        teacher_source = DualBranchSTImputer.from_config(teacher_cfg).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save({
                "model": teacher_source.state_dict(),
                "optimizer": None,
                "epoch": 3,
                "metrics": {"val_mae": 1.0},
                "config": teacher_cfg,
            }, path)
            cfg["teacher"] = {
                "enabled": True,
                "architecture": "v14_safe_c2f_moe",
                "checkpoint": str(path),
                "seed": cfg["seed"],
                "strict": True,
                "initialize_student": True,
            }
            student = DualBranchSTImputer.from_config(cfg)
            context = prepare_v16_teacher(cfg, student, torch.device("cpu"))
            self.assertIsNotNone(context.model)
            self.assertFalse(any(p.requires_grad for p in context.model.parameters()))
            self.assertEqual(
                context.metadata["student_backbone_tensors_initialized"],
                context.metadata["student_backbone_tensors_total"],
            )
            student_state = student.state_dict()
            teacher_state = context.model.state_dict()
            for name, value in student_state.items():
                prefix = "main_branch.student_backbone."
                if name.startswith(prefix):
                    source_name = "main_branch.main_backbone." + name[len(prefix):]
                    torch.testing.assert_close(value, teacher_state[source_name], atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
