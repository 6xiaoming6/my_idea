from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from stmoe_imputer.utils.checkpoint import load_checkpoint, save_checkpoint


class CheckpointWrapperTest(unittest.TestCase):
    def test_data_parallel_checkpoint_is_saved_without_module_prefix(self) -> None:
        source = torch.nn.DataParallel(torch.nn.Linear(3, 2))
        target = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(path, source, None, 3, {}, {})
            payload = torch.load(path, map_location="cpu")
            self.assertTrue(
                all(not key.startswith("module.") for key in payload["model"])
            )
            self.assertEqual(
                sorted(item.name for item in Path(directory).iterdir()),
                ["best.pt"],
            )
            load_checkpoint(path, target)
        for expected, actual in zip(
            source.module.parameters(), target.parameters()
        ):
            torch.testing.assert_close(expected, actual)

    def test_legacy_module_prefix_is_accepted(self) -> None:
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        legacy = {
            "model": {
                f"module.{key}": value
                for key, value in source.state_dict().items()
            },
            "optimizer": None,
            "epoch": 1,
            "metrics": {},
            "config": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(legacy, path)
            load_checkpoint(path, target)
        for expected, actual in zip(source.parameters(), target.parameters()):
            torch.testing.assert_close(expected, actual)


if __name__ == "__main__":
    unittest.main()
