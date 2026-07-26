from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from stmoe_imputer.utils.checkpoint import load_checkpoint, save_checkpoint


class CheckpointWrappersTest(unittest.TestCase):
    def test_data_parallel_checkpoint_loads_into_plain_model(self) -> None:
        source = torch.nn.Linear(3, 2)
        wrapped = torch.nn.DataParallel(source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(
                path,
                wrapped,
                optimizer=None,
                epoch=7,
                metrics={"val_mae": 1.0},
                cfg={"version": "test"},
            )
            self.assertTrue(path.is_file())
            self.assertEqual(list(path.parent.glob("*.pt")), [path])
            target = torch.nn.Linear(3, 2)
            checkpoint = load_checkpoint(path, target)
        self.assertEqual(checkpoint["epoch"], 7)
        for source_parameter, target_parameter in zip(
            source.parameters(),
            target.parameters(),
        ):
            torch.testing.assert_close(
                source_parameter,
                target_parameter,
            )


if __name__ == "__main__":
    unittest.main()
