"""Opt-in, CPU-only REAL one-window train/val/test checks, not accuracy evidence.

V22_REAL_SMOKE=1 PYTHONPATH=src:tests python -m unittest test_v22_real_smoke
Streams only the first NPY window from each NPZ (does not load full training data).
"""
import importlib.util
import io
import os
from pathlib import Path
import unittest
import zipfile

import numpy as np
import torch

from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.data.transforms import ensure_multiscale
from stmoe_imputer.engine import build_optimizer, evaluate, train_one_epoch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v22_real_runner", ROOT / "scripts/v22/train.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def first_window(path):
    with zipfile.ZipFile(path) as archive:
        name = "x_f_gt.npy" if "x_f_gt.npy" in archive.namelist() else "x_f.npy"
        with archive.open(name) as stream:
            version = np.lib.format.read_magic(stream)
            reader = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}[version]
            shape, fortran, dtype = reader(stream)
            if fortran or len(shape) != 5 or dtype.hasobject:
                raise ValueError("Real smoke requires C-order numeric [N,C,T,H,W] windows")
            count = int(np.prod(shape[1:]))
            raw = stream.read(count * dtype.itemsize)
            return torch.from_numpy(np.frombuffer(raw, dtype=dtype).copy().reshape(1, *shape[1:])).float()


def first_batch(cfg, npz, csv):
    x = first_window(ROOT / npz)
    _, _, t, h, w = x.shape
    flat = np.loadtxt(ROOT / csv, delimiter=",", dtype=np.float32, max_rows=1)
    m = torch.from_numpy(flat.copy()).reshape(1, 1, -1, h, w)
    if m.shape[2] == 1:
        m = m.expand(1, 1, t, h, w).clone()
    return ensure_multiscale({"x_f_gt": x, "m_f": m},
                            cfg["data"]["scales"]["fine_to_mid"], cfg["data"]["scales"]["fine_to_coarse"],
                            cfg["data"]["scales"].get("pooling_mode", "avg"))


@unittest.skipUnless(os.environ.get("V22_REAL_SMOKE") == "1", "Opt-in real-data CPU smoke")
class RealSmokeTest(unittest.TestCase):
    def test_real_splits_all_modes_and_datasets(self):
        torch.set_num_threads(2)
        suite = runner.load_suite()
        for dataset in suite["datasets"]:
            for pattern in ("fixed", "random"):
                cfg, paths = runner.build_config(suite, "moe", dataset, pattern, "0.4", 42, epochs=1, cpu=True)
                batches = {s: first_batch(cfg, p, cfg["data"]["mask"][f"{s}_csv"]) for s, p in paths.items()}
                for variant in suite["variants"]:
                    with self.subTest(dataset=dataset, pattern=pattern, variant=variant):
                        torch.manual_seed(42)
                        cfg["model"]["v22"]["mode"] = variant
                        model = DualBranchSTImputer.from_config(cfg)
                        optimizer = build_optimizer(model, cfg)
                        train = train_one_epoch(model, [batches["train"]], optimizer, torch.device("cpu"), cfg, 1)
                        val = evaluate(model, [batches["val"]], torch.device("cpu"), cfg, epoch=1)
                        checkpoint = io.BytesIO()
                        torch.save(model.state_dict(), checkpoint)
                        checkpoint.seek(0)
                        model.load_state_dict(torch.load(checkpoint, weights_only=True))
                        test = evaluate(model, [batches["test"]], torch.device("cpu"), cfg, epoch=1)
                        for result in (train, val, test):
                            self.assertTrue(all(np.isfinite(result[k]) for k in ("loss", "mae", "rmse")))
                        print(f"REAL_SMOKE {dataset} {pattern} {variant}: train_loss={train['loss']:.6f} val_mae={val['mae']:.6f} test_mae={test['mae']:.6f}", flush=True)


if __name__ == "__main__":
    unittest.main()
