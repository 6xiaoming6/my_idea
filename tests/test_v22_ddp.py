"""CPU/Gloo integration tests: real DDP collectives, no GPU allocation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from test_v22_coarsening_moe import config
from _v14_utils import make_batch
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.engine import evaluate, build_optimizer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v22_ddp", ROOT / "scripts/v22/train_ddp.py")
ddp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ddp)


def worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2)
    try:
        for mode in ("fixed", "fixed_stats", "single_local", "single_wide", "uniform", "moe", "moe_no_stats"):
            torch.manual_seed(42)
            cfg = config(mode)
            cfg["loss"].update(lambda_v22_mass=.001, lambda_v22_balance=.001)
            samples = [{k: v.squeeze(0) for k, v in make_batch(cfg, seed=i + 1).items()} for i in range(3)]
            raw = DualBranchSTImputer.from_config(cfg)
            raw.alpha.requires_grad_(False)
            optimizer = build_optimizer(raw, cfg)
            model = DistributedDataParallel(raw, broadcast_buffers=False)
            train_sampler = DistributedSampler(samples, num_replicas=2, rank=rank, shuffle=True, seed=42)
            train_sampler.set_epoch(1)
            # Two optimizer steps on each rank: catches unfinished reducer errors.
            train = ddp.epoch_pass(model, DataLoader(samples, batch_size=1, sampler=train_sampler), cfg,
                                   torch.device("cpu"), 1, optimizer, torch.amp.GradScaler("cpu", enabled=False))
            assert train["metric_missing_count"] > 0
            parameter = torch.cat([p.detach().flatten() for p in raw.parameters()])
            reference = parameter.clone()
            dist.broadcast(reference, 0)
            torch.testing.assert_close(reference, parameter, atol=0, rtol=0)
            for size in (3, 1):
                subset = samples[:size]
                result = ddp.epoch_pass(raw, DataLoader(subset, batch_size=1, sampler=ddp.ExactShardSampler(subset, rank, 2)),
                                       cfg, torch.device("cpu"), 1)
                if rank == 0:
                    serial = evaluate(raw, DataLoader(subset, batch_size=1), torch.device("cpu"), cfg, epoch=1)
                    for key in ("mae", "rmse", "mape", "wape", "metric_missing_count"):
                        torch.testing.assert_close(torch.tensor(result[key]), torch.tensor(serial[key]), rtol=1e-6, atol=1e-6)
                dist.barrier()
    finally:
        dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def test_unpadded_sampler(self):
        for size in (0, 1, 3, 8):
            shards = [list(ddp.ExactShardSampler(range(size), rank, 2)) for rank in range(2)]
            self.assertEqual(sorted(shards[0] + shards[1]), list(range(size)))
            self.assertFalse(set(shards[0]) & set(shards[1]))

    def test_two_process_training_and_exact_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(worker, args=(directory,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
