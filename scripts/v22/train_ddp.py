#!/usr/bin/env python3
"""V22-only DDP: sharded training, exact unpadded evaluation, rank-zero artifacts."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from tqdm import tqdm

from stmoe_imputer.data import build_datasets, build_test_dataset
from stmoe_imputer.engine import build_optimizer, build_scheduler, _append_model_diagnostics
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.metrics import MaskedMetricAccumulator
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils import set_seed
from stmoe_imputer.utils.checkpoint import save_checkpoint, load_checkpoint
from stmoe_imputer.utils.device import move_batch_to_device
from stmoe_imputer.utils.train_logger import TrainLogger


class ExactShardSampler(Sampler):
    """No duplicated evaluation samples, including splits smaller than world size."""
    def __init__(self, dataset, rank, world_size):
        self.indices = range(rank, len(dataset), world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def reduce_metrics(acc, device):
    names = ("absolute_error", "squared_error", "absolute_percentage_error", "absolute_target", "count")
    totals = torch.tensor([getattr(acc, k) for k in names], dtype=torch.float64, device=device)
    dist.all_reduce(totals)
    for key, value in zip(names, totals.cpu().tolist()):
        setattr(acc, key, value)
    return acc.compute()


def epoch_pass(model, loader, cfg, device, epoch, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    stats = defaultdict(float)
    acc = MaskedMetricAccumulator()
    n, missing_total = 0, 0.
    progress = tqdm(loader, desc=f"{'train' if training else 'eval'} epoch {epoch}", disable=dist.get_rank() != 0, leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        size = batch["x_f_gt"].shape[0]
        local_count = (1 - batch["m_f"]).sum() * batch["x_f_gt"].shape[1]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device.type, enabled=training and cfg["train"].get("amp", False) and device.type == "cuda"):
            outputs = model(batch)
            loss, logs = compute_main_stage_loss(outputs, batch, cfg, epoch)
            if training:
                # DDP averages gradients across ranks. Weight masked losses by
                # missing counts to reproduce a global masked reduction per step.
                global_count = local_count.detach().clone()
                dist.all_reduce(global_count)
                weight = dist.get_world_size() * local_count / global_count.clamp_min(1)
                normalized_pred = (outputs["x_hat_final"] - outputs["v22_center"]) / outputs["v22_scale"]
                normalized_gt = (batch["x_f_gt"] - outputs["v22_center"]) / outputs["v22_scale"]
                from stmoe_imputer.losses import masked_loss
                main = masked_loss(normalized_pred, normalized_gt, batch["m_f"], cfg["loss"].get("type", "smooth_l1"))
                loss = loss + (weight - 1) * main
        if training:
            finite = torch.tensor(int(torch.isfinite(loss).item()), device=device)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Non-finite loss on a DDP rank")
            scaler.scale(loss).backward()
            if cfg["train"].get("grad_clip_norm"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
        acc.update(outputs["x_hat_final"], batch["x_f_gt"], batch["m_f"])
        n += size
        count = float(local_count)
        missing_total += count
        stats["l_main"] += float(logs["l_main"]) * count
        for key in ("l_v22_mass", "l_v22_balance"):
            stats[key] += float(logs[key]) * size
        diagnostic = defaultdict(list)
        _append_model_diagnostics(diagnostic, outputs)
        for key, values in diagnostic.items():
            stats[key] += sum(values) / len(values) * size
        if training:
            progress.set_postfix(loss=float(logs["loss"]))
    # Empty evaluation shards still participate with zero numerators.
    key_sets = [None] * dist.get_world_size()
    dist.all_gather_object(key_sets, list(stats))
    keys = sorted(set().union(*key_sets))
    values = torch.tensor([n, missing_total, *[stats[k] for k in keys]], dtype=torch.float64, device=device)
    dist.all_reduce(values)
    samples, missing, *sums = values.cpu().tolist()
    result = {k: v / max(missing if k == "l_main" else samples, 1) for k, v in zip(keys, sums)}
    result.update(reduce_metrics(acc, device))
    result["loss"] = result.get("l_main", 0.) + cfg["loss"]["lambda_v22_mass"] * result.get("l_v22_mass", 0.) + cfg["loss"]["lambda_v22_balance"] * result.get("l_v22_balance", 0.)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--name", required=True)
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}_npz")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg["model"]["architecture"] != "v22_coarsening_moe":
        raise ValueError("This distributed entry point supports only V22")
    local_rank = int(os.environ["LOCAL_RANK"])
    cpu = cfg.get("device") == "cpu"
    device = torch.device("cpu" if cpu else f"cuda:{local_rank}")
    if not cpu:
        torch.cuda.set_device(device)
    dist.init_process_group("gloo" if cpu else "nccl", timeout=timedelta(seconds=cfg["distributed"]["timeout_seconds"]))
    logger = None
    status, done, best_epoch, best_mae = "failed", 0, 0, math.inf
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != cfg["distributed"]["world_size"]:
            raise ValueError("Configured DDP world size differs from torchrun")
        set_seed(cfg.get("seed", 42))
        train_ds, val_ds = build_datasets(cfg, args.train_npz, args.val_npz, args.synthetic)
        test_ds = build_test_dataset(cfg, args.test_npz, args.synthetic)
        if test_ds is None or not len(train_ds) or not len(val_ds) or not len(test_ds):
            raise ValueError("Nonempty train, val and test splits are required")
        sampler = DistributedSampler(train_ds, world, rank, shuffle=True, seed=cfg.get("seed", 42), drop_last=False)
        kwargs = dict(batch_size=cfg["data"]["batch_size"], num_workers=cfg["data"]["num_workers"], pin_memory=not cpu, drop_last=False)
        train_loader = DataLoader(train_ds, sampler=sampler, **kwargs)
        val_loader = DataLoader(val_ds, sampler=ExactShardSampler(val_ds, rank, world), **kwargs)
        test_loader = DataLoader(test_ds, sampler=ExactShardSampler(test_ds, rank, world), **kwargs)
        raw = DualBranchSTImputer.from_config(cfg).to(device)
        # Disabled auxiliary branch has an unused scalar; exclude it from DDP.
        raw.alpha.requires_grad_(False)
        optimizer = build_optimizer(raw, cfg)
        scheduler = build_scheduler(optimizer, cfg)
        model = DDP(raw, device_ids=None if cpu else [local_rank], broadcast_buffers=False)
        set_seed(cfg.get("seed", 42) + rank)
        scaler = torch.amp.GradScaler(device.type, enabled=not cpu and cfg["train"].get("amp", False))
        location = [None]
        if rank == 0:
            mask = cfg["data"]["mask"]
            location[0] = str(ROOT / cfg["output_dir"] / cfg["data"]["dataset_name"] / "ablation" / args.name.removeprefix("ablation_") / mask["pattern"] / f"rate{mask['missing_rate']}" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_seed{cfg['seed']}_ddp{world}_bs{cfg['data']['batch_size']}")
        dist.broadcast_object_list(location, src=0)
        run = Path(location[0])
        if rank == 0:
            run.mkdir(parents=True, exist_ok=False)
            (run / "config.json").write_text(json.dumps(cfg, indent=2))
            logger = TrainLogger(run / "logs")
            logger.log_header(cfg, extra={"run_dir": str(run), "world_size": world,
                "global_batch_size": cfg["distributed"]["global_batch_size"],
                "train_samples": len(train_ds), "val_samples": len(val_ds), "test_samples": len(test_ds),
                "train_padding_duplicates_per_epoch": len(sampler) * world - len(train_ds),
                "evaluation_padding": 0, "total_params": sum(p.numel() for p in raw.parameters())})
            logger.log_table_headers()
            print(f"[DDP] {world} ranks, batch={cfg['data']['batch_size']}/rank; output={run}", flush=True)
        dist.barrier()
        for epoch in range(1, cfg["train"]["epochs"] + 1):
            sampler.set_epoch(epoch)
            if not cpu:
                torch.cuda.reset_peak_memory_stats(device)
            start = time.monotonic()
            train = epoch_pass(model, train_loader, cfg, device, epoch, optimizer, scaler)
            trained = time.monotonic()
            val = None
            if epoch % cfg["train"]["val_epoch"] == 0 or epoch == cfg["train"]["epochs"]:
                # No DDP forward collectives: validation ranks may have unequal batch counts.
                val = epoch_pass(raw, val_loader, cfg, device, epoch)
                if not all(math.isfinite(val[k]) for k in ("loss", "mae", "rmse")):
                    raise FloatingPointError("Non-finite distributed validation metrics")
            finish = time.monotonic()
            train["lr"] = optimizer.param_groups[0]["lr"]
            improved = val is not None and val["mae"] < best_mae
            if improved:
                best_mae, best_epoch = val["mae"], epoch
            perf = torch.tensor([trained - start, finish - trained, finish - start, 0 if cpu else torch.cuda.max_memory_allocated(device) / 1024**3], device=device)
            dist.all_reduce(perf, op=dist.ReduceOp.MAX)
            if scheduler:
                scheduler.step()
            if rank == 0:
                logger.log_epoch(epoch, train, val, dict(zip(("train_time_sec", "val_time_sec", "epoch_time_sec", "peak_memory_gb"), perf.cpu().tolist())), improved)
                if improved:
                    temporary = run / "checkpoints/best.pt.tmp"
                    save_checkpoint(temporary, raw, optimizer, epoch, {"val_mae": best_mae}, cfg)
                    os.replace(temporary, run / "checkpoints/best.pt")
                    logger.log_best(epoch, best_mae)
                print(f"[DDP E {epoch}/{cfg['train']['epochs']}] loss={train['loss']:.5f} val_mae={val['mae'] if val else '-'} time={float(perf[2]):.1f}s", flush=True)
            done = epoch
            dist.barrier()
        load_checkpoint(run / "checkpoints/best.pt", raw, map_location=device)
        test = epoch_pass(raw, test_loader, cfg, device, best_epoch)
        if not all(math.isfinite(test[k]) for k in ("loss", "mae", "rmse")):
            raise FloatingPointError("Non-finite test metrics")
        if rank == 0:
            logger.log_test(test, {"best_epoch": best_epoch, "best_val_mae": best_mae, "checkpoint": str(run / "checkpoints/best.pt"), "test_samples": len(test_ds)})
        dist.barrier()
        status = "finished"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        if logger:
            logger.log_footer({"completed_epochs": done, "best_epoch": best_epoch, "best_val_mae": best_mae}, status=status)
            logger.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
