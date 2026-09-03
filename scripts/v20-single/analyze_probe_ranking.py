#!/usr/bin/env python3
"""Offline-only Probe-vs-oracle expert ranking analysis for a trained V20 run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from stmoe_imputer.data import build_loader, build_test_dataset, masked_pool2d_spatial
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.scale_utils import get_active_scales
from stmoe_imputer.utils.checkpoint import load_checkpoint
from stmoe_imputer.utils.device import move_batch_to_device


ROOT = Path(__file__).resolve().parents[2]
SCALE_MODULES = {
    "fine": ("embed_f", "routed_expert_pool"),
    "mid": ("embed_m", "routed_expert_pool_m"),
    "coarse": ("embed_c", "routed_expert_pool_c"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--test-npz", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def _rank(value: torch.Tensor) -> torch.Tensor:
    order = value.argsort(dim=-1)
    return order.argsort(dim=-1).float()


def _spearman(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_rank = _rank(left)
    right_rank = _rank(right)
    left_centered = left_rank - left_rank.mean(dim=-1, keepdim=True)
    right_centered = right_rank - right_rank.mean(dim=-1, keepdim=True)
    numerator = (left_centered * right_centered).sum(dim=-1)
    denominator = (
        left_centered.square().sum(dim=-1)
        * right_centered.square().sum(dim=-1)
    ).clamp_min(1e-12).sqrt()
    return numerator / denominator


def _topk_overlap(left: torch.Tensor, right: torch.Tensor, k: int) -> torch.Tensor:
    left_indices = left.topk(k, dim=-1, largest=False).indices
    right_indices = right.topk(k, dim=-1, largest=False).indices
    return (
        left_indices[:, :, None] == right_indices[:, None, :]
    ).any(dim=-1).float().sum(dim=-1) / float(k)


def _oracle_targets(batch: dict, cfg: dict) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    target = batch["x_f_gt"]
    ones = torch.ones_like(batch["m_f"])
    scale_cfg = cfg["data"]["scales"]
    mode = scale_cfg.get("pooling_mode", "avg")
    mid, _, _ = masked_pool2d_spatial(
        target,
        ones,
        kernel_size=int(scale_cfg["fine_to_mid"]),
        mode=mode,
        return_reliability=True,
    )
    coarse, _, _ = masked_pool2d_spatial(
        target,
        ones,
        kernel_size=int(scale_cfg["fine_to_coarse"]),
        mode=mode,
        return_reliability=True,
    )
    return {
        "fine": (target, 1.0 - batch["m_f"].float()),
        "mid": (mid, 1.0 - batch["r_m"].float()),
        "coarse": (coarse, 1.0 - batch["r_c"].float()),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    effective_config = run_dir / "effective_config.json"
    config_path = effective_config if effective_config.is_file() else run_dir / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("model", {}).get("architecture") != "v20_probe_validated_c2f_moe":
        raise ValueError(f"Not a V20 run: {run_dir}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = build_test_dataset(cfg, args.test_npz)
    if dataset is None:
        raise RuntimeError("A test dataset is required")
    loader = build_loader(dataset, cfg, shuffle=False)
    model = DualBranchSTImputer.from_config(cfg).to(device).eval()
    load_checkpoint(run_dir / "checkpoints/best.pt", model, map_location=device)
    wrapper = model.main_branch
    backbone = wrapper.main_backbone
    decoder = wrapper.probe_evaluator.probe_decoder
    scale_mode = backbone.scale_mode
    rows = []
    aggregate: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for batch_index, batch in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        oracle_targets = _oracle_targets(batch, cfg)
        scale_inputs = {
            "fine": (batch["x_f_obs"], batch["m_f"]),
            "mid": (batch["x_m_obs"], batch["m_m"]),
            "coarse": (batch["x_c_obs"], batch["m_c"]),
        }
        for scale in get_active_scales(scale_mode):
            embed_name, pool_name = SCALE_MODULES[scale]
            x, mask = scale_inputs[scale]
            features = getattr(backbone, pool_name).forward_all(
                getattr(backbone, embed_name)(x, mask)
            )
            prediction = decoder(features).float()
            target, weight = oracle_targets[scale]
            expanded_weight = weight[:, None].expand(
                -1, prediction.shape[1], target.shape[1], -1, -1, -1
            )
            oracle_error = (
                (prediction - target[:, None].float()).abs() * expanded_weight
            ).sum(dim=(2, 3, 4, 5)) / expanded_weight.sum(
                dim=(2, 3, 4, 5)
            ).clamp_min(1.0)
            probe_error = outputs["v20_probe"][scale]["raw_error"].float()
            spearman = _spearman(probe_error, oracle_error)
            top1 = (probe_error.argmin(dim=-1) == oracle_error.argmin(dim=-1)).float()
            top2 = _topk_overlap(probe_error, oracle_error, min(2, probe_error.shape[1]))
            valid = outputs["v20_probe"][scale]["valid"].bool()
            for sample in range(probe_error.shape[0]):
                if not valid[sample]:
                    continue
                row = {
                    "batch": batch_index,
                    "sample": sample,
                    "scale": scale,
                    "spearman": float(spearman[sample].cpu()),
                    "top1_agreement": float(top1[sample].cpu()),
                    "top2_overlap": float(top2[sample].cpu()),
                }
                for expert in range(probe_error.shape[1]):
                    row[f"probe_error_e{expert}"] = float(probe_error[sample, expert].cpu())
                    row[f"oracle_error_e{expert}"] = float(oracle_error[sample, expert].cpu())
                rows.append(row)
                for key in ("spearman", "top1_agreement", "top2_overlap"):
                    aggregate[scale][key].append(row[key])

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    per_sample = analysis_dir / "probe_oracle_per_sample.csv"
    if not rows:
        raise RuntimeError("No valid Probe samples were available for ranking analysis")
    with per_sample.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        scale: {
            key: sum(values) / len(values)
            for key, values in metrics.items()
            if values
        }
        for scale, metrics in aggregate.items()
    }
    summary["overall"] = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in ("spearman", "top1_agreement", "top2_overlap")
    }
    (analysis_dir / "probe_oracle_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    per_scale = analysis_dir / "probe_oracle_per_scale.csv"
    with per_scale.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("scale", "spearman", "top1_agreement", "top2_overlap"),
        )
        writer.writeheader()
        for scale, metrics in summary.items():
            writer.writerow({"scale": scale, **metrics})
    print(f"Wrote V20 Probe/oracle analysis under {analysis_dir}")


if __name__ == "__main__":
    main()
