#!/usr/bin/env python3
"""Train newly added upstream baselines on canonical grid-window data.

This module is deliberately an adapter: model classes are imported from the
vendored upstream repositories and are not edited or reimplemented here.
Only data loading, scaling, optimization, validation, checkpoint selection,
and original-scale evaluation live in this file.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml


BENCH = Path(__file__).resolve().parents[1]
NEURAL_MODELS = {"SAITS", "GRIN", "STCPA", "STAMImputer", "PAST"}
SIMPLE_MODELS = {"MeanFill", "HistoricalAverage"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(NEURAL_MODELS | SIMPLE_MODELS))
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-prefix", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--rate", required=True, type=float)
    parser.add_argument("--channel", default="0")
    return parser.parse_args()


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_splits(root: Path, mask: str, rate: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rate_text = format(rate, "g")
    path = root / f"true_data_{mask}_{rate_text}_v2.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {}
    with np.load(path, allow_pickle=False) as data:
        for split in ("train", "val", "test"):
            values = np.asarray(data[f"{split}_data"], dtype=np.float32)
            observed = np.asarray(data[f"{split}_mask"], dtype=np.float32)
            if values.shape != observed.shape or values.ndim != 3:
                raise ValueError(f"{split}: expected matching [B,T,N] data/mask, got {values.shape}/{observed.shape}")
            result[split] = values, observed
    return result


def metrics(pred: np.ndarray, true: np.ndarray, observed: np.ndarray) -> tuple[float, float, float]:
    hidden = observed < 0.5
    if not hidden.any():
        raise ValueError("Evaluation mask contains no artificially hidden values")
    error = pred[hidden].astype(np.float64) - true[hidden].astype(np.float64)
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    target = np.abs(true[hidden].astype(np.float64))
    valid = target > 1e-6
    mape = float(np.mean(np.abs(error[valid]) / target[valid]) * 100.0) if valid.any() else 0.0
    return mae, rmse, mape


def print_metrics(prefix: str, values: tuple[float, float, float], epoch: int | None = None) -> None:
    mae, rmse, mape = values
    if prefix == "Validation":
        print(f"Validation Epoch {epoch}: average Loss: {mae:.10f}", flush=True)
        print(f"Validation Metrics Epoch {epoch}: MAE: {mae:.10f} RMSE: {rmse:.10f} MAPE: {mape:.10f}", flush=True)
    else:
        print(f"TEST MAE: {mae:.10f}, RMSE: {rmse:.10f}, MAPE: {mape:.10f}", flush=True)


def channel_groups(manifest: dict, nodes: int) -> list[slice]:
    selected = manifest.get("selected_channels") or [0]
    count = max(1, len(selected))
    if nodes % count:
        return [slice(0, nodes)]
    width = nodes // count
    return [slice(index * width, (index + 1) * width) for index in range(count)]


def simple_run(model_name: str, splits: dict, data_root: Path) -> None:
    train, train_mask = splits["train"]
    manifest_path = data_root.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    if model_name == "MeanFill":
        prediction = np.empty((1, 1, train.shape[-1]), dtype=np.float32)
        global_observed = train[train_mask > 0.5]
        fallback = float(global_observed.mean()) if global_observed.size else 0.0
        for group in channel_groups(manifest, train.shape[-1]):
            values = train[..., group][train_mask[..., group] > 0.5]
            prediction[..., group] = float(values.mean()) if values.size else fallback

        def impute(values: np.ndarray) -> np.ndarray:
            return np.broadcast_to(prediction, values.shape).copy()
    else:
        available = train_mask.sum(axis=0)
        historical = (train * train_mask).sum(axis=0) / np.maximum(available, 1.0)
        node_available = train_mask.sum(axis=(0, 1))
        node_mean = (train * train_mask).sum(axis=(0, 1)) / np.maximum(node_available, 1.0)
        historical = np.where(available > 0, historical, node_mean[None])

        def impute(values: np.ndarray) -> np.ndarray:
            return np.broadcast_to(historical[None], values.shape).copy()

    for split, epoch in (("val", 0), ("test", None)):
        true, observed = splits[split]
        score = metrics(impute(true), true, observed)
        print_metrics("Validation" if split == "val" else "Test", score, epoch)


def import_upstream(model_name: str):
    roots = {
        "SAITS": BENCH / "SAITS", "GRIN": BENCH / "grin", "STCPA": BENCH / "STCPA",
        "STAMImputer": BENCH / "STAMImupter", "PAST": BENCH / "PAST",
    }
    sys.path.insert(0, str(roots[model_name]))
    if model_name == "SAITS":
        return importlib.import_module("modeling.saits").SAITS
    if model_name == "GRIN":
        return importlib.import_module("lib.nn.models.grin").GRINet
    if model_name == "STCPA":
        return importlib.import_module("model.model").STGAIN_Att
    if model_name == "STAMImputer":
        return importlib.import_module("model.MoE").SP_TSFormer_MoE_v2
    return importlib.import_module("model.model").Model


def adjacency(data_root: Path, nodes: int) -> np.ndarray:
    matrix = np.eye(nodes, dtype=np.float32)
    edge_path = data_root.parent / "grid_edges.csv"
    edges = np.loadtxt(edge_path, delimiter=",", skiprows=1, usecols=(0, 1), dtype=np.int64)
    edges = np.atleast_2d(edges)
    matrix[edges[:, 0], edges[:, 1]] = 1.0
    return matrix


def local_neighbors(adj: np.ndarray, width: int) -> np.ndarray:
    rows = []
    for index in range(adj.shape[0]):
        candidates = np.flatnonzero(adj[index] > 0)
        candidates = candidates[candidates != index]
        if not candidates.size:
            candidates = np.array([index])
        rows.append(np.resize(candidates, width))
    return np.stack(rows).astype(np.int64)


def build_model(name: str, cfg: dict, nodes: int, steps: int, data_root: Path, device: torch.device):
    cls = import_upstream(name)
    arch = cfg.get("architecture", {})
    if name == "SAITS":
        return cls(
            n_groups=arch.get("n_groups", 2), n_group_inner_layers=arch.get("n_group_inner_layers", 1),
            d_time=steps, d_feature=nodes, d_model=arch.get("d_model", 256),
            d_inner=arch.get("d_inner", 128), n_head=arch.get("n_head", 4),
            d_k=arch.get("d_k", 64), d_v=arch.get("d_v", 64), dropout=arch.get("dropout", 0.1),
            input_with_mask=True, param_sharing_strategy=arch.get("param_sharing_strategy", "inner_group"),
            MIT=True, device=device, diagonal_attention_mask=True,
        )
    adj = adjacency(data_root, nodes)
    if name == "GRIN":
        return cls(adj=adj, d_in=1, d_hidden=arch.get("d_hidden", 64), d_ff=arch.get("d_ff", 64),
                   ff_dropout=arch.get("ff_dropout", 0.0), n_layers=arch.get("n_layers", 1),
                   kernel_size=arch.get("kernel_size", 2), decoder_order=arch.get("decoder_order", 1),
                   d_emb=arch.get("d_emb", 8), layer_norm=arch.get("layer_norm", False), merge="mlp")
    if name == "STCPA":
        # The upstream STCPA pipeline is a two-generator cascade (fc then att).
        # Keep both original STGAIN_Att modules; this adapter only supplies the
        # grid data and coordinates their native detached hand-off.
        class STCPAPair(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n_temporal = arch.get("n_temporal", 3)
                kwargs = {"num_nodes": nodes, "n_blocks": arch.get("n_blocks", 5),
                          "n_temporal": self.n_temporal, "device": device}
                self.fc_model = cls(**kwargs)
                self.att_model = cls(**kwargs)
        return STCPAPair()
    if name == "STAMImputer":
        neighbors = local_neighbors(adj, arch.get("neighbor_width", 8))
        model = cls(input_dim=3, embed_dim=arch.get("embed_dim", 32), Tembed_dim=arch.get("tembed_dim", 64),
                    output_dim=1, num_nodes=nodes, num_series=steps, adj=neighbors,
                    num_heads=arch.get("num_heads", 4), mlp_ratio=arch.get("mlp_ratio", 4),
                    dropout=arch.get("dropout", 0.15), num_layers=arch.get("num_layers", 4))
        model.device = str(device)  # override hardcoded "cuda:0" in upstream model
        return model
    args = Namespace(node_num=nodes, hidden_dim=arch.get("hidden_dim", 64), dropout=arch.get("dropout", 0.1),
                     layer_num=arch.get("layer_num", 3), seq_len=steps,
                     alpha=arch.get("alpha", 0.1), order=arch.get("order", 1))
    return cls(args, torch.as_tensor(adj, dtype=torch.float32, device=device))


def time_features(batch: int, steps: int, device: torch.device) -> torch.Tensor:
    index = torch.arange(steps, device=device)
    result = torch.zeros((batch, steps, 1, 4), dtype=torch.long, device=device)
    result[..., 1] = (index % 7).view(1, -1, 1)
    result[..., 2] = (index % 24).view(1, -1, 1)
    result[..., 3] = ((index * 4 // max(steps, 1)) % 4).view(1, -1, 1)
    return result


def haar_parts(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = np.empty_like(x)
    for index in range(0, x.shape[1], 2):
        end = min(index + 2, x.shape[1])
        low[:, index:end] = x[:, index:end].mean(axis=1, keepdims=True)
    return low, x - low


def forward(name: str, model, x: torch.Tensor, mask: torch.Tensor, stage: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
    observed_x = x * mask
    if name == "SAITS":
        output = model({"X": observed_x.squeeze(-1), "missing_mask": mask.squeeze(-1),
                        "X_holdout": x.squeeze(-1), "indicating_mask": 1 - mask.squeeze(-1)}, stage)
        return output["imputed_data"].unsqueeze(-1), []
    if name == "GRIN":
        result = model(observed_x, mask.bool())
        if isinstance(result, tuple):
            imputed, predictions = result
            return imputed, [predictions[index] for index in range(predictions.shape[0])]
        return result, []
    if name == "STCPA":
        temporal, outputs, first_outputs = model.n_temporal, [], []
        for step in range(x.shape[1]):
            history = [observed_x[:, max(0, step - offset), :, 0] for offset in range(1, temporal + 1)]
            history = torch.stack(history, dim=1)
            first = model.fc_model(observed_x[:, step, :, 0], history)
            completed = mask[:, step, :, 0] * observed_x[:, step, :, 0] + (1 - mask[:, step, :, 0]) * first.detach()
            second = model.att_model(completed, history)
            first_outputs.append(first)
            outputs.append(second)
        first = torch.stack(first_outputs, dim=1).unsqueeze(-1)
        return torch.stack(outputs, dim=1).unsqueeze(-1), [first]
    if name == "STAMImputer":
        array = observed_x.detach().cpu().numpy()
        low, high = haar_parts(array)
        return model(array, low, high), []
    x_hat, t_hat = model(observed_x, mask.bool(), time_features(x.shape[0], x.shape[1], x.device))
    return x_hat + t_hat, []


def masked_loss(pred: torch.Tensor, true: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    hidden = observed < 0.5
    return torch.mean(torch.abs(pred[hidden] - true[hidden]))


def evaluate(name: str, model, loader: DataLoader, mean: float, std: float) -> tuple[float, float, float]:
    model.eval()
    preds, truths, masks = [], [], []
    with torch.no_grad():
        for x, mask in loader:
            output, _ = forward(name, model, x, mask, "test")
            preds.append((output * std + mean).cpu().numpy()[..., 0])
            truths.append((x * std + mean).cpu().numpy()[..., 0])
            masks.append(mask.cpu().numpy()[..., 0])
    return metrics(np.concatenate(preds), np.concatenate(truths), np.concatenate(masks))


def neural_run(name: str, cfg: dict, splits: dict, data_root: Path) -> None:
    train_cfg = cfg["training"]
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_values, train_mask = splits["train"]
    observed_values = train_values[train_mask > 0.5]
    if name == "STCPA" and observed_values.size:
        # Upstream SpeedDataset defaults to min/max scaling because STGAIN_Att
        # has a sigmoid output. Preserve that preprocessing contract.
        mean = float(observed_values.min())
        std = float(observed_values.max() - observed_values.min())
    else:
        mean = float(observed_values.mean()) if observed_values.size else 0.0
        std = float(observed_values.std()) if observed_values.size else 1.0
    std = max(std, 1e-6)

    def loader(split: str, shuffle: bool) -> DataLoader:
        values, mask = splits[split]
        x = torch.as_tensor((values - mean) / std, dtype=torch.float32, device=device).unsqueeze(-1)
        m = torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(-1)
        return DataLoader(TensorDataset(x, m), batch_size=int(train_cfg["batch_size"]), shuffle=shuffle)

    train_loader, val_loader, test_loader = loader("train", True), loader("val", False), loader("test", False)
    model = build_model(name, cfg, train_values.shape[-1], train_values.shape[1], data_root, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("lr", 0.001)),
                                 weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    epochs, val_epoch = int(train_cfg["epochs"]), int(train_cfg["val_epoch"])
    patience_limit = int(train_cfg.get("patience", 10))
    checkpoint = Path(cfg["output"]["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, patience = float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for x, mask in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output, auxiliary = forward(name, model, x, mask, "train")
            loss = masked_loss(output, x, mask)
            if auxiliary:
                loss = loss + sum(masked_loss(item, x, mask) for item in auxiliary)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name} produced a non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 5.0)))
            optimizer.step(); losses.append(float(loss.detach()))
        print(f"Train Epoch {epoch}: averaged Loss: {np.mean(losses):.10f}", flush=True)
        should_validate = epoch % val_epoch == 0 or epoch == epochs
        if not should_validate:
            continue
        score = evaluate(name, model, val_loader, mean, std)
        print_metrics("Validation", score, epoch)
        if score[0] < best:
            best, patience = score[0], 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_mae": best,
                        "normalization": {"mean": mean, "std": std}}, checkpoint)
            print(f"best loss is updated to {best:.10f} at {epoch}", flush=True)
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break
    if not checkpoint.is_file():
        raise RuntimeError("No validation checkpoint was produced")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    print(f"Loading best model from {checkpoint}", flush=True)
    print_metrics("Test", evaluate(name, model, test_loader, mean, std))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = Path(args.data_prefix).resolve()
    splits = load_splits(data_root, args.mask, args.rate)
    if args.model in SIMPLE_MODELS:
        simple_run(args.model, splits, data_root)
    else:
        neural_run(args.model, cfg, splits, data_root)


if __name__ == "__main__":
    main()
