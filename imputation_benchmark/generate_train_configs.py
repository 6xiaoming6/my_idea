#!/usr/bin/env python3
"""Generate native baseline configs from data settings and a JSON policy.

Model architecture and optimizer definitions remain in upstream templates.
Run-level budgets such as epochs, batch size, patience, validation interval,
and two-stage allocation are supplied by a unified JSON policy.
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "training_configs"
DATASETS = {
    "TaxiBJ": (32, 32, 2, 12),
    "BikeNYC": (24, 12, 2, 12),
    "CHAP": (32, 32, 1, 7),
}
DATASET_TIME = {
    "TaxiBJ": ("20210101 00:00:00", "5min"),
    "BikeNYC": ("20210101 00:00:00", "5min"),
    "CHAP": ("20180101 00:00:00", "1D"),
}
SOURCE_SPLITS = {
    "TaxiBJ": ("data/TaxiBJ/taxibj_train.npz", "data/TaxiBJ/taxibj_val.npz", "data/TaxiBJ/taxibj_test.npz"),
    "BikeNYC": ("data/BikeNYC/bikenyc_train.npz", "data/BikeNYC/bikenyc_val.npz", "data/BikeNYC/bikenyc_test.npz"),
    "CHAP": ("data/CHAP/beijing/chap_beijing_train.npz", "data/CHAP/beijing/chap_beijing_val.npz", "data/CHAP/beijing/chap_beijing_test.npz"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--mask", default="fixed", choices=("fixed", "random", "SR-TR", "SR-TC", "SC-TR", "SC-TC"))
    parser.add_argument("--rate", type=float, default=0.2)
    parser.add_argument("--channel", default="0", help="all or a zero-based channel index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy-json", default=str(ROOT / "policies" / "baseline_paper.json"),
        help="Unified JSON training policy translated into each baseline's native config format",
    )
    return parser.parse_args()


def label(rate: float) -> str:
    return format(rate, "g")


def adapted_dir(dataset: str, mask: str, rate: float, channel: str) -> Path:
    return ROOT / "data" / "adapted" / dataset / f"{mask}_{label(rate)}" / f"channel_{channel}"


def relative_data(model_dir: Path, dataset: str, mask: str, rate: float, channel: str) -> str:
    return os.path.relpath(adapted_dir(dataset, mask, rate, channel), model_dir).replace("\\", "/")


def relative_split_data(model_dir: Path, dataset: str, mask: str, rate: float, channel: str) -> str:
    return relative_data(model_dir, dataset, mask, rate, channel) + "/split"


def output_path(dataset: str, model: str, mask: str, rate: float, channel: str, suffix: str = "conf") -> Path:
    return OUT / dataset / f"{model}_{mask}_{label(rate)}_channel_{channel}_train.{suffix}"


def write_ini(template: Path, output: Path, updates: dict[str, dict[str, object]]) -> None:
    cfg = configparser.ConfigParser()
    if not cfg.read(template):
        raise FileNotFoundError(template)
    for section, values in updates.items():
        if not cfg.has_section(section):
            cfg.add_section(section)
        for key, value in values.items():
            cfg[section][key] = str(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        cfg.write(stream)


def source_split_sizes(dataset: str) -> tuple[int, int, int]:
    sizes = []
    for relative in SOURCE_SPLITS[dataset]:
        with np.load(ROOT.parent / relative, allow_pickle=False) as data:
            key = "x_f_gt" if "x_f_gt" in data.files else "x_f"
            sizes.append(int(data[key].shape[0]))
    return tuple(sizes)


def load_policy(path_text: str) -> tuple[Path, dict]:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (ROOT.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training policy not found: {path}")
    policy = json.loads(path.read_text(encoding="utf-8"))
    required = {"AGCRN", "ASTGNN", "BRITS", "CSDI", "E2GAN", "GAIN", "GCASTN",
                "IGNNK", "ImputeFormer", "mTAN", "PriSTI", "SSTBAN", "LAST", "LATC",
                "SAITS", "GRIN", "STCPA", "STAMImputer", "PAST", "MeanFill",
                "HistoricalAverage"}
    missing = required - set(policy.get("models", {}))
    if missing:
        raise ValueError(f"Policy {path} is missing models: {sorted(missing)}")
    for model, values in policy["models"].items():
        if not values.get("non_iterative") and not values.get("non_neural"):
            if int(values.get("val_epoch", 0)) < 1:
                raise ValueError(f"Policy {path}: {model}.val_epoch must be at least 1")
    return path, policy


def main() -> None:
    args = parse_args()
    policy_path, policy = load_policy(args.policy_json)
    strategy = policy["models"]
    if not 0 < args.rate < 1:
        raise ValueError("--rate must be between 0 and 1")
    height, width, channels, length = DATASETS[args.dataset]
    if args.channel == "all":
        selected_channels = channels
    else:
        channel = int(args.channel)
        if not 0 <= channel < channels:
            raise ValueError(f"{args.dataset} channel must be all or in [0, {channels - 1}]")
        selected_channels = 1
    nodes = height * width * selected_channels
    mask, rate, dataset, channel = args.mask, args.rate, args.dataset, args.channel
    seed = args.seed
    rate_text = label(rate)
    ignnk_nm = max(1, round(nodes * 60 / 307))
    ignnk_no = nodes - ignnk_nm
    train_count, val_count, test_count = source_split_sizes(dataset)
    split_total = train_count + val_count + test_count
    train_ratio = train_count / split_total
    val_ratio = val_count / split_total
    test_ratio = test_count / split_total
    large_grid = nodes >= 1000
    batches = {model: values["batch_large" if large_grid else "batch_small"]
               for model, values in strategy.items() if "batch_large" in values}

    specs = [
        ("AGCRN", ROOT / "AGCRN", "configurations/PEMS04.conf", {
            "data": {"dataset": dataset, "data_prefix": relative_split_data(ROOT / "AGCRN", dataset, mask, rate, channel),
                     "type": mask, "miss_rate": rate_text, "num_nodes": nodes, "seq_len": length},
            "train": {"use_nni": 0, "seed": seed, "epochs": strategy["AGCRN"]["epochs"],
                      "batch_size": batches["AGCRN"], "early_stop": strategy["AGCRN"]["early_stop"],
                      "val_epoch": strategy["AGCRN"]["val_epoch"],
                      "early_stop_patience": strategy["AGCRN"]["patience"]},
        }),
        ("IGNNK", ROOT / "IGNNK", "configurations/PEMS04.conf", {
            "file": {"data_prefix": relative_split_data(ROOT / "IGNNK", dataset, mask, rate, channel),
                     "distance_df_filename": relative_split_data(ROOT / "IGNNK", dataset, mask, rate, channel) + "/grid_edges.csv",
                     "save_prefix": f"./experiments/{dataset}_{mask}_{rate_text}_channel_{channel}"},
            "train": {"use_nni": 0, "type": mask, "miss_rate": rate_text, "num_of_vertices": nodes,
                      "no": ignnk_no, "nm": ignnk_nm, "time_dim": length,
                      "max_iter": strategy["IGNNK"]["epochs"], "val_epoch": strategy["IGNNK"]["val_epoch"],
                      "batch_size": batches["IGNNK"],
                      "patience": strategy["IGNNK"]["patience"]},
        }),
        ("mTAN", ROOT / "mTAN", "configurations/PEMS04_SC-TC_0.5.conf", {
            "Data": {"data_prefix": relative_split_data(ROOT / "mTAN", dataset, mask, rate, channel),
                     "save_prefix": f"./experiments/{dataset}_{mask}_{rate_text}_channel_{channel}",
                     "type": mask, "miss_rate": rate_text, "sample_len": length},
            "Training": {"use_nni": 0, "seed": seed, "epochs": strategy["mTAN"]["epochs"],
                         "val_epoch": strategy["mTAN"]["val_epoch"],
                         "batch_size": batches["mTAN"], "patience": strategy["mTAN"]["patience"]},
        }),
        ("GAIN", ROOT / "GAIN", "configurations/PEMS04.conf", {
            "file": {"data_prefix": relative_split_data(ROOT / "GAIN", dataset, mask, rate, channel),
                     "save_prefix": f"./experiments/{dataset}_{mask}_{rate_text}_channel_{channel}"},
            "train": {"use_nni": 0, "type": mask, "miss_rate": rate_text,
                      "sample_len": length, "framewise": 1, "epoch": strategy["GAIN"]["epochs"],
                      "val_epoch": strategy["GAIN"]["val_epoch"],
                      "batch_size": batches["GAIN"], "patience": strategy["GAIN"]["patience"]},
        }),
        ("E2GAN", ROOT / "E2GAN", "configurations/PEMS04.conf", {
            "file": {"data_prefix": relative_split_data(ROOT / "E2GAN", dataset, mask, rate, channel),
                     "save_prefix": f"./experiments/{dataset}_{mask}_{rate_text}_channel_{channel}"},
            "train": {"use_nni": 0, "type": mask, "miss_rate": rate_text, "sample_len": length,
                      "epoch": strategy["E2GAN"]["epochs"],
                      "pretrain_epoch": strategy["E2GAN"]["pretrain_epochs"],
                      "val_epoch": strategy["E2GAN"]["val_epoch"],
                      "batch_size": batches["E2GAN"], "patience": strategy["E2GAN"]["patience"]},
        }),
        ("ASTGNN", ROOT / "ASTGNN", "configurations/PEMS04_SR-TC_70.conf", {
            "Data": {"adj_filename": relative_data(ROOT / "ASTGNN", dataset, mask, rate, channel) + "/grid_edges.csv",
                     "graph_signal_matrix_filename": relative_data(ROOT / "ASTGNN", dataset, mask, rate, channel),
                     "miss_type": mask, "miss_rate": rate_text, "num_of_vertices": nodes,
                     "points_per_hour": length, "num_for_predict": length, "len_input": length,
                     "dataset_name": dataset, "train_ratio": train_ratio,
                     "val_ratio": val_ratio, "test_ratio": test_ratio},
            "Training": {"use_nni": 0, "epochs": strategy["ASTGNN"]["epochs"],
                         "fine_tune_epochs": strategy["ASTGNN"]["fine_tune_epochs"],
                         "val_epoch": strategy["ASTGNN"]["val_epoch"],
                         "batch_size": batches["ASTGNN"]},
        }),
        ("SSTBAN", ROOT / "SSTBAN" / "SSTBAN-imputation", "configurations/PEMS04.conf", {
            "Data": {"dataset_name": dataset,
                     "data_prefix": relative_data(ROOT / "SSTBAN" / "SSTBAN-imputation", dataset, mask, rate, channel),
                     "miss_type": mask, "miss_rate": rate_text, "num_of_vertices": nodes, "sample_len": 12,
                     "train_ratio": train_ratio, "val_ratio": val_ratio, "test_ratio": test_ratio},
            "Time": {"start": DATASET_TIME[dataset][0], "freq": DATASET_TIME[dataset][1]},
            "Training": {"use_nni": 0, "epochs": strategy["SSTBAN"]["epochs"],
                         "val_epoch": strategy["SSTBAN"]["val_epoch"],
                         "batch_size": batches["SSTBAN"], "patience": strategy["SSTBAN"]["patience"]},
        }),
    ]
    for model, model_dir, template, updates in specs:
        write_ini(model_dir / template, output_path(dataset, model, mask, rate, channel), updates)

    csdi = ROOT / "CSDI"
    write_ini(csdi / "config/PEMS04.conf", output_path(dataset, "CSDI", mask, rate, channel), {
        "file": {"data_prefix": relative_split_data(csdi, dataset, mask, rate, channel)},
        "train": {"type": mask, "miss_rate": rate_text, "sample_len": length, "use_nni": 0,
                  "epochs": strategy["CSDI"]["epochs"], "batch_size": batches["CSDI"],
                  "val_epoch": strategy["CSDI"]["val_epoch"]},
        "diffusion": {"val_nsample": strategy["CSDI"]["val_nsample"]},
    })

    brits = ROOT / "BRITS"
    prepared = f"./training_data/{dataset}_{mask}_{rate_text}_channel_{channel}"
    experiment = f"./experiments/{dataset}_{mask}_{rate_text}_channel_{channel}"
    write_ini(brits / "configurations/PEMS04_12_SR-TR_0.1_prepare.conf",
              output_path(dataset, "BRITS_prepare", mask, rate, channel), {
        "prepare": {"seq_len": length, "attributes": 1, "type": mask, "miss_rate": rate_text,
                    "file_prefix": prepared,
                    "ori_file_prefix": relative_data(brits, dataset, mask, rate, channel),
                    "val_ratio": val_ratio, "test_ratio": test_ratio},
    })
    write_ini(brits / "configurations/PEMS04_12_SR-TR_0.1.conf",
              output_path(dataset, "BRITS", mask, rate, channel), {
        "train": {"use_nni": 0, "nodes": nodes, "seq_len": length, "attributes": 1,
                  "type": mask, "miss_rate": rate_text, "file_prefix": prepared,
                  "experiment_path": experiment, "epochs": strategy["BRITS"]["epochs"],
                  "val_epoch": strategy["BRITS"]["val_epoch"],
                  "batch_size": batches["BRITS"], "patience": strategy["BRITS"]["patience"]},
    })

    gcastn = ROOT / "GCASTN" / "GCASTN-main" / "code_data_paper_632" / "GCASTN"
    write_ini(gcastn / "configurations/PEMS04.conf", output_path(dataset, "GCASTN", mask, rate, channel), {
        "Data": {"adj_filename": relative_data(gcastn, dataset, mask, rate, channel) + "/grid_edges.csv",
                 "graph_signal_matrix_filename": relative_data(gcastn, dataset, mask, rate, channel),
                 "miss_type": mask, "miss_rate": rate_text, "num_of_vertices": nodes,
                 "points_per_hour": length, "num_for_predict": length, "len_input": length,
                 "dataset_name": dataset, "train_ratio": train_ratio,
                 "val_ratio": val_ratio, "test_ratio": test_ratio},
        "Training": {"epochs": strategy["GCASTN"]["epochs"],
                     "fine_tune_epochs": strategy["GCASTN"]["fine_tune_epochs"],
                     "val_epoch": strategy["GCASTN"]["val_epoch"],
                     "batch_size": batches["GCASTN"]},
    })

    last = ROOT / "LAST"
    write_ini(last / "configurations/PEMS04.conf", output_path(dataset, "LAST", mask, rate, channel), {
        "Data": {"dataset_name": dataset, "data_prefix": relative_data(last, dataset, mask, rate, channel),
                 "miss_type": mask, "miss_rate": rate_text, "sample_len": length,
                 "train_ratio": train_ratio, "val_ratio": val_ratio, "test_ratio": test_ratio},
    })

    latc = ROOT / "LATC"
    data = relative_data(latc, dataset, mask, rate, channel)
    write_ini(latc / "configurations/PEMS04.conf", output_path(dataset, "LATC", mask, rate, channel), {
        "Data": {"graph_signal_matrix_filename": f"{data}/true_data_{mask}_{rate_text}_v2.npz",
                 "miss_graph_signal_matrix_filename": f"{data}/miss_data_{mask}_{rate_text}_v2.npz",
                 "points_per_day": length, "test_ratio": test_ratio},
        "Training": {"use_nni": 0, "maxiter": strategy["LATC"]["epochs"]},
    })

    pristi_dir = ROOT / "PriSTI" / "PriSTI-main"
    pristi = yaml.safe_load((pristi_dir / "config/pems04.yaml").read_text(encoding="utf-8"))
    pristi["file"].update({"data_prefix": relative_split_data(pristi_dir, dataset, mask, rate, channel),
                           "dataset": dataset.lower(), "miss_type": mask, "miss_rate": rate})
    pristi["diffusion"].update({"adj_file": relative_data(pristi_dir, dataset, mask, rate, channel) + "/grid_edges.csv",
                                "node_num": nodes})
    pristi["train"].update({"nni": False, "epochs": strategy["PriSTI"]["epochs"],
                            "batch_size": batches["PriSTI"],
                            "valid_epoch_interval": strategy["PriSTI"]["val_epoch"],
                            "val_epoch": strategy["PriSTI"]["val_epoch"],
                            "val_nsample": strategy["PriSTI"]["val_nsample"]})
    pristi_path = output_path(dataset, "PriSTI", mask, rate, channel, "yaml")
    pristi_path.write_text(yaml.safe_dump(pristi, sort_keys=False), encoding="utf-8")

    impute_dir = ROOT / "imputeformer"
    impute = yaml.safe_load((impute_dir / "configurations/PEMS04.yaml").read_text(encoding="utf-8"))
    impute.update({"epochs": strategy["ImputeFormer"]["epochs"],
                   "patience": strategy["ImputeFormer"]["patience"],
                   "val_epoch": strategy["ImputeFormer"]["val_epoch"]})
    impute_path = output_path(dataset, "ImputeFormer", mask, rate, channel, "yaml")
    impute_path.write_text(yaml.safe_dump(impute, sort_keys=False), encoding="utf-8")

    # The following YAML files configure only the adapter-owned training
    # protocol. Architecture values are copied from each upstream repository's
    # published/default configuration; the model implementations stay intact.
    architectures = {
        "SAITS": {"n_groups": 2, "n_group_inner_layers": 1, "param_sharing_strategy": "inner_group",
                  "d_model": 256, "d_inner": 128, "n_head": 4, "d_k": 64, "d_v": 64, "dropout": 0.1},
        "GRIN": {"d_hidden": 64, "d_emb": 8, "d_ff": 64, "ff_dropout": 0.0,
                 "kernel_size": 2, "decoder_order": 1, "n_layers": 1, "layer_norm": False},
        "STCPA": {"n_blocks": 5, "n_temporal": 3},
        "STAMImputer": {"embed_dim": 32, "tembed_dim": 64, "num_heads": 4,
                         "mlp_ratio": 4, "dropout": 0.15, "num_layers": 4, "neighbor_width": 8},
        "PAST": {"hidden_dim": 64, "layer_num": 3, "dropout": 0.1, "alpha": 0.1, "order": 1},
    }
    for model in ("SAITS", "GRIN", "STCPA", "STAMImputer", "PAST", "MeanFill", "HistoricalAverage"):
        settings = strategy[model]
        training = {"seed": seed}
        if not settings.get("non_neural"):
            training.update({
                "epochs": settings["epochs"], "val_epoch": settings["val_epoch"],
                "batch_size": batches[model], "patience": settings["patience"],
                "lr": settings["lr"], "weight_decay": settings.get("weight_decay", 0.0),
                "grad_clip": settings.get("grad_clip", 5.0),
            })
        payload = {
            "model": model,
            "data": {"dataset": dataset, "mask": mask, "rate": rate, "channel": channel,
                     "nodes": nodes, "window": length},
            "architecture": architectures.get(model, {}),
            "training": training,
            "output": {"checkpoint": str(
                ROOT / "experiments" / f"{dataset}_{model}_{mask}_{rate_text}_channel_{channel}" / "best_model.pth"
            )},
        }
        path = output_path(dataset, model, mask, rate, channel, "yaml")
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    print(f"Generated training configs under {OUT / dataset} using policy {policy['name']} ({policy_path})")


if __name__ == "__main__":
    main()
