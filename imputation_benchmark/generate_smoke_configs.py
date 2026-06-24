#!/usr/bin/env python3
"""Generate fixed-mask smoke configurations for all benchmark entry points.

Only dataset-dependent values and smoke runtime values are written.  Network
architecture hyperparameters remain those in each model's bundled template.
Run from the project root before executing the generated commands.
"""
from __future__ import annotations

import configparser
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "generated_configs"
DATASETS = {
    "TaxiBJ": (32, 32, 2, 12),
    "BikeNYC": (24, 12, 2, 12),
    "CHAP": (32, 32, 1, 7),
}
RATE = "0.2"

DATASET_TIME = {
    "TaxiBJ": ("20210101 00:00:00", "5min"),
    "BikeNYC": ("20210101 00:00:00", "5min"),
    "CHAP": ("20180101 00:00:00", "1D"),
}


def legacy_path(model_dir: Path, dataset: str) -> str:
    return str(Path(__import__("os").path.relpath(
        ROOT / "data" / "smoke" / dataset / f"fixed_{RATE}" / "channel_0", model_dir
    ))).replace("\\", "/")


def write_ini(template: Path, output: Path, updates: dict[str, dict[str, str]]) -> None:
    cfg = configparser.ConfigParser()
    cfg.read(template)
    for section, values in updates.items():
        if not cfg.has_section(section):
            cfg.add_section(section)
        for key, value in values.items():
            cfg[section][key] = str(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        cfg.write(f)


def main() -> None:
    commands: list[str] = []
    for dataset, (h, w, channels, length) in DATASETS.items():
        nodes = h * w
        # Graph/legacy models are validated per channel.  The data adapter
        # creates channel_0; channel_1 is generated separately for flow data.
        specs = [
            ("AGCRN", ROOT / "AGCRN", "configurations/PEMS04.conf", {"data": {"dataset": dataset, "data_prefix": legacy_path(ROOT / "AGCRN", dataset), "type": "fixed", "miss_rate": RATE, "num_nodes": nodes, "seq_len": length}, "train": {"epochs": "1"}}),
            ("IGNNK", ROOT / "IGNNK", "configurations/PEMS04.conf", {"file": {"data_prefix": legacy_path(ROOT / "IGNNK", dataset), "distance_df_filename": legacy_path(ROOT / "IGNNK", dataset) + "/grid_edges.csv", "save_prefix": "./smoke_experiments"}, "train": {"type": "fixed", "miss_rate": RATE, "num_of_vertices": nodes, "time_dim": length, "max_iter": "1"}}),
            ("mTAN", ROOT / "mTAN", "configurations/PEMS04_SC-TC_0.5.conf", {"Data": {"data_prefix": legacy_path(ROOT / "mTAN", dataset), "type": "fixed", "miss_rate": RATE, "sample_len": length, "save_prefix": "./smoke_experiments"}, "Training": {"epochs": "1", "batch_size": "1"}}),
            ("GAIN", ROOT / "GAIN", "configurations/PEMS04.conf", {"file": {"data_prefix": legacy_path(ROOT / "GAIN", dataset), "save_prefix": "./smoke_experiments"}, "train": {"use_nni": "0", "type": "fixed", "miss_rate": RATE, "epoch": "1", "batch_size": "1"}}),
            ("E2GAN", ROOT / "E2GAN", "configurations/PEMS04.conf", {"file": {"data_prefix": legacy_path(ROOT / "E2GAN", dataset), "save_prefix": "./smoke_experiments"}, "train": {"use_nni": "0", "type": "fixed", "miss_rate": RATE, "epoch": "1", "pretrain_epoch": "1", "batch_size": "1"}}),
            ("ASTGNN", ROOT / "ASTGNN", "configurations/PEMS04_SR-TC_70.conf", {"Data": {"adj_filename": legacy_path(ROOT / "ASTGNN", dataset) + "/grid_edges.csv", "graph_signal_matrix_filename": legacy_path(ROOT / "ASTGNN", dataset), "miss_type": "fixed", "miss_rate": RATE, "num_of_vertices": nodes, "points_per_hour": length, "num_for_predict": length, "len_input": length, "dataset_name": dataset}, "Training": {"epochs": "1", "fine_tune_epochs": "0", "batch_size": "1"}}),
            ("SSTBAN", ROOT / "SSTBAN" / "SSTBAN-imputation", "configurations/PEMS04.conf", {"Data": {"dataset_name": dataset, "data_prefix": legacy_path(ROOT / "SSTBAN" / "SSTBAN-imputation", dataset), "miss_type": "fixed", "miss_rate": RATE, "num_of_vertices": nodes, "sample_len": length}, "Time": {"start": DATASET_TIME[dataset][0], "freq": DATASET_TIME[dataset][1]}, "Training": {"epochs": "1", "batch_size": "1"}}),
        ]
        for name, model_dir, template_rel, updates in specs:
            output = OUT / dataset / f"{name}_fixed_{RATE}_smoke.conf"
            write_ini(model_dir / template_rel, output, updates)
            commands.append(f"# {name}\ncd {model_dir.relative_to(ROOT)} && python <train-entry> --config {output.relative_to(ROOT)}")
        # CSDI
        csdi_dir = ROOT / "CSDI"
        write_ini(csdi_dir / "config/PEMS04.conf", OUT / dataset / f"CSDI_fixed_{RATE}_smoke.conf", {
            "file": {"data_prefix": legacy_path(csdi_dir, dataset)},
            "train": {"type": "fixed", "miss_rate": RATE, "epochs": "1", "batch_size": "1", "sample_len": length},
        })
        commands.append(f"# CSDI\ncd CSDI && python train.py --config {OUT.relative_to(ROOT) / f'CSDI_fixed_{RATE}_smoke.conf'}")
        # BRITS needs a prepare config and a train config.
        brits_dir = ROOT / "BRITS"
        smoke_dir = f"./smoke_data/{dataset}_fixed_{RATE}"
        write_ini(brits_dir / "configurations/PEMS04_12_SR-TR_0.1_prepare.conf", OUT / dataset / f"BRITS_fixed_{RATE}_prepare_smoke.conf", {
            "prepare": {"seq_len": length, "attributes": "1", "type": "fixed", "miss_rate": RATE,
                        "test_ratio": "0.2", "val_ratio": "0.1", "file_prefix": smoke_dir,
                        "ori_file_prefix": legacy_path(brits_dir, dataset)},
        })
        write_ini(brits_dir / "configurations/PEMS04_12_SR-TR_0.1.conf", OUT / dataset / f"BRITS_fixed_{RATE}_smoke.conf", {
            "train": {"use_nni": "0", "epochs": "1", "batch_size": "1", "nodes": nodes,
                      "seq_len": length, "attributes": "1", "type": "fixed", "miss_rate": RATE,
                      "file_prefix": smoke_dir, "experiment_path": f"./smoke_experiments/{dataset}_fixed_{RATE}"},
        })
        commands.append(f"# BRITS (step 1: prepare JSON, step 2: train)\ncd BRITS && python input_process.py --config {OUT.relative_to(ROOT) / f'BRITS_fixed_{RATE}_prepare_smoke.conf'}\ncd BRITS && python main.py --config {OUT.relative_to(ROOT) / f'BRITS_fixed_{RATE}_smoke.conf'}")
        # GCASTN
        gcast_dir = ROOT / "GCASTN" / "GCASTN-main" / "code_data_paper_632" / "GCASTN"
        write_ini(gcast_dir / "configurations/PEMS04.conf", OUT / dataset / f"GCASTN_fixed_{RATE}_smoke.conf", {
            "Data": {"adj_filename": legacy_path(gcast_dir, dataset) + "/grid_edges.csv", "graph_signal_matrix_filename": legacy_path(gcast_dir, dataset), "miss_type": "fixed", "miss_rate": RATE, "num_of_vertices": nodes, "points_per_hour": length, "num_for_predict": length, "len_input": length, "dataset_name": dataset},
            "Training": {"epochs": "1", "fine_tune_epochs": "0", "batch_size": "1"},
        })
        commands.append(f"# GCASTN\ncd GCASTN/GCASTN-main/code_data_paper_632/GCASTN && python train_GCASTN.py --config {OUT.relative_to(ROOT) / f'GCASTN_fixed_{RATE}_smoke.conf'}")
        # LAST
        last_dir = ROOT / "LAST"
        write_ini(last_dir / "configurations/PEMS04.conf", OUT / dataset / f"LAST_fixed_{RATE}_smoke.conf", {
            "Data": {"dataset_name": dataset, "data_prefix": legacy_path(last_dir, dataset), "miss_type": "fixed", "miss_rate": RATE, "sample_len": length, "train_ratio": "0.7", "val_ratio": "0.1", "test_ratio": "0.2"},
        })
        commands.append(f"# LAST\ncd {last_dir.relative_to(ROOT)} && python main.py --config {OUT.relative_to(ROOT) / f'LAST_fixed_{RATE}_smoke.conf'} --miss_type fixed --miss_rate {RATE}")
        # PriSTI and ImputeFormer use YAML model templates; save dataset-aware copies.
        pristi_dir = ROOT / "PriSTI" / "PriSTI-main"
        pristi = yaml.safe_load((pristi_dir / "config/pems04.yaml").read_text())
        pristi["file"].update({"data_prefix": legacy_path(pristi_dir, dataset), "dataset": dataset.lower(), "miss_type": "fixed", "miss_rate": float(RATE)})
        pristi["train"].update({"epochs": 1, "batch_size": 1, "nni": False})
        output = OUT / dataset / f"PriSTI_fixed_{RATE}_smoke.yaml"; output.write_text(yaml.safe_dump(pristi, sort_keys=False))
        commands.append(f"# PriSTI\ncd PriSTI/PriSTI-main && python exe_survey.py --config {OUT.relative_to(ROOT) / f'PriSTI_fixed_{RATE}_smoke.yaml'} --targetstrategy random")
        imp_dir = ROOT / "imputeformer"
        imp = yaml.safe_load((imp_dir / "configurations/PEMS04.yaml").read_text())
        imp.update({"epochs": 1, "patience": 1})
        output = OUT / dataset / f"ImputeFormer_fixed_{RATE}_smoke.yaml"; output.write_text(yaml.safe_dump(imp, sort_keys=False))
        commands.append(f"# ImputeFormer\ncd imputeformer && python main.py --data_prefix {legacy_path(imp_dir, dataset)} --dataset {dataset} --miss_type fixed --miss_rate {RATE} --sample_len {length} --batch_size 1 --epochs 1")
        # LATC has no network architecture; use one iteration only for smoke.
        latc = OUT / dataset / f"LATC_fixed_{RATE}_smoke.conf"
        path = legacy_path(ROOT / "LATC", dataset)
        latc.parent.mkdir(parents=True, exist_ok=True)
        latc.write_text(
            f"[Data]\ngraph_signal_matrix_filename = {path}/true_data_fixed_{RATE}_v2.npz\n"
            f"miss_graph_signal_matrix_filename = {path}/miss_data_fixed_{RATE}_v2.npz\npoints_per_day = {length}\ntest_ratio = 0.2\n"
            "[Training]\nuse_nni = 0\nc = 1\ntheta = 5\nmaxiter = 1\n",
            encoding="utf-8",
        )
        commands.append(f"# LATC\ncd LATC && python train_LATC.py --config {OUT.relative_to(ROOT) / f'LATC_fixed_{RATE}_smoke.conf'}")
    (OUT / "COMMANDS.txt").write_text("\n\n".join(commands) + "\n", encoding="utf-8")
    print(f"Generated configs under {OUT}")


if __name__ == "__main__":
    main()
