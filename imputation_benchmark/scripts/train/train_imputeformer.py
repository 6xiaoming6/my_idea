#!/usr/bin/env python3
import os
from common import BASELINES, adapted_data, config, finish, parse_args, rate_label, recommended_batch, run_stages, window_length

def setup(args, cwd):
    (cwd / "logs").mkdir(parents=True, exist_ok=True)
    config_link = cwd / "configurations" / f"{args.dataset}.yaml"
    config_link.unlink(missing_ok=True)
    config_link.symlink_to(os.path.relpath(config(args, "ImputeFormer", "yaml"), config_link.parent))
    data = adapted_data(args, split=True)
    data.mkdir(parents=True, exist_ok=True)
    dataset_link = data / args.dataset
    if dataset_link.is_symlink():
        dataset_link.unlink()
    if not dataset_link.exists():
        dataset_link.symlink_to(".")

def main():
    args = parse_args("ImputeFormer")
    cwd = BASELINES / "imputeformer"
    finish(run_stages("ImputeFormer", args, cwd, [[
        args.python, "main.py", "--data_prefix", str(adapted_data(args, split=True).resolve()),
        "--dataset", args.dataset, "--miss_type", args.mask,
        "--miss_rate", rate_label(args.rate), "--sample_len", str(window_length(args)),
        "--batch_size", str(recommended_batch("ImputeFormer", args)),
    ]], setup=setup))

if __name__ == "__main__": main()
