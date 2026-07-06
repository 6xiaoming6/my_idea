#!/usr/bin/env python3
"""Shared entry point for the newly vendored baseline adapters."""
from __future__ import annotations

from common import BENCH, adapted_data, config, finish, parse_args, rate_label, run_stages


def run(model: str) -> None:
    args = parse_args(model)
    cfg = config(args, model, "yaml")
    command = [
        args.python, str(BENCH / "scripts" / "train" / "adapted_grid_runner.py"),
        "--model", model, "--config", cfg,
        "--data-prefix", str(adapted_data(args, split=True).resolve()),
        "--dataset", args.dataset, "--mask", args.mask,
        "--rate", rate_label(args.rate), "--channel", args.channel,
    ]
    finish(run_stages(model, args, BENCH, [command]))
