#!/usr/bin/env python3
"""Run the matched TaxiBJ random0.4 Top-2 depth/pool comparison on one GPU."""
from pathlib import Path

import run_depth_pool

POLICY = Path(__file__).resolve().parents[2] / "configs/v24/top2_pair_experiments.json"


if __name__ == "__main__":
    raise SystemExit(run_depth_pool.main(POLICY))
