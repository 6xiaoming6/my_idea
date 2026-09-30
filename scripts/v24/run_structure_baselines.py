#!/usr/bin/env python3
"""Run B1 -> B2 -> B4 -> B5 on one GPU; retain completed B3 as reference.

Uses the same no-feedback protocol, frozen-source runner and completion checks
as run_b3_c3.py. Restarting the unchanged command skips finished experiments.
"""
from run_b3_c3 import main

if __name__ == "__main__":
    main(default_variants=("B1", "B2", "B4", "B5"), suite_name="structure_baselines")
