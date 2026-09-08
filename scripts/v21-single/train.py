#!/usr/bin/env python3
"""Train V21 by reusing the stable V14 runner and applying one V21 patch."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "dual_state": "configs/v21-single/observation_moment_dual_state.json",
    "content_only": "configs/v21-single/ablations/content_only.json",
    "legacy_control": "configs/v21-single/ablations/legacy_pyramid.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANTS),
        default="dual_state",
        help="V21 scale-pyramid variant (default: dual_state).",
    )
    known, forwarded = parser.parse_known_args()
    if "--experiment-config" in forwarded:
        raise ValueError(
            "V21 owns --experiment-config; use --variant to select its controlled ablation."
        )
    if "--help" in forwarded or "-h" in forwarded:
        subprocess.run(
            [sys.executable, "scripts/v14-single/train.py", "--help"],
            cwd=ROOT,
            check=True,
        )
        print("\nV21-specific: --variant {dual_state,content_only,legacy_control}")
        return

    if "--run-name" not in forwarded:
        default_name = "full" if known.variant == "dual_state" else f"ablation_v21_{known.variant}"
        forwarded.extend(("--run-name", default_name))
    command = [
        sys.executable,
        "scripts/v14-single/train.py",
        *forwarded,
        "--experiment-config",
        VARIANTS[known.variant],
    ]
    print("[v21]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
