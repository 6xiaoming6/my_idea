#!/usr/bin/env python3
"""Train one V21.1 distortion-calibration variant through the stable V14 runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "distortion_calibrated": "configs/v21.1-single/distortion_calibrated.json",
    "evidence_gate_control": "configs/v21.1-single/ablations/evidence_gate_no_distortion.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANTS),
        default="distortion_calibrated",
    )
    known, forwarded = parser.parse_known_args()
    if "--experiment-config" in forwarded:
        raise ValueError(
            "V21.1 owns --experiment-config; select P3/P4 with --variant."
        )
    if "--help" in forwarded or "-h" in forwarded:
        subprocess.run(
            [sys.executable, "scripts/v14-single/train.py", "--help"],
            cwd=ROOT,
            check=True,
        )
        print("\nV21.1-specific: --variant {distortion_calibrated,evidence_gate_control}")
        return

    if "--run-name" not in forwarded:
        default = (
            "full"
            if known.variant == "distortion_calibrated"
            else "ablation_v21_1_p4_evidence_gate_control"
        )
        forwarded.extend(("--run-name", default))
    command = [
        sys.executable,
        "scripts/v14-single/train.py",
        *forwarded,
        "--experiment-config",
        VARIANTS[known.variant],
    ]
    print("[v21.1]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
