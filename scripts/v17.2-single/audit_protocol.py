#!/usr/bin/env python3
"""Audit that V17.2 differs from V17.1 E1 only by version metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_common import (
    DATASETS,
    RATES,
    ROOT,
    build_protocol_audit,
    build_resolved_config,
    load_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="TaxiBJ")
    parser.add_argument("--mask", choices=("fixed", "random"), default="random")
    parser.add_argument("--rate", choices=RATES, default="0.4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--full-config", default=None)
    parser.add_argument("--e1-config", default=None)
    parser.add_argument("--v17-2-config", default=None)
    parser.add_argument(
        "--output",
        default="outputs/v17.2-single/summary/protocol_audit.json",
    )
    return parser.parse_args()


def _resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    args = parse_args()
    explicit = (args.full_config, args.e1_config, args.v17_2_config)
    if any(explicit) and not all(explicit):
        raise ValueError(
            "--full-config, --e1-config and --v17-2-config must be supplied together"
        )
    if all(explicit):
        full = load_json(_resolve(args.full_config))
        e1 = load_json(_resolve(args.e1_config))
        candidate = load_json(_resolve(args.v17_2_config))
    else:
        common = (args.dataset, args.mask, args.rate, args.seed)
        full = build_resolved_config(*common, version="full", epochs=args.epochs)
        e1 = build_resolved_config(*common, version="e1", epochs=args.epochs)
        candidate = build_resolved_config(
            *common,
            version="v17_2",
            epochs=args.epochs,
        )

    report = build_protocol_audit(full, e1, candidate)
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    try:
        display_path = output.relative_to(ROOT)
    except ValueError:
        display_path = output
    print(f"[saved] {display_path}")
    if not report["passed"]:
        raise SystemExit("V17.2 protocol audit failed")


if __name__ == "__main__":
    main()
