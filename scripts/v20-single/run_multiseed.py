#!/usr/bin/env python3
from __future__ import annotations

from _matrix import common_parser, run_points


POINTS = [
    ("TaxiBJ", "fixed", "0.4"),
    ("TaxiBJ", "random", "0.4"),
    ("BikeNYC", "random", "0.4"),
    ("CHAP", "fixed", "0.4"),
]


def main() -> None:
    parser = common_parser("Run the V20 four-point three-seed confirmation.")
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 2026, 3407))
    args = parser.parse_args()
    run_points(args, POINTS, seeds=list(dict.fromkeys(args.seeds)))


if __name__ == "__main__":
    main()
