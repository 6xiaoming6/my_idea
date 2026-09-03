#!/usr/bin/env python3
from __future__ import annotations

from _matrix import common_parser, run_points


def main() -> None:
    parser = common_parser("Run the three V20 two-epoch smoke points.")
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 2
    run_points(args, [
        ("TaxiBJ", "fixed", "0.4"),
        ("BikeNYC", "random", "0.4"),
        ("CHAP", "fixed", "0.4"),
    ], run_name="smoke_v20")


if __name__ == "__main__":
    main()
