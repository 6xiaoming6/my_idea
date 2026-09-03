#!/usr/bin/env python3
from __future__ import annotations

from _matrix import DATASETS, RATES, common_parser, run_points


if __name__ == "__main__":
    parser = common_parser("Run all 24 formal V20 experiments.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--patterns", nargs="+", choices=("fixed", "random"), default=("fixed", "random"))
    parser.add_argument("--rates", nargs="+", choices=RATES, default=RATES)
    args = parser.parse_args()
    points = [
        (dataset, pattern, rate)
        for dataset in args.datasets
        for pattern in args.patterns
        for rate in args.rates
    ]
    run_points(args, points)
