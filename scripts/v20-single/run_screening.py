#!/usr/bin/env python3
from __future__ import annotations

from _matrix import common_parser, run_points


POINTS = [
    ("TaxiBJ", "fixed", "0.2"),
    ("TaxiBJ", "fixed", "0.4"),
    ("TaxiBJ", "random", "0.4"),
    ("TaxiBJ", "random", "0.8"),
    ("BikeNYC", "fixed", "0.6"),
    ("BikeNYC", "random", "0.4"),
    ("CHAP", "fixed", "0.4"),
    ("CHAP", "random", "0.4"),
]


if __name__ == "__main__":
    args = common_parser("Run the V20 eight-point screening.").parse_args()
    run_points(args, POINTS)
