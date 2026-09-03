#!/usr/bin/env python3
from __future__ import annotations

from _matrix import common_parser, run_points
from run_screening import POINTS


ABLATIONS = (
    "random_exam_only",
    "geometry_exam_only",
    "random_hybrid",
    "geometry_hybrid",
    "geometry_prior_only",
    "legacy_geometry_hybrid",
)


def main() -> None:
    parser = common_parser("Run V20 routing ablations on the screening points.")
    parser.add_argument("--ablations", nargs="+", choices=ABLATIONS, default=ABLATIONS)
    args = parser.parse_args()
    for ablation in args.ablations:
        run_points(args, POINTS, ablation=ablation)


if __name__ == "__main__":
    main()
