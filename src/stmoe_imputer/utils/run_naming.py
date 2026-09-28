"""Compact names for the experiment directory above mask and rate folders."""
from __future__ import annotations

import re


def experiment_label(name: str) -> str:
    """Keep the model/ablation identity, without version, data, mask, or seed."""
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip()).strip("_").lower()
    label = re.sub(r"^v\d+_", "", label)
    label = re.sub(r"^coe_", "", label)
    label = re.sub(r"_e\d+(?=_seed\d+$)", "", label)  # epoch suffix, not expert count
    mask = (r"(?:original_random|mixed_random_point|random_point|node_contiguous|"
            r"spatial_region|spatiotemporal_block|alt_mixed\d+|mixed\d*|random|fixed)")
    label = re.sub(r"_" + mask + r"_rate\d+(?:\.\d+)?(?:_updates)?(?=_|$)", "", label)
    label = re.sub(r"_seed\d+$", "", label)
    label = re.sub(r"(^|_)(?:taxibj|bikenyc)(?=_|$)", r"\1", label)
    label = re.sub(r"(?<=_)v\d+_", "", label)
    label = label.replace("partner_focus_focus_", "partner_focus_")
    label = label.replace("focus_focus_", "focus_")
    label = label.replace("partner_residual4_residual4_", "partner_residual4_")
    label = label.replace("partner_native4_residual4_", "partner_native4_")
    label = label.replace("team_accept_team_", "team_accept_")
    for repeated in ("abc", "chain4", "route20"):
        label = label.replace(f"{repeated}_{repeated}_", f"{repeated}_")
    return re.sub(r"_+", "_", label).strip("_") or "experiment"
