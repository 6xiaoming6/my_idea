#!/bin/bash
# Run ablation baselines on TaxiBJ and BikeNYC sequentially.
# fine_only and no_router already completed for TaxiBJ — skip.
set -e
cd "$(dirname "$0")/.."

TAXIBJ_ABLATIONS=(
  "ablation_fixed_scale_experts"
  "ablation_no_cross_scale"
  "ablation_routed_only"
  "ablation_shared_only"
)

BIKENYC_ABLATIONS=(
  "ablation_fine_only"
  "ablation_no_router"
  "ablation_fixed_scale_experts"
  "ablation_no_cross_scale"
  "ablation_routed_only"
  "ablation_shared_only"
)

echo "===== TaxiBJ baselines (remaining 4) ====="
for abl in "${TAXIBJ_ABLATIONS[@]}"; do
  echo "--- $abl ---"
  python scripts/train.py \
    -c configs/datasets/taxibj.json \
    --override_config "configs/ablations/${abl}.json" \
    --train_npz data/TaxiBJ/taxibj_train.npz \
    --val_npz data/TaxiBJ/taxibj_val.npz \
    -n "$abl" --no_plot
done

echo "===== BikeNYC baselines (all 6) ====="
for abl in "${BIKENYC_ABLATIONS[@]}"; do
  echo "--- $abl ---"
  python scripts/train.py \
    -c configs/datasets/bikenyc.json \
    --override_config "configs/ablations/${abl}.json" \
    --train_npz data/BikeNYC/bikenyc_train.npz \
    --val_npz data/BikeNYC/bikenyc_val.npz \
    -n "$abl" --no_plot
done

echo "===== All done ====="
