#!/bin/bash
# =============================================================================
#  补跑 BikeNYC full_random0.4（上次仅训练了1 epoch 即中断）
#  用法: bash scripts/archive/continue_bikenyc.sh --gpu 1
# =============================================================================
set -euo pipefail

GPU="1"
CONDA_ENV="difftdi"
PYTHON="conda run --no-capture-output -n ${CONDA_ENV} python"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu|-g) GPU="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

RATE="0.4"
SEED="42"
MASK_DIR="data/BikeNYC/random_mask/${RATE}"
TRAIN_MASK="${MASK_DIR}/train.csv"
VAL_MASK="${MASK_DIR}/val.csv"

cd "$ROOT"
$PYTHON scripts/generate_fixed_masks.py \
  --train_npz data/BikeNYC/bikenyc_train.npz \
  --val_npz data/BikeNYC/bikenyc_val.npz \
  --pattern random \
  --mask_rate "$RATE" \
  --seed "$SEED" \
  --output_dir "$MASK_DIR"

MASK_FILE="/tmp/bikenyc_full_random0.4_$$.json"
$PYTHON -c "
import json
override = {
    'data': {
        'mask': {
            'pattern': 'random',
            'missing_rate': 0.4,
            'train_csv': '${TRAIN_MASK}',
            'val_csv': '${VAL_MASK}'
        }
    }
}
with open('${MASK_FILE}', 'w') as f:
    json.dump(override, f, indent=2)
"

echo "=============================================="
echo "  BikeNYC full_random0.4 — GPU ${GPU}"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

$PYTHON scripts/train.py \
  -c configs/datasets/bikenyc.json \
  --override_config "$MASK_FILE" \
  --train_npz data/BikeNYC/bikenyc_train.npz \
  --val_npz data/BikeNYC/bikenyc_val.npz \
  -n "full" \
  --no_plot --quiet

rm -f "$MASK_FILE"

echo ""
echo "=============================================="
echo "  BikeNYC full_random0.4 完成！ $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
