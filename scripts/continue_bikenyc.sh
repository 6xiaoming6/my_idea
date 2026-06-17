#!/bin/bash
# =============================================================================
#  补跑 BikeNYC full_random0.4（上次仅训练了1 epoch 即中断）
#  用法: bash scripts/continue_bikenyc.sh --gpu 1
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
ROOT="$(dirname "$SCRIPT_DIR")"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

MASK_FILE="/tmp/bikenyc_full_random0.4_$$.json"
$PYTHON -c "
import json
override = {
    'data': {
        'mask': {
            'pattern': 'random',
            'missing_rate': 0.4
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

cd "$ROOT"
$PYTHON scripts/train.py \
  -c configs/bikenyc.json \
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
