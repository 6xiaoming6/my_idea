#!/bin/bash
# =============================================================================
#  续跑 TaxiBJ — fixed 模式
#  用法: bash scripts/continue_taxibj.sh --gpu 0
# =============================================================================
set -euo pipefail

GPU="0"
CONDA_ENV="difftdi"
PYTHON="conda run --no-capture-output -n ${CONDA_ENV} python"
FIXED_SEED="42"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu|-g) GPU="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$ROOT/outputs"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

DS="TaxiBJ"
CFG="configs/taxibj.json"
TRAIN_NPZ="data/TaxiBJ/taxibj_train.npz"
VAL_NPZ="data/TaxiBJ/taxibj_val.npz"

already_done() {
  local exp_name="$1" mask_suffix="$2"
  compgen -G "${OUTPUT_DIR}/${DS}/${exp_name}_${mask_suffix}_*" > /dev/null 2>&1
}

write_fixed_override() {
  local rate="$1" out="$2"
  local mask_dir="data/${DS}/fixed_mask/${rate}"
  local train_mask="${mask_dir}/train.csv"
  local val_mask="${mask_dir}/val.csv"
  $PYTHON scripts/generate_fixed_masks.py \
    --train_npz "$TRAIN_NPZ" \
    --val_npz "$VAL_NPZ" \
    --pattern fixed \
    --mask_rate "$rate" \
    --seed "$FIXED_SEED" \
    --output_dir "$mask_dir"
  $PYTHON -c "
import json
override = {
    'data': {
        'mask': {
            'pattern': 'fixed',
            'missing_rate': ${rate},
            'train_csv': '${train_mask}',
            'val_csv': '${val_mask}'
        }
    }
}
with open('${out}', 'w') as f:
    json.dump(override, f, indent=2)
"
}

make_combined() {
  local mask_file="$1" abl_name="$2" out="$3"
  $PYTHON -c "
import json
from stmoe_imputer.config import load_config, deep_update
combined = deep_update(load_config('${mask_file}'), load_config('configs/${abl_name}.json'))
with open('${out}', 'w') as f:
    json.dump(combined, f, indent=2)
"
}

run_exp() {
  local exp_name="$1" mask_override="$2"
  echo "[$(date '+%H:%M:%S')] ${DS} | ${exp_name}"
  cd "$ROOT"
  $PYTHON scripts/train.py \
    -c "$CFG" \
    --override_config "$mask_override" \
    --train_npz "$TRAIN_NPZ" \
    --val_npz "$VAL_NPZ" \
    -n "${exp_name}" \
    --no_plot --quiet
  echo "  => ${exp_name} 完成"
}

ABLATIONS=(
  "ablation_fine_only"
  "ablation_no_router"
  "ablation_fixed_scale_experts"
  "ablation_no_cross_scale"
  "ablation_routed_only"
  "ablation_shared_only"
)

run_block() {
  local rate="$1" skip_full="$2"
  local suffix="fixed${rate}"
  local mask_file="/tmp/taxibj_fixed_${rate}_$$.json"
  write_fixed_override "$rate" "$mask_file"

  echo ""
  echo "--- TaxiBJ | fixed | rate=${rate} ---"

  if [ "$skip_full" = true ]; then
    echo "[skip] full_${suffix}"
  elif already_done "full" "$suffix"; then
    echo "[skip] full_${suffix} (already exists)"
  else
    run_exp "full" "$mask_file"
  fi

  for abl in "${ABLATIONS[@]}"; do
    if already_done "$abl" "$suffix"; then
      echo "[skip] ${abl}_${suffix} (already exists)"
      continue
    fi
    local combined="/tmp/taxibj_${abl}_${rate}_$$.json"
    make_combined "$mask_file" "$abl" "$combined"
    run_exp "$abl" "$combined"
    rm -f "$combined"
  done

  rm -f "$mask_file"
}

echo "=============================================="
echo "  续跑 TaxiBJ (fixed) — GPU ${GPU}"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

run_block "0.2" true   # full已做, 消融自动跳过已做的; 实际只跑 shared_only
run_block "0.4" false
run_block "0.6" false
run_block "0.8" false

echo ""
echo "=============================================="
echo "  TaxiBJ 续跑完成！ $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
