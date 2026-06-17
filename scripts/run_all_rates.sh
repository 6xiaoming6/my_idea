#!/bin/bash
# =============================================================================
#  一键训练全部缺失率 — fixed 或 random 模式
#
#  用法:
#    bash scripts/run_all_rates.sh --dataset TaxiBJ --gpu 0 --mask_pattern fixed
#    bash scripts/run_all_rates.sh --dataset TaxiBJ --gpu 0 --mask_pattern random
#    bash scripts/run_all_rates.sh --dataset all --gpu 0 --mask_pattern all
#
#  参数:
#    --dataset       TaxiBJ | BikeNYC | all
#    --gpu           GPU id
#    --mask_pattern  fixed | random | all (默认: random, all = fixed + random)
#    --skip_full     跳过 Full Model，只跑消融
# =============================================================================
set -euo pipefail

DATASET=""
GPU="0"
MASK_PATTERN="random"
SKIP_FULL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset|-d)       DATASET="$2"; shift 2 ;;
    --gpu|-g)           GPU="$2"; shift 2 ;;
    --mask_pattern|-m)  MASK_PATTERN="$2"; shift 2 ;;
    --skip_full)        SKIP_FULL=true; shift ;;
    *) echo "Unknown: $1"
       echo "Usage: $0 --dataset {TaxiBJ|BikeNYC|all} --gpu <id> [--mask_pattern fixed|random|all] [--skip_full]"
       exit 1 ;;
  esac
done

if [ -z "$DATASET" ]; then
  echo "Usage: $0 --dataset {TaxiBJ|BikeNYC|all} --gpu <id> [--mask_pattern fixed|random|all] [--skip_full]"
  echo ""
  echo "Examples:"
  echo "  bash $0 --dataset TaxiBJ --gpu 0 --mask_pattern fixed"
  echo "  bash $0 --dataset TaxiBJ --gpu 0 --mask_pattern random"
  echo "  bash $0 --dataset TaxiBJ --gpu 0 --mask_pattern all"
  echo "  bash $0 --dataset all --gpu 0 --mask_pattern fixed"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ABL="$SCRIPT_DIR/run_all_ablations.sh"

RATES=(0.2 0.4 0.6 0.8)
GLOBAL_START=$(date +%s)

# Resolve patterns to run
if [ "$MASK_PATTERN" = "all" ]; then
  PATTERNS=("fixed" "random")
else
  PATTERNS=("$MASK_PATTERN")
fi

echo ""
echo "=============================================="
echo "  全部缺失率训练"
echo "  数据集:       ${DATASET}"
echo "  GPU:          cuda:${GPU}"
echo "  Mask Pattern: ${PATTERNS[*]}"
echo "  Rates:        ${RATES[*]}"
echo "  Skip Full:    ${SKIP_FULL}"
echo "  开始时间:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo ""

TOTAL_RATES=${#RATES[@]}
TOTAL_PATTERNS=${#PATTERNS[@]}
TOTAL_STEPS=$((TOTAL_PATTERNS * TOTAL_RATES))
STEP=0

for pattern in "${PATTERNS[@]}"; do
  for rate in "${RATES[@]}"; do
    STEP=$((STEP + 1))
    echo "##############################################"
    echo "##  [$STEP/$TOTAL_STEPS] pattern=${pattern}  rate=${rate}"
    echo "##############################################"

    SKIP_FLAG=""
    if [ "$SKIP_FULL" = true ]; then
      SKIP_FLAG="--skip_full"
    fi

    bash "$RUN_ABL" \
      --dataset "$DATASET" \
      --gpu "$GPU" \
      --mask_pattern "$pattern" \
      --mask_rate "$rate" \
      $SKIP_FLAG

    echo ""
  done
done

ELAPSED=$(( $(date +%s) - GLOBAL_START ))
echo "=============================================="
echo "  全部完成！"
echo "  数据集:       ${DATASET}"
echo "  Mask Pattern: ${PATTERNS[*]}"
echo "  Rates:        ${RATES[*]}"
echo "  耗时:         ${ELAPSED}s ($((ELAPSED/60))m$((ELAPSED%60))s)"
echo "  结束时间:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
