#!/bin/bash
# =============================================================================
#  通用消融实验运行脚本 — Full Model + 6 组消融
#
#  用法:
#    bash scripts/run_all_ablations.sh --dataset TaxiBJ --gpu 0 --mask_pattern random --mask_rate 0.4
#    bash scripts/run_all_ablations.sh --dataset BikeNYC --gpu 1
#    bash scripts/run_all_ablations.sh --dataset TaxiBJ --gpu 0 --mask_pattern fixed --mask_rate 0.4
#
#  参数:
#    --dataset          TaxiBJ | BikeNYC | all
#    --gpu              GPU id
#    --mask_pattern     random | fixed (默认: random)
#    --mask_rate        缺失率 (默认: 0.4)
#    --fixed_seed       fixed 模式的 CSV seed (默认: 42)
#    --skip_full        跳过 Full Model，只跑消融
# =============================================================================
set -euo pipefail

# ── 参数解析 ──────────────────────────────────────────────────────────────
DATASET=""
GPU="0"
MASK_PATTERN="random"
MASK_RATE="0.4"
FIXED_SEED="42"
SKIP_FULL=false
CONDA_ENV="difftdi"
PYTHON="conda run --no-capture-output -n ${CONDA_ENV} python"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset|-d)       DATASET="$2"; shift 2 ;;
    --gpu|-g)           GPU="$2"; shift 2 ;;
    --mask_pattern|-m)  MASK_PATTERN="$2"; shift 2 ;;
    --mask_rate|-r)     MASK_RATE="$2"; shift 2 ;;
    --fixed_seed)       FIXED_SEED="$2"; shift 2 ;;
    --skip_full)        SKIP_FULL=true; shift ;;
    *) echo "Unknown: $1"
       echo "Usage: $0 --dataset {TaxiBJ|BikeNYC|all} --gpu <id> [--mask_pattern <p>] [--mask_rate <r>] [--fixed_seed <s>] [--skip_full]"
       exit 1 ;;  esac
done

if [ -z "$DATASET" ]; then
  echo "Usage: $0 --dataset {TaxiBJ|BikeNYC|all} --gpu <id> [--mask_pattern <p>] [--mask_rate <r>] [--fixed_seed <s>] [--skip_full]"
  echo ""
  echo "Examples:"
  echo "  bash $0 --dataset TaxiBJ --gpu 0"
  echo "  bash $0 --dataset TaxiBJ --gpu 0 --mask_pattern random --mask_rate 0.4"
  echo "  bash $0 --dataset TaxiBJ --gpu 0 --mask_pattern fixed --mask_rate 0.4"
  echo "  bash $0 --dataset TaxiBJ --gpu 0 --mask_pattern fixed --mask_rate 0.6 --fixed_seed 42"
  echo "  bash $0 --dataset all --gpu 0 --mask_pattern random --mask_rate 0.3"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"

cd "$(dirname "$0")/.."
SCRIPT_DIR="$(pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── 生成 Mask Override Config ──────────────────────────────────────────────
MASK_OVERRIDE_FILE="/tmp/mask_override_${MASK_PATTERN}_${MASK_RATE}_${TIMESTAMP}.json"

if [ "$MASK_PATTERN" != "fixed" ] && [ "$MASK_PATTERN" != "random" ]; then
  echo "Unsupported --mask_pattern ${MASK_PATTERN}. Use fixed or random."
  exit 1
fi

cleanup_mask_override() {
  rm -f "$MASK_OVERRIDE_FILE"
}
trap cleanup_mask_override EXIT

# ── 数据集列表 ────────────────────────────────────────────────────────────
if [ "$DATASET" = "all" ]; then
  DATASETS=("TaxiBJ" "BikeNYC")
else
  DATASETS=("$DATASET")
fi

# 6 组消融
ABLATIONS=(
  "ablation_fine_only"
  "ablation_no_router"
  "ablation_fixed_scale_experts"
  "ablation_no_cross_scale"
  "ablation_routed_only"
  "ablation_shared_only"
)

declare -A ABL_DESC=(
  ["ablation_fine_only"]="仅细尺度"
  ["ablation_no_router"]="无动态路由"
  ["ablation_fixed_scale_experts"]="固定尺度专家"
  ["ablation_no_cross_scale"]="无跨尺度共享"
  ["ablation_routed_only"]="仅路由分支"
  ["ablation_shared_only"]="仅共享分支"
)

declare -A DS_CONFIG=(
  ["TaxiBJ"]="configs/taxibj.json"
  ["BikeNYC"]="configs/bikenyc.json"
)
declare -A DS_TRAIN=(
  ["TaxiBJ"]="data/TaxiBJ/taxibj_train.npz"
  ["BikeNYC"]="data/BikeNYC/bikenyc_train.npz"
)
declare -A DS_VAL=(
  ["TaxiBJ"]="data/TaxiBJ/taxibj_val.npz"
  ["BikeNYC"]="data/BikeNYC/bikenyc_val.npz"
)

# ── 显示配置 ──────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  消融实验运行脚本"
echo "  数据集:       ${DATASETS[*]}"
echo "  GPU:          cuda:${GPU}"
echo "  Mask Pattern: ${MASK_PATTERN}"
echo "  Mask Rate:    ${MASK_RATE}"
echo "  Skip Full:    ${SKIP_FULL}"
echo "  开始时间:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo ""

# ── 检查 CUDA ──────────────────────────────────────────────────────────────
$PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

# ── 逐数据集运行 ──────────────────────────────────────────────────────────
GLOBAL_START=$(date +%s)

for ds in "${DATASETS[@]}"; do
  cfg="${DS_CONFIG[$ds]}"
  train_npz="${DS_TRAIN[$ds]}"
  val_npz="${DS_VAL[$ds]}"

  echo "##############################################"
  echo "##  $ds  (mask=${MASK_PATTERN}, rate=${MASK_RATE})"
  echo "##############################################"
  # Generate per-dataset offline masks and mask override with CSV paths.
  DS_MASK_OVERRIDE="$MASK_OVERRIDE_FILE"
  DS_MASK_OVERRIDE="/tmp/mask_override_${MASK_PATTERN}_${ds}_${TIMESTAMP}.json"
  MASK_DIR="data/${ds}/${MASK_PATTERN}_mask/${MASK_RATE}"
  TRAIN_MASK_CSV="${MASK_DIR}/train.csv"
  VAL_MASK_CSV="${MASK_DIR}/val.csv"

  echo "[info] generating ${MASK_PATTERN} masks for ${ds} rate=${MASK_RATE}"
  $PYTHON scripts/generate_fixed_masks.py \
    --train_npz "$train_npz" \
    --val_npz "$val_npz" \
    --pattern "$MASK_PATTERN" \
    --mask_rate "$MASK_RATE" \
    --seed "$FIXED_SEED" \
    --output_dir "$MASK_DIR"

  $PYTHON -c "
import json
override = {
    'data': {
        'mask': {
            'pattern': '${MASK_PATTERN}',
            'missing_rate': ${MASK_RATE},
            'train_csv': '${TRAIN_MASK_CSV}',
            'val_csv': '${VAL_MASK_CSV}'
        }
    }
}
with open('${DS_MASK_OVERRIDE}', 'w') as f:
    json.dump(override, f, indent=2)
print(f'[info] mask override for ${ds} → ${DS_MASK_OVERRIDE}')
"

  echo ""

  # ── Full Model ────────────────────────────────────────────────────────
  if [ "$SKIP_FULL" = false ]; then
    echo "[$(date '+%H:%M:%S')] $ds / Full Model"
    $PYTHON scripts/train.py \
      -c "$cfg" \
      --override_config "$DS_MASK_OVERRIDE" \
      --train_npz "$train_npz" \
      --val_npz "$val_npz" \
      -n "full" \
      --no_plot --quiet
    echo "  => Full Model 完成"
    echo ""
  fi

  # ── Ablations ──────────────────────────────────────────────────────────
  TOTAL_ABL=${#ABLATIONS[@]}
  echo "--- $ds 消融实验 (${TOTAL_ABL} 组) ---"
  for i in "${!ABLATIONS[@]}"; do
    abl="${ABLATIONS[$i]}"
    idx=$((i + 1))
    echo "[$(date '+%H:%M:%S')] [$idx/$TOTAL_ABL] $ds / $abl  —  ${ABL_DESC[$abl]}"

    # 合并 mask override + ablation override
    # 先生成合并后的 config（deep_update: base → mask_override → ablation_override）
    COMBINED_OVERRIDE="/tmp/combined_override_${abl}_${ds}_${TIMESTAMP}.json"
    $PYTHON -c "
import json
from stmoe_imputer.config import load_config, deep_update
mask_cfg = load_config('${DS_MASK_OVERRIDE}')
abl_cfg = load_config('configs/${abl}.json')
combined = deep_update(mask_cfg, abl_cfg)
with open('${COMBINED_OVERRIDE}', 'w') as f:
    json.dump(combined, f, indent=2)
"

    $PYTHON scripts/train.py \
      -c "$cfg" \
      --override_config "$COMBINED_OVERRIDE" \
      --train_npz "$train_npz" \
      --val_npz "$val_npz" \
      -n "${abl}" \
      --no_plot --quiet

    rm -f "$COMBINED_OVERRIDE"
    echo "  => $abl 完成"
    echo ""
  done

  rm -f "$DS_MASK_OVERRIDE"
done

ELAPSED=$(( $(date +%s) - GLOBAL_START ))
echo "=============================================="
echo "  全部完成！"
echo "  数据集:       ${DATASETS[*]}"
echo "  Mask Pattern: ${MASK_PATTERN}"
echo "  Mask Rate:    ${MASK_RATE}"
echo "  耗时:         ${ELAPSED}s ($((ELAPSED/60))m$((ELAPSED%60))s)"
echo "  结束时间:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo ""
echo "汇总分析: 跑完后告诉我，我帮你生成对比报告"
