#!/bin/bash
# =============================================================================
#  V2 完整运行脚本 — Full Model + 6 组消融实验
#
#  支持双 GPU 并行:
#    # 终端 1
#    bash scripts/run_all_v2.sh --dataset TaxiBJ --gpu 0
#
#    # 终端 2
#    bash scripts/run_all_v2.sh --dataset BikeNYC --gpu 1
#
#  也支持单 GPU 全部跑:
#    bash scripts/run_all_v2.sh --dataset all --gpu 0
# =============================================================================
set -euo pipefail

# ── 参数解析 ──────────────────────────────────────────────────────────────
DATASET=""
GPU="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset|-d) DATASET="$2"; shift 2 ;;
    --gpu|-g)     GPU="$2"; shift 2 ;;
    *) echo "Unknown: $1"; echo "Usage: $0 --dataset {TaxiBJ|BikeNYC|all} --gpu <id>"; exit 1 ;;
  esac
done

if [ -z "$DATASET" ]; then
  echo "Usage: $0 --dataset {TaxiBJ|BikeNYC|all} --gpu <id>"
  echo ""
  echo "Examples:"
  echo "  bash $0 --dataset TaxiBJ --gpu 0     # 终端1"
  echo "  bash $0 --dataset BikeNYC --gpu 1    # 终端2"
  echo "  bash $0 --dataset all --gpu 0        # 单GPU全跑"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"

cd "$(dirname "$0")/.."

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

declare -A DESC=(
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

# ── 检查 ──────────────────────────────────────────────────────────────────
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

echo "=============================================="
echo "  V2 Full + Ablation"
echo "  数据集: ${DATASETS[*]}"
echo "  GPU:    cuda:${GPU}"
echo "  开始:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo ""

# ── 逐数据集运行 ──────────────────────────────────────────────────────────
GLOBAL_START=$(date +%s)

for ds in "${DATASETS[@]}"; do
  cfg="${DS_CONFIG[$ds]}"
  train_npz="${DS_TRAIN[$ds]}"
  val_npz="${DS_VAL[$ds]}"

  echo "##############################################"
  echo "##  $ds"
  echo "##############################################"
  echo ""

  # ── Full Model ──────────────────────────────────────────────────────
  echo "[$(date '+%H:%M:%S')] $ds / Full Model"
  python scripts/train.py \
    -c "$cfg" \
    --train_npz "$train_npz" \
    --val_npz "$val_npz" \
    -n full \
    --no_plot
  echo "  => Full Model 完成"
  echo ""

  # ── Ablations ────────────────────────────────────────────────────────
  TOTAL_ABL=${#ABLATIONS[@]}
  echo "--- $ds 消融实验 (${TOTAL_ABL} 组) ---"
  for i in "${!ABLATIONS[@]}"; do
    abl="${ABLATIONS[$i]}"
    idx=$((i + 1))
    echo "[$(date '+%H:%M:%S')] [$idx/$TOTAL_ABL] $ds / $abl  —  ${DESC[$abl]}"
    python scripts/train.py \
      -c "$cfg" \
      --override_config "configs/${abl}.json" \
      --train_npz "$train_npz" \
      --val_npz "$val_npz" \
      -n "${abl}" \
      --no_plot
    echo "  => $abl 完成"
    echo ""
  done
done

ELAPSED=$(( $(date +%s) - GLOBAL_START ))
echo "=============================================="
echo "  全部完成！"
echo "  数据集: ${DATASETS[*]}"
echo "  耗时:   $((ELAPSED/60))m$((ELAPSED%60))s"
echo "  结束:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo ""
echo "汇总分析: 跑完后告诉我，我帮你生成对比报告"
