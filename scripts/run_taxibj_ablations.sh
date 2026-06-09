#!/bin/bash
# ============================================================================
# TaxiBJ 消融实验 — 补充运行脚本
#
# 已完成（跳过）:
#   - ablation_fine_only      (120 ep, MAE 已完成)
#   - ablation_no_router      (117 ep, MAE 已完成)
#
# 本轮运行:
#   - ablation_fixed_scale_experts  (上一轮无训练数据, 重新跑)
#   - ablation_no_cross_scale
#   - ablation_routed_only
#   - ablation_shared_only
#
# 用法: cd v2 && bash scripts/run_taxibj_ablations.sh [GPU_ID]
# ============================================================================
set -euo pipefail

GPU="${1:-0}"
BASE_CONFIG="configs/taxibj.json"
TRAIN_NPZ="data/TaxiBJ/taxibj_train.npz"
VAL_NPZ="data/TaxiBJ/taxibj_val.npz"

ABLATIONS=(
  "ablation_fixed_scale_experts"
  "ablation_no_cross_scale"
  "ablation_routed_only"
  "ablation_shared_only"
)

declare -A DESC=(
  ["ablation_fixed_scale_experts"]="固定尺度专家 (每尺度独立专家池, 不共享)"
  ["ablation_no_cross_scale"]="无跨尺度共享专家"
  ["ablation_routed_only"]="仅路由分支 (无共享分支)"
  ["ablation_shared_only"]="仅共享分支 (无路由分支)"
)

# ── 切换到 v2 目录 ──────────────────────────────────────────────────────
cd "$(dirname "$0")/.."
echo "工作目录: $(pwd)"
echo "使用 GPU: cuda:${GPU}"
echo ""

# ── 前置检查 ────────────────────────────────────────────────────────────
if [ ! -f "$TRAIN_NPZ" ] || [ ! -f "$VAL_NPZ" ]; then
  echo "[错误] TaxiBJ 数据文件不存在"
  echo "  需要: $TRAIN_NPZ / $VAL_NPZ"
  exit 1
fi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'" 2>/dev/null || {
  echo "[错误] PyTorch 或 CUDA 不可用"
  exit 1
}

# ── 开始运行 ────────────────────────────────────────────────────────────
TOTAL=${#ABLATIONS[@]}
START_TIME=$(date +%s)

echo "=========================================="
echo "  TaxiBJ 消融实验 (${TOTAL} 组, 80 epochs/组)"
echo "=========================================="
echo ""

for i in "${!ABLATIONS[@]}"; do
  abl="${ABLATIONS[$i]}"
  idx=$((i + 1))
  desc="${DESC[$abl]:-}"

  echo "──────────────────────────────────────────"
  echo "[${idx}/${TOTAL}] ${abl}  -  ${desc}"
  echo "──────────────────────────────────────────"

  python scripts/train.py \
    -c "$BASE_CONFIG" \
    --override_config "configs/${abl}.json" \
    --train_npz "$TRAIN_NPZ" \
    --val_npz "$VAL_NPZ" \
    -n "${abl}" \
    --no_plot

  echo "[${idx}/${TOTAL}] ${abl} 完成 ($(date '+%H:%M:%S'))"
  echo ""
done

# ── 汇总 ────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MIN=$((ELAPSED / 60))
SEC=$((ELAPSED % 60))

echo "=========================================="
echo "  TaxiBJ ${TOTAL} 组消融实验全部完成！"
echo "  总耗时: ${MIN} 分 ${SEC} 秒"
echo "  输出目录: outputs/TaxiBJ/"
echo "=========================================="
ls -dt outputs/TaxiBJ/ablation_* 2>/dev/null | head -4 | while read d; do
  echo "  $(basename "$d")"
done
