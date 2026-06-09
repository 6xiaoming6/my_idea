#!/bin/bash
# ============================================================================
# BikeNYC 消融实验一键运行脚本
# 用法: cd v2 && bash scripts/run_bikenyc_ablations.sh
# ============================================================================
set -euo pipefail

# ── 配置 ──────────────────────────────────────────────────────────────────
GPU="${1:-0}"                           # 默认使用 GPU 0，可传参覆盖: bash ...sh 1
BASE_CONFIG="configs/bikenyc.json"
TRAIN_NPZ="data/BikeNYC/bikenyc_train.npz"
VAL_NPZ="data/BikeNYC/bikenyc_val.npz"

# 6 组消融实验
ABLATIONS=(
  "ablation_fine_only"
  "ablation_no_router"
  "ablation_fixed_scale_experts"
  "ablation_no_cross_scale"
  "ablation_routed_only"
  "ablation_shared_only"
)

# 消融说明
declare -A DESC=(
  ["ablation_fine_only"]="仅细尺度 (无多尺度, 无跨尺度共享)"
  ["ablation_no_router"]="无动态路由 (均匀门控)"
  ["ablation_fixed_scale_experts"]="固定尺度专家 (每尺度独立, 不共享)"
  ["ablation_no_cross_scale"]="无跨尺度共享专家"
  ["ablation_routed_only"]="仅路由分支 (无共享分支)"
  ["ablation_shared_only"]="仅共享分支 (无路由分支)"
)

# ── 切换到 v2 目录 ───────────────────────────────────────────────────────
cd "$(dirname "$0")/.."
echo "工作目录: $(pwd)"
echo "使用 GPU: cuda:${GPU}"
echo ""

# ── 检查前置条件 ─────────────────────────────────────────────────────────
if [ ! -f "$TRAIN_NPZ" ] || [ ! -f "$VAL_NPZ" ]; then
  echo "[错误] BikeNYC 数据文件不存在: $TRAIN_NPZ / $VAL_NPZ"
  exit 1
fi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'" 2>/dev/null || {
  echo "[错误] PyTorch 或 CUDA 不可用，请先安装"
  exit 1
}

# ── 开始运行 ─────────────────────────────────────────────────────────────
TOTAL=${#ABLATIONS[@]}
START_TIME=$(date +%s)

echo "=========================================="
echo "  BikeNYC 消融实验 (共 ${TOTAL} 组)"
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

# ── 汇总 ─────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MIN=$((ELAPSED / 60))
SEC=$((ELAPSED % 60))

echo "=========================================="
echo "  全部 ${TOTAL} 组消融实验完成！"
echo "  总耗时: ${MIN} 分 ${SEC} 秒"
echo "  输出目录: outputs/BikeNYC/"
echo "=========================================="
ls -d outputs/BikeNYC/ablation_* 2>/dev/null | while read d; do
  echo "  $(basename "$d")"
done
