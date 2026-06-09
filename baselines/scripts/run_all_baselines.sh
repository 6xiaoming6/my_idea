#!/bin/bash
# =============================================================================
#  Baselines 完整运行脚本 — TaxiBJ + BikeNYC
#
#  用法:
#    cd baselines
#    bash scripts/run_all_baselines.sh [GPU_ID]
#
#  包含:
#    统计类 (3): mean, historical, linear
#    深度模型 (6): fine_only, conv3d_unet, transformer, ms_concat,
#                  fixed_experts, no_router
#    (可选) SAITS: 如已安装 pypots 则自动运行
#
#  输出: runs/taxibj/ 和 runs/bikenyc/ 下的 result.json + best.pt
# =============================================================================
set -euo pipefail

# ── 配置 ──────────────────────────────────────────────────────────────────
GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

STAT_METHODS=("mean" "historical" "linear")
DEEP_MODELS=("fine_only" "conv3d_unet" "transformer" "ms_concat" "fixed_experts" "no_router")

TAXIBJ_CONFIG="configs/taxibj.json"
BIKENYC_CONFIG="configs/bikenyc.json"
TAXIBJ_NPY="data/taxibj_flow.npy"
TAXIBJ_GRID_DIR="../v2/data/TaxiBJ"

# ── 工具函数 ──────────────────────────────────────────────────────────────
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
elapsed() {
  local s=$(( $(date +%s) - $1 ))
  printf "%dm%ds" $((s/60)) $((s%60))
}
lowercase() { echo "$1" | tr '[:upper:]' '[:lower:]'; }

# ── 前置检查 & 目录准备 ───────────────────────────────────────────────────
echo "=============================================="
echo "  Baselines 完整运行 — TaxiBJ + BikeNYC"
echo "  开始时间: $(timestamp)"
echo "=============================================="
echo ""

python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)')" || {
  echo "[FATAL] PyTorch 不可用"; exit 1
}
echo ""

# 提前创建 runs 目录，避免 tee 写文件失败
mkdir -p runs/taxibj runs/bikenyc

# ── TaxiBJ 数据准备 ───────────────────────────────────────────────────────
if [ ! -f "$TAXIBJ_NPY" ]; then
  echo "[数据准备] 转换 TaxiBJ .grid → .npy ..."
  if [ ! -f "$TAXIBJ_GRID_DIR/TAXIBJ2013.grid" ]; then
    echo "[FATAL] TaxiBJ .grid 文件不存在: $TAXIBJ_GRID_DIR"
    exit 1
  fi
  python -m st_impute.data.parse_libcity_taxibj \
    --data_dir "$TAXIBJ_GRID_DIR" \
    --save_path "$TAXIBJ_NPY"
  echo "[数据准备] TaxiBJ .npy 已生成: $TAXIBJ_NPY"
else
  echo "[数据准备] TaxiBJ .npy 已存在，跳过转换"
fi

# ── BikeNYC 数据准备 (pad 到尺寸可被 4 整除) ──────────────────────────────
BIKENYC_NPY="data/bikenyc_flow.npy"
BIKENYC_RAW="../v2/data/BikeNYC/flow_data.npy"
if [ ! -f "$BIKENYC_NPY" ]; then
  echo "[数据准备] BikeNYC .npy 尺寸对齐 (H 必须整除 4) ..."
  python -c "
import numpy as np
d = np.load('$BIKENYC_RAW')
print(f'  Original: {d.shape}')
# Pad H 到下一个 4 的倍数
H = d.shape[2]
pad_h = ((H + 3) // 4) * 4 - H
d_pad = np.pad(d, ((0,0),(0,0),(0,pad_h),(0,0)), mode='constant', constant_values=0)
print(f'  Padded:   {d_pad.shape}')
np.save('$BIKENYC_NPY', d_pad)
"
  echo "[数据准备] BikeNYC .npy 已生成: $BIKENYC_NPY"
else
  echo "[数据准备] BikeNYC .npy 已存在，跳过"
fi
echo ""

# ── 统计 Baseline ─────────────────────────────────────────────────────────
echo "=============================================="
echo "  阶段 1/3: 统计 Baseline (3 methods x 2 datasets)"
echo "=============================================="
echo ""

for ds_name in "TaxiBJ" "BikeNYC"; do
  if [ "$ds_name" = "TaxiBJ" ]; then cfg="$TAXIBJ_CONFIG"; else cfg="$BIKENYC_CONFIG"; fi
  ds_lower=$(lowercase "$ds_name")
  echo "--- $ds_name ---"
  for method in "${STAT_METHODS[@]}"; do
    echo "[$(timestamp)] $ds_name / $method"
    # 保存完整 JSON 到独立文件，同时输出到终端
    python eval_statistical.py --config "$cfg" --method "$method" 2>&1 | tee "runs/$ds_lower/${method}_result.txt"
    # 提取 test 指标行追加到汇总文件
    grep '"test"' "runs/$ds_lower/${method}_result.txt" | head -1 >> "runs/$ds_lower/_stat_results.jsonl" || true
    echo ""
  done
done

# ── 深度 Baseline ─────────────────────────────────────────────────────────
echo "=============================================="
echo "  阶段 2/3: 深度 Baseline (6 models x 2 datasets)"
echo "=============================================="
echo ""

TOTAL_DEEP=$((${#DEEP_MODELS[@]} * 2))
DEEP_IDX=0
START_DEEP=$(date +%s)

for ds_name in "TaxiBJ" "BikeNYC"; do
  if [ "$ds_name" = "TaxiBJ" ]; then cfg="$TAXIBJ_CONFIG"; else cfg="$BIKENYC_CONFIG"; fi
  ds_lower=$(lowercase "$ds_name")
  echo "--- $ds_name ---"
  for model in "${DEEP_MODELS[@]}"; do
    DEEP_IDX=$((DEEP_IDX + 1))
    echo "[$(timestamp)] [$DEEP_IDX/$TOTAL_DEEP] $ds_name / $model"
    python train_deep.py --config "$cfg" --model "$model"
    # 打印本次结果
    res_file="runs/$ds_lower/$model/result.json"
    if [ -f "$res_file" ]; then
      python -c "
import json
d = json.load(open('$res_file'))
t = d['test']
print(f'  => MAE={t[\"mae\"]:.4f}  RMSE={t[\"rmse\"]:.4f}  MAPE={t[\"mape\"]:.4f}')
" 2>/dev/null || true
    fi
    echo ""
  done
done

echo "深度 Baseline 总耗时: $(elapsed $START_DEEP)"
echo ""

# ── SAITS (可选) ──────────────────────────────────────────────────────────
echo "=============================================="
echo "  阶段 3/3: SAITS (可选, 需 pypots)"
echo "=============================================="
echo ""

HAS_PYPOTS=$(python -c "import pypots; print('yes')" 2>/dev/null || echo "no")
if [ "$HAS_PYPOTS" = "yes" ]; then
  for ds_name in "TaxiBJ" "BikeNYC"; do
    if [ "$ds_name" = "TaxiBJ" ]; then cfg="$TAXIBJ_CONFIG"; else cfg="$BIKENYC_CONFIG"; fi
    echo "[$(timestamp)] $ds_name / SAITS"
    python train_saits_pypots.py --config "$cfg" 2>&1 | tail -1
    echo ""
  done
else
  echo "[跳过] pypots 未安装, 如需 SAITS 请: pip install pypots"
  echo ""
fi

# ── 汇总 ──────────────────────────────────────────────────────────────────
echo "=============================================="
echo "  全部完成！结果汇总"
echo "  结束时间: $(timestamp)"
echo "=============================================="
echo ""

for ds_name in "TaxiBJ" "BikeNYC"; do
  ds_lower=$(lowercase "$ds_name")
  rundir="runs/$ds_lower"
  echo "=== $ds_name ==="
  printf "  %-22s %10s %10s %10s\n" "Method" "MAE" "RMSE" "MAPE"
  echo "  ------------------------------------------------"

  # 统计类 (从各 method_result.txt 提取)
  for method in "${STAT_METHODS[@]}"; do
    txt="$rundir/${method}_result.txt"
    if [ -f "$txt" ]; then
      python -c "
import json, re
text = open('$txt').read()
m = re.search(r'\{.*\}', text, re.DOTALL)
if m:
    d = json.loads(m.group())
    t = d['test']
    print(f'  {\"$method\":<22} {t[\"mae\"]:>10.4f} {t[\"rmse\"]:>10.4f} {t[\"mape\"]:>10.2f}')
" 2>/dev/null || true
    fi
  done

  # 深度模型
  for model in "${DEEP_MODELS[@]}"; do
    res_file="$rundir/$model/result.json"
    if [ -f "$res_file" ]; then
      python -c "
import json
d = json.load(open('$res_file'))
t = d['test']
print(f'  {\"$model\":<22} {t[\"mae\"]:>10.4f} {t[\"rmse\"]:>10.4f} {t[\"mape\"]:>10.2f}')
" 2>/dev/null || true
    fi
  done
  echo ""
done

echo "详细结果: runs/taxibj/  &  runs/bikenyc/"
echo "完成！"
