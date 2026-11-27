#!/bin/bash
# Fresh training for csfi_lfca_mgain_topk_v2.
# Does not reuse a csfi_lfca_v1 run name or checkpoint as a resume.
# Loss stays L1 + MS-SSIM at 1:1. This is not the short-finetune 9:1 recipe.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-/root/data/datasets/lol-v2-real}"
MODEL_VARIANT="csfi_lfca_mgain_topk_v2"
RUN_NAME="${RUN_NAME:-${MODEL_VARIANT}_lolv2_$(date +%Y%m%d_%H%M%S)}"
SAVE_ROOT="${SAVE_ROOT:-/root/data/experiments/TTSR-lowlight-block-amend}"
SAVE_DIR="${SAVE_DIR:-$SAVE_ROOT/$RUN_NAME}"
TRAIN_REF="${TRAIN_REF:-nanobanana_ref_v2}"
TEST_REF="${TEST_REF:-nanobanana_ref_v3}"
INIT_SCOPE="${INIT_SCOPE:-fresh}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
ALLOW_PARTIAL_COVERAGE="${ALLOW_PARTIAL_COVERAGE:-0}"
PYTHON="${PYTHON:-python}"

case "$ALLOW_PARTIAL_COVERAGE" in
  0)
    COVERAGE_ARGS=()
    COVERAGE_LABEL="strict coverage"
    ;;
  1)
    COVERAGE_ARGS=(--allow_partial_coverage)
    COVERAGE_LABEL="explicit partial coverage"
    ;;
  *)
    echo "ERROR: ALLOW_PARTIAL_COVERAGE must be 0 or 1, got '$ALLOW_PARTIAL_COVERAGE'" >&2
    exit 1
    ;;
esac

echo "============================================"
echo " block-amend $MODEL_VARIANT"
echo " mode=train  init_scope=$INIT_SCOPE"
echo " gain=multiscale_spatial_v1  channels=32,64  res=2,2,1  tokens=400"
echo " gain_grid_policy=scaled_budget_v1"
echo " match=topk_soft  k=4  temperature=0.1  chunk=128  recompute=true"
echo " loss=L1+MS-SSIM 1:1  Adam 1e-4  batch=4  crop=160  epochs=100"
echo " coverage=$COVERAGE_LABEL"
echo " data=$DATA"
echo " train_ref=$TRAIN_REF  test_ref=$TEST_REF"
echo " save_dir=$SAVE_DIR"
echo " code=$ROOT"
echo "============================================"

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ttsr 2>/dev/null || true
cd "$ROOT"
export TORCH_HOME="${TORCH_HOME:-/root/data/cache/torch}"
export PYTHONUNBUFFERED=1

INIT_ARGS=(--init_scope "$INIT_SCOPE")
if [[ "$INIT_SCOPE" == "lte_only" ]]; then
  if [[ -z "$INIT_CHECKPOINT" || ! -f "$INIT_CHECKPOINT" ]]; then
    echo "ERROR: INIT_SCOPE=lte_only requires existing INIT_CHECKPOINT" >&2
    exit 1
  fi
  INIT_ARGS+=(--init_checkpoint "$INIT_CHECKPOINT")
elif [[ -n "$INIT_CHECKPOINT" ]]; then
  echo "ERROR: fresh init must not set INIT_CHECKPOINT ($INIT_CHECKPOINT)" >&2
  exit 1
fi

"$PYTHON" block_amend_main.py \
    --mode train \
    --model_variant "$MODEL_VARIANT" \
    --gain_estimator multiscale_spatial_v1 \
    --gain_channels 32,64 \
    --gain_res_blocks 2,2,1 \
    --gain_attn_heads 4 \
    --gain_attn_depth 1 \
    --gain_attn_ffn_expansion 2 \
    --gain_max_tokens 400 \
    --gain_downsample binomial5_stride2 \
    --gain_readout_scale 4 \
    --gain_grid_policy scaled_budget_v1 \
    --matching_mode topk_soft \
    --match_topk 4 \
    --match_temperature 0.1 \
    --match_query_chunk_size 128 \
    --match_transfer_recompute true \
    --dataset_dir "$DATA" \
    --train_ref_subdir "$TRAIN_REF" \
    --test_ref_subdir "$TEST_REF" \
    --save_dir "$SAVE_DIR" \
    --batch_size 4 \
    --train_crop_size 160 \
    --lr_rate 1e-4 \
    --num_epochs 100 \
    --decay 50 \
    --gamma 0.5 \
    --val_ratio 0.1 \
    --split_seed 42 \
    --print_every 10 \
    --val_every 1 \
    --num_workers 4 \
    --seed 42 \
    --num_gpu 1 \
    "${COVERAGE_ARGS[@]}" \
    "${INIT_ARGS[@]}"
