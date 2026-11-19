#!/bin/bash
# block-amend CSFI-U-Net on LOLv2-Real (generated Nano refs).
# Model identity: csfi_lfca_v1 (LFCA restoration blocks at coarse H/4 and
# middle H/2; depth 1,1; heads 1). Recipe/data protocol unchanged from the
# IGAB baseline. IGAB-variant checkpoints cannot be resumed/evaluated here.
# Default: fresh init (code MainNet + ImageNet LTE, frozen). No silent P2 load.
# No --reset. Non-empty save_dir is rejected by block_amend_main.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-/root/data/datasets/lol-v2-real}"
MODEL_VARIANT="csfi_lfca_v1"
RUN_NAME="${RUN_NAME:-${MODEL_VARIANT}_lolv2_$(date +%Y%m%d_%H%M%S)}"
SAVE_ROOT="${SAVE_ROOT:-/root/data/experiments/TTSR-lowlight-block-amend}"
SAVE_DIR="${SAVE_DIR:-$SAVE_ROOT/$RUN_NAME}"
TRAIN_REF="${TRAIN_REF:-nanobanana_ref_v2}"
TEST_REF="${TEST_REF:-nanobanana_ref_v3}"
# Optional LTE-only warm-start (explicit). Leave empty for pure fresh.
INIT_SCOPE="${INIT_SCOPE:-fresh}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"

echo "============================================"
echo " block-amend CSFI-U-Net ($MODEL_VARIANT)"
echo " mode=train  init_scope=$INIT_SCOPE"
echo " backbone=csfi_unet  num_res_blocks=8+8+4+2"
echo " model_variant=$MODEL_VARIANT  restoration_block=lfca  blocks=1,1  heads=1"
echo " data=$DATA  train_ref=$TRAIN_REF"
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

python block_amend_main.py \
    --mode train \
    --model_variant "$MODEL_VARIANT" \
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
    "${INIT_ARGS[@]}"
