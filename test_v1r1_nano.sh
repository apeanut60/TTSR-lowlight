#!/bin/bash
# Per-camera RGB eval for v1r1 nano adapt (nanobanana ref, PSNR/SSIM=RGB).
# Usage: bash test_v1r1_nano.sh Huawei|Nikon [epoch]

set -euo pipefail
CAM="${1:-}"
EPOCH="${2:-}"
if [[ "$CAM" != "Huawei" && "$CAM" != "Nikon" ]]; then
  echo "Usage: $0 Huawei|Nikon [epoch]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")" && pwd)"
# Prefer newest recipe dir that has checkpoints.
SAVE_DIR=""
for cand in \
  "/root/data/experiments/retry/TTSR-lowlight-v1r5-nano-${CAM}" \
  "/root/data/experiments/retry/TTSR-lowlight-v1r4-nano-${CAM}" \
  "/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-${CAM}" \
  "/root/data/experiments/retry/TTSR-lowlight-v1r2-nano-${CAM}" \
  "/root/data/experiments/retry/TTSR-lowlight-v1r1-nano-${CAM}"
do
  if [[ -d "$cand/model" ]] && ls "$cand"/model/model_*.pt >/dev/null 2>&1; then
    SAVE_DIR="$cand"
    break
  fi
done
if [[ -z "$SAVE_DIR" ]]; then
  echo "no v1r* checkpoints for $CAM" >&2
  exit 1
fi

if [[ -z "$EPOCH" ]]; then
  # pick latest model_*.pt
  LATEST="$(ls -1 "$SAVE_DIR"/model/model_*.pt 2>/dev/null | sort | tail -1 || true)"
  if [[ -z "$LATEST" ]]; then
    echo "no checkpoints in $SAVE_DIR/model" >&2
    exit 1
  fi
  EPOCH="$(basename "$LATEST" | sed -E 's/model_0*([0-9]+)\.pt/\1/')"
fi
EPO_TAG="$(printf '%05d' "$EPOCH")"
MODEL_PATH="$SAVE_DIR/model/model_${EPO_TAG}.pt"

if [[ "$CAM" == "Huawei" ]]; then
  REF_DIR="/root/data/datasets/data1/Eval/Huawei/nanobanana_ref"
else
  REF_DIR="/root/data/datasets/data1/Eval/Nikon/nanobanana_ref_Nikon"
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "missing checkpoint: $MODEL_PATH" >&2
  exit 1
fi

OUT_SUB="save_results_nanobanana_${CAM}_e${EPO_TAG}"
LOG="test_${CAM}_nanobanana_rgb_e${EPO_TAG}.log"
mkdir -p "$SAVE_DIR/$OUT_SUB"

echo "============================================"
echo " v1r1 nano eval  camera=$CAM  epoch=$EPOCH"
echo " Model: $MODEL_PATH"
echo " Ref  : $REF_DIR"
echo " Metric: PSNR-RGB / SSIM-RGB"
echo "============================================"

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ttsr 2>/dev/null || true
cd "$ROOT"

RESULTS="$SAVE_DIR/save_results"
if [[ -e "$RESULTS" && ! -L "$RESULTS" ]]; then
  mv "$RESULTS" "$SAVE_DIR/save_results.bak_$(date +%s)" 2>/dev/null || true
fi
rm -f "$RESULTS"
ln -sfn "$SAVE_DIR/$OUT_SUB" "$RESULTS"

python main.py \
    --dataset data1 \
    --dataset_dir /root/data/datasets/data1 \
    --data1_camera "$CAM" \
    --ref_dir "$REF_DIR" \
    --save_dir "$SAVE_DIR" \
    --reset False \
    --log_file_name "$LOG" \
    --eval True \
    --eval_save_results True \
    --eval_mean_align False \
    --enhance_mode True \
    --num_res_blocks 10+10+6+4 \
    --n_feats 64 \
    --res_scale 1.0 \
    --ref_degrade False \
    --model_path "$MODEL_PATH" \
    --num_workers 4 \
    --num_gpu 1

echo "Log: $SAVE_DIR/$LOG"
grep -h 'PSNR:' "$SAVE_DIR/$LOG" | tail -3 || true
