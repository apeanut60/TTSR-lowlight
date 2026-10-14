#!/bin/bash
# data1 (LSRW) test-set eval for TTSR-lowlight-v14 with fixed protocol:
#   1) nanobanana ref (clean; no HR-as-ref, no ref_degrade)
#   2) PSNR = RGB three-channel
#   3) SSIM = RGB three-channel (mean of R/G/B SSIM)
#
# Usage: bash test_v14_data1.sh [epoch]
# Example: bash test_v14_data1.sh 35

set -euo pipefail
EPOCH="${1:-40}"
VER=v14
ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="/root/data/experiments/TTSR-lowlight-${VER}/model/model_$(printf '%05d' "$EPOCH").pt"
SAVE_DIR="/root/data/experiments/TTSR-lowlight-${VER}"
EPO_TAG="$(printf '%05d' "$EPOCH")"

HUAWEI_REF="/root/data/datasets/data1/Eval/Huawei/nanobanana_ref"
NIKON_REF="/root/data/datasets/data1/Eval/Nikon/nanobanana_ref_Nikon"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "missing checkpoint: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -d "$HUAWEI_REF" || ! -d "$NIKON_REF" ]]; then
  echo "missing nanobanana ref dirs" >&2
  exit 1
fi

echo "============================================"
echo " TTSR-lowlight v14 data1 Eval (protocol fix)"
echo " Epoch : $EPOCH"
echo " Model : $MODEL_PATH"
echo " Code  : $ROOT @ $(git -C "$ROOT" rev-parse --short HEAD)"
echo " Ref   : nanobanana (Huawei + Nikon), clean"
echo " Metric: PSNR-RGB / SSIM-RGB"
echo "============================================"

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ttsr 2>/dev/null || true
cd "$ROOT"

run_cam() {
  local CAM="$1"
  local REF_DIR="$2"
  local LOG="test_data1_${CAM}_nanobanana_rgb_e${EPO_TAG}.log"
  local OUT_SUB="save_results_nanobanana_${CAM}_e${EPO_TAG}"

  mkdir -p "$SAVE_DIR/$OUT_SUB"

  echo ""
  echo "---- camera=$CAM  ref=$REF_DIR ----"
  # trainer always writes to save_dir/save_results — isolate per camera.
  local RESULTS="$SAVE_DIR/save_results"
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
  echo "Out: $SAVE_DIR/$OUT_SUB/"
}

run_cam Huawei "$HUAWEI_REF"
run_cam Nikon "$NIKON_REF"

echo ""
echo "Done. Primary lines look like:"
echo "  data1  PSNR: <RGB>  PSNRy: <ignored>  SSIM: <RGB> ..."
grep -h 'PSNR:' "$SAVE_DIR"/test_data1_{Huawei,Nikon}_nanobanana_rgb_e${EPO_TAG}.log || true
