#!/bin/bash
# P0: Huawei v1r3 zero-shot on LOLv2-Real Test100 (GT-degrade + nano_v3).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt"
DATA="/root/data/datasets/lol-v2-real"
OUT_ROOT="/root/data/experiments/TTSR-lowlight-lolv2real/p0_zeroshot"
NANO_REF="$DATA/Test/nanobanana_ref_v3"

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ttsr 2>/dev/null || true
cd "$ROOT"
export TORCH_HOME=/root/data/cache/torch
export PYTHONUNBUFFERED=1

run_eval() {
  local name="$1"
  local ref_degrade="$2"
  local ref_dir="$3"
  local save="$OUT_ROOT/$name"
  mkdir -p "$save/save_results"
  # trainer writes to save_dir/save_results — symlink for cleanliness
  rm -rf "$save/save_results"
  mkdir -p "$save/save_results"

  echo "=== P0 eval $name  ref_degrade=$ref_degrade ref_dir=${ref_dir:-GT} ==="
  python main.py \
    --dataset lolv2real \
    --dataset_dir "$DATA" \
    --ref_dir "$ref_dir" \
    --save_dir "$save" \
    --reset False \
    --log_file_name "eval_${name}.log" \
    --eval True \
    --eval_save_results True \
    --eval_mean_align False \
    --eval_per_image True \
    --enhance_mode True \
    --num_res_blocks 10+10+6+4 \
    --n_feats 64 \
    --res_scale 1.0 \
    --ref_degrade "$ref_degrade" \
    --ref_color_jitter 0.2 \
    --ref_shift_range 4 \
    --ref_blur_sigma 2.0 \
    --eval_degrade_seed 1234 \
    --model_path "$CKPT" \
    --query_illum_align False \
    --texture_fusion False \
    --tile_size 256 \
    --tile_overlap 96 \
    --tile_window cosine \
    --num_gpu 1 \
    --num_workers 4

  # mean-Y of preds vs GT
  python - <<PY
import os, json, numpy as np
from imageio import imread
pred_dir = "$save/save_results"
gt_dir = "$DATA/Test/Normal"
ys_p, ys_g = [], []
files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
# trainer may save as 00000.png indexed — also try basename match
print('n_pred_files', len(files))
# Prefer per-image json if present
pj = None
for root, _, fs in os.walk("$save"):
    for f in fs:
        if f.endswith('.json') and 'per' in f.lower():
            pj = os.path.join(root, f)
if files:
    # map by order: TestSet sorted low*.png → normal*.png
    lows = sorted([f for f in os.listdir("$DATA/Test/Low") if f.endswith('.png')])
    for i, f in enumerate(files):
        pred = imread(os.path.join(pred_dir, f)).astype(np.float64)
        if pred.ndim == 3 and pred.shape[2] > 3:
            pred = pred[:,:,:3]
        if i < len(lows):
            gname = lows[i].replace('low', 'normal')
            gt = imread(os.path.join(gt_dir, gname)).astype(np.float64)
            if gt.ndim == 3 and gt.shape[2] > 3:
                gt = gt[:,:,:3]
            # resize if needed
            if pred.shape[:2] != gt.shape[:2]:
                import cv2
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_LINEAR)
            yp = 0.299*pred[:,:,0]+0.587*pred[:,:,1]+0.114*pred[:,:,2]
            yg = 0.299*gt[:,:,0]+0.587*gt[:,:,1]+0.114*gt[:,:,2]
            ys_p.append(yp.mean()); ys_g.append(yg.mean())
    print('meanY_pred=%.2f meanY_gt=%.2f ratio=%.3f n=%d' % (
        np.mean(ys_p), np.mean(ys_g), np.mean(ys_p)/max(np.mean(ys_g),1e-6), len(ys_p)))
    open("$save/mean_y.txt","w").write(
        "meanY_pred=%.4f\nmeanY_gt=%.4f\nratio=%.4f\nn=%d\n" % (
            float(np.mean(ys_p)), float(np.mean(ys_g)),
            float(np.mean(ys_p)/max(np.mean(ys_g),1e-6)), len(ys_p)))
PY
  grep -E 'PSNR:|SSIM:' "$save/eval_${name}.log" | tail -5 || true
}

run_eval "gt_degrade" True ""
run_eval "nano_v3" False "$NANO_REF"

echo "P0 done → $OUT_ROOT"
