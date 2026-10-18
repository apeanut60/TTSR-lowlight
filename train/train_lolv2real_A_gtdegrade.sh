#!/bin/bash
# LOLv2-Real Stage A: GT-degraded FT from data1 Huawei v1r3.
# Cross-domain: freeze LTE, L1+MS-SSIM only (no exposure — v1 hurt nano PSNR).
# Select ckpt by lolv2real_nanobanana Test100.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAVE_DIR="/root/data/experiments/TTSR-lowlight-lolv2real/A_gtdegrade_v2_noexp"
PRETRAIN="/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt"
DATA="/root/data/datasets/lol-v2-real"
NANO_EVAL="/root/data/datasets/lol-v2-real/Test/nanobanana_ref_v3"

if [[ ! -f "$PRETRAIN" ]]; then
  echo "missing pretrain: $PRETRAIN" >&2
  exit 1
fi

mkdir -p "$SAVE_DIR"
echo "============================================"
echo " LOLv2-Real Stage A v2  GT-degrade"
echo " init=$PRETRAIN"
echo " lr=1e-5  freeze_lte  exposure_w=0  (v1 exposure hurt nano)"
echo " save_dir=$SAVE_DIR"
echo " code=$ROOT @ $(git -C "$ROOT" rev-parse --short HEAD)"
echo "============================================"

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ttsr 2>/dev/null || true
cd "$ROOT"
export TORCH_HOME=/root/data/cache/torch
export PYTHONUNBUFFERED=1

python main.py \
    --dataset lolv2real \
    --dataset_dir "$DATA" \
    --save_dir "$SAVE_DIR" \
    --reset True \
    --log_file_name "train_lolv2real_A_gtdegrade_v2.log" \
    --enhance_mode True \
    --num_res_blocks 10+10+6+4 \
    --n_feats 64 \
    --res_scale 1.0 \
    --batch_size 8 \
    --train_crop_size 160 \
    --num_init_epochs 0 \
    --num_epochs 15 \
    --print_every 10 \
    --save_every 1 \
    --val_every 1 \
    --decay 8 \
    --gamma 0.5 \
    --lr_rate 1e-5 \
    --lr_rate_lte 1e-6 \
    --rec_w 1.0 \
    --rec_loss_type l1 \
    --ms_ssim_w 1.0 \
    --per_w 0.0 \
    --tpl_w 0.0 \
    --adv_w 0.0 \
    --illum_smooth_w 0.0 \
    --color_w 0.0 \
    --exposure_w 0.0 \
    --ref_degrade True \
    --ref_color_jitter 0.2 \
    --ref_shift_range 4 \
    --ref_blur_sigma 2.0 \
    --eval_ref_degrade True \
    --eval_mean_align False \
    --eval_lolv2_nanobanana True \
    --lolv2_nanobanana_eval_real_ref_dir "$NANO_EVAL" \
    --mixed_lolv2_real_dir "$DATA" \
    --load_pretrain True \
    --pretrain_path "$PRETRAIN" \
    --freeze_lte True \
    --freeze_stages "" \
    --query_illum_align False \
    --texture_fusion False \
    --tile_size 256 \
    --tile_overlap 96 \
    --tile_window cosine \
    --num_gpu 1 \
    --num_workers 4 \
    --seed 42
