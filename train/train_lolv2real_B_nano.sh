#!/bin/bash
# LOLv2-Real Stage B: Nano FT from Stage A best.
# Usage: bash train/train_lolv2real_B_nano.sh [pretrain_pt]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAVE_DIR="/root/data/experiments/TTSR-lowlight-lolv2real/B_nano_from_huawei_lr2e6"
PRETRAIN="${1:-}"
DATA="/root/data/datasets/lol-v2-real"
NANO_EVAL="/root/data/datasets/lol-v2-real/Test/nanobanana_ref_v3"
LR="${LR:-2e-6}"

if [[ -z "$PRETRAIN" || ! -f "$PRETRAIN" ]]; then
  echo "Usage: $0 <init.pt>" >&2
  exit 1
fi

mkdir -p "$SAVE_DIR"
echo "============================================"
echo " LOLv2-Real Stage B  Nano FT (cross-domain gentle)"
echo " init=$PRETRAIN"
echo " ref=nanobanana_ref_v2  lr=$LR  freeze_lte"
echo " save_dir=$SAVE_DIR"
echo "============================================"

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ttsr 2>/dev/null || true
cd "$ROOT"
export TORCH_HOME=/root/data/cache/torch
export PYTHONUNBUFFERED=1

python main.py \
    --dataset lolv2_nanobanana \
    --dataset_dir "$DATA" \
    --lolv2_nanobanana_subset real \
    --lolv2_nanobanana_ref_subdir nanobanana_ref_v2 \
    --save_dir "$SAVE_DIR" \
    --reset True \
    --log_file_name "train_lolv2real_B_nano.log" \
    --enhance_mode True \
    --num_res_blocks 10+10+6+4 \
    --n_feats 64 \
    --res_scale 1.0 \
    --batch_size 8 \
    --train_crop_size 192 \
    --num_init_epochs 0 \
    --num_epochs 15 \
    --print_every 5 \
    --save_every 1 \
    --val_every 1 \
    --decay 8 \
    --gamma 0.5 \
    --lr_rate "$LR" \
    --lr_rate_lte 1e-7 \
    --rec_w 1.0 \
    --rec_loss_type l1 \
    --ms_ssim_w 1.0 \
    --per_w 0.0 \
    --tpl_w 0.0 \
    --adv_w 0.0 \
    --illum_smooth_w 0.0 \
    --color_w 0.0 \
    --exposure_w 0.0 \
    --ref_degrade False \
    --eval_mean_align False \
    --eval_lolv2_nanobanana True \
    --lolv2_nanobanana_eval_real_ref_dir "$NANO_EVAL" \
    --lolv2_nanobanana_eval_syn_ref_dir "" \
    --mixed_lolv2_real_dir "$DATA" \
    --data1_nanobanana_eval_huawei_ref_dir "" \
    --data1_nanobanana_eval_nikon_ref_dir "" \
    --ref_dir "$NANO_EVAL" \
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
