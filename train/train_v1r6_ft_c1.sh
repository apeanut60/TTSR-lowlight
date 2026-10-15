#!/bin/bash
# v1r6-FT C1: gamma-aligned Query FT (alpha=0.75), from v1r3e4
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAVE_DIR="/root/data/experiments/retry/v1r6-ft-C1-gamma075"
PRETRAIN="/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt"
EVAL_H="/root/data/datasets/data1/Eval/Huawei/nanobanana_ref"

mkdir -p "$SAVE_DIR"
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ttsr 2>/dev/null || true
cd "$ROOT"

echo "============================================"
echo " v1r6-FT C1 gamma  align=ON alpha=0.75  seed=42"
echo " from=$PRETRAIN"
echo " save=$SAVE_DIR"
echo "============================================"

python main.py \
    --dataset data1_nanobanana \
    --dataset_dir /root/data/datasets/data1 \
    --data1_camera Huawei \
    --ref_dir "$EVAL_H" \
    --nanobanana_ref_subdir nanobanana_ref \
    --nanobanana_manifest_dir /root/data/datasets/data1/.nanobanana_sample_manifest_500 \
    --save_dir "$SAVE_DIR" \
    --reset True \
    --log_file_name train_v1r6_ft_C1.log \
    --seed 42 \
    --enhance_mode True \
    --num_res_blocks 10+10+6+4 \
    --n_feats 64 \
    --res_scale 1.0 \
    --batch_size 8 \
    --train_crop_size 192 \
    --num_init_epochs 0 \
    --num_epochs 5 \
    --print_every 5 \
    --save_every 1 \
    --val_every 1 \
    --decay 999999 \
    --gamma 0.5 \
    --lr_scheduler step \
    --lr_rate 3e-6 \
    --lr_rate_lte 3e-7 \
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
    --eval_per_image True \
    --eval_data1 False \
    --query_illum_align True \
    --query_illum_alpha 0.75 \
    --query_gamma_min 0.3 \
    --query_gamma_max 1.0 \
    --load_pretrain True \
    --pretrain_path "$PRETRAIN" \
    --freeze_lte True \
    --freeze_stages "" \
    --tile_size 256 \
    --tile_overlap 96 \
    --tile_window cosine \
    --num_gpu 1 \
    --num_workers 4

echo "C1 done. Log: $SAVE_DIR/train_v1r6_ft_C1.log"
