#!/bin/bash
# LOLv2-Real brightness adapt: freeze backbone, train only global_illum_head
# from data1 Huawei v1r3. Addresses meanY_ratio~0.62 without wrecking texture.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAVE_DIR="/root/data/experiments/TTSR-lowlight-lolv2real/A_illum_only_from_huawei"
PRETRAIN="/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt"
DATA="/root/data/datasets/lol-v2-real"
NANO_EVAL="/root/data/datasets/lol-v2-real/Test/nanobanana_ref_v3"

mkdir -p "$SAVE_DIR"
echo "============================================"
echo " LOLv2-Real A  illum-head only  from Huawei"
echo " freeze=sfe,stage1-3,merge + LTE; lr_illum=1e-4"
echo "============================================"

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ttsr 2>/dev/null || true
cd "$ROOT"
export TORCH_HOME=/root/data/cache/torch PYTHONUNBUFFERED=1

python main.py \
    --dataset lolv2real \
    --dataset_dir "$DATA" \
    --save_dir "$SAVE_DIR" \
    --reset True \
    --log_file_name "train_lolv2real_A_illum_only.log" \
    --enhance_mode True \
    --num_res_blocks 10+10+6+4 \
    --n_feats 64 \
    --res_scale 1.0 \
    --batch_size 8 \
    --train_crop_size 160 \
    --num_init_epochs 0 \
    --num_epochs 10 \
    --print_every 10 \
    --save_every 1 \
    --val_every 1 \
    --decay 5 \
    --gamma 0.5 \
    --lr_rate 1e-6 \
    --lr_rate_illum 1e-4 \
    --lr_rate_lte 1e-8 \
    --rec_w 1.0 \
    --rec_loss_type l1 \
    --ms_ssim_w 0.5 \
    --per_w 0.0 \
    --tpl_w 0.0 \
    --adv_w 0.0 \
    --illum_smooth_w 0.0 \
    --color_w 0.0 \
    --exposure_w 0.3 \
    --ref_degrade True \
    --ref_color_jitter 0.15 \
    --ref_shift_range 2 \
    --ref_blur_sigma 1.0 \
    --eval_ref_degrade True \
    --eval_mean_align False \
    --eval_lolv2_nanobanana True \
    --lolv2_nanobanana_eval_real_ref_dir "$NANO_EVAL" \
    --lolv2_nanobanana_eval_syn_ref_dir "" \
    --mixed_lolv2_real_dir "$DATA" \
    --ref_dir "$NANO_EVAL" \
    --load_pretrain True \
    --pretrain_path "$PRETRAIN" \
    --freeze_lte True \
    --freeze_stages "sfe,stage1,stage2,stage3,merge" \
    --query_illum_align False \
    --texture_fusion False \
    --tile_size 256 \
    --tile_overlap 96 \
    --tile_window cosine \
    --num_gpu 1 \
    --num_workers 4 \
    --seed 42
