#!/bin/bash
# block-amend: train CSFI-U-Net Stage-A on LOLv2-Real from scratch (new blocks).
# LTE may warm-start from Huawei v1r3; MainNetEnhance weights are NOT transferable.
# Uses a dedicated save_dir — do not overwrite P2 / L1@100 experiment trees.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAVE_DIR="${SAVE_DIR:-/root/data/experiments/TTSR-lowlight-block-amend/csfi_lolv2_scratch}"
# Optional: LTE (+ RefCorrection if shapes match) from old P2 init
PRETRAIN="${PRETRAIN:-/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt}"
DATA="${DATA:-/root/data/datasets/lol-v2-real}"
NANO_EVAL="${NANO_EVAL:-/root/data/datasets/lol-v2-real/Test/nanobanana_ref_v3}"

mkdir -p "$SAVE_DIR"
echo "============================================"
echo " block-amend CSFI Stage-A  LOLv2-Real"
echo " backbone=csfi_unet  num_res_blocks=8+8+4+2"
echo " LTE warm-start (require_core=False): $PRETRAIN"
echo " save_dir=$SAVE_DIR"
echo " code=$ROOT"
echo "============================================"

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ttsr 2>/dev/null || true
cd "$ROOT"
export TORCH_HOME=/root/data/cache/torch
export PYTHONUNBUFFERED=1

PRETRAIN_ARGS=()
if [[ -f "$PRETRAIN" ]]; then
  PRETRAIN_ARGS=(
    --load_pretrain True
    --pretrain_path "$PRETRAIN"
    --pretrain_require_core False
    --freeze_lte True
  )
else
  echo "WARN: pretrain missing ($PRETRAIN); training LTE from ImageNet VGG init"
  PRETRAIN_ARGS=(--load_pretrain False --freeze_lte False)
fi

python main.py \
    --dataset lolv2real \
    --dataset_dir "$DATA" \
    --save_dir "$SAVE_DIR" \
    --reset True \
    --log_file_name "train_block_amend_csfi_lolv2.log" \
    --enhance_mode True \
    --backbone csfi_unet \
    --num_res_blocks 8+8+4+2 \
    --n_feats 64 \
    --res_scale 1.0 \
    --igab_blocks 1,1 \
    --igab_heads 1 \
    --enable_lightup True \
    --use_global_illum False \
    --use_ref_illum False \
    --batch_size 4 \
    --train_crop_size 160 \
    --num_init_epochs 0 \
    --num_epochs 100 \
    --print_every 10 \
    --save_every 1 \
    --val_every 1 \
    --decay 50 \
    --gamma 0.5 \
    --lr_rate 1e-4 \
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
    --freeze_stages "" \
    --query_illum_align False \
    --texture_fusion False \
    --stage_mode_lv3 normal \
    --stage_mode_lv2 normal \
    --stage_mode_lv1 normal \
    --tile_size 256 \
    --tile_overlap 96 \
    --tile_window cosine \
    --num_gpu 1 \
    --num_workers 4 \
    --seed 42 \
    "${PRETRAIN_ARGS[@]}"
