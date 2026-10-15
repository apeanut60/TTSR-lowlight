#!/bin/bash
# v1r5: large-crop short push from v1r3 Huawei best (e4)
#   crop=256, batch=4, lr=8e-6, 8 epochs, L1+MS-SSIM, freeze LTE
# Usage: bash train/train_v1r5_nano.sh [Huawei]

set -euo pipefail
CAM="${1:-Huawei}"
if [[ "$CAM" != "Huawei" ]]; then
  echo "v1r5 is Huawei-only for now" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAVE_DIR="/root/data/experiments/retry/TTSR-lowlight-v1r5-nano-${CAM}"
PRETRAIN="/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt"
EVAL_H="/root/data/datasets/data1/Eval/Huawei/nanobanana_ref"

if [[ ! -f "$PRETRAIN" ]]; then
  echo "missing pretrain: $PRETRAIN" >&2
  exit 1
fi

mkdir -p "$SAVE_DIR"
echo "============================================"
echo " v1r5 large-crop  camera=$CAM"
echo " from=$PRETRAIN"
echo " crop=256 batch=4 lr=8e-6 epochs=8"
echo " save_dir=$SAVE_DIR"
echo "============================================"

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ttsr 2>/dev/null || true
cd "$ROOT"

python main.py \
    --dataset data1_nanobanana \
    --dataset_dir /root/data/datasets/data1 \
    --data1_camera "$CAM" \
    --nanobanana_ref_subdir nanobanana_ref \
    --nanobanana_manifest_dir /root/data/datasets/data1/.nanobanana_sample_manifest_500 \
    --save_dir "$SAVE_DIR" \
    --reset True \
    --log_file_name "train_v1r5_nano_${CAM}.log" \
    --enhance_mode True \
    --num_res_blocks 10+10+6+4 \
    --n_feats 64 \
    --res_scale 1.0 \
    --batch_size 4 \
    --train_crop_size 256 \
    --num_init_epochs 0 \
    --num_epochs 8 \
    --print_every 10 \
    --save_every 1 \
    --val_every 1 \
    --decay 4 \
    --gamma 0.5 \
    --lr_rate 8e-6 \
    --lr_rate_lte 8e-7 \
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
    --eval_data1 True \
    --early_stop_on_data1 False \
    --data1_nanobanana_eval_huawei_ref_dir "$EVAL_H" \
    --data1_nanobanana_eval_nikon_ref_dir "" \
    --load_pretrain True \
    --pretrain_path "$PRETRAIN" \
    --freeze_lte True \
    --freeze_stages "" \
    --tile_size 256 \
    --tile_overlap 96 \
    --tile_window cosine \
    --num_gpu 1 \
    --num_workers 4
