#!/bin/bash
# v1r3: continue Huawei nano adapt from v1r2 best (e4, RGB PSNR~19.88)
#   same L1+MS-SSIM, freeze LTE, push toward PSNR 20
# Usage: bash train/train_v1r3_nano_continue.sh [Huawei]

set -euo pipefail
CAM="${1:-Huawei}"
if [[ "$CAM" != "Huawei" ]]; then
  echo "v1r3 continue is Huawei-only for now" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAVE_DIR="/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-${CAM}"
PRETRAIN="/root/data/experiments/retry/TTSR-lowlight-v1r2-nano-Huawei/model/model_00004.pt"
EVAL_H="/root/data/datasets/data1/Eval/Huawei/nanobanana_ref"

if [[ ! -f "$PRETRAIN" ]]; then
  echo "missing pretrain: $PRETRAIN" >&2
  exit 1
fi

mkdir -p "$SAVE_DIR"
echo "============================================"
echo " v1r3 continue  camera=$CAM"
echo " from=$PRETRAIN"
echo " loss=L1+MS-SSIM  freeze=LTE  lr=1e-5"
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
    --nanobanana_manifest_dir /root/data/datasets/data1/.nanobanana_sample_manifest \
    --save_dir "$SAVE_DIR" \
    --reset True \
    --log_file_name "train_v1r3_nano_${CAM}.log" \
    --enhance_mode True \
    --num_res_blocks 10+10+6+4 \
    --n_feats 64 \
    --res_scale 1.0 \
    --batch_size 8 \
    --train_crop_size 192 \
    --num_init_epochs 0 \
    --num_epochs 10 \
    --print_every 5 \
    --save_every 1 \
    --val_every 1 \
    --decay 5 \
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
