#!/bin/bash
# Phase4: ref-only ±5% brightness/saturation aug; best prior recipe (mss_w=2.0, cosine, full 271)
set -euo pipefail
EXP_NAME="P4-refaug-mss2"
SEED=42
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAVE_DIR="/root/data/experiments/retry/SOTA-Nano-${EXP_NAME}"
PRETRAIN="/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt"
MANIFEST="/root/data/datasets/data1/.nanobanana_sample_manifest_full"
EVAL_H="/root/data/datasets/data1/Eval/Huawei/nanobanana_ref"
N_TRAIN="$(wc -l < "$MANIFEST/Huawei.txt" | tr -d ' ')"

mkdir -p "$SAVE_DIR"
cat <<EOF
============================================
Experiment name: SOTA-Nano-${EXP_NAME}
Train samples:   ${N_TRAIN} (Huawei Nano full)
Validation samples: 30
Reference source: Nano + ref_photo_aug ±5%
Metric: RGB PSNR / RGB SSIM
Crop: 192
Loss: L1 + 2.0*(1-MS-SSIM)
Freeze: LTE only
Learning rate: 1e-5 cosine / 10ep
Seed: ${SEED}
============================================
EOF

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ttsr 2>/dev/null || true
cd "$ROOT"

python main.py \
    --dataset data1_nanobanana \
    --dataset_dir /root/data/datasets/data1 \
    --data1_camera Huawei \
    --nanobanana_ref_subdir nanobanana_ref \
    --nanobanana_manifest_dir "$MANIFEST" \
    --save_dir "$SAVE_DIR" \
    --reset True \
    --log_file_name "train_${EXP_NAME}.log" \
    --enhance_mode True \
    --num_res_blocks 10+10+6+4 \
    --n_feats 64 --res_scale 1.0 \
    --batch_size 8 --train_crop_size 192 \
    --num_init_epochs 0 --num_epochs 10 \
    --print_every 10 --save_every 1 --val_every 1 \
    --lr_scheduler cosine --lr_rate 1e-5 --lr_rate_lte 1e-6 --eta_min 1e-7 \
    --seed "$SEED" \
    --rec_w 1.0 --rec_loss_type l1 --ms_ssim_w 2.0 \
    --per_w 0 --tpl_w 0 --adv_w 0 --illum_smooth_w 0 --color_w 0 --exposure_w 0 \
    --ref_degrade False --ref_photo_aug True --ref_photo_aug_range 0.05 \
    --eval_mean_align False --eval_data1 True --early_stop_on_data1 False \
    --data1_nanobanana_eval_huawei_ref_dir "$EVAL_H" \
    --data1_nanobanana_eval_nikon_ref_dir "" \
    --load_pretrain True --pretrain_path "$PRETRAIN" \
    --freeze_lte True --freeze_stages "" \
    --tile_size 256 --tile_overlap 96 --tile_window cosine \
    --num_gpu 1 --num_workers 4
