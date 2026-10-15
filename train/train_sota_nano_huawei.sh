#!/bin/bash
# SOTA-Nano Huawei optimization runs (Phase 2/3).
# Usage:
#   bash train/train_sota_nano_huawei.sh <exp_name> <ms_ssim_w>
# Example:
#   bash train/train_sota_nano_huawei.sh P2-cosine-full-mss1 1.0

set -euo pipefail
EXP_NAME="${1:?exp_name required}"
MS_W="${2:?ms_ssim_w required}"
SEED="${3:-42}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAVE_DIR="/root/data/experiments/retry/SOTA-Nano-${EXP_NAME}"
PRETRAIN="/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt"
MANIFEST="/root/data/datasets/data1/.nanobanana_sample_manifest_500"
EVAL_H="/root/data/datasets/data1/Eval/Huawei/nanobanana_ref"
N_TRAIN="$(wc -l < "$MANIFEST/Huawei.txt" | tr -d ' ')"

if [[ ! -f "$PRETRAIN" ]]; then
  echo "missing pretrain: $PRETRAIN" >&2
  exit 1
fi

mkdir -p "$SAVE_DIR"

cat <<EOF
============================================
Experiment name: SOTA-Nano-${EXP_NAME}
Train samples:   ${N_TRAIN} (Huawei Nano full manifest)
Validation samples: 30 (Huawei Eval)
Reference source: Nano
Metric: RGB PSNR / RGB SSIM
Crop: 192
Loss: L1 + ${MS_W}*(1-MS-SSIM)
Freeze: LTE only
Learning rate: 1e-5 cosine (10 epochs, eta_min=1e-7)
Seed: ${SEED}
Start ckpt: v1r3e4
save_dir: ${SAVE_DIR}
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
    --n_feats 64 \
    --res_scale 1.0 \
    --batch_size 8 \
    --train_crop_size 192 \
    --num_init_epochs 0 \
    --num_epochs 10 \
    --print_every 10 \
    --save_every 1 \
    --val_every 1 \
    --lr_scheduler cosine \
    --lr_rate 1e-5 \
    --lr_rate_lte 1e-6 \
    --eta_min 1e-7 \
    --seed "$SEED" \
    --rec_w 1.0 \
    --rec_loss_type l1 \
    --ms_ssim_w "$MS_W" \
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
