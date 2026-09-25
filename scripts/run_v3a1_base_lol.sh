#!/usr/bin/env bash
# V3-A.1 方案 A: retrain the LOL-specific reference-free base.
#
# Why a new base at all: with the data1/LSRW base the LOL Train and Test splits
# sit 4.83 dB apart for the SAME frozen model (15.74 vs 20.57 RGB PSNR), so a
# refiner trained on Train learns a ~1.7x larger correction than Test needs and
# overshoots. A LOL-trained base has a normal +1.2 dB train/test gap.
#
# The 50 train images that have no Nano-v2 reference are held out as validation;
# the 100-image test set is NOT used for checkpoint selection.
set -euo pipefail

SEED=${SEED:-42}
EPOCHS=${EPOCHS:-60}
PY=${PY:-/root/miniconda3/envs/ttsr/bin/python}
ROOT=${ROOT:-/root/data/experiments/v3a1_lolv2real}
DATA=${DATA:-$ROOT/base_data}
SAVE="$ROOT/base_lol"

cd "$(dirname "$0")/.." || exit 1
if [[ -e "$SAVE" ]]; then
  echo "refusing to overwrite $SAVE" >&2
  exit 1
fi
mkdir -p "$ROOT/logs"

echo "=== V3-A.1 base (LOL)  epochs=$EPOCHS seed=$SEED -> $SAVE"
"$PY" -W ignore main.py \
  --dataset lolv2real_v3a --dataset_dir "$DATA" --v3a_ref_variant nanobanana_ref_v2 \
  --enhance_mode True --enhance_backbone retinexformer \
  --retinex_n_feat 40 --retinex_num_blocks 1,2,2 \
  --freeze_stages= --no_reference True --no_ref_texture True --no_global_illum True \
  --ref_correction False --freeze_lte True --load_pretrain False --num_gpu 1 \
  --batch_size 8 --train_crop_size 128 --num_workers 8 \
  --num_init_epochs 2 --num_epochs "$EPOCHS" \
  --lr_rate 1e-4 --lr_rate_lte 0 --lr_rate_refillum 1e-4 \
  --decay 20 --gamma 0.5 \
  --rec_w 1.0 --per_w 0.1 --tpl_w 0.0 --adv_w 0.0 \
  --ref_correct_w 0.0 --illum_match_w 0.0 \
  --illum_smooth_w 1.0 --color_w 0.5 --exposure_w 0.0 \
  --print_every 100 --save_every 5 --val_every 5 \
  --save_dir "$SAVE" --seed "$SEED" 2>&1 | tee "$ROOT/logs/base_lol.log"
echo "=== base (LOL) done"
