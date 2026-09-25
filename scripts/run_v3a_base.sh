#!/usr/bin/env bash
# V3-A Stage 1: train the LOL-v2-real reference-free Base-R and freeze it.
#
# Differences from run_retinex_v1.sh (all deliberate, see the V3-A plan):
#   * --dataset lolv2real_v3a  (never dataset/lolv2real: its TrainSet does
#     Ref = HR.copy(), i.e. GT-as-reference)
#   * --exposure_w 0.0         (run_retinex_v1.sh uses 1.0; §10 of the
#     retinex findings shows exposure fights the reconstruction target)
#   * --v3a_ref_variant        so the TestSet reads Test/<variant>, not Train's
#   * explicit --decay/--gamma (otherwise the LR never decays)
#
# The checkpoint used downstream is fixed a priori as the LAST epoch; the
# intermediate evaluations are reported as a trajectory only, never used to
# pick a checkpoint.
set -uo pipefail

SEED=${SEED:-42}
EPOCHS=${EPOCHS:-150}
PY=${PY:-/root/miniconda3/envs/ttsr/bin/python}
ROOT=${ROOT:-/root/data/experiments/v3a_lolv2real}
DATA=${DATA:-/root/data/datasets/lol-v2-real}
VARIANT=${VARIANT:-nanobanana_ref_v2}
SAVE="$ROOT/base_r"

cd "$(dirname "$0")/.." || exit 1
if [[ -e "$SAVE" ]]; then
  echo "refusing to overwrite $SAVE" >&2
  exit 1
fi
mkdir -p "$ROOT/logs"

echo "=== Base-R  epochs=$EPOCHS  seed=$SEED  -> $SAVE"
"$PY" -W ignore main.py \
  --dataset lolv2real_v3a --dataset_dir "$DATA" --v3a_ref_variant "$VARIANT" \
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
  --print_every 100 --save_every 25 --val_every 25 \
  --save_dir "$SAVE" --seed "$SEED" 2>&1 | tee "$ROOT/logs/base_r.log"
echo "=== Base-R done; frozen checkpoint = $SAVE/model/model_$(printf %05d "$EPOCHS").pt"
