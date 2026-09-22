#!/usr/bin/env bash
# =============================================================================
# Retinexformer + TTSR-reference V1 — paired runs
#
# Runs four models that share one training recipe so the changes made in this
# round can be attributed one at a time:
#
#   R0  retinexformer . no texture   no ref-illum   -> no-reference baseline
#   R1  retinexformer . no texture   ref-illum on   -> illumination increment
#   R2  retinexformer . texture lv3  ref-illum on   -> full V1
#   R3  ttsr backbone . texture lv3  ref-illum on   -> CONTROL (not in the plan)
#
# R3 is the extra control: it keeps the OLD backbone but restricts the texture
# path to the T_lv3 injection only, matching R2's fusion topology. Without it,
# R2 vs the previous E group mixes "swapped the backbone" with "changed the
# texture injection from three points to one".
#
# Recipe (shared): frozen clean LTE, tpl/adv/ref_correct/illum_match off,
# RefCorrection and GlobalIllumHead off, explicit --lr_rate_refillum.
# Query is the raw low-light image; argmax / top-k / S mapping are unchanged.
#
# Usage
#   tmux new -s retinex
#   cd /root/projects/TTSR-lowlight
#   bash run_retinex_v1.sh 2>&1 | tee /root/data/experiments/retinex_v1_driver.log
#
# Environment knobs: SEED (42), EPOCHS (48), OUT_ROOT
# =============================================================================
set -uo pipefail

SEED=${SEED:-42}
EPOCHS=${EPOCHS:-48}
PY=${PY:-/root/miniconda3/envs/ttsr/bin/python}
OUT_ROOT=${OUT_ROOT:-/root/data/experiments/retinex_v1_s${SEED}}
DATA=/root/data/datasets/LOLdataset

cd "$(dirname "$0")" || exit 1

if [[ ! -x "$PY" ]]; then
  echo "python not found: $PY" >&2; exit 1
fi
if [[ -e "$OUT_ROOT" ]]; then
  echo "Output dir already exists: $OUT_ROOT" >&2
  echo "Move it aside or set OUT_ROOT=<new dir>." >&2
  exit 1
fi
mkdir -p "$OUT_ROOT/logs"

echo "=== retinex_v1 seed=$SEED epochs=$EPOCHS -> $OUT_ROOT  $(date +%F' '%T) ==="
echo "=== python: $PY"
"$PY" -c "import torch,einops;print('torch',torch.__version__,'| einops',einops.__version__,'| cuda',torch.cuda.is_available())"

# Shared by all four runs.
COMMON=(
  --dataset LOL --dataset_dir "$DATA"
  --enhance_mode True
  --freeze_stages=
  --batch_size 8 --train_crop_size 128 --num_workers 8
  --num_init_epochs 2 --num_epochs "$EPOCHS"
  --print_every 50 --save_every 10 --val_every 10
  --lr_rate 1e-4 --lr_rate_lte 0 --lr_rate_refillum 1e-4
  --freeze_lte True --load_pretrain False --num_gpu 1
  --rec_w 1.0 --per_w 0.1 --tpl_w 0.0
  --adv_w 0.0 --ref_correct_w 0.0 --illum_match_w 0.0
  --illum_smooth_w 1.0 --color_w 0.5 --exposure_w 1.0
  --ref_correction False --no_global_illum True
  --ref_illum_const_ref False --oracle_matching off
  --ref_degrade True --eval_ref_degrade False
  --eval_lol_nanobanana True
  --seed "$SEED"
)

# New backbone args (num_res_blocks/n_feats are parsed but unused there).
RETINEX=( --enhance_backbone retinexformer --retinex_n_feat 40 --retinex_num_blocks 1,2,2 )
# Old backbone args, kept identical to the E-group recipe for comparability.
TTSR=( --enhance_backbone ttsr --num_res_blocks 8+8+4+2 --n_feats 64 --res_scale 1.0 )

STATUS=()

run_one () {
  local tag="$1"; shift
  local log="$OUT_ROOT/logs/$tag.log"
  echo
  echo "──────────────────────────────────────────────────────────────"
  echo ">>> $tag   start $(date +%T)"
  echo "──────────────────────────────────────────────────────────────"
  if "$PY" -W ignore main.py "${COMMON[@]}" "$@" \
        --save_dir "$OUT_ROOT/$tag" 2>&1 | tee "$log"; then
    STATUS+=("OK   $tag")
  else
    STATUS+=("FAIL $tag  (see $log)")
  fi
  echo ">>> $tag   done  $(date +%T)"
}

run_one R0_noref          "${RETINEX[@]}" --no_reference True  --no_ref_texture True
run_one R1_refillum       "${RETINEX[@]}" --no_reference False --no_ref_texture True
run_one R2_refillum_tex3  "${RETINEX[@]}" --no_reference False --no_ref_texture False
run_one R3_ttsr_tex3only  "${TTSR[@]}"    --no_reference False --no_ref_texture False \
                                          --texture_lv3_only True

echo
echo "══════════════════════════════════════════════════════════════"
echo "ALL DONE  $(date +%F' '%T)"
echo "══════════════════════════════════════════════════════════════"
for s in "${STATUS[@]}"; do echo "  $s"; done

echo
echo "── final (last) evaluation available in each log ──"
for tag in R0_noref R1_refillum R2_refillum_tex3 R3_ttsr_tex3only; do
  echo "── $tag"
  grep -E "INFO: (LOL|lol_nanobanana)  PSNR:" "$OUT_ROOT/logs/$tag.log" 2>/dev/null | tail -2 || echo "   (no eval lines)"
done

echo
echo "── optimizer groups actually used (must list ref_illum separately) ──"
for tag in R0_noref R1_refillum R2_refillum_tex3 R3_ttsr_tex3only; do
  echo -n "  $tag : "
  grep -m1 "optimizer group 1/" "$OUT_ROOT/logs/$tag.log" 2>/dev/null | sed 's/.*INFO: //' || echo "(missing)"
done

echo
echo "Compare, per reference setting, the per-image PSNR of R1-R0 (illumination"
echo "increment), R2-R1 (texture increment), R2-R0 (full) and R2-R3 (backbone"
echo "swap, isolated). Use the Nano-referenced numbers as the main result and"
echo "the HR-referenced ones only as a diagnostic."
