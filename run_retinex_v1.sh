#!/usr/bin/env bash
# =============================================================================
# Retinexformer + TTSR-reference V1 — paired runs on **data1 (LSRW)**
#
# Four models, one shared recipe, so each change can be attributed:
#
#   R0  retinexformer . no texture   no ref-illum   -> no-reference baseline
#   R1  retinexformer . no texture   ref-illum on   -> illumination increment
#   R2  retinexformer . texture lv3  ref-illum on   -> full V1
#   R3  ttsr backbone . texture lv3  ref-illum on   -> CONTROL (extra, not in the plan)
#
# R3 keeps the OLD backbone but restricts the texture path to the T_lv3
# injection only, i.e. the same fusion topology as R2. Without it, R2 vs the
# earlier E group mixes "swapped the backbone" with "changed the texture
# injection from three points to one". Drop R3 if you only want speed.
#
# Recipe (shared with the E group where it applies):
#   frozen clean LTE; tpl / adv / ref_correct / illum_match off;
#   RefCorrection and GlobalIllumHead off; explicit --lr_rate_refillum.
#   Query is the raw low-light image; argmax / top-k / S mapping unchanged.
#
# Device notes (measured on this box):
#   data1 = 5600 train pairs (Huawei 2450 + Nikon 3150) vs LOL 485, so one
#   data1 epoch is ~11.5x one LOL epoch. Measured ~2m45s per epoch
#   (700 batches @ batch 8) on the retinexformer backbone with frozen LTE.
#   The pipeline is disk-bound, NOT CPU-bound: running two trainings in
#   parallel made each ~2x slower, so aggregate throughput is unchanged.
#   Tiled eval (tile 256) keeps the 43200-candidate matching matrix off the
#   GPU; the 50-image HR-referenced eval takes ~1 min, the two extra
#   Nano-referenced loaders add ~1 min more.
#
# Usage
#   tmux new -s retinex
#   cd /root/projects/TTSR-lowlight
#   bash run_retinex_v1.sh 2>&1 | tee /root/data/experiments/retinex_v1_driver.log
#
# Environment knobs:
#   EPOCHS=40   SEED=42   OUT_ROOT=/root/data/experiments/retinex_v1_data1_s42
#   SKIP_R3=1   (run only R0/R1/R2)
# =============================================================================
set -uo pipefail

SEED=${SEED:-42}
EPOCHS=${EPOCHS:-40}
SKIP_R3=${SKIP_R3:-0}
VAL_EVERY=${VAL_EVERY:-10}
PRINT_EVERY=${PRINT_EVERY:-100}
PY=${PY:-/root/miniconda3/envs/ttsr/bin/python}
OUT_ROOT=${OUT_ROOT:-/root/data/experiments/retinex_v1_data1_s${SEED}}
DATA=${DATA:-/root/data/datasets/data1}

cd "$(dirname "$0")" || exit 1

if [[ ! -x "$PY" ]]; then echo "python not found: $PY" >&2; exit 1; fi
if [[ ! -d "$DATA/Training data" ]]; then echo "data1 not found under $DATA" >&2; exit 1; fi
if [[ -e "$OUT_ROOT" ]]; then
  echo "Output dir already exists: $OUT_ROOT" >&2
  echo "Move it aside or set OUT_ROOT=<new dir>." >&2
  exit 1
fi
mkdir -p "$OUT_ROOT/logs"

N_RUNS=4; [[ "$SKIP_R3" == "1" ]] && N_RUNS=3
# Measured on this box: 137 s per data1 epoch (700 batches @ batch 8) and
# ~70 s per evaluation (100 images across the three loaders), plus startup.
PER_RUN_MIN=$(awk -v e="$EPOCHS" -v v="$VAL_EVERY" \
  'BEGIN{printf "%.0f", (e+2)*137/60 + int(e/v)*70/60 + 0.5}')
TOTAL_MIN=$(( PER_RUN_MIN * N_RUNS ))

echo "════════════════════════════════════════════════════════════════"
echo " retinex_v1 / data1    seed=$SEED  epochs=$EPOCHS  runs=$N_RUNS"
echo " output : $OUT_ROOT"
echo " estimate: ~${PER_RUN_MIN} min per run  ->  ~$(( TOTAL_MIN / 60 ))h$(( TOTAL_MIN % 60 ))m total"
echo "════════════════════════════════════════════════════════════════"
"$PY" -c "import torch, einops; print('torch', torch.__version__, '| einops', einops.__version__, '| cuda', torch.cuda.is_available())"
echo "train pairs: $(ls "$DATA/Training data"/*/low 2>/dev/null | wc -l)   eval: $(ls "$DATA/Eval"/*/low 2>/dev/null | wc -l)"
echo

# Shared by all runs.
COMMON=(
  --dataset data1 --dataset_dir "$DATA"
  --enhance_mode True
  --freeze_stages=
  --batch_size 8 --train_crop_size 128 --num_workers 8
  --num_init_epochs 2 --num_epochs "$EPOCHS"
  --print_every "$PRINT_EVERY" --save_every 10 --val_every "$VAL_EVERY"
  --lr_rate 1e-4 --lr_rate_lte 0 --lr_rate_refillum 1e-4
  # LR schedule. Without --decay the option default is 999999, i.e. the LR
  # never decays and nothing converges in the late epochs: the first run of
  # this script omitted it and R1 peaked at epoch 20 while R3 regressed
  # 0.58 dB between epoch 30 and 40. Repo convention is 10 or 20; 20 is what
  # the most recent experiments used.
  --decay 20 --gamma 0.5
  --freeze_lte True --load_pretrain False --num_gpu 1
  --rec_w 1.0 --per_w 0.1 --tpl_w 0.0
  --adv_w 0.0 --ref_correct_w 0.0 --illum_match_w 0.0
  --illum_smooth_w 1.0 --color_w 0.5 --exposure_w 1.0
  --ref_correction False --no_global_illum True
  --ref_illum_const_ref False --oracle_matching off
  # Training sees degraded references; every eval loader sees clean ones.
  --ref_degrade True --eval_ref_degrade False
  # data1 eval: the raw TestSet is HR-referenced (diagnostic, 50 images);
  # --eval_data1 also injects the two Nano-referenced loaders (main result).
  --eval_data1 True
  --data1_camera all
  --seed "$SEED"
)

# New backbone args (num_res_blocks / n_feats are parsed but unused there).
RETINEX=( --enhance_backbone retinexformer --retinex_n_feat 40 --retinex_num_blocks 1,2,2 )
# Old backbone args, kept comparable to the E-group recipe.
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
if [[ "$SKIP_R3" != "1" ]]; then
  run_one R3_ttsr_tex3only "${TTSR[@]}"   --no_reference False --no_ref_texture False \
                                          --texture_lv3_only True
fi

echo
echo "══════════════════════════════════════════════════════════════"
echo "ALL DONE  $(date +%F' '%T)"
echo "══════════════════════════════════════════════════════════════"
for s in "${STATUS[@]}"; do echo "  $s"; done

TAGS=( R0_noref R1_refillum R2_refillum_tex3 )
[[ "$SKIP_R3" != "1" ]] && TAGS+=( R3_ttsr_tex3only )

echo
echo "── last evaluation per run (data1 = HR ref / diagnostic; *_nanobanana_* = main) ──"
for tag in "${TAGS[@]}"; do
  echo "── $tag"
  grep -E "INFO: (data1|data1_nanobanana)" "$OUT_ROOT/logs/$tag.log" 2>/dev/null \
    | tail -6 || echo "   (no eval lines)"
done

echo
echo "── optimizer groups actually used (ref_illum must be its own group) ──"
for tag in "${TAGS[@]}"; do
  echo -n "  $tag : "
  grep -m1 "optimizer group 1/" "$OUT_ROOT/logs/$tag.log" 2>/dev/null | sed 's/.*INFO: //' || echo "(missing)"
done

echo
echo "Read the results in this order, per reference setting (Nano first):"
echo "  R1 - R0 : does the reference illumination path help at all?"
echo "  R2 - R1 : does the lv3 texture adapter add anything on top?"
echo "  R2 - R0 : full reference increment"
echo "  R2 - R3 : backbone swap, isolated from the fusion-topology change"
