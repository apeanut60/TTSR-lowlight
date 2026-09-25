#!/usr/bin/env bash
# V3-A stage 2: train both refiner arms sequentially, then evaluate.
#
# Base-R is the existing audited frozen N0 (retinex_v11 N0_fixed_s42) rather
# than a freshly trained LOL-specific base: the LOL-specific base was trained
# and REGRESSED (raw PSNR 20.37 -> 18.86 from epoch 25 to 75 while SSIM rose
# and mean-aligned PSNR stayed flat), so it was worse than the existing N0 at
# every epoch. See findings_v3a.md.
set -euo pipefail

PY=${PY:-/root/miniconda3/envs/ttsr/bin/python}
ROOT=${ROOT:-/root/data/experiments/v3a_lolv2real}
STEPS=${STEPS:-3000}

cd "$(dirname "$0")/.." || exit 1
mkdir -p "$ROOT/logs"

for arm in r1 r2; do
  echo "=== [$(date '+%F %T')] V3-A $arm ($STEPS steps)"
  "$PY" -W ignore scripts/train_v3a_lolv2real.py --arm "$arm" --root "$ROOT" \
        --steps "$STEPS" 2>&1 | tee "$ROOT/logs/train_$arm.log"
done

echo "=== [$(date '+%F %T')] final comparison + oracle gate"
"$PY" -W ignore scripts/eval_v3a_lolv2real.py --root "$ROOT" \
      --step "$STEPS" 2>&1 | tee "$ROOT/logs/eval_v3a.log"
echo "=== [$(date '+%F %T')] done"
