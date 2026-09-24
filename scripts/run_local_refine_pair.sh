#!/usr/bin/env bash
# V2 local-reference-refinement: sequential Self -> Nano -> five-condition eval.
#
# The plan (section G2) forbids running the two arms in parallel: the cgroup
# CPU quota is shared, so parallelism buys nothing and only adds noise. Any
# failure stops the run before the next stage.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-/root/miniconda3/envs/ttsr/bin/python}"
OUT_ROOT="${OUT_ROOT:-/root/data/experiments/retinex_v2_localref}"
STEPS="${STEPS:-3000}"

echo "=== [$(date '+%F %T')] V2-Self (${STEPS} steps)"
"$PY" -W ignore scripts/train_local_refine.py --arm self \
      --out_root "$OUT_ROOT" --steps "$STEPS"

echo "=== [$(date '+%F %T')] V2-Nano (${STEPS} steps)"
"$PY" -W ignore scripts/train_local_refine.py --arm nano \
      --out_root "$OUT_ROOT" --steps "$STEPS"

echo "=== [$(date '+%F %T')] five-condition eval @ step ${STEPS}"
"$PY" -W ignore scripts/eval_local_refine.py --out_root "$OUT_ROOT" \
      --step "$STEPS" --out "$OUT_ROOT/per_image_v2.csv"

echo "=== [$(date '+%F %T')] done"
