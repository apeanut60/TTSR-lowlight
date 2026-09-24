#!/usr/bin/env bash
# V2 seed replication: repeat the Self/Nano pair on fresh seeds.
#
# Same protocol as run_local_refine_pair.sh, but each seed gets its own refiner
# initialisation (refiner_init_s<seed>.pt) so the replication varies both the
# initialisation and the augmentation/data draw -- not just data order.
#
# Sequential on purpose: the cgroup CPU quota is shared, so parallel arms only
# add noise. Any failure aborts the whole sweep.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-/root/miniconda3/envs/ttsr/bin/python}"
OUT_ROOT="${OUT_ROOT:-/root/data/experiments/retinex_v2_localref}"
STEPS="${STEPS:-3000}"
SEEDS="${SEEDS:-7 123}"

for s in $SEEDS; do
  echo "=== [$(date '+%F %T')] seed $s : V2-Self"
  "$PY" -W ignore scripts/train_local_refine.py --arm self \
        --out_root "$OUT_ROOT" --steps "$STEPS" --seed "$s"
  echo "=== [$(date '+%F %T')] seed $s : V2-Nano"
  "$PY" -W ignore scripts/train_local_refine.py --arm nano \
        --out_root "$OUT_ROOT" --steps "$STEPS" --seed "$s"
  echo "=== [$(date '+%F %T')] seed $s : five-condition eval"
  "$PY" -W ignore scripts/eval_local_refine.py --out_root "$OUT_ROOT" \
        --step "$STEPS" --seed "$s" --out "$OUT_ROOT/per_image_v2_s${s}.csv"
done

echo "=== [$(date '+%F %T')] seed sweep done: $SEEDS"
