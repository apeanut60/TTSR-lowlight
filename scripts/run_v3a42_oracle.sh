#!/usr/bin/env bash
# V3-A.4.2 §24 run order: acceptance tests -> artifact lock -> full sweep.
#
#   tmux new -s v3a42 'bash projects/TTSR-lowlight/scripts/run_v3a42_oracle.sh'
#
# The lock pins the current git HEAD, so COMMIT FIRST (and re-run setup, which
# step 2 does) -- otherwise the run is attributed to the wrong revision.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/root/miniconda3/envs/ttsr/bin/python
ROOT=/root/data/experiments/v3a42_blockwise_oracle
GRIDS=1,2,4,8,16
mkdir -p "$ROOT/logs"

echo "=== [1/3] acceptance tests (tests/test_v3a42_oracle.py) ==="
"$PY" tests/test_v3a42_oracle.py

echo "=== [2/3] artifact lock (pins HEAD + every input SHA) ==="
"$PY" scripts/setup_v3a42_oracle.py --root "$ROOT" --grids "$GRIDS"

echo "=== [3/3] full sweep: dev64 + train575, G = $GRIDS ==="
"$PY" -W ignore scripts/diagnose_v3a42_blockwise_oracle.py \
    --root "$ROOT" --splits dev,train --grids "$GRIDS" \
    2>&1 | tee "$ROOT/logs/diagnose_blockwise_oracle.full.log"

echo "=== done: $ROOT/oracle/{summary.json,per_image.csv,capture_curve.csv} ==="
