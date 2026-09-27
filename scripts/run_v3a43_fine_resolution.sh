#!/usr/bin/env bash
# V3-A.4.3 run order: acceptance tests -> artifact lock -> fine-resolution sweep.
#
#   tmux new -s v3a43 -d 'bash scripts/run_v3a43_fine_resolution.sh'
#
# The lock pins the current git HEAD and the frozen V3-A.4.2 summary, so COMMIT
# FIRST -- step 2 re-runs setup and refuses a dirty protocol.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/root/miniconda3/envs/ttsr/bin/python
ROOT=/root/data/experiments/v3a43_fine_resolution
GRIDS=1,2,4,8,16,32,64
LEVELS=16,32,64
mkdir -p "$ROOT/logs"

echo "=== [1/3] acceptance tests (tests/test_v3a43_fine_resolution.py) ==="
"$PY" tests/test_v3a43_fine_resolution.py

echo "=== [2/3] artifact lock (HEAD + every input SHA + V3-A.4.2 anchor) ==="
"$PY" scripts/setup_v3a43_fine_resolution.py --root "$ROOT" \
    --grids "$GRIDS" --levels "$LEVELS"

echo "=== [3/3] fine-resolution sweep: dev64 + train575, G = $GRIDS ==="
"$PY" -W ignore scripts/diagnose_v3a43_fine_resolution.py \
    --root "$ROOT" --splits dev,train --grids "$GRIDS" --levels "$LEVELS" \
    2>&1 | tee "$ROOT/logs/diagnose_fine_resolution.full.log"

echo "=== done: $ROOT/oracle/{summary.json,per_image.csv,fine_capture_curve.csv,nesting.json} ==="
