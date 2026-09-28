#!/usr/bin/env bash
# V3-A.5A run order (plan §32): tests -> lock -> §31 acceptance -> A0 -> A1 -> dev64.
#
#   tmux new -s v3a5 -d 'bash scripts/run_v3a5.sh'
#
# COMMIT FIRST: a formal run (--limit 0) refuses a dirty working tree, because
# the lock pins repo_commit and cannot vouch for uncommitted code.
#
# V3-A.5B (global + regional) is NOT run here: plan §22/§32 only enter it after
# the 5A verdict passes.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/root/miniconda3/envs/ttsr/bin/python
ROOT=/root/data/experiments/v3a5_g64_verifier
mkdir -p "$ROOT/logs"

echo "=== [1/6] acceptance tests (tests/test_v3a5_verifier.py) ==="
"$PY" tests/test_v3a5_verifier.py

echo "=== [2/6] targets + energy mask + artifact lock ==="
"$PY" scripts/setup_v3a5_verifier.py --root "$ROOT" \
    2>&1 | tee "$ROOT/logs/setup_v3a5.log"

echo "=== [3/6] §31 acceptance: target algebra / G64 repro / frozen proposal / step0 ==="
"$PY" -W ignore scripts/check_v3a5_targets.py --root "$ROOT" \
    2>&1 | tee "$ROOT/logs/check_v3a5_targets.log"

echo "=== [4/6] train A0 dense-BlockH4 ==="
"$PY" -W ignore scripts/train_v3a5_verifier.py --root "$ROOT" --arm A0_dense_blockh4 \
    2>&1 | tee -a "$ROOT/logs/train_A0_dense_blockh4.log"

echo "=== [5/6] train A1 G64 ==="
"$PY" -W ignore scripts/train_v3a5_verifier.py --root "$ROOT" --arm A1_g64 \
    2>&1 | tee -a "$ROOT/logs/train_A1_g64.log"

echo "=== [6/6] dev64 (and train575) comparison ==="
"$PY" -W ignore scripts/eval_v3a5_dev.py --root "$ROOT" --splits dev,train \
    2>&1 | tee "$ROOT/logs/eval_dev.log"

echo "=== done: $ROOT/diagnostics/{summary.json,per_image.csv,gate_metrics.csv,recovery64.csv} ==="
