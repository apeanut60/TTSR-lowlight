#!/usr/bin/env bash
# V3-A.5C runner (plan §39):
#   [0] N=4 micro sanity
#   [1] build tiny16 + cache
#   [2] acceptance tests
#   [3] target/oracle stats (produced by setup)
#   [4] train C0
#   [5] eval C0 + verdict
#   [6-7] C1 only if C0 fails/partial
#
#   bash scripts/run_v3a5c.sh
#   UPDATES=2000 bash scripts/run_v3a5c.sh   # shorter smoke
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/root/miniconda3/envs/ttsr/bin/python
ROOT=${ROOT:-/root/data/experiments/v3a5c_tiny_overfit}
UPDATES=${UPDATES:-20000}
MICRO_UPDATES=${MICRO_UPDATES:-200}
mkdir -p "$ROOT/logs"

echo "=== [2 early] unit tests ==="
"$PY" tests/test_v3a5c_tiny.py

echo "=== [1] setup tiny16 + micro4 cache + lock ==="
"$PY" -W ignore scripts/setup_v3a5c_tiny.py --root "$ROOT" \
    2>&1 | tee "$ROOT/logs/setup_v3a5c.log"

echo "=== [0] N=4 micro sanity (short C0) ==="
"$PY" -W ignore scripts/train_v3a5c_tiny.py --root "$ROOT" \
    --arm C0_current_loss --micro --updates "$MICRO_UPDATES" \
    2>&1 | tee "$ROOT/logs/train_C0_micro.log"
echo "=== [4] train C0 current-loss (updates=$UPDATES) ==="
"$PY" -W ignore scripts/train_v3a5c_tiny.py --root "$ROOT" \
    --arm C0_current_loss --updates "$UPDATES" \
    2>&1 | tee "$ROOT/logs/train_C0_current_loss.log"

echo "=== [5] eval C0 + verdict ==="
"$PY" -W ignore scripts/eval_v3a5c_tiny.py --root "$ROOT" \
    --arm C0_current_loss \
    2>&1 | tee "$ROOT/logs/eval_C0.log"

LABEL=$("$PY" -c "import json; print(json.load(open('$ROOT/C0_current_loss/final_eval.json'))['verdict']['label'])")
ACTION=$("$PY" -c "import json; print(json.load(open('$ROOT/C0_current_loss/final_eval.json'))['verdict']['action'])")
echo "C0 verdict: $LABEL / $ACTION"

if [[ "$ACTION" == "run_c1" ]]; then
  echo "=== [6] train C1 gate-only ==="
  "$PY" -W ignore scripts/train_v3a5c_tiny.py --root "$ROOT" \
      --arm C1_gate_only --updates "$UPDATES" \
      2>&1 | tee "$ROOT/logs/train_C1_gate_only.log"
  echo "=== [7] eval C1 + verdict ==="
  "$PY" -W ignore scripts/eval_v3a5c_tiny.py --root "$ROOT" \
      --arm C1_gate_only \
      2>&1 | tee "$ROOT/logs/eval_C1.log"
  "$PY" -c "import json; v=json.load(open('$ROOT/C1_gate_only/final_eval.json'))['verdict']; print('C1 verdict:', v['label'], v['action'])"
else
  echo "C0 succeeded — skipping C1 per plan."
fi

echo "DONE. artifacts under $ROOT"
