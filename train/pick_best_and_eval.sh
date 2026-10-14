#!/bin/bash
# Pick best Huawei-nano epoch from train log and run official RGB eval.
# Usage: bash train/pick_best_and_eval.sh <save_dir>

set -euo pipefail
SAVE_DIR="${1:?save_dir required}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$(ls -1 "$SAVE_DIR"/train_*.log 2>/dev/null | head -1)"
if [[ -z "$LOG" ]]; then
  echo "no train log in $SAVE_DIR" >&2
  exit 1
fi

BEST="$(python3 - <<PY
import re
log=open("$LOG").read().splitlines()
best_ep, best_p, best_s = None, -1.0, -1.0
ep=None
for line in log:
    m=re.search(r'Current epoch learning rate', line)
    # epochs are inferred from evaluate order of data1_nanobanana_huawei
    m=re.search(r'data1_nanobanana_huawei\s+PSNR:\s*([0-9.]+)\s+SSIM:\s*([0-9.]+)', line)
    if m:
        # count occurrences
        pass
rows=[]
for line in log:
    m=re.search(r'data1_nanobanana_huawei\s+PSNR:\s*([0-9.]+)\s+SSIM:\s*([0-9.]+)', line)
    if m:
        rows.append((float(m.group(1)), float(m.group(2))))
if not rows:
    raise SystemExit('no huawei nano metrics')
# epoch index 1-based in appearance order
best_i=max(range(len(rows)), key=lambda i:(rows[i][0], rows[i][1]))
print(best_i+1, rows[best_i][0], rows[best_i][1])
for i,(p,s) in enumerate(rows,1):
    mark=' *' if i==best_i+1 else ''
    print(f'ep{i}: PSNR={p:.3f} SSIM={s:.4f}{mark}', file=__import__('sys').stderr)
PY
)"

read -r BEST_EP BEST_P BEST_S <<<"$BEST"
echo "Selected epoch=$BEST_EP PSNR=$BEST_P SSIM=$BEST_S"

# Official eval into same save_dir
CAM=Huawei
EPO_TAG="$(printf '%05d' "$BEST_EP")"
MODEL_PATH="$SAVE_DIR/model/model_${EPO_TAG}.pt"
REF_DIR="/root/data/datasets/data1/Eval/Huawei/nanobanana_ref"
OUT_SUB="save_results_nanobanana_${CAM}_e${EPO_TAG}"
LOG_OUT="test_${CAM}_nanobanana_rgb_e${EPO_TAG}.log"
mkdir -p "$SAVE_DIR/$OUT_SUB"
RESULTS="$SAVE_DIR/save_results"
if [[ -e "$RESULTS" && ! -L "$RESULTS" ]]; then
  mv "$RESULTS" "$SAVE_DIR/save_results.bak_$(date +%s)" 2>/dev/null || true
fi
rm -f "$RESULTS"
ln -sfn "$SAVE_DIR/$OUT_SUB" "$RESULTS"

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ttsr 2>/dev/null || true
cd "$ROOT"

python main.py \
    --dataset data1 \
    --dataset_dir /root/data/datasets/data1 \
    --data1_camera Huawei \
    --ref_dir "$REF_DIR" \
    --save_dir "$SAVE_DIR" \
    --reset False \
    --log_file_name "$LOG_OUT" \
    --eval True \
    --eval_save_results True \
    --eval_mean_align False \
    --enhance_mode True \
    --num_res_blocks 10+10+6+4 \
    --n_feats 64 \
    --res_scale 1.0 \
    --ref_degrade False \
    --model_path "$MODEL_PATH" \
    --freeze_stages "" \
    --num_workers 4 \
    --num_gpu 1

echo "BEST_EPOCH=$BEST_EP"
echo "VAL_PSNR=$BEST_P VAL_SSIM=$BEST_S"
grep 'PSNR:' "$SAVE_DIR/$LOG_OUT" | tail -1
