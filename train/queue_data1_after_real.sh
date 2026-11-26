#!/bin/bash
# After the LOLv2-Real run in tmux lfca_real finishes successfully, train
# data1 Huawei, then data1 Nikon, with the same csfi_lfca_v1 recipe.
# Fresh init each time. Does not write into /root/data/datasets/data1.
# Official Eval is linked only so preflight can see the Test tree; this
# queue does not run Official Test. Eval refs are .png and lows are .jpg,
# so preflight will report those Test refs as missing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="${SAVE_ROOT:-/root/data/experiments/TTSR-lowlight-block-amend}"
DATA1="/root/data/datasets/data1"
REAL_DIR="$EXP/csfi_lfca_v1_lolv2_partial_20261119"
REAL_SESSION="${REAL_SESSION:-lfca_real}"
QUEUE_LOG="$EXP/lfca_queue.log"

mkdir -p "$EXP"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"
}

make_view() {
  local camera="$1"
  local eval_ref="$2"
  local view="$EXP/views/data1_$(echo "$camera" | tr '[:upper:]' '[:lower:]')"
  local train_src="$DATA1/Training data/$camera"
  local eval_src="$DATA1/Eval/$camera"
  mkdir -p "$view/Train" "$view/Test"
  ln -sfn "$train_src/low" "$view/Train/Low"
  ln -sfn "$train_src/high" "$view/Train/Normal"
  ln -sfn "$train_src/nanobanana_ref" "$view/Train/nanobanana_ref"
  ln -sfn "$eval_src/low" "$view/Test/Low"
  ln -sfn "$eval_src/high" "$view/Test/Normal"
  ln -sfn "$eval_src/$eval_ref" "$view/Test/nanobanana_ref"
  # The dataset root must not be a symlink target we mutate. Views are
  # directories of symlinks only.
  echo "$view"
}

run_finished() {
  local save_dir="$1"
  local log_file="$save_dir/block_amend.log"
  [[ -f "$log_file" ]] && grep -q "val epoch 100" "$log_file" \
    && [[ -f "$save_dir/checkpoint_last.pt" ]]
}

run_camera() {
  local camera="$1"
  local view="$2"
  local tag
  tag="$(echo "$camera" | tr '[:upper:]' '[:lower:]')"
  local preflight_dir="$EXP/preflight_data1_${tag}"
  local run_name="csfi_lfca_v1_data1_${tag}_partial_20261119"
  local save_dir="$EXP/$run_name"

  if [[ -e "$preflight_dir" ]]; then
    log "ERROR: preflight dir already exists: $preflight_dir"
    exit 1
  fi
  if [[ -e "$save_dir" ]]; then
    log "ERROR: train dir already exists: $save_dir"
    exit 1
  fi

  log "preflight $camera  data=$view"
  DATA="$view" \
  TRAIN_REF=nanobanana_ref \
  TEST_REF=nanobanana_ref \
  SAVE_DIR="$preflight_dir" \
  ALLOW_PARTIAL_COVERAGE=1 \
  INIT_SCOPE=fresh \
  bash -c '
    set -euo pipefail
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate ttsr
    cd "'"$ROOT"'"
    export TORCH_HOME="${TORCH_HOME:-/root/data/cache/torch}"
    python block_amend_main.py --mode preflight \
      --model_variant csfi_lfca_v1 \
      --dataset_dir "$DATA" \
      --train_ref_subdir "$TRAIN_REF" \
      --test_ref_subdir "$TEST_REF" \
      --save_dir "$SAVE_DIR" \
      --init_scope fresh \
      --allow_partial_coverage
  '
  if ! grep -q "Preflight OK" "$preflight_dir/block_amend.log"; then
    log "ERROR: $camera preflight did not finish cleanly"
    exit 1
  fi

  log "train $camera -> $save_dir"
  DATA="$view" \
  TRAIN_REF=nanobanana_ref \
  TEST_REF=nanobanana_ref \
  RUN_NAME="$run_name" \
  SAVE_DIR="$save_dir" \
  ALLOW_PARTIAL_COVERAGE=1 \
  INIT_SCOPE=fresh \
  INIT_CHECKPOINT= \
  bash "$ROOT/train/train_block_amend_csfi_lolv2.sh"

  if ! run_finished "$save_dir"; then
    log "ERROR: $camera training did not reach epoch 100"
    exit 1
  fi
  log "$camera training finished: $save_dir"
}

log "queue started; real session=$REAL_SESSION"
HUAWEI_VIEW="$(make_view Huawei nanobanana_ref)"
NIKON_VIEW="$(make_view Nikon nanobanana_ref_Nikon)"
log "huawei view=$HUAWEI_VIEW"
log "nikon view=$NIKON_VIEW"

log "waiting for tmux session $REAL_SESSION to exit"
while tmux has-session -t "$REAL_SESSION" 2>/dev/null; do
  sleep 30
done
log "$REAL_SESSION is gone"

if ! run_finished "$REAL_DIR"; then
  log "ERROR: real run is not a finished epoch-100 checkpoint at $REAL_DIR"
  log "data1 training will not start"
  exit 1
fi
log "real finished; starting Huawei then Nikon"

run_camera Huawei "$HUAWEI_VIEW"
run_camera Nikon "$NIKON_VIEW"
log "queue finished: Huawei then Nikon"
