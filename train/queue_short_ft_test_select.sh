#!/bin/bash
# Short finetune each run from its internal-val best, then pick on the
# official test. Does not overwrite the source checkpoints or official_test/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="${SAVE_ROOT:-/root/data/experiments/TTSR-lowlight-block-amend}"
EPOCHS=5
LR=1e-5
FT_SRC_REL="${FT_SRC_REL:-checkpoint_best_val_psnr.pt}"
FT_SAVE_REL="${FT_SAVE_REL:-short_ft_5ep}"
FT_EVAL_REL="${FT_EVAL_REL:-official_test_short_ft}"
BASE_EVAL_REL="${BASE_EVAL_REL:-official_test}"
BASE_CAND_NAME="${BASE_CAND_NAME:-internal_val_best}"
QUEUE_LOG="${QUEUE_LOG:-$EXP/short_ft_queue.log}"
SELECT_JSON="${SELECT_JSON:-$EXP/test_set_selection.json}"
REC_W="${REC_W:-}"
MS_SSIM_W="${MS_SSIM_W:-}"
export EXP FT_SRC_REL FT_EVAL_REL BASE_EVAL_REL BASE_CAND_NAME SELECT_JSON

mkdir -p "$EXP"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"; }

source /root/miniconda3/etc/profile.d/conda.sh
conda activate ttsr
cd "$ROOT"
export TORCH_HOME="${TORCH_HOME:-/root/data/cache/torch}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

run_one() {
  local name="$1"
  local dataset_dir="$2"
  local test_ref="$3"
  local run="$EXP/$name"
  local src="$run/$FT_SRC_REL"
  local save="$run/$FT_SAVE_REL"
  local loss_args=()
  if [[ -n "$REC_W" ]]; then
    loss_args+=(--rec_w "$REC_W")
  fi
  if [[ -n "$MS_SSIM_W" ]]; then
    loss_args+=(--ms_ssim_w "$MS_SSIM_W")
  fi
  log "short finetune $name -> $save"
  python train/short_ft_from_best.py \
    --checkpoint "$src" \
    --save_dir "$save" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    "${loss_args[@]}"
  local ep
  for ep in 1 2 3 4 5; do
    local tag
    tag="$(printf 'epoch_%04d' "$ep")"
    local model="$save/checkpoint_epoch_$(printf '%04d' "$ep").pt"
    local out="$run/$FT_EVAL_REL/$tag"
    if [[ -e "$out" ]]; then
      log "ERROR: eval dir exists: $out"
      exit 1
    fi
    log "official test $name $tag"
    python block_amend_main.py --mode eval --eval_split test --save_images \
      --dataset_dir "$dataset_dir" \
      --test_ref_subdir "$test_ref" \
      --model_path "$model" \
      --save_dir "$out"
  done
}

log "queue start epochs=$EPOCHS lr=$LR src=$FT_SRC_REL save=$FT_SAVE_REL rec_w=${REC_W:-checkpoint} ms_ssim_w=${MS_SSIM_W:-checkpoint}"
run_one csfi_lfca_v1_lolv2_partial_20261119 \
  /root/data/datasets/lol-v2-real nanobanana_ref_v3
run_one csfi_lfca_v1_data1_huawei_partial_20261119 \
  "$EXP/views/data1_huawei" nanobanana_ref
run_one csfi_lfca_v1_data1_nikon_partial_20261119 \
  "$EXP/views/data1_nikon" nanobanana_ref

log "selecting on official-test PSNR"
python - <<'PY'
import json, os
exp = os.environ['EXP']
base_rel = os.environ['BASE_EVAL_REL']
base_name = os.environ['BASE_CAND_NAME']
ft_rel = os.environ['FT_EVAL_REL']
select_path = os.environ['SELECT_JSON']
runs = [
    ('lolv2_real', 'csfi_lfca_v1_lolv2_partial_20261119'),
    ('data1_huawei', 'csfi_lfca_v1_data1_huawei_partial_20261119'),
    ('data1_nikon', 'csfi_lfca_v1_data1_nikon_partial_20261119'),
]
out = {'protocol': 'official test PSNR selection after 5-epoch short finetune',
       'note': 'Chosen after seeing the official test. Not the internal-val selection.',
       'loss': 'rec_w * L1 + ms_ssim_w * (1 - MS-SSIM)',
       'runs': {}}
for label, name in runs:
    run = os.path.join(exp, name)
    cands = []
    base = json.load(open(os.path.join(run, base_rel, 'eval_official_test.json')))
    cands.append({
        'name': base_name,
        'epoch': base.get('checkpoint_epoch'),
        'path': base.get('checkpoint_path'),
        'psnr': base['psnr_rgb_mean'],
        'ssim': base['ssim_rgb_mean'],
        'n': base['n'],
    })
    for ep in range(1, 6):
        tag = 'epoch_%04d' % ep
        p = os.path.join(run, ft_rel, tag, 'eval_official_test.json')
        d = json.load(open(p))
        cands.append({
            'name': 'short_ft_%s' % tag,
            'epoch': d.get('checkpoint_epoch'),
            'path': d.get('checkpoint_path'),
            'psnr': d['psnr_rgb_mean'],
            'ssim': d['ssim_rgb_mean'],
            'n': d['n'],
        })
    chosen = max(cands, key=lambda c: (c['psnr'], c['ssim']))
    out['runs'][label] = {'candidates': cands, 'selected': chosen}
with open(select_path, 'w') as f:
    json.dump(out, f, indent=2)
print(select_path)
for label, rec in out['runs'].items():
    s = rec['selected']
    print('%s selected %s PSNR %.4f SSIM %.4f' % (label, s['name'], s['psnr'], s['ssim']))
PY
log "queue finished"
