# Block-amend training baseline

Engineering baseline for the CSFI-U-Net Stage-A port on LOLv2-Real with
**generated Nano references**. Reproducible start point — not a performance claim.

**Checkpoint schema:** `checkpoint_schema_version = 3` (LFCA; adds a top-level
`model_variant` and the restoration-block identity to `config`). Schema 2 was
amend-2 (IGAB variant).  
Older / incomplete / other-variant packs are **rejected** on resume (no
warm-start fallback). See `docs/BLOCK_AMEND_LFCA.md` for the current model
identity (`csfi_lfca_v1`) and commands.

## What is implemented

| Area | Behavior |
|---|---|
| Init | `fresh` / `lte_only` (LTE.* only) / strict pack load for resume & eval |
| Data | ID-mapped Low/GT/Nano; no GT-as-Ref; paired crop/flip/rot |
| Epoch aug | `set_epoch(epoch)`; seed = SHA256(base_seed\|epoch\|sample_id); `persistent_workers=False` |
| Manifest | Identity hash **v2** over relative paths + `ref_subdir` (not basename-only; not content hash) |
| Tiled infer | Shared `model/tiled_infer.py`; no silent `apply_*_cfg` |
| Metrics | `rgb_float_clipped_v1` |
| Scheduler | One `StepLR.step()` after each epoch’s optimizer updates |
| Checkpoint | Atomic `tmp_ckpt_*.pt` → `os.replace`; validate → update best → save best (if improved) → save last |
| Resume | Restores model/opt/sched/RNG/loader generators; locks split identity & config |

## Fixed training recipe (unchanged by the LFCA replacement)

```
backbone=csfi_unet  num_res_blocks=8+8+4+2  n_feats=64
model_variant=csfi_lfca_v1  restoration_block=lfca
restoration_blocks=1,1  restoration_heads=1  enable_lightup=True
use_global_illum=False  use_ref_illum=False
ref_correction=True  freeze_lte=True  freeze_stages=""
query_illum_align=False  texture_fusion=False
stage_mode_lv*=normal
batch_size=4  train_crop_size=160  lr=1e-4
num_epochs=100  StepLR(50, 0.5)
rec_w=1  ms_ssim_w=1  other losses off
seed=42  val_ratio=0.1  split_seed=42
ref_degrade=False
```

## Checkpoint pack (schema 3)

Required fields: `checkpoint_schema_version`, `model_variant`, `model_state_dict`,
`optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `global_step`,
`best_val_psnr`, `best_epoch`, `validation_completed`, `config` (full resolved),
`metric_protocol`, `init_scope`, `run_id`, `split_identity`, `rng`
(python / numpy / torch CPU / CUDA / train&val generators).

**Save order each epoch:** train → `scheduler.step()` → optional val →
update best if **strictly** better → build pack → save best (if this epoch) →
save last. `last.best_val_psnr` is the true historical best after that epoch.

**Resume:** checkpoint config wins; only explicit CLI ops overrides
(`cpu`, `num_gpu`, `num_workers`, `num_epochs` extend, paths/logs). Locked
architecture/loss/split/seed/crop/batch keys cannot be silently defaulted over.
Missing split manifests or hash mismatch → hard fail (no re-split).

**Eval:** builds model from checkpoint `config` when present; bare weights
require explicit architecture CLI (no silent defaults).

**Compatibility:** schema 3 / `csfi_lfca_v1` only. Older incomplete checkpoints
and IGAB-variant (schema 2) checkpoints cannot resume or be evaluated here; do
not auto-convert.

## Manifest identity (v2)

Hash input: sorted `{id, split, ref_subdir, low/normal/ref relative paths}`.  
Same basenames under a different `ref_subdir` → different hash.  
This is **not** a file-content fingerprint (optional `file_sha256` exists but is
not recomputed every epoch).

## Tests run (CPU, synthetic / stub)

```text
Python 3.10.20  PyTorch 1.12.1+cu113
python tests/test_block_amend_round2.py
→ all block-amend round2 tests passed
```

Covered: atomic save create/replace/failure-safe; epoch-aware paired aug;
manifest overlap/dup/batch/ref_subdir hash; grad NaN blocks step; last/best
20→21→20.5 sequence; resume rejects missing split; mini train+resume
scheduler/epoch; val_every metadata; tampered manifest; plus LTE-only / no-ref /
PNG / tiled regressions.

**Stub:** `_StubLTE` replaces VGG LTE (no download).  
**Not verified here:** full LOLv2 training, Official Test, real ImageNet VGG
bitwise continuity across machines/PyTorch versions, mid-epoch resume.

## Commands (do not execute as formal runs here)

### Preflight
```bash
cd /root/projects/TTSR-lowlight-v1-data1SOTA && conda activate ttsr
export TORCH_HOME=/root/data/cache/torch
python block_amend_main.py --mode preflight \
  --dataset_dir /root/data/datasets/lol-v2-real \
  --save_dir /root/data/experiments/TTSR-lowlight-block-amend/preflight_check \
  --init_scope fresh
```

### Fresh train
```bash
bash train/train_block_amend_csfi_lolv2.sh
# or:
python block_amend_main.py --mode train \
  --dataset_dir /root/data/datasets/lol-v2-real \
  --save_dir /root/data/experiments/TTSR-lowlight-block-amend/$RUN_NAME \
  --init_scope fresh
```

### Resume
```bash
python block_amend_main.py --mode resume \
  --dataset_dir /root/data/datasets/lol-v2-real \
  --save_dir /root/data/experiments/TTSR-lowlight-block-amend/<RUN> \
  --resume_checkpoint .../<RUN>/checkpoint_last.pt \
  --split_dir .../<RUN>/splits
```

### Official Test / export
```bash
python block_amend_main.py --mode eval --eval_split test --save_images \
  --dataset_dir /root/data/datasets/lol-v2-real \
  --test_ref_subdir nanobanana_ref_v3 \
  --save_dir .../<RUN>/official_test \
  --model_path .../<RUN>/checkpoint_best_val_psnr.pt
```
