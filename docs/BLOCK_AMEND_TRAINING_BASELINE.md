# Block-amend training baseline

Engineering baseline for the CSFI-U-Net Stage-A port on LOLv2-Real with
**generated Nano references**. This is a reproducible starting point, not a
claim of optimal hyperparameters or PSNR vs old P2.

## What changed (this fix)

| Area | Change |
|---|---|
| Init | `fresh` / `lte_only` / strict `inference`/`resume` via `model/weight_init.py`. No silent shape-filtered MainNet load from old P2. |
| Data | `dataset/block_amend_lolv2.py`: ID-mapped triplets, no GT-as-Ref, paired crop/flip/rot, train/val manifests. |
| Tiled infer | Shared `model/tiled_infer.py`; trainer + `v2_p2_runtime` use it. **No** `apply_p2_cfg` / `apply_csfi_cfg` during inference. |
| No-ref | `TTSREnhance` accepts `ref=refsr=None` (skips RefCorrection / ST / inject). |
| Train entry | `block_amend_main.py` (no Official Test / extra_test during train). |
| Metrics | Protocol `rgb_float_clipped_v1` in `utils_metrics_block_amend.py`. |
| Scheduler | Epoch-level `scheduler.step()` **after** optimizer updates. |
| Checkpoints | Atomic `checkpoint_last.pt` / `checkpoint_best_val_psnr.pt` packs. |

## Unchanged model / loss

- Architecture: TTSREnhance + LTE + SearchTransfer + CSFIUNet (H, H/2, H/4), IGAB×2, light-up base + RGB residual, optional RefCorrection.
- Matching: SearchTransfer / `sigmoid(S)` as soft-attention modulation (not a calibrated reliability gate).
- Loss recipe: L1 (`rec_w=1`) + MS-SSIM (`ms_ssim_w=1`); other losses off.
- Train hyperparams: `batch=4`, `crop=160`, `lr=1e-4`, `epochs=100`, StepLR `50/0.5`, `seed=42`.

## Fixed config (explicit)

```
backbone=csfi_unet
num_res_blocks=8+8+4+2
n_feats=64  res_scale=1.0
igab_blocks=1,1  igab_heads=1
enable_lightup=True
use_global_illum=False  use_ref_illum=False
ref_correction=True  freeze_lte=True  freeze_stages=""
query_illum_align=False  texture_fusion=False
stage_mode_lv*=normal
eval_mean_align=False  eval_chroma_gain=1.0  eval_tta=False
ref_degrade=False  (default off for this baseline)
```

## Init modes

| Mode | Behavior |
|---|---|
| `fresh` (default) | Code init for MainNet (soft inject / zero Estimator readout) + ImageNet VGG LTE, frozen. |
| `lte_only` | Load **only** `LTE.*` from `--init_checkpoint`. MainNet/RefCorrection must be byte-identical; missing/mismatch LTE keys error. |
| `resume` | Full training pack (`model_state_dict` + optimizer + scheduler + epoch). |
| `eval` / inference | Strict weight load; architecture mismatch errors out. |

## Data

- Train refs: `Train/nanobanana_ref_v2` (override `--train_ref_subdir`).
- Official Test refs: `Test/nanobanana_ref_v3` (eval only).
- Missing Nano → error (or `--allow_partial_coverage` writes include/exclude manifests).
- Internal val: `val_ratio=0.1`, `split_seed=42`, saved under `save_dir/splits/`.
- Best ckpt selected on **internal val** `psnr_rgb` only. Official Test is a separate `--mode eval --eval_split test` command.

## Metrics / checkpoints

- Protocol name: **`rgb_float_clipped_v1`** (float PSNR/SSIM on clipped [0,1]→[0,255] float; PNG is clip+round uint8).
- Saves: `run_meta.json`, `checkpoint_last.pt`, `checkpoint_best_val_psnr.pt`, per-epoch `val_metrics_epXXXXX.json`.

## Tests run

```bash
conda activate ttsr
python tests/test_block_amend_baseline.py
# all block-amend baseline tests passed
```

Covered: LTE-only isolation, no cfg rewrite, tiled stitch (square/short/transpose/odd/small), single-tile≡direct, no-ref, missing-ref error, paired aug, GT≠Ref input, tiny train step + frozen LTE, PNG no wrap, ckpt/scheduler restore, fresh rejects non-empty dir, unknown stage_mode.

**Not run:** full-dataset training, Official Test eval, real VGG end-to-end (LTE stubbed in unit tests). Requires local LOLv2 + Nano trees and `TORCH_HOME` VGG cache for real runs.

## Commands (do not execute here as formal runs)

### 1. Preflight (no train)

```bash
cd /root/projects/TTSR-lowlight-v1-data1SOTA
conda activate ttsr
export TORCH_HOME=/root/data/cache/torch
python block_amend_main.py \
  --mode preflight \
  --dataset_dir /root/data/datasets/lol-v2-real \
  --save_dir /root/data/experiments/TTSR-lowlight-block-amend/preflight_check \
  --train_ref_subdir nanobanana_ref_v2 \
  --test_ref_subdir nanobanana_ref_v3 \
  --init_scope fresh
```

### 2. Fresh training

```bash
RUN_NAME=csfi_lolv2_fresh_$(date +%Y%m%d_%H%M%S) \
  bash train/train_block_amend_csfi_lolv2.sh
# or:
python block_amend_main.py --mode train \
  --dataset_dir /root/data/datasets/lol-v2-real \
  --save_dir /root/data/experiments/TTSR-lowlight-block-amend/$RUN_NAME \
  --init_scope fresh
```

Optional LTE-only:

```bash
INIT_SCOPE=lte_only INIT_CHECKPOINT=/path/to/weights.pt \
  RUN_NAME=csfi_lolv2_lte_only bash train/train_block_amend_csfi_lolv2.sh
```

### 3. Resume

```bash
python block_amend_main.py --mode resume \
  --dataset_dir /root/data/datasets/lol-v2-real \
  --save_dir /root/data/experiments/TTSR-lowlight-block-amend/<RUN> \
  --resume_checkpoint /root/data/experiments/TTSR-lowlight-block-amend/<RUN>/checkpoint_last.pt \
  --split_dir /root/data/experiments/TTSR-lowlight-block-amend/<RUN>/splits
```

### 4. Final Official Test / export

```bash
python block_amend_main.py --mode eval --eval_split test --save_images \
  --dataset_dir /root/data/datasets/lol-v2-real \
  --test_ref_subdir nanobanana_ref_v3 \
  --save_dir /root/data/experiments/TTSR-lowlight-block-amend/<RUN>/official_test \
  --model_path /root/data/experiments/TTSR-lowlight-block-amend/<RUN>/checkpoint_best_val_psnr.pt
```
