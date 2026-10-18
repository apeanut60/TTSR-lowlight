# TTSR inference / evaluation consistency fix

Branch: `fix/ttsr-inference-consistency`  
Base: `a4f651239bb3ab2deb56fd805b69b31fbdde8c70`  
This change does not add a head, does not edit checkpoints, and does not start training.

## What changed

1. **One tiled backbone.** `forward_p2_tiled_backbone` in `v2_p2_runtime.py` computes the full-image query gamma/LUT, tiles the backbone, and blends. It does not run `RefIllumTransfer` or `GlobalIllumHead`. `tiled_p2` and `scripts/train_lolv2syn_illum.py:illum_forward` both call it, then `apply_p2_photometric_postprocess` (RefIllum, then GlobalIllumHead). The illum trainer keeps the backbone under `no_grad` and leaves the two photometric modules on the autograd graph.

2. **Border weights.** The old cosine ramp was applied to every tile edge, including the image border, so the corner weight was 0 and `accum / weight.clamp(min=1e-8)` wrote 0 there. Feathering is now only on edges that touch another tile. The accumulated weight must be `> 0` or the forward raises. A short axis (`H < tile` or `W < tile`) is one span of the true length; origins are never negative.

3. **RefIllum reference.** `TTSREnhance.forward` feeds `RefCorrection(ref)` into RefIllum. The old tiled path called `apply_ref_illum(..., raw ref)` after the stitch. `get_effective_reference` applies RefCorrection once on the full reference (or returns the raw reference when the head is off). It does not read GT. Tile texture features can still differ from a single full-image forward; that is unchanged.

4. **Cache identity.** `begin_cache` / `assert_cache_compatible` require checkpoint SHA256, manifest SHA256, `P2_CONFIG`, `PREPROCESS_VERSION`, dtype, range, and the sample-id set. A mismatch raises and does not rewrite `metadata.json`. `--rebuild_cache` is the only switch that deletes `.npy` and metadata first. `PREPROCESS_VERSION` was bumped, so an S1@1400 cache cannot be reused as an S1c@400 cache.

5. **RGB metrics.** LOLv2 domain eval, the illum trainer, and Nikon export score uint8 RGB PSNR/SSIM through `score_rgb_tensors` / `channel_metrics`. No GT mean alignment and no TTA. RGB SSIM is the mean of the three channels, not Y SSIM.

6. **Val vs Official Test.** `scripts/lolv2_split.py` holds out a fixed fit/val split (seed + ratio persisted). Early stopping reads that val split only. Official Test is opened only by `--eval_split test --final_eval`, and that path scores a checkpoint without using Test for early stopping. LOLv2 manifests have no scene id; the split is per image and the metadata says scene isolation is unconfirmed. The old full-train crop schedule is not the default anymore (`random_crop_fit_s{seed}.csv`). Historical `docs/TTSR_LOWLIGHT_LOLV2*_RESULTS.md` numbers are left as Test-selected results from the old protocol.

## Root cause

| Issue | Cause |
|---|---|
| A. Border | Cosine ramp included 0 at both ends and was applied on the outer image edge |
| B. Reference | Stitched RefIllum saw the raw nano; `forward()` saw `RefCorrection(ref)` |
| C. Cache | Existing `.npy` files were skipped, then metadata was rewritten for the new checkpoint |
| D. Metrics | Labels said "Y PSNR" while the number was already RGB; Y SSIM exists beside RGB SSIM |
| Test selection | The training loop scored Official Test every 100 steps and kept the best |

## Code

- `v2_p2_runtime.py`
- `model/TTSREnhance.py` (`get_effective_reference` only; no new parameters)
- `scripts/train_lolv2syn_illum.py`
- `scripts/train_lolv2syn_domain.py`
- `scripts/train_lolv2_domain.py`
- `scripts/lolv2_split.py`
- `scripts/prepare_lolv2_p2_cache.py`
- `scripts/prepare_lolv2syn_p2_cache.py`
- `local_refine_runtime.py`
- `scripts/eval_nikon_export.py` (`score_rgb_tensors`)
- `tests/test_tiled_blending.py`
- `tests/test_p2_ref_postprocess.py`
- `tests/test_p2_cache_invalidation.py`
- `tests/test_lolv2_eval_consistency.py`

`P2_CONFIG` is unchanged (asserted in `test_p2_config_unchanged`). S1c `checkpoint_00400.pt` still loads the same `MainNet` / `RefCorrection` keys.

## Does this change historical numbers?

Yes, if those checkpoints are scored again with the new forward. The published tables were **not** edited.

One LOLv2-Syn Test image, S1c@400, 384×384, `r00816405t` (not a dataset score):

| Comparison | max abs | border mean abs | interior mean abs | this image PSNR_RGB |
|---|---:|---:|---:|---|
| A. old blend vs new blend (backbone only) | 0.959 | 0.0464 | 0 | 21.655 → 21.971 |
| B. raw ref vs RefCorrection(ref), new blend | 0.00013 | 0.00002 | 0.00004 | 21.988 → 21.987 |
| A+B old full vs new full | 0.961 | 0.0465 | 0.00005 | 21.672 → 21.987 |

Legacy weight at the corner was 0. Interior pixels match, so A is the border fix, not a texture-model change. B is about 1e-4 and does not explain the PSNR move. No cache was rebuilt (C) and the metric function on this pair was already uint8 RGB (D). **Do not read the +0.3 dB on this one image as a model improvement or as a new Test100 number.**

## What to re-evaluate, and whether to retrain

Re-score with the new forward before comparing to the old tables:

- Syn P0 / S1 / S1b / S1c / S1d
- Real P0 / L1 / composed H3
- Any Y0 `.npy` cache (rebuild with `--rebuild_cache` into a new directory or the same directory explicitly)

No retrain was run. S1d's photometric weights were fit with the old border and the raw reference. A future illum run would use the shared forward and the val split; that is a new experiment, not part of this fix.

## Tests

14/14 passed (`tests/test_tiled_blending.py`, `test_p2_ref_postprocess.py`, `test_p2_cache_invalidation.py`, `test_lolv2_eval_consistency.py`):

constant image, border weight > 0, irregular H/W, one call each and fixed order, RefCorrection on/off, train forward equals eval forward, photometric grads finite and nonzero, frozen weights unchanged, S1 cache rejected for an S1c SHA, RGB entries agree, Official Test blocked unless `--eval_split test --final_eval`.

## Commit

Implementation commit SHA is filled in the commit that adds this file's final hash line below.

`FIX_COMMIT_SHA`
