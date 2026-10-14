# v1r6 — Reference-Guided Query Illumination Alignment

**Branch:** `experiment/v1r6-query-illum-align`  
**Checkpoint:** `.../TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt` (v1r3e4)  
**Eval:** LSRW Huawei Nano Ref, 30 images, `ref_degrade=False`, `eval_mean_align=False`  
**Date:** 2026-10-14

## 1. Goal

Before SearchTransfer matching, brightness-align the **Query** (Low VGG features) toward the **Nano Ref** luminance, without changing MainNet Low input, Ref Value, or pretrained weight shapes.

Hypothesis: reducing illumination mismatch improves patch correspondence. This is tested zero-shot; PSNR gain is not assumed.

## 2. Code changes

| File | Role |
|---|---|
| `model/query_illum_align.py` | `align_query_illumination`, binary-search gamma, `estimate_query_gamma` (full-image for tiles) |
| `model/TTSREnhance.py` | Optional Query path: `LTE(low_query)`; MainNet still gets raw `lr` |
| `model/SearchTransfer.py` | Optional `return_diag` match stats; normal return interface unchanged |
| `option.py` | `--query_illum_align`, `--query_illum_alpha`, `--query_gamma_min/max` (default **off**) |
| `trainer.py` | Tiled inference: one full-image gamma, reused per tile |
| `scripts/eval_v1r6_query_illum.py` | A0–A4 ablation + diagnostics |
| `tests/test_query_illum_align.py` | Unit / consistency tests |

**Unchanged:** MainNet, LTE weights, SearchTransfer cosine/argmax, RGB PSNR/SSIM口径, Ref Value path.

### Audit notes (this tree)

- `lrsr` / `LR_sr`: copy of Low in `[-1,1]`; LTE input `(x+1)/2` → `[0,1]`.
- `ref` / `refsr`: Nano Ref (same when `ref_degrade=False`); Value = `LTE(ref)`, Key = `LTE(refsr)`.
- Missing Nano Ref under `--ref_dir` raises `FileNotFoundError` (no silent GT fallback).

## 3. Method

```
low01, ref01 ∈ [0,1]
L = 0.2126 R + 0.7152 G + 0.0722 B
γ* = argmin_γ | mean(L(low01^γ)) − mean(L(ref01)) |   # binary search on [γ_min, γ_max]
low_γ = 2·low01^γ* − 1
low_query = (1−α)·low + α·low_γ     # Query only
```

Defaults: `γ ∈ [0.3, 1.0]` (brighten-only; never darken past γ=1), `α` configurable. GT is never used.

## 4. Tests

| Check | Result |
|---|---|
| `α=0` vs flag-off output | **bit-identical** (`max_diff=0`) |
| Checkpoint load / weight shapes | 266 matching tensors; no new params |
| Gamma NaN/Inf on black/dark/normal | pass |
| Batch + size-mismatched ref | pass |
| Missing Nano Ref | raises (existing loader) |
| Tiled gamma | full-image `estimate_query_gamma` + `query_gamma_override` |

## 5. Baseline reproduction (A0)

| Metric | Reported baseline | A0 (this run) |
|---|---|---|
| RGB PSNR | 20.162 | **20.162** |
| RGB SSIM | 0.6324 | **0.6324** |
| N | 30 | 30 |

## 6. Zero-shot ablation (same ckpt, same 30 images)

| Run | Method | α | RGB PSNR | RGB SSIM | ΔPSNR vs A0 |
|---|---|---|---|---|---|
| A0 | Original Query | 0.00 | **20.162** | **0.6324** | — |
| A1 | Gamma align | 0.25 | 20.113 | 0.6322 | −0.050 |
| A2 | Gamma align | 0.50 | 20.074 | 0.6317 | −0.088 |
| A3 | Gamma align | 0.75 | 20.068 | 0.6315 | −0.094 |
| A4 | Gamma align | 1.00 | 20.090 | 0.6316 | −0.072 |

Raw JSON: `/root/data/experiments/retry/v1r6-query-illum-align/ablation/`.

**No zero-shot RGB PSNR/SSIM gain.** Best frozen metric remains A0 / v1r3e4.

## 7. Gamma & luminance

Same γ distribution for all runs (per-image, Low vs Nano Ref):

| Stat | Value |
|---|---|
| mean γ | 0.405 |
| std | 0.089 |
| min / max | 0.300 / 0.663 |
| q10 / q50 / q90 | 0.300 / 0.401 / 0.512 |

Several images hit `γ_min=0.3` (very dark Low vs brighter Nano).

| | A0 (α=0) | A4 (α=1) |
|---|---|---|
| mean \|L_low − L_ref\| | 0.228 | 0.234 (pre) |
| mean \|L_aligned − L_ref\| | (= before) | **0.041** |

Alignment closes the luma gap as intended when α>0.

## 8. Matching diagnostics

Identity stats are **auxiliary** (Nano Ref may have geometric drift; grids are same HxW after resize, so identity is defined but not a correctness oracle).

| Run | max cos ↑ | id cos ↑ | match L2 ↓ | ≤1 px | ≤2 | ≤4 | ≤8 |
|---|---|---|---|---|---|---|---|
| A0 | 0.354 | 0.218 | 26.48 | 0.092 | 0.105 | 0.134 | 0.193 |
| A1 | 0.393 | 0.259 | 24.31 | 0.118 | 0.133 | 0.167 | 0.235 |
| A2 | 0.423 | 0.288 | 23.01 | 0.130 | 0.146 | 0.183 | 0.258 |
| A3 | 0.451 | 0.314 | **21.83** | **0.138** | **0.155** | **0.192** | **0.274** |
| A4 | **0.477** | **0.332** | 22.05 | 0.132 | 0.147 | 0.183 | 0.264 |

Feature norms (mean over images):

| Run | Query ‖·‖ | Key ‖·‖ |
|---|---|---|
| A0 | 1.23 | 1.88 |
| A4 | **1.94** | 1.88 |

Query features move toward Key scale; cosine similarity and spatial concentration improve monotonically with α (peak spatial locality at **α=0.75**).

## 9. Per-image PSNR (A0 vs A4)

| idx | A0 | A4 | Δ | γ |
|---|---|---|---|---|
| 00 | 21.209 | 20.210 | −1.00 | 0.464 |
| 01 | 21.980 | 21.426 | −0.55 | 0.392 |
| 02 | 20.393 | 20.733 | +0.34 | 0.528 |
| 03 | 20.689 | 20.224 | −0.47 | 0.448 |
| 04 | 23.652 | 22.911 | −0.74 | 0.463 |
| 05 | 23.532 | 23.734 | +0.20 | 0.338 |
| 06 | 21.135 | 21.392 | +0.26 | 0.300 |
| 07 | 17.023 | 16.794 | −0.23 | 0.300 |
| 08 | 20.257 | 19.915 | −0.34 | 0.429 |
| 09 | 16.680 | 16.676 | −0.00 | 0.353 |
| 10 | 17.371 | 17.423 | +0.05 | 0.413 |
| 11 | 14.927 | 15.009 | +0.08 | 0.446 |
| 12 | 24.121 | 23.776 | −0.35 | 0.502 |
| 13 | 19.860 | 19.921 | +0.06 | 0.350 |
| 14 | 21.527 | 21.743 | +0.22 | 0.525 |
| 15 | 16.886 | 17.202 | +0.32 | 0.336 |
| 16 | 26.355 | 24.814 | −1.54 | 0.300 |
| 17 | 18.770 | 18.202 | −0.57 | 0.416 |
| 18 | 21.900 | 21.931 | +0.03 | 0.386 |
| 19 | 16.400 | 17.022 | +0.62 | 0.300 |
| 20 | 18.656 | 18.886 | +0.23 | 0.300 |
| 21 | 22.457 | 22.205 | −0.25 | 0.329 |
| 22 | 24.382 | 24.293 | −0.09 | 0.315 |
| 23 | 18.593 | 18.400 | −0.19 | 0.333 |
| 24 | 20.530 | 21.187 | +0.66 | 0.339 |
| 25 | 16.896 | 17.744 | +0.85 | 0.511 |
| 26 | 25.299 | 26.403 | +1.10 | 0.411 |
| 27 | 16.441 | 16.080 | −0.36 | 0.466 |
| 28 | 19.112 | 18.925 | −0.19 | 0.663 |
| 29 | 17.833 | 17.531 | −0.30 | 0.506 |

Gains and losses cancel; large drops (e.g. idx 16) dominate the mean.

## 10. Decision

| Question | Answer |
|---|---|
| Zero-shot PSNR improve? | **No** (−0.05 to −0.09 dB) |
| Matching improve? | **Yes** (max cos +0.12, id cos +0.11, L2 −4.6, ≤4 px +6 pp at α=0.75) |
| Method failed? | **Not declared failed** — correspondence improves; MainNet/Search consumers were trained on dark Queries |
| Best α (metrics) | A0 for PSNR |
| Best α (match) | **α=0.75 (A3)** spatial; **α=1.0 (A4)** cosine |
| Large-scale retrain? | **No** |
| Short FT worth trying? | **Yes, limited** — freeze LTE, keep α fixed at **0.75**, short run from v1r3e4; stop if no PSNR lift over A0 |

### Recommended next step (not executed here)

1. Fix `query_illum_align=True`, `query_illum_alpha=0.75`, γ∈[0.3,1.0].
2. Short Huawei nano adapt from `model_00004.pt` (LTE frozen, same L1+MS-SSIM recipe as v1r3), few epochs only.
3. Re-eval 30-image Huawei Nano; accept only if RGB PSNR > 20.162.
4. If still flat: abandon query-gamma as a PSNR lever; keep diagnostics code for future matching work.

## 11. Limits respected

- No LLFormer / Retinexformer / V3-B0 / LocalRefine  
- No MainNet redesign / LTE retrain from scratch  
- No GT in matching; no checkpoint overwrite  
- No unverified “SOTA” claim; zero-shot does **not** beat 20.162  

## 12. How to reproduce

```bash
cd /root/projects/TTSR-lowlight-v1-data1SOTA
python tests/test_query_illum_align.py
python scripts/eval_v1r6_query_illum.py \
  --ckpt /root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt \
  --out_dir /root/data/experiments/retry/v1r6-query-illum-align/ablation
```
