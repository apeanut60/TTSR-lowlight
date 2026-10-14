# v1r7 — Global Gamma + Partial Luminance Histogram Alignment (B2)

**Branch:** `experiment/v1r7-gamma-hist-align`  
**Init / zero-shot ckpt:** `.../TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt` (v1r3e4, **not overwritten**)  
**Eval:** Huawei Nano Ref, 30 images, `ref_degrade=False`, `eval_mean_align=False`

## 1. Code changes

| File | Change |
|---|---|
| `model/query_illum_align.py` | `build_luma_hist_lut`, `apply_luma_hist_lut`, `estimate_query_align_maps`, `cdf_l1_distance`; `align_query_illumination(..., beta, lut_override)` |
| `model/TTSREnhance.py` | Pass `query_hist_beta` / `query_hist_lut_override` (Query only) |
| `dataset/data1_nanobanana.py` | Full-image γ + hist LUT **before** crop; store `query_gamma`, `query_hist_lut` |
| `dataset/data1.py` | `ToTensor` for `query_hist_lut` |
| `trainer.py` | Train + tiled: reuse full-image γ/LUT overrides |
| `option.py` | `--query_hist_beta` (default 0), `--query_hist_bins` (256) |
| `scripts/eval_v1r7_gamma_hist.py` | Zero-shot A0/A3/B2 harness |
| `train/train_v1r7_ft_b2.sh` | Short FT for B2 |
| `tests/test_query_hist_align.py` | Unit tests |

**Unchanged:** MainNet, LTE, SearchTransfer cosine/argmax, Ref Key/Value, RGB metric口径. Default: align **off**.

## 2. Method (B2)

1. Full-image Rec.709 γ (binary search) so mean luma(Low^γ) ≈ mean luma(Nano Ref).  
2. Blend: `Qγ = (1−α)·Low + α·Low^γ` with **α=0.75**.  
3. Build 256-bin monotonic CDF-matching LUT: luma(Qγ) → luma(Nano Ref) on the **full** pair.  
4. Partial hist: `Y' = (1−β)·Y_Qγ + β·H(Y_Qγ)`.  
5. Apply to RGB via safe chroma-preserving scale `RGB·(Y'/Y)`.  
6. Only Search Query uses the result; MainNet gets original Low; Key/Value untouched.  
7. **Never GT.**

### Full-image policy

- **Train:** estimate γ+LUT on full Low/Nano before crop; apply to Low Query crop; Ref keeps independent crop/aug.  
- **Eval / tiles:** estimate once on full image; reuse overrides for all tiles.

## 3. Tests

| Check | Result |
|---|---|
| β=0 ≡ gamma-only | pass |
| α=0, β=0 ≡ original | pass (bit-identical on model) |
| Checkpoint shapes | unchanged (266/266 load) |
| LUT monotonic / no NaN | pass (black/dark/normal/sat) |
| Full-image maps → crop | pass |
| A0 / A3 reproduce | **20.162 / 20.068** exact prior |

## 4. Zero-shot results (v1r3e4)

| Run | α | β | RGB PSNR | RGB SSIM | max cos | id cos | L2 | ≤4 px | CDF L1 |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 0 | 0 | **20.162** | **0.6324** | 0.354 | 0.218 | 26.5 | 0.134 | 0.228 |
| A3 | 0.75 | 0 | 20.068 | 0.6315 | 0.451 | 0.314 | 21.8 | 0.192 | 0.076 |
| B2-25 | 0.75 | 0.25 | 20.103 | 0.6317 | 0.472 | 0.340 | 20.7 | 0.211 | 0.055 |
| B2-50 | 0.75 | 0.50 | **20.137** | 0.6319 | **0.486** | **0.353** | **20.3** | **0.219** | **0.036** |

ΔPSNR vs A0: A3 −0.094 · B2-25 −0.059 · B2-50 **−0.025**.  
ΔPSNR vs A3: B2-25 +0.035 · B2-50 **+0.070**.

Luma means (B2-50): Low 0.055 → γ-aligned 0.225 → final 0.263 → Ref 0.289.

**Cosine / L2 / CDF are diagnostics only** — not treated as proof of better correspondence.

### B2-50 vs A0 per-image (±0.01 dB)

15↑ / 14↓. Worst drops: idx0 −1.02, idx16 −0.95, idx4 −0.64. Best gains: idx25 +0.82, idx26 +0.74, idx19 +0.73.

## 5. Short FT (β=0.50)

Triggered: PSNR recovers from A3 toward A0; matching diagnostics improve; no one-sided collapse.

Recipe: from v1r3e4, 250 Huawei Nano pairs, crop 192, freeze LTE, L1+MS-SSIM, Adam 3e-6, seed=42, 5 epochs.  
Control C0 = prior same-budget no-align FT (`v1r6-ft-C0-control`).

| Epoch | C0 PSNR | C0 SSIM | B2 FT PSNR | B2 FT SSIM | B2−C0 | B2−A0 |
|---|---|---|---|---|---|---|
| 0 (zs) | 20.162 | 0.6324 | 20.137 | 0.6319 | −0.025 | −0.025 |
| 1 | 20.002 | 0.6309 | 20.060 | 0.6324 | +0.058 | −0.102 |
| 2 | 20.057 | 0.6314 | 20.124 | 0.6330 | +0.067 | −0.038 |
| 3 | 20.082 | 0.6321 | 20.140 | 0.6338 | +0.058 | −0.022 |
| 4 | 19.984 | 0.6298 | 20.084 | 0.6333 | +0.100 | −0.078 |
| 5 | 20.097 | 0.6333 | **20.150** | **0.6347** | +0.053 | **−0.012** |

Best B2 FT: e5 → `/root/data/experiments/retry/v1r7-ft-B2-b050/model/model_00005.pt` (**20.150**).  
Still **below** v1r3e4 by 0.012 dB (< 0.01 claimed as “effective” only with caution; does **not** beat 20.162).

B2e5 vs A0 images: mixed (not a clean win).

## 6. Conclusions

| Question | Answer |
|---|---|
| Does B2 improve zero-shot vs A3? | **Yes** (+0.07 @ β=0.50) |
| Does B2 beat A0 20.162 zero-shot? | **No** (best zs 20.137) |
| Does short FT beat 20.162? | **No** (best 20.150) |
| Does B2 FT beat same-budget C0? | **Yes** (~+0.05), but both under baseline |
| Recommended production ckpt | **v1r3e4** (`model_00004.pt`), align **off** |
| Keep B2 as default? | **No** — optional research flag only |

Histogram matching further reduces CDF distance and raises diagnostic match scores beyond gamma alone, but Huawei RGB PSNR remains capped at/below the frozen v1r3e4 checkpoint under this recipe.

## 7. Next suggestions

1. Do **not** ship query γ/hist as default for Huawei PSNR.  
2. If pursuing matching further: learned query tone with e2e FT, or fix large negative outliers (idx 0/16) rather than stronger global hist.  
3. Prefer levers outside Search Query (tone head / data) for >20.162.

## 8. Artifacts

```
/root/data/experiments/retry/v1r7-gamma-hist-align/zeroshot/
/root/data/experiments/retry/v1r7-ft-B2-b050/
/root/data/experiments/retry/v1r6-ft-C0-control/   # same-budget control
```

Reproduce zero-shot:

```bash
python scripts/eval_v1r7_gamma_hist.py --runs A0,A3,B2-25,B2-50
```
