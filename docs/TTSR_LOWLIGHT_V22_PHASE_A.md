# TTSR-lowlight v2.2 — Phase A: Full-Image DC Diagnosis

Zero-training. H3@`checkpoint_02000`. Float32 P2 Y0 caches from v2.

Artifacts: `diagnostics/v22/{train500,eval30}_dc_analysis.csv`, `crop_full_comparison.csv`, `dc_distribution_summary.json`.

## Eval30 regression

H3 mean PSNR **20.466** (matches v2.0). Δ vs P2 mean **+0.201**.

## Train500 vs Eval30 DC behaviour

| Stat | Train500 | Eval30 |
|---|---|---|
| Mean ΔH3 vs P2 | +0.119 | +0.201 |
| Mean ΔCentered | +0.124 | +0.154 |
| Mean ‖m‖ | 0.054 | 0.060 |
| Mean e_mean_frac | 0.538 | 0.623 |
| Frac H3 positive | 61.2% | 66.7% |
| Frac H3 negative | 37.8% | 33.3% |
| Frac harmful DC (`dot(e,m)>0`) | **40.6%** | **40.0%** |
| Frac “excessive” ‖m‖ | 8.6% | 3.3% |

**Train500 exhibits the same class of harmful global RGB corrections** as Eval30 (≈40% harmful DC alignment). Eval30 failures are not an out-of-distribution quirk of the 30-image set alone.

Focus Eval30 (from summary): worst cases remain mean-dominated (2053 etc.); best cases often use corrective `m` opposing P2 bias.

## Crop vs Full prediction (frozen H3, n=20 Train crops)

| Metric | Value |
|---|---|
| Mean MAE(D_full-crop, D_crop-forward) | 0.0023 |
| Mean cosine | **0.989** |
| Mean \|Δm\| | 0.0015 |
| MAE border (8px) | 0.0087 |
| MAE interior | 0.0003 |

Predictions are **close but not identical**; discrepancy concentrates on **borders** (convolution/match context). Even when predictions are similar, **crop MSE cannot supervise the true full-image RGB mean target** `m* = mean_HW(GT−Y0)`.

## Phase A answers

1. **Does Train500 also show wrong global RGB corrections?** Yes (~40% harmful DC).
2. **Do Eval30 issues have Train counterparts?** Yes — same statistics and mechanism.
3. **Are Full vs Crop predictions markedly different?** Mild overall (cosine≈0.99), but border-sensitive; more importantly the **training objective** differs for DC.
4. **Is Full-Image Training a justified mechanism test?** **Yes** — proceed to Phase B.

Protocol consistent (H3@2000 reproduced). No blockers.
