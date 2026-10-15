# TTSR-lowlight v2.5 — Fitting & Gradient Audit (v2.4 A3)

## Q1 verdict (evidence-based)

**Primary: Loss–PSNR objective conflict, with train-objective still improving.**  
Not a simple underfit. Mild train/eval divergence on PSNR; SSIM rises on both.

| Signal | Trend A3 0→2000 | Interpretation |
|---|---|---|
| Train80 **crop** PSNR | 20.73 → **20.91** | Training objective spatial fit ↑ |
| Train80 crop SSIM | 0.672 → **0.681** | Structure term working |
| Train80 crop L1 | 0.165 → **0.162** | L1 still decreasing |
| Train80 crop MS-SSIM loss | 0.139 → **0.134** | MS-SSIM still decreasing |
| Train80 **full** PSNR | 19.99 → **20.06** | Slight full-res train gain |
| Eval30 PSNR | 20.47 → **20.37** | Dev PSNR ↓ |
| Eval30 SSIM | 0.633 → **0.638** | Dev SSIM ↑ |
| Eval P2-only | 20.265 → **20.112** | Joint FT degrades P2 base |

**Not** “training insufficient”: crop L1/MS-SSIM/PSNR/SSIM all move in the optimized direction through 2000 steps.  
**Not** classic pure overfit alone: train full-res also edges up while Eval PSNR falls — consistent with **optimizing MS-SSIM-friendly solutions that hurt RGB PSNR on Eval30**, plus **P2-only drift**.

Uncertain: how much of Eval PSNR drop is P2 drift vs H3 residual vs metric mismatch. Mixed causes are plausible.

> Train80 / Train500 are **not** holdouts (used in prior P2/H3 training). Eval30 is a reused development set.

## Gradient balance @ init (fixed crop)

On one locked Train80 crop, separate backward for `L1` vs `Ls=1−MS-SSIM`:

| Module | ‖g₁‖ | ‖gₛ‖ | cos(g₁,gₛ) | r(λ=1) | r(0.3) | r(0.1) |
|---|---:|---:|---:|---:|---:|---:|
| P2 | 7.74 | 3.63 | **+0.15** | 0.47 | 0.14 | 0.047 |
| Encoder | 0.130 | 0.080 | +0.86 | 0.61 | 0.18 | 0.061 |
| Matcher | 0.083 | 0.032 | +0.93 | 0.38 | 0.11 | 0.038 |
| **Head** | 0.350 | 0.075 | **−0.22** | 0.21 | 0.064 | 0.021 |

- Raw `Ls` (0.41) > `L1` (0.14) on this crop, but **gradient norms** matter more.
- **Head gradients conflict** (negative cosine): L1 and MS-SSIM push the residual head differently → lowering λ is motivated.
- P2 directions only weakly aligned; at λ=1 MS-SSIM is a large fraction of P2’s update pressure (r≈0.47).

## Protocol note for W0

v2.4 A3 used `hash(sample_id)` crops (process-unstable).  
v2.5 locks **`md5(seed|sample_id)`** crops for W0/W1/W2. **W0 is re-trained** (not a byte-identical A3 clone) under the corrected schedule; A3 remains the historical λ=1 reference on Eval30.

Raw JSON: `/root/data/experiments/TTSR-lowlight-v25/diagnostics/a3_fitting_grad_audit.json`
