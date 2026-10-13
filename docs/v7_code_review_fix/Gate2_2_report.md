# Gate 2.2 — Alignment Information Quality

- **actual commit:** `e5b068df0d9f7109d8abdf995b3b50d3cca96e8e`
- **G2.1 code commit:** `e597d47648d72b34ad2781b4140d6e9c7e02e3a0`
- **G2.1 ckpt:** `/root/data/experiments/V7_CODE_REVIEW_FIX/gate2_1/ckpt_microset_win3_fixed.pt`
- **seed/lr/updates/n:** 42 / 0.0001 / 200 / 8

## 1. Fixed-batch overfit (supersedes step-loss claim)

| | init | G2.1 |
|---|---|---|
| mean MSE (8 imgs) | 0.020378 | 0.010242 |
| overfit_ok | | True |

Old G2.1 compared `loss_first`/`loss_last` on **different** images in the cycle — invalid for overfit.

## 2. Attention audit (no train)

| metric | value |
|---|---|
| mean weight_max | 0.1115 (uniform=0.1111) |
| mean entropy / uniform | 1.0000 |
| Δ‖q‖ / Δ‖k‖ / Δ‖v‖ | 0.107 / 0.1436 / 0.9483 |
| grad q/k/v @G2.1 | 1.947e-06 / 5.75e-06 / 0.01441 |
| **attention_selection_real** | **False** |

## 3. G2.1 align arm — PSNR / sensitivity

### microset (n=8)

| PSNR Base | Normal | Shuffle | N−Base | N−Shuf | mae_swap | mae_shuf |
|---|---|---|---|---|---|---|
| 23.510 | 26.176 | 26.175 | +2.6665 | +0.0012 | 6.100e-04 | 2.030e-04 |

### microset-heldout (n=16)

| PSNR Base | Normal | Shuffle | N−Base | N−Shuf | mae_swap | mae_shuf |
|---|---|---|---|---|---|---|
| 27.024 | 24.868 | 24.866 | -2.1563 | +0.0021 | 5.370e-04 | 2.012e-04 |

## 4. Single-variable: A2 = tex['h4'] (bypass align)

| batch MSE @0 | @200 | overfit |
|---|---|---|
| 0.020378 | 0.010078 | True |

### microset (n=8)

| PSNR Base | Normal | Shuffle | N−Base | N−Shuf | mae_swap | mae_shuf |
|---|---|---|---|---|---|---|
| 23.510 | 26.274 | 25.998 | +2.7645 | +0.2760 | 4.960e-03 | 1.070e-02 |

### microset-heldout (n=16)

| PSNR Base | Normal | Shuffle | N−Base | N−Shuf | mae_swap | mae_shuf |
|---|---|---|---|---|---|---|
| 27.024 | 24.702 | 24.747 | -2.3219 | -0.0448 | 4.024e-03 | 8.591e-03 |

clear_ref_utilization (microset criteria): **True**

### Interpretation

1. **G2.1 + random align:** microset mean MSE drops (0.0204→0.0102) and PSNR rises vs Base (~+2.7 dB), but Normal≈Shuffle and `mae_swapA2~6e-4` → residual learning **without** Ref identity. Attention weights ≈ uniform (`wmax≈1/9`, entropy frac≈1) despite nonzero Δθ on Q/K/V — **no real spatial selection**.
2. **A2=`tex['h4']` (only change):** microset Normal−Shuffle **+0.28 dB**, `mae_swapA2~5e-3`, `mae_shuffleRef~1e-2` → **clear Ref content use**. Heldout shows the same sensitivity (swap/shuffle MAE) but **no** gain vs Base (overfit / domain of 8-image microset).
3. **Bottleneck:** frozen random Match/Flow/DCN Alignment destroys usable A2; Fusion+Texture can use Ref when A2 is co-located `tex['h4']`.
4. **C2 audit:** skipped — Ref utilization already demonstrated under `tex_h4`. No Setting B / long train / Official Test.

## Verdict

**Gate 2.2 PASS** (alignment info-quality diagnosis complete; microset Ref utilization unlocked by bypassing align)

Artifacts:
- `G2_2_align_info_quality.json`
- `gate2_2/per_image_psnr.csv`
- `gate2_2/attention_per_image.json`
- `Gate2_2_report.md`

## Repro

```bash
cd /root/projects/TTSR-lowlight && /root/miniconda3/envs/ttsr/bin/python -m unittest tests.test_v7_river -q
/root/miniconda3/envs/ttsr/bin/python scripts/gate2_2_align_info_quality.py --device cuda
```
