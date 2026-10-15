# TTSR-lowlight v2.1 — H3 Residual Failure Diagnosis

Zero-training diagnosis on frozen **P2 Y0 cache** + **H3@checkpoint_02000**.  
No model changes, no retrain, no production-config edits. GT used only for metrics / attribution.

Artifacts:

- `diagnostics/v2/h3_residual_decomposition.csv`
- `diagnostics/v2/h3_diag_summary.json`
- `diagnostics/v2/h3_diag_2053.json`
- `diagnostics/v2/h3_diag_2053/` (P2 / H3 / GT / D×8 / MeanOnly / Centered / overview)

Script: `scripts/diag_v21_h3_residual.py`

---

## Protocol

- Y0 from `cache_p2_eval` float32 `[-1,1]` (not PNG screenshots).
- Y3 = Y0 + V3B0ResidualFusion(Encoder(Y0), Match(Y0, R_eff)).
- Eval: Huawei Eval30, same RGB PSNR/SSIM as v1/v2 (`calc_psnr_and_ssim`).
- Decomposition: `D = Y3−Y0`, `D_mean = spatial_mean(D)` per RGB, `D_spatial = D−D_mean`.
- Variants: P2, H3, MeanOnly=`Y0+D_mean`, Centered=`Y0+D_spatial`.
- Second-path ablations (P2 Nano still baked into Y0): Correct Nano / Self / Mismatch / Zero-T / Bypass.

---

## 1. Focus: 2053.jpg

| Variant | PSNR | vs P2 |
|---|---|---|
| P2 (Y0) | **24.379** | — |
| H3 (Correct Nano) | 21.520 | **−2.860** |
| MeanOnly | 21.480 | −2.899 |
| Centered | **24.459** | **+0.079** |
| H3 Self | 21.485 | −2.894 |
| H3 Mismatch | 22.343 | −2.036 |
| H3 Zero-T | 25.613 | +1.234 |
| Bypass | 24.379 | 0 |

`D_mean` RGB ([-1,1] scale): **R=+0.061, G=+0.066, B=+0.052** (global brightening / greenish lift).

Residual energy on 2053: **97.3% mean / 2.7% spatial**.  
Frequency: `ratio_low ≈ 0.9999`.

MSE identity check (`ΔMSE = mean(2(Y0−GT)D + D²)`):

| Term | Value |
|---|---|
| ΔMSE (H3−P2) | +0.01360 |
| cross `2(Y0−GT)·D` | +0.00989 (73%) |
| `D²` | +0.00370 (27%) |
| ΔMSE MeanOnly | +0.01386 |
| ΔMSE Centered | **−0.00026** (slightly helpful) |
| identity residual | ~1e−9 |

P2 was already positively biased vs GT on all channels; H3’s positive `D_mean` **amplifies** that bias → cross term dominates the MSE rise.

**Visual:** MeanOnly ≈ H3; Centered ≈ P2 (slightly better). See `diagnostics/v2/h3_diag_2053/2053_overview.png`.

---

## 2. Eval30 aggregates

| Condition | Mean PSNR | Δ vs P2 |
|---|---|---|
| P2 | 20.265 | — |
| **H3 Correct Nano** | **20.466** | **+0.201** |
| MeanOnly | 20.302 | +0.037 |
| **Centered** | **20.419** | **+0.154** |
| H3 Self | 20.253 | −0.012 |
| H3 Mismatch | 19.571 | −0.694 |
| H3 Zero-T | 18.636 | −1.629 |

Mean residual energy share: **e_mean_frac ≈ 0.623**, spatial ≈ 0.377.  
Mean frequency: **ratio_low ≈ 0.999** (residual is essentially low-frequency).

Mean MSE decomposition (H3 vs P2):

| Term | Mean over 30 |
|---|---|
| ΔMSE | −0.00405 (PSNR↑ on average) |
| cross | −0.00604 |
| D² | +0.00199 |
| ΔMSE MeanOnly | −0.00258 |
| ΔMSE Centered | −0.00147 |

### Worst failures (all mean-bias dominated)

| Image | ΔH3 | MeanOnly | Centered | e_mean_frac |
|---|---|---|---|---|
| 2053 | −2.86 | −2.90 | +0.08 | **0.973** |
| 2059 | −1.54 | −1.57 | +0.05 | 0.887 |
| 2055 | −1.44 | −1.48 | +0.06 | 0.846 |
| 2058 | −1.26 | −1.18 | −0.11 | 0.936 |
| 2042 | −1.16 | −1.10 | −0.07 | 0.931 |

### Strong gains (often MeanOnly-led)

| Image | ΔH3 | MeanOnly | Centered | e_mean_frac |
|---|---|---|---|---|
| 2052 | +1.74 | +1.68 | +0.04 | high mean share, **corrective** bias |
| 2057 | +1.68 | +1.64 | +0.03 | same pattern |
| 2044 | +1.51 | +1.28 | +0.17 | mean + some spatial |
| 2061 | +1.37 | +0.93 | +0.35 | mixed |
| 2056 | +1.28 | +1.06 | +0.17 | mixed |

Failures: `D_mean` pushes further from GT. Gains: `D_mean` often **corrects** existing P2 bias (sign opposes P2 signed error).

---

## 3. Direct answers

### Q1. Is 2053’s −2.86 dB mainly whole-image RGB bias?

**Yes.** MeanOnly alone reproduces −2.90 dB; Centered alone is **+0.08 dB**. Residual energy is 97% DC mean. Cross-term with existing P2 positive bias explains most ΔMSE.

### Q2. Does H3’s +0.201 dB depend on correct Nano in the *new* path?

**Yes, for the Eval30 mean.**  

- Correct Nano: **+0.201**  
- Self: **−0.012** (≈P2; no net gain)  
- Mismatch: **−0.694**  
- Zero-T: **−1.629**  

Caveat: this isolates only H3’s second path; P2’s Y0 already used correct Nano. On 2053 specifically, Correct Nano is harmful via bias while Zero-T helps locally — sample-dependent, not a contradiction of the mean.

### Q3. Which residual class contributes most to Eval30 PSNR?

**On average, the zero-mean spatial residual (`Centered`, +0.154 dB) contributes more of the mean gain than MeanOnly (+0.037).**  
But **tail risk / worst drops are almost entirely mean-bias**, and ~62% of residual *energy* is still DC. Frequency-wise the residual is near-pure low-frequency (~99.9%).

Practical reading: H3’s benefit is mostly a **low-frequency correction** that sometimes includes a helpful global shift and sometimes a damaging one; it is **not** high-frequency texture recovery.

---

## 4. Implications (no action taken this round)

- Do **not** describe H3 as recovering fine texture.  
- Optional future (not started): clamp / remove `D_mean` at inference, or train with mean-free residual target — would likely heal 2053-class failures while retaining much of Centered’s gain.  
- Keep P2 as default until such a control is validated.

*Diagnosis only. No training. No production default change.*
