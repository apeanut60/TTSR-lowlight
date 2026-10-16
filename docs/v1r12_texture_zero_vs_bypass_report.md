# v1r12 — Texture-Zero vs Injection-Bypass Audit

**Branch:** `experiment/v1r12-texture-zero-vs-bypass`  
**Ckpt:** v1r3e4 · **Texture:** v1r9 M1 (α=0.75, β=0.50, λ=0.25) · **Gate:** original A0 S (not G2)  
**Scope:** Zero-shot diagnostic. No training, no new modules, production defaults unchanged.

Artifacts: `diagnostics/v1r12/` · `REPRO.md`

---

## Phase A — Code audit

| Texture | Stage | Head | T channels | Stream | Operation |
|---|---|---|---|---|---|
| T_lv3 | 1 | `conv11_head` | 256 | `x11`←SFE | `x11 += G·R(x11,T)` |
| T_lv2 | 2 | `conv22_head` | 128 | `x22`←SFE | `x22 += G·R(x22,T)` |
| T_lv1 | 3 | `conv33_head` | 64 | `x33`←SFE | `x33 += G·R(x33,T)` |

```text
R(F,T) = conv_head(concat(F, T_up))     # T_up = bicubic↑ to Low HxW
gated  = R * G                            # G = sigmoid(bicubic↑ S)
F ← F + gated
```

**Math distinction (confirmed):**

- **Texture-Zero:** `F_zero = F + G·R(F, 0)` — branch active; T replaced by zeros.  
- **Branch-Bypass:** `F_bypass = F` — no `conv_head`, no add.

`R(F,0) ≠ 0`: stream channels + **bias** in `conv*_head` still produce a residual. Local decomp on C0: `‖R_zero‖/‖R_normal‖ ≈ 0.65–0.69` (L2), `cos(R_n,R_z)≈0.51–0.64`, `R_n = R_z + R_texture` (recon err = 0).

`code_audit.json`.

---

## Correctness / regression

Tests: `tests/test_v1r12_texture_bypass.py` → **ALL PASS**.

| Run | Target | Got | OK |
|---|---|---|---|
| C0 | 20.1805 / 0.6327 | 20.1805 / 0.6327 | ✓ |
| C1 | 20.234 | 20.234 | ✓ |
| C3 | 20.212 | 20.212 | ✓ |
| C5 | 20.145 | 20.145 | ✓ |

---

## Seven-run results

| Run | lv3 | lv2 | lv1 | PSNR | SSIM | Δ vs C0 | ↑/↓ |
|---|---|---|---|---|---|---|---|
| **C0** | normal | normal | normal | **20.181** | **0.6327** | — | — |
| **C1** | **T-zero** | normal | normal | **20.234** | 0.6324 | **+0.054** | 13/16 |
| **C2** | **bypass** | normal | normal | 19.959 | 0.6296 | **−0.222** | 14/16 |
| **C3** | normal | **T-zero** | normal | 20.212 | 0.6334 | +0.032 | 13/16 |
| **C4** | normal | **bypass** | normal | **20.264** | **0.6341** | **+0.084** | 20/9 |
| **C5** | normal | normal | **T-zero** | 20.145 | 0.6307 | −0.035 | 13/16 |
| **C6** | normal | normal | **bypass** | 19.770 | 0.6191 | **−0.411** | 14/16 |

### Texture-Zero − Bypass (isolates `R(F,0)`)

| Contrast | Mean ΔPSNR | Median | Exploratory 95% CI |
|---|---|---|---|
| C1−C2 (lv3) | **+0.275** | +0.182 | [−0.06, +0.63] |
| C3−C4 (lv2) | −0.052 | −0.140 | [−0.20, +0.11] |
| C5−C6 (lv1) | **+0.376** | +0.062 | [−0.07, +0.84] |

Positive C1−C2 / C5−C6 ⇒ retaining stream-only residual beats killing the branch.

### C0 local residual decomposition (mean over 30, tile-avg)

| Stage | ‖R_z‖/‖R_n‖ | ‖R_t‖/‖R_n‖ | cos(R_n,R_z) |
|---|---|---|---|
| lv3 | 0.657 | 0.749 | 0.639 |
| lv2 | 0.650 | 0.853 | 0.512 |
| lv1 | 0.688 | 0.809 | 0.591 |

---

## Focus PSNR

| File | C0 | C1 T0 | C2 byp | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|---|
| `2037.jpg` | 20.427 | **22.639** | 22.440 | 20.340 | 21.101 | 20.724 | 20.664 |
| `2053.jpg` | 25.178 | 26.086 | **27.311** | 25.503 | 24.379 | 25.044 | 21.511 |
| `2041.jpg` | 23.348 | 23.751 | 20.778 | 23.708 | 23.816 | 23.421 | 23.170 |
| `2063.jpg` | 25.533 | 24.361 | 25.619 | 25.810 | 24.791 | 25.018 | 21.954 |
| `2056.jpg` | 17.209 | 16.109 | 15.944 | 16.773 | 17.287 | 17.051 | 17.661 |
| `2062.jpg` | 17.459 | 16.201 | 14.771 | 17.371 | 16.663 | 17.437 | 17.598 |

C1 mean gain is **not robust**: median Δ(C1−C0)=**−0.046**; top gains `2037` (+2.21), `2040` (+2.05). `2063` loses −1.17 under lv3 T-zero.

Viz: `diagnostics/v1r12/focus/`, residual maps: `residual_maps/`.

---

## Interpretation

**Q1 — Is lv3 texture harmful?**  
On mean, **yes** (C1 > C0 by +0.054), but that is the **texture-dependent** part. Keeping `R(F,0)` matters: C1 ≫ C2 (+0.275).

**Q2 — Is the entire lv3 injection residual harmful?**  
**No.** Full bypass (C2) **drops** mean PSNR by 0.22. Removing the branch is worse than zeroing T.

**Q3 — Is R(F,0) beneficial?**  
**lv3 & lv1: yes** (C1>C2, C5>C6). **lv2: weak opposite** (C4 ≳ C3); mid-stage stream-only residual is less valuable / slightly harmful vs bypass on this set.

**Q4 — Stage differences?**  
- **Coarse:** T often harmful; branch still useful.  
- **Mid:** bypass can beat both C0 and T-zero (C4 best mean **20.264**).  
- **Fine:** T helpful; bypass catastrophic (−0.41).

**Q5 — Robustness?**  
lv3 T-zero mean↑ / median↓ — **driven by a few large winners**; many images regress. Do not treat +0.05 dB as stable.

---

## Decision rules

| Case | Evidence | Applies |
|---|---|---|
| **A** lv3 T-zero > Bypass | C1 20.234 ≫ C2 19.959 | **Yes** |
| **B** Bypass > T-zero | lv2: C4 > C3 | Mild yes for lv2 only |
| **C** T-zero ≈ Bypass | — | No for lv3/lv1 |
| **D** Bypass/T-zero hurts positives | `2063` under C1; many C2 losses | **Yes** |
| **E** no convincing gain | — | No for closing the direction |

**Overlap A+D for lv3:** Prefer suppressing **texture-dependent** coarse residual while **keeping** `R(F,0)` / the branch — not a permanent full bypass, and not a new GateNet without further evidence.

---

## One recommended next experiment (not started)

**Eval-only “lv3 texture-null residual”:** keep M1 and A0 S; at stage1 replace injection with `G·R(F,0)` only (≡ C1), optionally combine with lv2 bypass (C4-style) in a **single** frozen combo check — still zero-shot, no training. Do **not** default-disable the lv3 branch.

---

## Deliverables

| Item | Path |
|---|---|
| Report | `docs/v1r12_texture_zero_vs_bypass_report.md` |
| Metrics | `diagnostics/v1r12/per_image_metrics.csv` |
| Summary | `diagnostics/v1r12/summary.json` |
| Audit / tests | `code_audit.json`, `correctness_tests.json` |
| Scripts | `scripts/run_v1r12_texture_bypass_audit.py`, `tests/test_v1r12_texture_bypass.py` |
