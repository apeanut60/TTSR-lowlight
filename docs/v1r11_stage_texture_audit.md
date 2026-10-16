# v1r11 — Stage-wise Texture Injection Audit

**Branch:** `experiment/v1r11-stage-texture-audit`  
**Ckpt:** v1r3e4 (frozen)  
**Texture / gate:** v1r9 M1 — `T_mix=0.75·T_B2+0.25·T_id`, S=A0, α=0.75, β=0.50, λ=0.25  
**Scope:** Zero-shot diagnostic only. No training, no S-gate / γ / hist / λ changes, production defaults unchanged.

Artifacts: `diagnostics/v1r11/` · `diagnostics/v1r11/REPRO.md`

---

## Phase 1 — Injection audit

| Stage | Stream | T native → up | Residual head | Gate | Add |
|---|---|---|---|---|---|
| **lv3 / stage1** | `x11` ← SFE | `T_lv3` [N,C,H/4,W/4] → bicubic H×W | `conv11_head(concat(x11,T_up))` | `G=sigmoid(↑S)` × residual | `x11 += gated` |
| **lv2 / stage2** | `x22` ← SFE | `T_lv2` [N,C,H/2,W/2] → H×W | `conv22_head` | same G | `x22 += gated` |
| **lv1 / stage3** | `x33` ← SFE | `T_lv1` [N,C,H,W] → H×W | `conv33_head` | same G | `x33 += gated` |

Feature streams after SFE are all `[N, n_feats, H, W]` (Low resolution).  
**Intervention:** `disable_T_lv*` zeros `T_*_up` **before** concat / residual generation (not after the gate).

JSON: `diagnostics/v1r11/code_audit.json`.

---

## Phase 2–3 — Results

S0 reproduces M1: **20.1805 / 0.6327** (target 20.181).  
All single-stage disables exceeded |Δ|=0.05 for at least one stage → Phase 3 ran.

| Run | Enabled T | PSNR | SSIM | Δ vs S0 |
|---|---|---|---|---|
| **S0** | lv3+lv2+lv1 | **20.181** | **0.6327** | — |
| **S1** | lv2+lv1 (no lv3) | **20.234** | 0.6324 | **+0.054** |
| **S2** | lv3+lv1 (no lv2) | 20.212 | **0.6334** | +0.032 |
| **S3** | lv3+lv2 (no lv1) | 20.145 | 0.6307 | **−0.035** |
| **S4** | lv3 only | 20.182 | 0.6316 | +0.002 |
| **S5** | lv2 only | 20.168 | 0.6302 | −0.013 |
| **S6** | lv1 only | 20.190 | 0.6318 | +0.009 |
| S7≡S3 | lv3+lv2 | 20.145 | — | −0.035 |
| S8≡S2 | lv3+lv1 | 20.212 | — | +0.032 |
| S9≡S1 | lv2+lv1 | 20.234 | — | +0.054 |

Leave-one-out style contribution (approx; not additive):

| Stage | Δ when disabled | Approx contrib to S0 |
|---|---|---|
| **lv3** | **+0.054** | **−0.054** (net harmful) |
| **lv2** | +0.032 | −0.032 (net mildly harmful) |
| **lv1** | −0.035 | **+0.035** (net helpful) |

CSV: `diagnostics/v1r11/per_image_metrics.csv` · `summary.json`.

### Focus PSNR

| File | S0 | S1 (¬lv3) | S2 (¬lv2) | S3 (¬lv1) | S4 lv3 | S5 lv2 | S6 lv1 |
|---|---|---|---|---|---|---|---|
| `2037.jpg` | 20.427 | **22.639** | 20.340 | 20.724 | 20.637 | **22.724** | 22.637 |
| `2053.jpg` | 25.178 | 26.086 | 25.503 | 25.044 | 25.385 | 25.865 | **26.289** |
| `2041.jpg` | 23.348 | 23.751 | 23.708 | 23.421 | **23.806** | 23.766 | 23.618 |
| `2063.jpg` | **25.533** | 24.361 | 25.810 | 25.018 | 25.256 | 24.008 | 24.504 |

Viz: `diagnostics/v1r11/focus/`, residual maps: `stage_maps/`.

---

## Interpretation

**1. Which stage provides positive PSNR contribution?**  
**lv1 (fine)** — disabling it drops mean PSNR by 0.035 dB. lv1-only (S6) stays near S0 (+0.009).

**2. Which stage causes negative transfer?**  
**lv3 (coarse)** on average — disabling it **raises** mean PSNR by **0.054** to **20.234**. lv2 is mildly negative (+0.032 when removed). Not universal: on `2063.jpg`, removing lv3 **hurts** (−1.17 dB).

**3. Are failures one scale or global misuse?**  
**Scale-dependent and image-dependent.** Hard M1 losers (`2037`, `2053`) improve sharply when lv3 is removed or when only fine/mid textures remain. `2063` (M1 winner) needs coarse texture. Not pure global Ref misuse.

**4. Does M1 improvement depend on a specific level?**  
M1’s modest +0.018 vs A0 is **not** driven by lv3; under M1, **coarse injection is the liability**. Best zero-shot ablation in this suite is **S1 / S9 (no lv3): 20.234**.

---

## Recommended next experiment (not started)

**Eval-only “M1 − lv3”:** keep frozen M1 T/S/γ/hist, set `disable_T_lv3=True` (lv2+lv1 only), confirm 20.234 on Huawei Nano, and document as optional inference mode — still no training. If promoting further, consider image-adaptive coarse gating later; do not change production default yet.

---

## Constraints honored

No training · no S-gate / α / β / λ changes · no architecture modules · no ckpt overwrite · defaults `disable_T_lv*=False`.
