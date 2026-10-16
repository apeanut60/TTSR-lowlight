# v1r14 — Independent Generalization Validation

**Verdict:** **Outcome D** (cross-domain Nikon only) + **do not promote P2**. Same-domain Huawei independent validation is **blocked by data availability**. On Nikon Eval20, mean RGB PSNR for P2 is below both A0 and M1; gains are sample-dependent and unstable.

Checkpoint frozen: `model_00004.pt` (v1r3e4 Huawei Nano). No training, architecture, gate, or α/β/λ changes.

---

## 1. Primary question

> Does lv2 Branch-Bypass (P2) consistently improve RGB PSNR over A0 and M1 on independent data?

**Answer on available independent data:** **No** (Nikon Eval20 cross-domain). Mean P2 < mean A0 and mean M1. Same-domain Huawei generalization remains **unvalidated** (no eligible holdout).

---

## 2. Dataset provenance (Phase A)

Full audit: `diagnostics/v1r14/dataset_audit.md`, `dataset_provenance.csv` (537 rows).

| Pool | Role |
|---|---|
| Huawei Eval30 | Method-selection set for v1r6–v1r13 — **regression only** |
| Huawei man250 / man271 | Checkpoint training |
| Huawei Training Nano leftovers (216) | SHA≠train/Eval30 but live under Training — **excluded** (not a declared holdout) |
| Nikon Eval20 + `nanobanana_ref_Nikon` | **Eligible Priority-2 cross-domain** |

Exact SHA1(1MB) collisions between leftovers and man271/Eval30: **0**. Filename differences alone do not prove scene independence; leftovers were still excluded.

---

## 3. Eligibility decision (Phase B)

| Priority | Status |
|---|---|
| 1 — Independent Huawei Nano Eval holdout | **Unavailable** → same-domain validation blocked |
| 2 — Independent Nikon Nano Eval | **Available (n=20)** → evaluated as **cross-domain** |

Artifacts: `eligible_samples.csv`, `exclusion_reasons.md`, `eligibility_decision.json`.

Huawei Training leftovers were **not** used as independent test data.

---

## 4. Frozen configurations (Phase C)

| ID | Description |
|---|---|
| **A0** | Original checkpoint + SearchTransfer + original S; no Gamma/Hist/Identity; all stages normal |
| **M1** | α=0.75, β=0.50, λ=0.25, `T_mix=0.75 T_B2 + 0.25 T_id`, gate=A0 S; stages normal |
| **P2** | M1 + **lv2 `branch_bypass`** (skip `conv22_head` residual only); lv3/lv1 normal |

Not used: v1r10 G2, v1r11 lv3 Texture-Zero, P3 combo.

---

## 5. Pre-evaluation correctness (Phase D)

Historical **Huawei Eval30** (not in independent aggregates):

| Run | Target PSNR / SSIM | Got | Pass |
|---|---|---|---|
| A0 | 20.162 / 0.6324 | 20.1621 / 0.6324 | ✓ |
| M1 | 20.1805 / 0.6327 | 20.1805 / 0.6327 | ✓ |
| P2 | 20.264 / 0.6341 | 20.2641 / 0.6341 | ✓ |

Unit tests (`tests/test_v1r14_independent_validation.py`): P2 skips `conv22_head`; M1 and P2 share identical T/S tensors.

Protocol: `ref_degrade=False`, no mean-align, tile 256 / overlap 96, full-image Gamma/LUT shared across tiles, Nano resized per TestSet semantics.

---

## 6. Independent results — Nikon Eval20 (Phase E)

**Label: CROSS-DOMAIN.** Do not treat as same-domain Huawei confirmation.

Domain snapshot (`nikon_domain_stats.json`): Low/GT all **960×640**; Nano refs all **1264×843** (resized by loader). Absolute level much lower than Huawei Eval30 (~16.7 vs ~20.2 PSNR).

| Run | Mean RGB PSNR | Mean SSIM | Luma MAE | Luma bias | Chroma MAE |
|---|---|---|---|---|---|
| A0 | **16.716** | 0.5086 | 0.1182 | −0.0205 | 0.0097 |
| M1 | 16.702 | 0.5085 | 0.1189 | −0.0243 | 0.0097 |
| P2 | 16.688 | **0.5098** | 0.1193 | −0.0118 | 0.0102 |

Primary summary = mean of per-image RGB PSNR (`per_image_metrics.csv`).

---

## 7. Paired comparisons (Phase F)

From `paired_analysis.json` (n=20; exploratory bootstrap; possible scene clustering):

### M1 − A0 (does texture fusion generalize?)

| Statistic | PSNR |
|---|---|
| Mean | **−0.014** |
| Median | +0.013 |
| Std | 0.171 |
| 10% trimmed mean | −0.012 |
| Mean excl. top-2 positive | −0.047 |
| Wins / losses / ties (|Δ|>0.01) | 11 / 9 / 0 |
| Bootstrap 95% CI | [−0.090, +0.061] |
| LOO mean range | 0.037 |

Best gains: 3013 (+0.37), 3006 (+0.19), 3011 (+0.19).  
Worst losses: 3018 (−0.34), 3010 (−0.32), 3019 (−0.23).

**Interpretation:** No consistent M1 advantage on Nikon; mean slightly negative, median slightly positive.

### P2 − M1 (does lv2 bypass help the fused model?)

| Statistic | PSNR |
|---|---|
| Mean | **−0.014** |
| Median | +0.129 |
| Std | **0.405** |
| 10% trimmed mean | +0.001 |
| Mean excl. top-2 positive | −0.075 |
| Wins / losses | 11 / 9 |
| Bootstrap 95% CI | [−0.183, +0.167] |

Best: 3018 (+0.55), 3017 (+0.52), 3019 (+0.43).  
Worst: 3006 (−0.69), 3001 (−0.68), 3016 (−0.48).

**Interpretation:** Highly sample-dependent; mean negative once top gains removed.

### P2 − A0 (complete candidate vs original)

| Statistic | PSNR |
|---|---|
| Mean | **−0.028** |
| Median | +0.078 |
| Std | 0.298 |
| 10% trimmed mean | −0.012 |
| Mean excl. top-2 positive | −0.071 |
| Wins / losses | 11 / 9 |
| Bootstrap 95% CI | [−0.156, +0.101] |
| LOO mean range | 0.052 |

Best: 3017 (+0.40), 3007 (+0.33), 3012 (+0.31).  
Worst: 3001 (−0.59), 3006 (−0.50), 3016 (−0.45).

SSIM: P2 − A0 mean **+0.0012** (small; all |Δ|≤0.01 → 20 ties under the 0.01 threshold). No large SSIM trade-off, but also no meaningful SSIM rescue of the PSNR mean loss.

---

## 8. Failure / domain notes (Phase G)

Focus panels: `diagnostics/v1r14/focus/NikonEval20_cross_domain/`  
Metric notes: `focus_failure_notes.json`

| Image | Role | P2−A0 PSNR | Metric-linked observations |
|---|---|---|---|
| 3001 | worst P2−A0 / P2−M1 | −0.59 / −0.68 | Higher \|luma bias\| and luma MAE vs A0 |
| 3006 | large P2 loss (M1 had helped A0) | −0.50 / −0.69 | Higher luma MAE; M1 gain reversed by bypass |
| 3007 | P2 gain | +0.33 | Lower \|luma bias\| vs A0; chroma MAE up slightly |
| 3017 | best P2−A0 | +0.40 | Luma MAE down; overall still mid-teens PSNR |
| 3018 | best P2−M1 | +0.21 / +0.55 | M1 had hurt A0; bypass recovers |

Do **not** infer causal texture/geometry mechanisms from PSNR alone. Nano geometry (1264×843 → Eval resize) and device shift are uncontrolled domain factors. No tuning on Nikon was performed.

---

## 9. Decision rules (Phase H)

| Outcome | Applies? |
|---|---|
| A — Positive same-domain Huawei validation | **No** — no eligible Huawei holdout |
| B — Mean↑ but unstable | **N/A for Huawei**; on Nikon mean does **not** rise |
| C — P2 does not beat A0/M1 | **Yes on Nikon** (mean PSNR) |
| **D — Only cross-domain Nikon** | **Yes — primary label** |
| E — No independent data | Partial: Priority-1 blocked; Priority-2 used |

### Recommendation

**Reject automatic promotion of P2.** Retain original checkpoint **A0** behavior as the production default until a curated independent **Huawei** Low/GT/Nano holdout can re-test same-domain generalization.

Cross-domain evidence: P2’s Eval30 +0.10 dB edge over A0 **did not transfer** as a mean gain on Nikon Eval20. Median P2−A0 is positive but fragile (negative after dropping top-2 contributors; wide CI crossing zero).

---

## 10. Deliverables

| Path | Content |
|---|---|
| `docs/v1r14_independent_validation_report.md` | This report |
| `diagnostics/v1r14/dataset_provenance.csv` | Full provenance table |
| `diagnostics/v1r14/dataset_audit.md` | Audit narrative |
| `diagnostics/v1r14/eligible_samples.csv` | Eligibility list |
| `diagnostics/v1r14/exclusion_reasons.md` | Explicit exclusions |
| `diagnostics/v1r14/per_image_metrics.csv` | Eval30 regression + Nikon metrics |
| `diagnostics/v1r14/paired_analysis.json` | Paired stats |
| `diagnostics/v1r14/summary.json` | Machine-readable summary |
| `diagnostics/v1r14/nikon_domain_stats.json` | Resolution / domain snapshot |
| `diagnostics/v1r14/focus/` | Extreme-case panels |
| `diagnostics/v1r14/predictions/NikonEval20_cross_domain/` | A0/M1/P2 PNGs |
| `diagnostics/v1r14/REPRO.md` | Commands |
| `scripts/run_v1r14_independent_validation.py` | Eval driver |
| `tests/test_v1r14_independent_validation.py` | Wiring / T–S / bypass tests |

---

## 11. What would unlock Outcome A/C on Huawei

A declared held-out Huawei Eval (or holdout) split with Low/GT/Nano that:

1. Was never in man250/man271 or FT,
2. Was never used for v1r6–v1r13 selection,
3. Has documented scene de-duplication vs Eval30,
4. Uses Nano refs not derived from GT.

Until then, **same-domain generalization of P2 remains unvalidated**.

---

*v1r15 not started.*
