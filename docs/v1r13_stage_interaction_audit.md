# v1r13 — Multi-Stage Texture Intervention Interaction Audit

**Branch:** `experiment/v1r13-stage-interaction-audit`  
**Ckpt:** v1r3e4 · **Texture:** v1r9 M1 · **Gate:** original A0 S  
**Interventions (reused v1r12 API):** lv3 `texture_zero` · lv2 `branch_bypass` · lv1 `normal`  
**Scope:** Zero-shot only. No training, sweeps, or production default changes.

Artifacts: `diagnostics/v1r13/` · `REPRO.md`

---

## Phase A — Intervention audit

| Run | lv3 | lv2 | lv1 | ≡ v1r12 |
|---|---|---|---|---|
| P0 | normal | normal | normal | C0 |
| P1 | texture_zero | normal | normal | C1 |
| P2 | normal | branch_bypass | normal | C4 |
| P3 | texture_zero | branch_bypass | normal | new |

- **lv3 Texture-Zero:** `F3 ← F3 + G·R(F3, 0)` (branch active).  
- **lv2 Bypass:** skip only `conv22_head` residual; RB/CSFI/merge unchanged.  
- **Live forward:** stage-2 receives features after modified stage-1 (verified in tests).

`code_audit.json`.

---

## Correctness

`tests/test_v1r13_stage_interaction.py` → **ALL PASS**  
(P0–P3 distinct; P3 ≠ P1/P2; conv11 runs / conv22 skipped / conv33 runs; S/T identical; params unchanged.)

| Run | Target | Got | OK |
|---|---|---|---|
| P0 | 20.1805 / 0.6327 | 20.1805 / 0.6327 | ✓ |
| P1 | ~20.234 | 20.234 | ✓ |
| P2 | ~20.264 | 20.264 | ✓ |

---

## Four-run results

| Run | PSNR | SSIM | vs A0 20.162 | vs P0 |
|---|---|---|---|---|
| **P0** | 20.181 | 0.6327 | +0.018 | — |
| **P1** | 20.234 | 0.6324 | +0.072 | +0.054 |
| **P2** | **20.264** | **0.6341** | **+0.102** | **+0.084** |
| **P3** | 20.220 | 0.6337 | +0.058 | +0.040 |

### Paired comparisons (n=30, ±0.01 win/loss)

| Contrast | Mean | Median | W/L/T | Mean excl. top-2 + | Exploratory 95% CI |
|---|---|---|---|---|---|
| P1−P0 | +0.054 | **−0.046** | 13/16/1 | −0.095 | (from v1r12 pattern) |
| P2−P0 | +0.084 | **+0.194** | 20/9/1 | +0.037 | — |
| **P3−P2** | **−0.044** | **−0.043** | **12/18/0** | **−0.173** | [−0.32, +0.26] |
| P3−P1 | −0.014 | +0.074 | 19/11/0 | −0.054 | — |
| P3−P0 | +0.040 | −0.054 | 15/15/0 | −0.136 | — |

**Central decision metric:** P3 does **not** improve on P2.

Top-2 positives for P3−P2 (`2040` +1.85, `2037` +1.68) dominate the sum; after excluding them, mean Δ = **−0.173**.

CSV: `per_image_metrics.csv` · `robustness_analysis.json`.

---

## Interaction

`I = PSNR(P3) − PSNR(P1) − PSNR(P2) + PSNR(P0)`

| | |
|---|---|
| Mean I | **−0.097** |
| Median I | −0.044 |
| I from run means | −0.097 |

**Negative interaction:** combining the two interventions yields **less** than the sum of their separate mean effects. P3 (20.220) sits **between** P1 and P2 and below P2.

`interaction_analysis.json`.

---

## Focus images

| File | P0 | P1 | P2 | P3 | P3−P2 | I |
|---|---|---|---|---|---|---|
| `2037.jpg` | 20.427 | 22.639 | 21.101 | **22.777** | +1.676 | −0.535 |
| `2040.jpg` | 20.448 | 22.501 | 21.237 | **23.091** | +1.855 | −0.199 |
| `2053.jpg` | 25.178 | 26.086 | 24.379 | 25.289 | +0.909 | +0.002 |
| `2041.jpg` | 23.348 | 23.751 | **23.816** | 23.802 | −0.014 | −0.417 |
| `2063.jpg` | **25.533** | 24.361 | 24.791 | 23.795 | **−0.996** | +0.177 |
| `2056.jpg` | 17.209 | 16.109 | **17.287** | 16.149 | −1.138 | −0.038 |
| `2062.jpg` | **17.459** | 16.201 | 16.663 | 15.524 | −1.139 | +0.119 |

lv2 Bypass does **not** neutralize lv3 T-zero failures (`2063`, `2056`, `2062` still worse under P3 than under P2). On winners (`2037`/`2040`), P3 amplifies P1 but that does not lift the Eval mean above P2.

Viz: `diagnostics/v1r13/focus/`.

---

## Decision

### **Case D — P3 is worse than P2**

- P3−P2 mean **−0.044**, median **−0.043**, 18 losses vs 12 wins.  
- Negative interaction I ≈ −0.10.  
- Robustness stress (drop top-2) makes P3−P2 even worse.

**Recommendation:** **Reject the combination.** Retain **P2 (lv2 branch-bypass only)** as the primary optional inference candidate from this ablation series. Do not add lv3 Texture-Zero on top of it by default. Do not train a stage selector / GateNet from these results.

---

## One recommended next step (not started)

**Hold P2 for optional inference documentation / independent-set check only** — freeze `stage_mode_lv2=branch_bypass` with M1 otherwise unchanged, and evaluate on a **non-Eval30** Huawei holdout or Nikon split if available. Do not stack lv3 T-zero. Do not start v1r14 automatically.

---

## Deliverables

| Item | Path |
|---|---|
| Report | `docs/v1r13_stage_interaction_audit.md` |
| Metrics | `diagnostics/v1r13/per_image_metrics.csv` |
| Interaction | `diagnostics/v1r13/interaction_analysis.json` |
| Robustness | `diagnostics/v1r13/robustness_analysis.json` |
| Tests / script | `tests/test_v1r13_stage_interaction.py`, `scripts/run_v1r13_stage_interaction.py` |
