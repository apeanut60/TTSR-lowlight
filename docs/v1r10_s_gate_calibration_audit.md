# v1r10 — S-Gate Calibration Audit

**Branch:** `experiment/v1r10-s-gate-audit`  
**Ckpt (frozen):** `/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt`  
**Texture (frozen M1):** α=0.75, β=0.50, λ=0.25, `T_mix=0.75·T_B2+0.25·T_id`, S source A0 unless noted  
**Scope:** Diagnostic zero-shot only — no training, no GateNet, no α/β/λ changes, production default unchanged.

Artifacts: `diagnostics/v1r10/` · `diagnostics/v1r10/REPRO.md`

---

## Phase A — Code audit

### Gate formula (confirmed)

```text
G = sigmoid( bicubic_upsample(S, size=MainNet_HW) )
```

Same `G` multiplies the **conv-head residual** at all three injection stages (`MainNetEnhance.py`).

### Injection order (each stage)

1. `concat(stream, T_*_up)`  
2. `conv*_head` → residual  
3. `residual *= G`  
4. `stream += gated residual`  
5. Subsequent RBs / CSFI / merge / illum heads  

The gate does **not** multiply raw `T`; it scales the generated residual.

### Read-only gate statistics (M1 / G0, 30 images, tile-averaged)

| Quantity | A0 | B2 |
|---|---|---|
| Raw S mean | ~0.37–0.45 (per-image; see CSV) | higher under align |
| Sigmoid gate mean | **0.590** | **0.618** |
| Gate std | **0.021** | **0.028** |
| Gate spatial CV | **0.036** | **0.046** |
| A0↔B2 gate corr | **0.30** | |
| A0↔B2 gate MAD | **0.038** | |

**Finding:** Under this checkpoint the A0 gate is **nearly spatially flat** (CV ≈ 3.6%). Spatial modulation amplitude is tiny relative to the mean strength (~0.59).

JSON: `diagnostics/v1r10/gate_distribution_stats.json`, `code_audit.json`.

---

## Correctness

`python tests/test_v1r10_s_gate_audit.py` → **ALL PASS**  
(`diagnostics/v1r10/correctness_tests.json`)

| # | Check | Result |
|---|---|---|
| 1 | G0 = v1r9 M1 | **20.1805 / 0.6327** vs target 20.181 / 0.6327 (Δ&lt;0.02) |
| 2 | Identical T across G0/G1 | T_mix independent of `s_source` |
| 3 | MainNet←Low | pass |
| 4 | Gate replaced after sigmoid(upsample) | pass |
| 5–8 | G2/G3/G4 mean/var/distribution | pass |
| 9–14 | Finite [0,1], ckpt unchanged, tiles, full-image γ/LUT, default off, no GT | pass |

---

## Phase B — Counterfactual results

All runs share identical M1 `T_lv*`. Only `G` (or S→G source) changes.

| Run | Gate | PSNR | SSIM | Δ vs G0 | ↑/↓ vs G0 (±0.01) |
|---|---|---|---|---|---|
| **G0** | Original A0 | **20.181** | **0.6327** | — | — |
| **G1** | B2 | 20.142 | 0.6322 | **−0.038** | 9 / 18 |
| **G2** | Constant mean(A0) | 20.189 | 0.6328 | **+0.009** | 15 / 5 |
| **G3** | Smooth A0 (k=31, mean-preserve) | 20.171 | 0.6328 | −0.009 | 12 / 18 |
| **G4a** | Shift (32,0) | 20.184 | 0.6327 | +0.003 | 6 / 6 |
| **G4b** | Shift (0,32) | 20.181 | 0.6327 | +0.001 | 4 / 6 |
| **G4c** | Shift (24,24) | 20.182 | 0.6327 | +0.002 | 6 / 6 |
| G4 agg. | mean of shifts | 20.182 ± 0.001 | — | +0.002 | — |

Paired bootstrap (exploratory, n=30, images reused in design):

| Run | Mean Δ | Median Δ | Std | 95% CI |
|---|---|---|---|---|
| G1 | −0.038 | −0.056 | 0.173 | [−0.098, +0.025] |
| G2 | +0.009 | +0.010 | 0.019 | [+0.002, +0.015] |
| G3 | −0.009 | −0.033 | 0.102 | [−0.045, +0.028] |

Threshold: **&lt;0.02 dB is not labeled a meaningful gain.**

Per-image CSV: `diagnostics/v1r10/per_image_metrics.csv`.

### Focus PSNR

| File | G0 | G1 | G2 | G3 | G4a |
|---|---|---|---|---|---|
| `2037.jpg` | 20.427 | 20.175 | **20.466** | 20.249 | 20.442 |
| `2041.jpg` | 23.348 | 22.929 | **23.414** | 23.219 | 23.388 |
| `2053.jpg` | 25.178 | 25.291 | 25.161 | **25.299** | 25.164 |
| `2056.jpg` | 17.209 | 17.218 | 17.207 | 17.242 | 17.207 |
| `2062.jpg` | 17.459 | **17.686** | 17.499 | 17.585 | 17.512 |
| `2063.jpg` | 25.533 | **26.035** | 25.548 | 25.841 | 25.508 |

Viz: `diagnostics/v1r10/focus/`, stage residual maps: `diagnostics/v1r10/stage_maps/`.

On `2037` / `2053`, alternate gates do **not** close the remaining gap to A0 F0 (~21.2 / ~26.4). Changes are small texture-strength tweaks, not photometric rescues.

---

## Decision rules

| Case | Evidence | Applies? |
|---|---|---|
| **A** G2 ≈ G0 | Δ=+0.009 &lt; 0.02; gate CV only 0.036 | **Yes** — spatial variation adds little |
| **B** G2 ≫ G0 | No meaningful gain | No |
| **C** G0 ≫ G2 and G4 | G0 ≉ better than G2/G4 | No (G4 ≈ G0) |
| **D** G3 ≫ G0 | G3 slightly worse | No |
| **E** All close | G2/G3/G4 within ~0.01 of G0; only G1 clearly hurts | **Yes** |

**Overlap A+E:** The soft-attention map is already almost constant. Replacing it with its mean is nearly a no-op; displacing it barely matters; smoothing does not help. Switching to **B2 G (G1) hurts** — that is a confidence-*level* effect (higher mean gate ≈ 0.62 vs 0.59), not a spatial-structure effect.

---

## Go / No-Go

### **No-Go on gate redesign**

Do **not** introduce a new GateNet, constant-gate production change, or smoothing pass.

**Retain v1r9 M1** (A0 S + fused T, λ=0.25) as the best documented zero-shot optional configuration. Keep production default = original path (`texture_fusion=False`, `gate_mode=None`).

---

## One recommended next step (not executed)

**Stop S-gate modification.** If pursuing further PSNR beyond ~20.18, choose a lever **outside** SearchTransfer soft-attention (e.g. MainNet illumination head / data), not another gate counterfactual.

---

## Deliverables

| Item | Path |
|---|---|
| Report | `docs/v1r10_s_gate_calibration_audit.md` |
| Metrics | `diagnostics/v1r10/per_image_metrics.csv` |
| Summary + bootstrap | `diagnostics/v1r10/summary.json` |
| Gate distributions | `diagnostics/v1r10/gate_distribution_stats.json` |
| Code audit | `diagnostics/v1r10/code_audit.json` |
| Tests | `tests/test_v1r10_s_gate_audit.py`, `correctness_tests.json` |
| Focus viz | `diagnostics/v1r10/focus/` |
| Eval script | `scripts/run_v1r10_s_gate_audit.py` |
