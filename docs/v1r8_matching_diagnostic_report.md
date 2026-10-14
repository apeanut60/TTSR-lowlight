# v1r8 — Gamma / Histogram Matching Diagnostic

**Branch:** `experiment/v1r8-matching-diagnostic`  
**Baseline ckpt (frozen, not modified):**  
`/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt` (v1r3e4)  
**Eval:** Huawei Nano Ref, 30 images, production RGB PSNR/SSIM (tiled), `ref_degrade=False`, `eval_mean_align=False`  
**Scope:** Diagnostic only — no training, no α/β sweep, no architecture changes, GT features offline only.

Artifacts: `diagnostics/v1r8/` (see `diagnostics/v1r8/REPRO.md`).

---

## 0. Index → filename (Eval order)

| Group | Index | Filename |
|---|---|---|
| Neg | 0 | `2037.jpg` |
| Neg | 16 | `2053.jpg` |
| Neg | 4 | `2041.jpg` |
| Pos | 25 | `2062.jpg` |
| Pos | 26 | `2063.jpg` |
| Pos | 19 | `2056.jpg` |

Full map: `diagnostics/v1r8/index_to_filename.json`.

---

## 1. MainNet / SearchTransfer mechanics (confirmed in source)

`TTSREnhance` uses `MainNetEnhance` (not 4× `MainNet`).

- **Query-only path:** γ / γ+hist alter `lr_sr` → LTE Query. MainNet still receives **original Low**. Key/Value from Nano Ref unchanged. Never GT.
- **SearchTransfer:** dense cosine `R_lv3`, hard argmax index + raw max similarity `S` (folded).
- **MainNetEnhance soft-gate** (`model/MainNetEnhance.py`):

```343:350:model/MainNetEnhance.py
        S_up = F.interpolate(S, size=(H, W), mode='bicubic')
        S_up = torch.sigmoid(S_up)  # [0,1] soft-gate, consistent with tpl_loss
        ...
        x11_res = x11_res * S_up
```

Same `sigmoid(S)` multiplies residual texture injection at stages 1–3 (`T_lv3` / `T_lv2` / `T_lv1`). Mean `sigmoid(S)` on Eval is ~0.62–0.63 (not near 0/1 saturation).

**Implication:** Changing Query shifts both **where** Value is copied (`match_arg`) and **how hard** it is gated (`S` → sigmoid). MainNet was trained under A0 Query statistics.

---

## 2. Phase A — Per-image visual diagnostics

For each focus file, contact sheet includes: Low, Nano Ref, GT, Query γ, Query γ+hist, A0/A3/B2 preds, |RGB| error maps (shared vmax), luma histograms, displacement heatmaps (shared scale), plus separate match-arrow viz with **identical Query sample coordinates** across methods.

| File | Contact sheet |
|---|---|
| `2037.jpg` | `diagnostics/v1r8/contact_sheets/00_2037.png` |
| `2041.jpg` | `diagnostics/v1r8/contact_sheets/04_2041.png` |
| `2053.jpg` | `diagnostics/v1r8/contact_sheets/16_2053.png` |
| `2056.jpg` | `diagnostics/v1r8/match_viz/` + `contact_sheets/19_2056.png` |
| `2062.jpg` | `diagnostics/v1r8/contact_sheets/25_2062.png` |
| `2063.jpg` | `diagnostics/v1r8/contact_sheets/26_2063.png` |

Match arrows: `diagnostics/v1r8/match_viz/{idx}_{stem}_{A0|A3|B2-50}.png`.

### Focus PSNR (v1r3e4, zero-shot)

| Filename | A0 | A3 | B2-50 | B2−A0 |
|---|---|---|---|---|
| `2037.jpg` | 21.209 | 20.291 | 20.187 | **−1.022** |
| `2041.jpg` | 23.652 | 22.889 | 23.016 | **−0.637** |
| `2053.jpg` | 26.355 | 25.768 | 25.409 | **−0.946** |
| `2056.jpg` | 16.400 | 16.769 | 17.130 | **+0.730** |
| `2062.jpg` | 16.896 | 17.631 | 17.720 | **+0.824** |
| `2063.jpg` | 25.299 | 26.376 | 26.043 | **+0.744** |

Negatives: matching tightens (↑ id-cos, ↓ L2, ↑ ≤4 px) while **SR luma MAE vs GT rises**.  
Positives: SR luma MAE falls (or barely changes) and PSNR rises — even when match change is large (`2056.jpg` changes ~98% of indices).

---

## 3. Phase B — Matching quality (all 30)

### Protocol note

- **PSNR/SSIM / error maps:** full-res tiled SR (production-equivalent).  
- **Match statistics & viz:** `max_side=512` (full-res `R_lv3` OOMs). Absolute cos/L2 differ from v1r7 full-res tables; **ordering A0 → A3 → B2** is consistent.

### Method means (v1r3e4)

| Metric | A0 | A3 | B2-50 |
|---|---|---|---|
| RGB PSNR | **20.162** | 20.068 | 20.137 |
| RGB SSIM | **0.6324** | 0.6315 | 0.6319 |
| max cos | 0.477 | 0.534 | **0.557** |
| identity cos | 0.312 | 0.423 | **0.464** |
| match L2 (lv3 px @512) | 38.0 | 27.2 | **21.8** |
| frac ≤1 / ≤2 / ≤4 / ≤8 px | 0.213 / 0.222 / 0.240 / 0.275 | 0.320 / 0.335 / 0.363 / 0.413 | **0.385 / 0.403 / 0.434 / 0.491** |
| match index changed vs A0 | 0 | 0.792 | **0.801** |
| mean sigmoid(S) | 0.617 | 0.630 | 0.635 |
| T_lv3↔GT feat cos-dist ↓ | 0.682 | 0.584 | **0.549** |
| SR luma MAE | **0.0782** | 0.0797 | 0.0787 |
| SR chroma MAE | 0.0168 | 0.0168 | **0.0166** |

B2 vs A0 per-image (±0.01 dB): **15↑ / 14↓ / 1 flat** — near coin-flip; mean PSNR −0.025 dB.

Offline GT-feature distances (T vs LTE(GT) at diagnostic scale): B2 reduces mean cos-distance at lv3/lv2/lv1 vs A0. **Not** treated as correspondence GT; used only as an exploratory proxy.

CSVs: `per_image_metrics.csv`, `matching_statistics.csv`.

### Correlations (n=30, exploratory) — ΔPSNR(B2−A0)

| Predictor | Spearman ρ |
|---|---|
| ↓ SR luma MAE | **+0.968** |
| ↑ identity cos | −0.076 |
| ↓ match L2 | −0.093 |
| match changed fraction | +0.036 |
| ↓ T3–GT cos-dist | −0.059 |
| ↑ max cos | +0.035 |

Plot: `diagnostics/v1r8/plots/correlation_scatter.png`.

**Reading:** Match-quality gains do **not** track PSNR. Almost all per-image ΔPSNR is explained by SR luminance error vs GT. Because MainNet input Low is fixed, that photometric/structural shift must arrive through **changed T/S injection**, not through painting Query γ onto the output directly.

---

## 4. Phase C — Fine-tuning regression (existing runs only)

Sources:  
`v1r3` train log · `v1r6-ft-C0-control` · `v1r6-ft-C1-gamma075` · `v1r7-ft-B2-b050`  
JSON: `diagnostics/v1r8/ft_regression/ft_regression_analysis.json`  
Plots: `ft_eval_trajectories.png`, `ft_train_l1.png`.

**Missing evidence:** none of the requested ckpts/logs were absent. Original v1r3 train L1/MS-SSIM epoch table was empty in the parsed log fields (Eval traj for v1r3 is present).

### Checklist

| # | Question | Finding |
|---|---|---|
| 1 | Ckpt load / shape | 266/266 shared keys; LTE soft-load policy unchanged; LTE weights **bit-identical** after FT (12 LTE tensors unchanged). |
| 2 | Trainable params | LTE frozen; updates in **MainNet** (248 tensors) + **RefCorrection** (6); max ‖Δ‖ ~3.6–3.7e−4. |
| 3 | Optimizer / scheduler restore | **Fresh Adam**; no v1r3 optimizer state. StepLR decay=999999 → constant lr. |
| 4 | Recipe vs v1r3 | v1r3 continue: lr **1e−5** (with decays), longer run, seed unset in script. FT: lr **3e−6**, 5 epochs, crop 192, L1+MS-SSIM, seed 42. |
| 5 | LR behavior | FT lr flat at 3e−6 all 5 epochs. |
| 6 | Train L1 / MS-SSIM | C0/C1/B2 nearly lockstep; no B2-specific train-loss collapse. |
| 7 | Eval traj | All three FT runs: e1 drops vs 20.162, then slow climb; **SSIM ends above** baseline, **PSNR stays below**. |
| 8 | Aug / full-image γ·LUT | C1/B2: full-image maps before crop (same as zs policy). C0: align off. |
| 9 | Stochastic comparability | Shared seed 42 + Generator + worker_init + cudnn deterministic. **Caveat:** Query γ/hist adds torch ops in `__getitem__`/forward → RNG stream can desync vs C0; Ref independent crop uses python/`np.random` — not guaranteed bit-identical across align on/off. Directionally comparable, not strictly identical. |
| 10 | Brightness vs structure | Pattern fits **PSNR↓ / SSIM↑**: small weight moves improve structure slightly while photometric fit to GT worsens vs lucky v1r3e4 snapshot. |

### Eval trajectories (RGB)

| Epoch | C0 PSNR/SSIM | C1 | B2 FT |
|---|---|---|---|
| zs (e4) | 20.162 / 0.6324 | same | 20.137 / 0.6319 |
| 1 | 20.002 / 0.6309 | 20.004 / 0.6323 | 20.060 / 0.6324 |
| 5 | 20.097 / 0.6333 | 20.112 / 0.6346 | **20.150 / 0.6347** |

e5 vs baseline: C0 **−0.065**, C1 **−0.050**, B2 **−0.012** PSNR; all three **SSIM ≥** baseline.

v1r3 history itself oscillates (many epochs &gt;20.162 and &lt;20.162); e4 is a high-PSNR snapshot, not a stable plateau under continued optimization.

---

## 5. Phase D — Hypotheses

### H1 — Brightness / photometric distortion

| | |
|---|---|
| **Support** | ΔPSNR ≈ −Δ(SR luma MAE) (ρ≈0.97). Negatives (`2037`/`2053`/`2041`) show ↑ luma MAE with B2; positives show ↓. Chroma MAE nearly flat. Contact sheets: error maps on losers are broad midtone/highlight residuals, not only edges. |
| **Contradict** | Query RGB never enters MainNet; mean Query brightness change alone cannot paint SR. Overall mean luma MAE A0→B2 is only +0.0005 — effect is **per-image redistribution**, not global over-brightening. |
| **Verdict** | **Plausible (downstream photometric/structural shift via T/S)**, not “Query LUT burned into pixels.” |
| **Cheap discrim.** | Report mean(Y_sr)−mean(Y_gt) signed bias per image for A0 vs B2 (bias vs MAE). Already half-answered by luma MAE correlation. |

### H2 — Incorrect matching (high cos, wrong place)

| | |
|---|---|
| **Support** | ~80% indices change under B2. Pos `2056.jpg`: tiny id-cos gain, L2 slightly **worse**, yet +0.73 dB — match “quality” metrics are unreliable proxies. Offline T–GT distance improves on average but ρ with ΔPSNR ≈ 0. |
| **Contradict** | Identity cos ↑, L2 ↓, ≤k-px ↑, T–GT dist ↓ simultaneously — if matches were systematically worse, these should not all move “better.” Visual arrows on winners often look more local. |
| **Verdict** | **Plausible for a subset of losers; not the global explanation.** Better cosine ≠ correct semantic correspondence. |
| **Cheap discrim.** | Zero-shot **match_arg swap**: compute B2 Query features but transfer Value with **A0 `match_arg`** (and conversely). |

### H3 — Value / MainNet mismatch (S / T stats vs trained gate)

| | |
|---|---|
| **Support** | MainNet trained on A0 SearchTransfer; B2 changes 80% of indices and raises mean sigmoid(S) 0.617→0.635 (more texture admitted). Top FT weight deltas sit in merge/RB heads that fuse T. Zero-shot A3 (γ only) already −0.09 dB with large index churn. |
| **Contradict** | FT under B2 (adapting MainNet) recovers toward baseline and beats C0, but still &lt;20.162 — adaptation helps a little, does not prove zero-shot mismatch is the sole zs gap. S shift is small in absolute gate space. |
| **Verdict** | **Plausible**, especially jointly with H2. |
| **Cheap discrim.** | Factorized zero-shot: (B2 `match_arg` + A0 `S`) vs (A0 `match_arg` + B2 `S`) vs full B2. Isolates index vs confidence. |

### H4 — Fine-tuning optimization regression

| | |
|---|---|
| **Support** | **C0 (no align)** also drops to 20.002@e1 and ends 20.097@e5 &lt; 20.162 while SSIM rises. Fresh Adam, lower constant lr, short horizon. v1r3 Eval already oscillates around e4. B2 FT &gt; C0 FT ⇒ align is not the unique cause of the drop from e4. |
| **Contradict** | None material for “FT can regress PSNR independent of B2.” |
| **Verdict** | **Confirmed** for the continue-FT vs v1r3e4 PSNR gap. |
| **Cheap discrim.** | (If ever training again) restore Adam moments from v1r3e4 or eval intermediate FT epochs more densely — **out of scope here**. |

### Cross-summary

| Question | Answer |
|---|---|
| Why better Query/Key sim ≠ higher RGB PSNR? | Match proxies improve, but ΔPSNR tracks **SR luminance error**, not cos/L2. ~80% re-indexing + mild S↑ retunes T injection; winners/losers cancel (~−0.025 dB). |
| Why continue-FT &lt; v1r3e4? | **H4 confirmed:** same-budget C0 without B2 also regresses PSNR / gains SSIM. B2 FT is best among FT runs but still short of the e4 snapshot. |

Correlations are exploratory (n=30); not causal proof.

---

## 6. Ranked recommendation — **one** next experiment

**Do not run it in this task.**

### #1 (only): Zero-shot `match_arg` / `S` factorization on v1r3e4

Instrument SearchTransfer (eval-only hook) to emit A0 and B2 soft-attention, then evaluate four forwards **without training / without αβ changes**:

1. **A0** (baseline)  
2. **B2 full** (current)  
3. **B2 indices + A0 `S`**  
4. **A0 indices + B2 `S`** (optional fifth: B2 Query features but force A0 indices **and** A0 `S` → must equal A0 if Query only affects matching)

**Decision rule:**

- (3) ≈ A0 and (2) bad → **H2** (wrong locations dominate).  
- (4) ≈ A0 and (2) bad → **H3** (confidence / gate mismatch).  
- (3) and (4) both intermediate → joint H2+H3.  
- All B2-index runs bad while match proxies look great → stop chasing global γ/hist for PSNR; prioritize photometric/MainNet levers instead.

This is the cheapest test that separates “better cosine” from “usable soft-attention for this MainNet,” and does not touch checkpoints or production defaults (align remains off).

---

## 7. Deliverables checklist

| Deliverable | Path |
|---|---|
| Report | `docs/v1r8_matching_diagnostic_report.md` |
| Contact sheets (6) | `diagnostics/v1r8/contact_sheets/` |
| Match viz | `diagnostics/v1r8/match_viz/` |
| Per-image metrics | `diagnostics/v1r8/per_image_metrics.csv` |
| Matching statistics | `diagnostics/v1r8/matching_statistics.csv` |
| Correlations | `diagnostics/v1r8/correlations.json`, `plots/correlation_scatter.png` |
| FT analysis | `diagnostics/v1r8/ft_regression/` |
| Reproduce | `diagnostics/v1r8/REPRO.md`, `scripts/run_v1r8_matching_diagnostic.py`, `scripts/analyze_v1r8_ft_regression.py` |

**Production recommendation unchanged:** keep v1r3e4, Query align **off** by default.
