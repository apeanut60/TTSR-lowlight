# v1r9 — Identity-Guided Texture Fusion (Zero-Shot)

**Branch:** `experiment/v1r9-identity-texture-fusion` (from `experiment/v1r8-matching-diagnostic`)  
**Ckpt (frozen):** `/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt`  
**Eval:** Huawei Nano, 30 images, RGB PSNR/SSIM, no mean-align, `ref_degrade=False`, production tiled inference  
**Scope:** Zero-shot only — no training, no α/β sweep, no MainNet/LTE changes. Default path unchanged (`texture_fusion=False`).

Artifacts: `diagnostics/v1r9/` · reproduce: `diagnostics/v1r9/REPRO.md`

---

## Phase 0 — Geometry audit

| Check | Result |
|---|---|
| Pairing | Exact basename stem: `low/2037.jpg` ↔ `nanobanana_ref/2037.png` |
| Native sizes | Low/GT **960×720**; Nano **1195×896** (aspect ≈ identical) |
| Eval preprocess | `TestSet` resizes Ref → Low HxW (`cv2.INTER_LINEAR`), crops both to multiples of 4 |
| Tiles | Same `(y0,x0,y1,x1)` on Low and Ref; full-image γ/LUT once, reused per tile |
| VGG scales | lv1 H×W, lv2 H/2×W/2, lv3 H/4×W/4 — Identity indices at lv3 are 1:1 when H=Hr, W=Wr |
| Unfold/fold | Identity reuses the same 3/6/12 kernels, strides, and border normalization as global transfer |

**Identity is a spatial prior on the resized Nano grid**, not oracle GT alignment. Native Nano resolution differs; after resize, Query/Key/Value share one coordinate system, so Identity indexing is well-defined. Mismatched lv3 shapes raise `ValueError` (no silent remap).

Audit JSON: `diagnostics/v1r9/phase0_geometry_audit.json`.

---

## Implementation

| Piece | Behavior |
|---|---|
| `SearchTransfer.identity_match_arg` | `arange(H*W)` when shapes match; else error |
| `SearchTransfer.transfer_from_arg` / `transfer_identity` | Same Value unfold → gather → fold → normalize as global path |
| `TTSREnhance` + `texture_fusion` | Computes A0 Search (for **S_A0**), B2 Search (for **T_B2**), Identity **T_id**; `T_mix=(1−λ)T_B2+λ T_id`; **S_mix=S_A0** |
| Defaults | `texture_fusion=False` → original single SearchTransfer path |
| MainNet input | Always original Low; Ref Value from Nano only; **no GT** |

---

## Tests (Phase 3)

`python tests/test_v1r9_identity_fusion.py` → **ALL PASS**

1. Fusion off: bit-identical twin forwards  
2–4. λ=0 → T_B2; λ=1 → T_id; intermediate λ exact linear mix; S stays A0  
5–7. Finite Identity tensors, border-safe indices, tile-local Identity  
8–10. MainNet←Low; Value←Ref; no GT  
11. Ckpt 266-core load  
12. Eval reproduces A0 **20.162 / 0.6324** and B2 **20.137 / 0.6319** (F0/F1)

---

## Zero-shot results (Phase 4–5)

| Run | Texture | S | RGB PSNR | RGB SSIM | Δ vs F0 | Δ vs M0 | ↑/↓ vs F0 | Luma MAE | Luma bias | Chroma MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| **F0** | A0 | A0 | **20.162** | **0.6324** | — | −0.013 | — | 0.0782 | −0.0311 | 0.0168 |
| **F1** | B2 | B2 | 20.137 | 0.6319 | −0.025 | −0.038 | 15/14 | 0.0787 | −0.0329 | 0.0166 |
| **M0** | 100% B2 | A0 | 20.175 | 0.6324 | **+0.013** | — | 15/15 | 0.0781 | −0.0308 | 0.0167 |
| **M1** | 75% B2 / 25% Id | A0 | **20.181** | **0.6327** | **+0.018** | **+0.005** | 15/15 | **0.0780** | **−0.0307** | 0.0166 |
| **M2** | 50/50 | A0 | 20.179 | 0.6327 | +0.017 | +0.004 | 15/15 | 0.0781 | −0.0310 | 0.0165 |
| **M3** | 25/75 | A0 | 20.175 | 0.6324 | +0.013 | −0.000 | 14/15 | 0.0783 | −0.0316 | 0.0164 |
| **M4** | 100% Identity | A0 | 20.169 | 0.6318 | +0.007 | −0.006 | 14/16 | 0.0786 | −0.0324 | 0.0164 |

Per-image CSV: `diagnostics/v1r9/per_image_metrics.csv`.

### Focus images (PSNR)

| File | F0 | F1 | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|---|---|
| `2037.jpg` (B2−) | 21.209 | 20.187 | 20.436 | 20.427 | 20.449 | 20.500 | **20.568** |
| `2041.jpg` (B2−) | **23.652** | 23.016 | 23.403 | 23.348 | 23.256 | 23.144 | 23.054 |
| `2053.jpg` (B2−) | **26.355** | 25.409 | 25.287 | 25.178 | 25.174 | 25.334 | 25.633 |
| `2056.jpg` (B2+) | 16.400 | 17.130 | 17.116 | 17.209 | 17.290 | 17.333 | **17.338** |
| `2062.jpg` (B2+) | 16.896 | **17.720** | 17.495 | 17.459 | 17.435 | 17.425 | 17.437 |
| `2063.jpg` (B2+) | 25.299 | **26.043** | 25.561 | 25.533 | 25.511 | 25.510 | 25.541 |

Side-by-side preds + error maps: `diagnostics/v1r9/focus_compare/`.

### T_B2 vs T_id (scaled @512, mean over 30)

| Level | Rel. L2 ‖Δ‖/‖T_B2‖ | Cosine(T_B2, T_id) |
|---|---|---|
| lv3 | 0.63 | 0.80 |
| lv2 | 0.58 | 0.85 |
| lv1 | 0.33 | 0.95 |

Fusion changes texture **substantially** at coarse levels (not a near-no-op blend). Spatial maps: `diagnostics/v1r9/t_diff_maps/`.

---

## Interpretation (Phase 6)

**Q1. Does M0 outperform F1?**  
**Yes.** B2 texture + A0 S (20.175) beats full B2 (20.137) by **+0.038 dB**. Under this checkpoint, keeping **A0 soft-attention** is preferable to B2’s raised `sigmoid(S)` when using B2 matches — consistent with v1r8 H3 (gate / confidence mismatch).

**Q2. Does any M1–M3 outperform M0?**  
**Yes — M1 and M2.** Best mean is **M1 (λ=0.25): 20.181 / 0.6327** (+0.005 vs M0, **+0.018 vs F0**). Identity supplies a small but real complementary prior when lightly blended with B2 texture under A0 S. Gains are modest; 15↑/15↓ vs F0 — not a one-sided win.

**Q3. Does pure Identity (M4) help?**  
M4 (20.169) beats F0 slightly (+0.007) but **loses to M0/M1**. 100% Identity is not optimal; Nano is not geometric GT. On hard B2 losers, Identity helps unevenly (`2037` recovers toward F0; `2053`/`2041` remain below F0).

**Q4. Beat the 20.162 baseline?**  
**Yes, zero-shot:** M1 **20.181** (+0.018), M2 **20.179**, M0 **20.175**. SSIM also edges up (M1/M2 **0.6327**). Magnitude is small relative to eval noise / historical v1r3 epoch swing, but direction is consistent across M0–M2 and A0/B2 reproduction is exact.

---

## Conclusions

1. **A0 S + B2 T (M0)** already recovers the F1 regression and slightly exceeds F0 — soft-attention source matters as much as match indices.  
2. **Light Identity mix (M1, λ=0.25)** is the best zero-shot config in this sweep.  
3. Default production path should stay **fusion off** until a follow-up decides to promote M1 as an optional eval flag.  
4. Do **not** treat Identity as GT correspondence; native Nano resize remains a caveat.

---

## Ranked next step (not started)

**Optional eval flag / short confirm:** freeze M1 (`λ=0.25`, S=A0, B2 Query α=0.75 β=0.50) as a documented optional inference mode and re-check on the same 30-image protocol only — no training. If a larger gain is required, the cheaper research lever is still outside Query matching (illum head / data), not stronger global hist.

---

## Deliverables

| Item | Path |
|---|---|
| Report | `docs/v1r9_identity_texture_fusion_report.md` |
| Metrics | `diagnostics/v1r9/per_image_metrics.csv`, `summary.json` |
| Focus compares | `diagnostics/v1r9/focus_compare/` |
| T diffs | `diagnostics/v1r9/t_diff_maps/`, `tdiff_summary.json` |
| Geometry audit | `diagnostics/v1r9/phase0_geometry_audit.json` |
| Tests | `tests/test_v1r9_identity_fusion.py` |
| Eval script | `scripts/run_v1r9_identity_fusion_eval.py` |
