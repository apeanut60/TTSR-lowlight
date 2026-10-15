# TTSR-lowlight v2.3 — Reference Feature Joint Fine-Tuning Results

**Verdict: Case D — joint training of Matcher/Encoder does not beat head-only FT.**  
H1/H2 vs paired H0: **+0.0003 / +0.0007 dB** on Huawei Eval30 (development set). Keep **H3@2000** as the research baseline; H0@1000 (= v2.2 V1) remains the best head-only FT checkpoint.

Eval30 has been reused across v1/v2 rounds — **not** an independent test set. Do not claim new SOTA from these numbers alone.

---

## 1. Setup

| Item | Value |
|---|---|
| Branch | `experiment/ttsr-lowlight-v23` (`8d6547e`) |
| Exp dir | `/root/data/experiments/TTSR-lowlight-v23/` |
| Init | H3@2000 `…/v2/…/H3_v3b0_s42/checkpoint_02000.pt` |
| P2 | Frozen; float32 Y0 Train500/Eval30 caches |
| Data | Huawei Nano Train500 / Eval30 only |
| Loss | RGB MSE only (no DC / perceptual / GAN) |
| Seed / Opt / Steps | 42 / Adam / 1000 |
| Protocol | **Full-Image**, batch=1 |
| Match | radius=4, τ=0.1, cosine; chunk=1024 (H0) / 4096 (H1/H2); dense≡chunk verified |

### Trainable groups & LR

| Arm | Trainable | Params | LR |
|---|---|---:|---|
| H0 | Head | 115 811 | head 1e-5 |
| H1 | Matcher + Head | 118 883 | matcher 2e-6, head 1e-5 |
| H2 | Encoder + Matcher + Head | 138 275 | enc 1e-6, matcher 2e-6, head 1e-5 |

SharedEncoder is a **single** shared instance for Y0 and Nano paths.

---

## 2. Phase A checks

| Check | Result |
|---|---|
| Step0 Eval30 PSNR | **20.466** for H0/H1/H2 (matches H3@2000) |
| H0/H1/H2 step0 forward identical | Pass (toy + Eval30) |
| H0 grads | Head ≠0; Enc/Mat = None |
| H1 grads | Head ≠0; Matcher ≠0; Enc = None |
| H2 grads | Enc/Mat/Head all ≠0 |
| Dense vs chunk fwd+grad | max \|Δ\| ~ 2e-7 (24×32 probe) |
| Chunk autograd fix | `LocalRefine.py`: cat chunks instead of in-place write |
| Peak VRAM (train) | H0 652 MB / H1 4.3 GB / H2 4.8 GB |
| Wall time | H0 4.7 min / H1 6.4 min / H2 6.4 min |

---

## 3. Eval30 summary (all checkpoints)

| Model | Step | PSNR | SSIM | Δ vs H3@2000 | Δ vs matched H0 |
|---|---:|---:|---:|---:|---:|
| P2 | — | 20.265 | 0.6340 | — | — |
| H3@2000 | — | 20.466 | 0.6331 | 0 | — |
| v2.2 V1@1000 | 1000 | 20.479 | 0.6334 | +0.013 | — |
| H0 | 250 | 20.474 | 0.6331 | +0.008 | 0 |
| H0 | 500 | 20.471 | 0.6333 | +0.005 | 0 |
| H0 | 750 | 20.466 | 0.6330 | +0.000 | 0 |
| **H0** | **1000** | **20.479** | **0.6334** | **+0.013** | **0** |
| H1 | 250 | 20.474 | 0.6332 | +0.008 | +0.000 |
| H1 | 500 | 20.471 | 0.6333 | +0.005 | −0.000 |
| H1 | 750 | 20.466 | 0.6330 | −0.000 | −0.000 |
| **H1** | **1000** | **20.480** | **0.6334** | **+0.014** | **+0.0003** |
| H2 | 250 | 20.474 | 0.6332 | +0.008 | +0.000 |
| H2 | 500 | 20.471 | 0.6333 | +0.005 | +0.000 |
| H2 | 750 | 20.466 | 0.6330 | −0.000 | −0.001 |
| **H2** | **1000** | **20.480** | **0.6334** | **+0.014** | **+0.0007** |

H0@1000 **exactly reproduces** v2.2 V1@1000 (20.479), confirming head-only Full-Image MSE protocol.

Curves: all three arms rise ~+0.008 by 250, dip near 750, peak at 1000 — same shape; joint modules do not change the trajectory.

### H1/H2 vs H0 @1000 (per-image)

| | mean Δ | median Δ | win/loss | p10 Δ | worst Δ |
|---|---:|---:|---:|---:|---:|
| H1 − H0 | +0.0003 | +0.0005 | 17/13 | −0.002 | −0.006 |
| H2 − H0 | +0.0007 | +0.0012 | 17/13 | −0.009 | −0.027 |

`<−1 dB` vs P2: still **6** images for H0/H1/H2 (unchanged vs H3).

---

## 4. Parameter drift @1000

| Arm | ‖ΔEnc‖ | ‖ΔMat‖ | ‖ΔHead‖ | rel Enc/Mat/Head |
|---|---:|---:|---:|---|
| H0 | 0 | 0 | 0.081 | 0 / 0 / 0.0023 |
| H1 | 0 | 0.0054 | 0.081 | 0 / 1e-6 / 0.0023 |
| H2 | 0.0029 | 0.0054 | 0.081 | 1e-6 / 1e-6 / 0.0024 |

Matcher/Encoder **do update**, but magnitude is tiny under the registered LRs. Head dominates learning.

---

## 5. Phase D — reference dependence (best @1000)

| Arm | Correct | Self | Mismatch | Zero-T | Bypass | C−S | C−M | C−Z |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H3 | 20.466 | 20.253 | 19.571 | 18.636 | 20.265 | +0.21 | +0.90 | +1.83 |
| H0 | 20.479 | 20.245 | 19.673 | 18.979 | 20.265 | +0.23 | +0.81 | +1.50 |
| H1 | 20.480 | 20.245 | 19.673 | 18.980 | 20.265 | +0.23 | +0.81 | +1.50 |
| H2 | 20.480 | 20.245 | 19.678 | 18.994 | 20.265 | +0.23 | +0.80 | +1.49 |

Joint FT **preserves** Nano reference utility (Correct ≫ Zero-T / Mismatch). It does **not** measurably increase Correct−Self beyond H0.

### Residual DC mechanism

| Arm | Full | MeanOnly | Centered | DC energy share |
|---|---:|---:|---:|---:|
| H3 | 20.466 | 20.302 | 20.419 | 0.623 |
| H0 | 20.479 | 20.298 | 20.432 | 0.633 |
| H1 | 20.480 | 20.298 | 20.433 | 0.634 |
| H2 | 20.480 | 20.297 | 20.433 | 0.636 |

Answers:

1. Unfreezing Matcher does **not** change global RGB correction (MeanOnly / DCshare ≈ H0).
2. Unfreezing Encoder likewise — residual remains ~63% DC energy.
3. PSNR gain vs H3 is still the same small head-only FT effect, mostly shared with H0.
4. **2053** remains badly harmed (P2 24.38 → H0/H1/H2 ≈ 21.39); joint FT does not fix it.
5. Failure mechanism matches original H3 (bad global RGB mean shift).

### Focus samples @1000 (PSNR)

| ID | P2 | H3 | H0 | H1 | H2 | H2−H0 |
|---|---:|---:|---:|---:|---:|---:|
| 2053 | 24.38 | 21.52 | 21.39 | 21.39 | 21.39 | −0.002 |
| 2059 | 24.38 | 22.85 | 22.71 | 22.71 | 22.71 | −0.008 |
| 2055 | 21.27 | 19.84 | 19.78 | 19.77 | 19.77 | −0.006 |
| 2058 | 22.01 | 20.76 | 20.68 | 20.68 | 20.68 | −0.002 |
| 2042 | 23.62 | 22.46 | 22.38 | 22.37 | 22.37 | −0.003 |
| 2052 | 17.74 | 19.49 | 19.56 | 19.56 | 19.56 | +0.003 |
| 2057 | 19.51 | 21.19 | 21.22 | 21.22 | 21.22 | −0.001 |
| 2044 | 17.24 | 18.75 | 18.86 | 18.86 | 18.86 | +0.000 |
| 2061 | 21.35 | 22.72 | 22.81 | 22.81 | 22.81 | +0.007 |
| 2056 | 17.29 | 18.57 | 18.61 | 18.61 | 18.61 | −0.001 |

---

## 6. Go / No-Go

**Case D.** Joint optimization of SharedEncoder + LocalSoftMatch under the registered budgets does **not** improve Huawei Nano reference enhancement beyond freezing those modules and training the RGB residual head alone.

Evidence-backed interpretation:

- Not a gradient-path bug: Phase A grads, dense/chunk equivalence, and non-zero ‖ΔMat‖/‖ΔEnc‖ all pass.
- Not an LR explosion: updates are tiny and stable; no NaN.
- Matcher features are **not** destroyed (ref ablations intact).
- Most likely: under RGB MSE + group LRs + 1000 steps, the frozen H3 Encoder/Matcher features are already near-sufficient for the head; head adaptation absorbs the residual gain (~+0.013 dB), while Matcher/Encoder contribute ≪0.01 dB.

**Do not** auto-unfreeze P2 or add GateNet/Verifier. Next experiments (if any) should be separate: longer H2 schedule with seed repeats, or a different scientific lever (e.g. DC-aware residual control from v2.1/v2.2), not deeper unfreeze by default.

---

## 7. Artifacts & reproducibility

### Checkpoints (SHA256)

| Arm | Path | SHA256 |
|---|---|---|
| H0@1000 | `…/v23/checkpoints/H0_head_only_s42/checkpoint_01000.pt` | `a487054a379a8997415e03b77dc2f551b394c423e534694fca3b90757fe48c1a` |
| H1@1000 | `…/v23/checkpoints/H1_matcher_head_s42/checkpoint_01000.pt` | `df7dd3e6499d84e515d2bce0c9f13d1c649fa49911b8f83cb0515c8a0310d63d` |
| H2@1000 | `…/v23/checkpoints/H2_encoder_matcher_head_s42/checkpoint_01000.pt` | `b19ede8f12d624596e392e19a12ae342d3de7f3bcb65cffa63086fb8f8d9dd19` |

### Commands

```bash
export TORCH_HOME=/root/data/cache/torch
cd /root/projects/TTSR-lowlight-v1-data1SOTA

python tests/test_v23_matcher_gradients.py
python tests/test_v23_trainable_groups.py
python tests/test_v23_step0_equivalence.py

python scripts/train_v23_joint_ft.py --arm H0 --steps 1000 --seed 42
python scripts/train_v23_joint_ft.py --arm H1 --steps 1000 --seed 42 --chunk 4096
python scripts/train_v23_joint_ft.py --arm H2 --steps 1000 --seed 42 --chunk 4096

python scripts/eval_v23_formal.py
python scripts/diag_v23_joint_features.py
```

### Delivered files

- `docs/TTSR_LOWLIGHT_V23_PLAN.md`, `docs/TTSR_LOWLIGHT_V23_RESULTS.md`
- `scripts/train_v23_joint_ft.py`, `eval_v23_formal.py`, `diag_v23_joint_features.py`
- `tests/test_v23_{trainable_groups,matcher_gradients,step0_equivalence}.py`
- `model/LocalRefine.py` — differentiable chunked matching (cat vs in-place)

Metrics/diagnostics live under `/root/data/experiments/TTSR-lowlight-v23/{metrics,diagnostics,logs}/`.
