# TTSR-lowlight v2.2 — Results

**Verdict: Case C (with a tiny mean bump on V1).**  
Full-image DC-aware fine-tuning did **not** fix harmful global RGB corrections. V1 Full edges V0 by **+0.013 dB** mean PSNR but **worsens** 2053-class tails. V2 (λ_DC=2) regresses the mean. Keep **H3@2000** and **Centered** as research baselines; do not promote V1/V2 as robustness upgrades.

Eval30 is a **reused development set** — not independent generalization.

---

## Identity

| Item | Value |
|---|---|
| Branch | `experiment/ttsr-lowlight-v22` |
| H3 init | `…/H3_v3b0_s42/checkpoint_02000.pt` sha `742d366b…` |
| V1 final | `…/V1_full_mse_s42/checkpoint_01000.pt` sha `c9a5d511…` |
| V2 final | `…/V2_full_dc2_s42/checkpoint_01000.pt` sha `0e7a4ab3…` |
| Train | Huawei500 full-image, batch=1, Adam lr=1e-5, 1000 steps, seed=42 |
| Peak VRAM | **586 MB** (true full-image; no crop fake) |
| Frozen | P2, Encoder, LocalSoftMatch |

Phase A: `docs/TTSR_LOWLIGHT_V22_PHASE_A.md`  
Tests: `tests/test_v22_dc_loss.py`, `tests/test_v22_fullimage_equivalence.py` (PASS; λ_DC=1 ≡ MSE + matching grads).

---

## Main Eval30 table (final checkpoints)

| Run | PSNR | SSIM | Δ vs H3(V0) | Worst Δ vs P2 | `<−1dB` |
|---|---|---|---|---|---|
| P2 | 20.265 | 0.6340 | — | — | — |
| V0 Full H3 | **20.466** | 0.6331 | — | −2.860 (2053) | 6 |
| V0 Centered | 20.419 | — | −0.047 | — | — |
| **V1 Full** (λ=1) | **20.479** | 0.6334 | **+0.013** | **−2.986** (2053) | 6 |
| V1 Centered | 20.432 | — | — | — | — |
| V2 Full (λ=2) | 20.441 | 0.6327 | −0.025 | −2.802 (2053) | 6 |
| V2 Centered | 20.387 | — | — | — | — |

### Training curves (Eval30 Full PSNR)

| Step | V1 | V2 |
|---|---|---|
| 0 | 20.466 | 20.466 |
| 250 | 20.474 | 20.462 |
| 500 | 20.471 | 20.448 |
| 750 | 20.466 | 20.429 |
| 1000 | **20.479** | 20.441 |

---

## Tail / focus samples (Full PSNR)

| ID | P2 | V0 | V1 | V2 | V0 Centered |
|---|---|---|---|---|---|
| **2053** | 24.379 | 21.520 | **21.394** ↓ | 21.577 | **24.459** |
| 2059 | 24.383 | 22.848 | 22.713 ↓ | 22.917 | 24.431 |
| 2055 | 21.273 | 19.835 | 19.776 ↓ | 19.867 | 21.335 |
| 2058 | 22.012 | 20.755 | 20.681 ↓ | 20.787 | 21.906 |
| 2042 | 23.616 | 22.456 | 22.376 ↓ | 22.454 | 23.545 |
| 2052 (gain) | 17.744 | 19.487 | **19.557** ↑ | 19.421 | 17.788 |
| 2057 | 19.510 | 21.193 | 21.222 ↑ | 21.119 | 19.542 |
| 2044 | 17.240 | 18.751 | 18.859 ↑ | 18.782 | 17.410 |

2053 `D_mean` RGB: V0 (+0.061,+0.066,+0.052) → V1 (**larger** bias) → V2 (similar to V0).  
Full-image MSE fine-tune did **not** shrink the harmful DC on the worst failure; λ_DC=2 only mildly trimmed 2053’s drop while hurting the mean.

Viz: `/root/data/experiments/TTSR-lowlight-v22/visualizations/2053/`

---

## Reference-path ablation (second path only)

| Condition | V0 | V1 | V2 |
|---|---|---|---|
| Correct Nano | **20.466** | **20.479** | 20.441 |
| Self | 20.253 | 20.245 | 20.246 |
| Mismatch | 19.571 | 19.673 | 19.710 |
| Zero-T | 18.636 | 18.979 | 19.066 |

Correct Nano remains best. Fine-tuning slightly **closes** the gap to Zero-T/Mismatch (less exclusive Nano advantage) — **not** evidence of stronger reference use.

---

## Phase A recap

- Train500 also ~40% harmful DC (same mechanism as Eval30).
- Crop↔Full prediction cosine ≈0.99; border-sensitive; crop loss cannot supervise full-image `m*`.
- True full-image training is feasible (586 MB).

---

## Answers to the scientific question

> Adjust training scale + RGB DC supervision without new networks?

**Partial / negative on the failure mode.** V1 gives a **small mean PSNR gain (+0.013)** with Nano still preferred, but **does not reduce severe global-bias failures** (2053 worse). V2’s stronger DC weight **hurts** mean PSNR. Post-hoc **Centered** still heals 2053-class samples far better than any full-output fine-tune here.

---

## Go / No-Go

| Case | Fit? |
|---|---|
| A — clear gain + better tails | **No** (mean↑ tiny; tails worse on V1) |
| B — flat mean, better tails | **No** (V2 tails only marginally better; mean↓) |
| **C — Centered still more robust; full fine-tune unhelpful for the bug** | **Yes** |
| D — overall collapse | No |

**Retain:** P2 default; H3@2000 research candidate; Centered as robustness diagnostic / optional inference variant (not newly trained).  
**Do not** start GateNet / Verifier / λ sweeps from this result.

---

## Reproduce

```bash
conda activate ttsr
cd /root/projects/TTSR-lowlight-v1-data1SOTA
git checkout experiment/ttsr-lowlight-v22

python tests/test_v22_dc_loss.py
python tests/test_v22_fullimage_equivalence.py
python scripts/diag_v22_fullimage_dc.py          # Phase A
python scripts/train_v22_fullimage.py --arm V1
python scripts/train_v22_fullimage.py --arm V2
python scripts/eval_v22_formal.py
```

P2 caches symlinked from v2; checkpoints under `/root/data/experiments/TTSR-lowlight-v22/` (not in git).

*v2.2 complete. No automatic next experiment.*
