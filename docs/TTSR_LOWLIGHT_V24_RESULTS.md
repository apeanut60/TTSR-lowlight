# TTSR-lowlight v2.4 — End-to-End P2 + H3 Results

**Verdict: Case B — joint P2 + MSE helps; L1+MS-SSIM does not for PSNR.**  
Best research candidate: **A2@250** (joint P2, RGB MSE) at **20.494 dB / 0.6355 SSIM** (+0.028 vs H3@2000, +0.025 vs A0@250).  
A3 is **not** synergistic; longer A2 training regresses. Eval30 is a reused **development set**.

---

## 1. Loss audit (Phase 0)

P2 v1r3 used **`L1 + 1.0·(1−MS-SSIM)`**, RGB, inputs `[-1,1]→[0,1]`, LTE frozen.  
See `TTSR_LOWLIGHT_V24_LOSS_AUDIT.md`. Task label “L1+SSIM” = this audited MS-SSIM recipe (not provisional `0.1·SSIM`).

## 2. Protocol

| Item | Value |
|---|---|
| Branch | `experiment/ttsr-lowlight-v24` |
| Init | P2 `model_00004.pt` (M1+lv2 bypass) + H3@2000 |
| Train | **192×192 crop**, batch=1, seed=42, Adam, **2000** updates |
| Why crop | Full-Image joint A2 OOM (~21 GB+); crops fit ~2.2 GB |
| Eval | Full-image tiled P2 (256/96) + H3; loads **per-arm P2** |
| LR | P2 1e-6 / Enc 1e-6 / Match 2e-6 / Head 1e-5 |
| P2 trainables | ~4.94 M (LTE frozen; `conv22_head` excluded — lv2 bypass) |

A0 crop@1000 = 20.471 vs v2.3 Full-Image H0@1000 = 20.479 — small protocol gap; do not mix.

## 3. Step0 / gradients

All arms step0: **Y=20.466**, **P2-only=20.265**.  
A2/A3: `Y0.requires_grad=True`, non-zero P2 grads (252/252), LTE unchanged.  
Details: `TTSR_LOWLIGHT_V24_GRADIENT_AUDIT.md`.

## 4. Eval30 — all checkpoints

| Arm | Step | PSNR | SSIM | P2-only | Δ vs H3 | Δ vs A0@step |
|---|---:|---:|---:|---:|---:|---:|
| P2 | — | 20.265 | 0.6340 | 20.265 | — | — |
| H3@2000 | — | 20.466 | 0.6331 | 20.265 | 0 | — |
| v2.3 H0@1000 | 1000 | 20.479 | 0.6334 | — | +0.013 | — |
| A0 | 250 | 20.468 | 0.6335 | 20.265 | +0.002 | 0 |
| A0 | 500 | 20.465 | 0.6334 | 20.265 | −0.001 | 0 |
| A0 | 1000 | 20.471 | 0.6334 | 20.265 | +0.005 | 0 |
| **A0** | **1500** | **20.473** | **0.6336** | 20.265 | **+0.007** | 0 |
| A0 | 2000 | 20.471 | 0.6337 | 20.265 | +0.005 | 0 |
| **A1** | **250** | **20.469** | **0.6340** | 20.265 | +0.003 | +0.001 |
| A1 | 500 | 20.458 | 0.6348 | 20.265 | −0.008 | −0.007 |
| A1 | 1000 | 20.454 | 0.6355 | 20.265 | −0.012 | −0.017 |
| A1 | 1500 | 20.448 | 0.6359 | 20.265 | −0.018 | −0.025 |
| A1 | 2000 | 20.442 | 0.6362 | 20.265 | −0.024 | −0.029 |
| **A2** | **250** | **20.494** | **0.6355** | **20.267** | **+0.028** | **+0.025** |
| A2 | 500 | 20.463 | 0.6362 | 20.219 | −0.003 | −0.001 |
| A2 | 1000 | 20.481 | 0.6367 | 20.245 | +0.015 | +0.010 |
| A2 | 1500 | 20.460 | 0.6365 | 20.235 | −0.006 | −0.014 |
| A2 | 2000 | 20.439 | 0.6365 | 20.206 | −0.027 | −0.032 |
| **A3** | **250** | **20.472** | **0.6364** | 20.219 | +0.006 | +0.004 |
| A3 | 500 | 20.385 | 0.6368 | 20.088 | −0.081 | −0.080 |
| A3 | 1000 | 20.421 | 0.6374 | 20.152 | −0.045 | −0.050 |
| A3 | 1500 | 20.396 | 0.6379 | 20.138 | −0.070 | −0.078 |
| A3 | 2000 | 20.371 | 0.6381 | 20.112 | −0.095 | −0.100 |

### 2×2 contrasts @250 (peak A2 window)

| Contrast | ΔPSNR |
|---|---:|
| ΔP2,MSE = A2−A0 | **+0.025** |
| ΔLoss,Frozen = A1−A0 | +0.001 |
| ΔLoss,Joint = A3−A2 | **−0.022** |
| Interaction | **−0.022** |

Unfreezing P2 under MSE helps; switching to L1+MS-SSIM under joint training **hurts** PSNR (negative interaction).

## 5. Interpretation

| Question | Answer |
|---|---|
| 1. Unfreeze P2 → better? | **Yes early** (A2@250 +0.025 vs A0). Gains fade / reverse by 2000 as P2-only drifts down. |
| 2. L1+MS-SSIM better than MSE? | **For SSIM yes, for PSNR no.** A1 SSIM↑ to 0.636 and `<−1dB` 6→2, but mean PSNR falls. |
| 3. Synergy? | **No.** A3 ≤ A2 on PSNR at every step after 0. |
| 4. Fix global RGB errors? | **Partially for A1 tail count**; DC share still ~0.59–0.62. A2@250 still fails 2053 (21.31). |

PSNR gain on A2@250 is mostly **final Y**, not a large P2-only jump (P2-only 20.267 ≈ 20.265). Longer joint FT **degrades the P2 base**.

## 6. Phase D (best ckpt each arm)

| Arm | Correct | Self | Mismatch | Zero-T | Bypass | C−S | Full / Mean / Cent | DCshare | P2-only |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| A0@1500 | 20.473 | 20.247 | 19.654 | 18.873 | 20.265 | +0.23 | 20.47 / 20.30 / 20.43 | 0.610 | 20.265 |
| A1@250 | 20.469 | 20.276 | 19.723 | 19.023 | 20.265 | +0.19 | 20.47 / 20.29 / 20.43 | 0.588 | 20.265 |
| A2@250 | 20.494 | 20.300 | 19.598 | 18.588 | 20.267 | +0.19 | 20.49 / 20.33 / 20.42 | 0.616 | 20.267 |
| A3@250 | 20.472 | 20.325 | 19.672 | 18.663 | 20.219 | +0.15 | 20.47 / 20.31 / 20.37 | 0.621 | 20.219 |

Reference dependence retained (Correct ≫ Zero-T). A3 Correct−Self shrinks; P2-only drops.

### Focus PSNR (best ckpts)

| ID | P2 | H3 | A0 | A1 | A2 | A3 |
|---|---:|---:|---:|---:|---:|---:|
| 2053 | 24.38 | 21.52 | 21.48 | **21.77** | 21.31 | 21.94 |
| 2059 | 24.38 | 22.85 | 22.82 | 23.00 | 22.44 | 22.77 |
| 2055 | 21.27 | 19.84 | 19.84 | 20.04 | 19.84 | 20.17 |
| 2052 | 17.74 | 19.49 | 19.50 | 19.28 | **19.62** | 19.20 |
| 2061 | 21.35 | 22.72 | 22.72 | 22.58 | **22.80** | 22.54 |

## 7. Runtime

| Arm | Wall | Peak VRAM |
|---|---:|---:|
| A0 | 10.2 min | 1.4 GB |
| A1 | 10.3 min | 1.4 GB |
| A2 | 12.5 min | 2.2 GB |
| A3 | 12.9 min | 2.2 GB |

## 8. Go / No-Go

**Case B.** Prefer **joint P2 + MSE** with **early stopping (~250 updates)** over L1+MS-SSIM joint.  
Do **not** promote A3. Do **not** run 2000-step joint by default.

Next (separate experiments only if desired):

1. A2-style joint MSE with Eval30 early-stop / multi-seed on a real holdout.
2. Keep MSE (or PSNR-oriented) loss when unfreezing P2; treat L1+MS-SSIM as SSIM-oriented frozen-H3 fine-tune only.
3. Do not auto-add GateNet / new heads.

## 9. Artifacts

### Best checkpoints (SHA256)

| Arm | Step | SHA256 |
|---|---:|---|
| A0 | 1500 | `ca232d5734f0b3e4c1888227513ca18d0da37a84312bc6fc9492a00427373dcd` |
| A1 | 250 | `996d9572f6f196f31b25acc378dac12ae58a5ecb9f799feb9989e3e7b360deac` |
| A2 | 250 | `7d219fbda797d7b368812fa5c183e227d0193604bb5a3215cd804d89db6e8624` |
| A3 | 250 | `2d5610063c8327c061d7dbfeeacb3b542a1e9dcdd74fa975166076fb0887ed90` |

### Reproduce

```bash
export TORCH_HOME=/root/data/cache/torch
cd /root/projects/TTSR-lowlight-v1-data1SOTA

python tests/test_v24_loss.py
python tests/test_v24_freeze_groups.py
python tests/test_v24_gradient_flow.py

for arm in A0 A1 A2 A3; do
  python scripts/train_v24_end2end.py --arm $arm --steps 2000 --seed 42 --crop 192
done
python scripts/eval_v24_formal.py
python scripts/diag_v24_residual.py
```

Docs: `TTSR_LOWLIGHT_V24_{PLAN,LOSS_AUDIT,GRADIENT_AUDIT,RESULTS}.md`  
Metrics: `/root/data/experiments/TTSR-lowlight-v24/{metrics,diagnostics,logs}/`
