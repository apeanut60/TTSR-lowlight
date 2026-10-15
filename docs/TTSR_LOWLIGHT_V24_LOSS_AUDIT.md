# TTSR-lowlight v2.4 — Phase 0: P2 Loss Audit

## Verdict

**P2 (v1r3 Huawei Nano) was trained with L1 + MS-SSIM, not plain single-scale SSIM.**

$$
L = 1.0\cdot\mathrm{L1}(Y,GT) + 1.0\cdot\bigl(1-\mathrm{MS\text{-}SSIM}(Y,GT)\bigr)
$$

All other aux losses were **off** (`per_w=tpl_w=adv_w=illum_smooth_w=color_w=exposure_w=0`).

The user recollection “L1 + SSIM” maps to this **MS-SSIM** recipe in the actual codebase.  
v2.4 A1/A3 therefore **reproduce the audited P2 recipe**, not the provisional `L1+0.1(1-SSIM)` fallback.

---

## Evidence

| Source | Finding |
|---|---|
| `train/train_v1r3_nano_continue.sh` | `--rec_w 1.0 --rec_loss_type l1 --ms_ssim_w 1.0` |
| `/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/args.txt` | `rec_loss_type=l1`, `ms_ssim_w=1.0`, aux weights 0 |
| `train_v1r3_nano_Huawei.log` | logs both `rec_loss` and `ms_ssim_w * ms_ssim_loss` every print |
| `trainer.py` | `loss += ms_ssim_w * MSSSIMLoss(sr, hr)` when `ms_ssim_w≠0` |
| `loss/loss_enhance.py::MSSSIMLoss` | differentiable `1 - MS-SSIM` |

## MSSSIMLoss details (as used)

| Item | Value |
|---|---|
| Channels | **RGB** (C=3), not Y-only |
| Input range | model tensors **[-1, 1]** → internally `(x+1)/2` clamped to **[0, 1]** |
| `data_range` | 1.0 |
| Window | Gaussian, size 11 (shrinks to 7 if `min(H,W)≤160`), σ=1.5 |
| C1 / C2 | `(0.01·R)²`, `(0.03·R)²` with R=`data_range` |
| Pyramid weights | `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]` (levels shrink if crop too small) |
| Form | returns **`1 - MS-SSIM`** (higher = worse) |
| Init vs train | `num_init_epochs=0` for v1r3 — **same loss** throughout |

## LTE

`--freeze_lte True` during v1r3. VGG/LTE must stay frozen in v2.4 joint FT unless a separate experiment says otherwise.

## v2.4 implication

| Arm Loss label | Implementation |
|---|---|
| RGB MSE | `F.mse_loss(Y, GT)` |
| “L1 + SSIM” (task name) | **Audited** `L1 + 1.0·(1−MS-SSIM)` via `ReconstructionLoss('l1')` + `MSSSIMLoss` |

Provisional `λ_s=0.1` single-scale SSIM was **not** used.
