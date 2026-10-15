# TTSR-lowlight v2.4 — End-to-End P2 + H3 Joint Fine-Tuning

## 2×2 matrix

| Arm | P2 | H3 (Enc+Match+Head) | Loss |
|---|---|---|---|
| A0 | frozen | train | RGB MSE |
| A1 | frozen | train | L1 + 1.0·(1−MS-SSIM) |
| A2 | unfreeze (no LTE; lv2 head frozen) | train | RGB MSE |
| A3 | unfreeze (same) | train | L1 + 1.0·(1−MS-SSIM) |

## Protocol

- Init: P2 `model_00004.pt` (M1 + lv2 bypass) + H3@2000
- Seed 42, Adam, **2000** updates
- LR: P2 1e-6 / Enc 1e-6 / Match 2e-6 / Head 1e-5
- **Unified 192×192 crop training** (Full-Image A2 OOM ~21GB+); Eval30 remains full-image tiled
- Huawei Train500 / Eval30 only

## Loss audit

See `TTSR_LOWLIGHT_V24_LOSS_AUDIT.md` — original P2 used L1+MS-SSIM(w=1), not 0.1·SSIM.
