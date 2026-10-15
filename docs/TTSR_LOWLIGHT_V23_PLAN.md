# TTSR-lowlight v2.3 — Reference Feature Joint Fine-Tuning

## Hypothesis

H3 only trains `V3B0ResidualFusion` while SharedEncoder and LocalSoftMatch stay frozen.
Unfreezing Matcher (H1) and SharedEncoder+Matcher (H2) may adapt Nano features to Huawei RGB reconstruction and improve Eval30 PSNR beyond head-only FT.

## Arms (all from H3@2000, P2 frozen)

| Arm | Trainable | LR groups |
|---|---|---|
| H0 | Head | head 1e-5 |
| H1 | Matcher Q/K/V + Head | matcher 2e-6, head 1e-5 |
| H2 | SharedEncoder + Matcher + Head | enc 1e-6, matcher 2e-6, head 1e-5 |

## Protocol

- Seed 42, Adam, 1000 updates, RGB MSE only
- Full-image batch=1 preferred; Crop 128 only if VRAM forces *all* arms
- H1/H2 must keep differentiable Encoder/Matcher paths (no detached F0/T for trainable modules)
- Huawei Train500 / Eval30 only; Eval30 is a reused development set

## Init

`/root/data/experiments/TTSR-lowlight-v2/checkpoints/H3_v3b0_s42/checkpoint_02000.pt`
