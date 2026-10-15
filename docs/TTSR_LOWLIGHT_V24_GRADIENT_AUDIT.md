# TTSR-lowlight v2.4 — Gradient Audit

## Protocol decision

| Probe | Peak VRAM | Result |
|---|---:|---|
| Crop 128 joint MSE | ~1.0 GB | OK, P2 grads 252/252 trainable |
| Crop 192 joint MSE / L1+MS-SSIM | ~2.2 GB | OK |
| Crop 256 joint | ~3.9 GB | OK |
| Full 720×960 joint | OOM (~21 GB+) | **Fail** |

**Unified training protocol: 192×192 aligned crops** (matches v1r3 `train_crop_size=192`).  
Eval30 remains full-image tiled P2 (tile=256, overlap=96).

## Freeze / train groups

| Component | A0/A1 | A2/A3 |
|---|---|---|
| P2 MainNet / SearchTransfer / RefCorrection | frozen | **train** @ 1e-6 |
| P2 LTE / LTE_copy (VGG) | frozen | frozen |
| `MainNet.conv22_head` (lv2 bypass unused) | frozen | frozen (excluded) |
| H3 Enc / Match / Head | train | train |

Effective P2 trainable params (A2/A3): **~4.94 M** (of 6.05 M total; LTE ~1.11 M frozen; lv2 head excluded).

## Tests

| Test | Result |
|---|---|
| `test_v24_loss.py` | MS-SSIM(X,X)≈0; L1+MS-SSIM & MSE grads finite |
| `test_v24_freeze_groups.py` | A0: P2 no grads; A2: 252/252 P2 grads + optimizer updates |
| `test_v24_gradient_flow.py` | MSE and L1+MS-SSIM: Loss→H3→Y0→P2 connected |
| Step0 (training `save_eval(0)`) | Expect Y≈20.466, P2-only≈20.265 |

## Notes

- Query/hist maps from `estimate_query_align_maps` are computed under `no_grad` (non-parametric); M1 formulas unchanged.
- A0/A1: `Y0` detached; A2/A3: `Y0.requires_grad=True`.
- No detached-cache path for joint arms; crops preclude full-image Y0 cache reuse.
