# TTSR-lowlight v2.5 — L1 / MS-SSIM Weight Balance

## Formula (only variable: λ)

\[
L=\mathrm{L1}(Y,GT)+\lambda\bigl(1-\mathrm{MS\text{-}SSIM}(Y,GT)\bigr)
\]

| Arm | λ |
|---|---:|
| W0 | 1.0 (v2.4 A3 recipe; reuse if protocol-identical) |
| W1 | 0.3 |
| W2 | 0.1 |

No MSE / perceptual / aux losses. Joint P2+H3, crop-192, seed 42, 2000 steps, same LRs as v2.4 A3.
