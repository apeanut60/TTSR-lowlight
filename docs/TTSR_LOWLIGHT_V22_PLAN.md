# TTSR-lowlight v2.2 — Full-Image DC-Aware Residual Training

Branch: `experiment/ttsr-lowlight-v22`  
Parent: `experiment/ttsr-lowlight-v2`  
Experiment root: `/root/data/experiments/TTSR-lowlight-v22/`  
P2 Y0 caches: symlinked from v2 (not copied/overwritten)

## Question

Can Full-Image training + explicit RGB DC loss weight reduce H3’s harmful global color/exposure shifts while keeping Correct-Nano gains — without GateNet / Verifier / new modules?

## Arms

| Arm | Init | Train | λ_DC |
|---|---|---|---|
| V0 | H3@2000 | none | — |
| V1 | H3@2000 | Full-image, 1000 steps, lr=1e-5, Adam, seed=42 | 1 (≡ MSE) |
| V2 | same | same | 2 |

Only `V3B0ResidualFusion` is trained. Encoder / LocalSoftMatch / P2 frozen.

## Loss

\[
L = \mathrm{MSE}(D_s, D_s^*) + \lambda_{\mathrm{DC}}\,\mathrm{MSE}(m, m^*)
\]

Verified: λ_DC=1 ⇒ L = MSE(Y0+D, GT).

## Deliverables

- Phase A: `docs/TTSR_LOWLIGHT_V22_PHASE_A.md`, `diagnostics/v22/`
- Train: `scripts/train_v22_fullimage.py`
- Eval: `scripts/eval_v22_formal.py`
- Results: `docs/TTSR_LOWLIGHT_V22_RESULTS.md`
