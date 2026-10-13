# Gate 2.3 — Attention Selection & C2 Geometry

## Locked context

- **run HEAD:** `95e4c846a032f2a24016cff5c83fadabb1e32211`
- **G2.1 ckpt SHA256:** `362e55f4c5010f85cae9aa94613c8a86f0b83c2a925ae209852538c8c618ef39`
- **tex_h4_200 ckpt SHA256:** `0f4141749a7667c18fe7edeb602a3e7cfd27c15e59a03befbd08dc5b6ac95e3e`
- **C2 feature_extraction SHA256:** `09609b61283680829ffcd59707b080abd901d4623f917ddc3ce35b37fe927b00`
- **G2.2 lock:** microset N−Shuf **+0.276 dB**; heldout N−Shuf **−0.045 dB**; heldout N−Base **−2.322 dB**.
- G2.2 was **ref-path diagnosis only**, not heldout generalization.

## Task A — Attention (no train)

| | G2.1 align | tex_h4 |
|---|---|---|
| weight_max | 0.1115 | 0.1132 |
| entropy_norm | 1.0000 | 0.9999 |
| Δq / Δk / Δv | 0.107 / 0.144 / 0.948 | 0.76 / 0.734 / 1.07 |
| grad q/k/v | 1.95e-06 / 5.75e-06 / 0.0144 | 5.03e-05 / 3.62e-05 / 0.0245 |
| selection_real | False | False |
| V/Conv content transfer | True | False |

**Verdict:** `no_spatial_selection` — both arms keep Softmax ≈ uniform (`wmax≈1/9`). G2.1 is classic **V/Conv content transfer** (Δv+Δfusion ≫ Δq/Δk). `tex_h4` updates Q/K more, but weights stay flat → Ref sensitivity is still **not** from peaked spatial attention; treat as non-selective V/window aggregation + fusion.

Viz: `/root/data/experiments/V7_CODE_REVIEW_FIX/gate2_3/attn_viz`

## Task B — Pretrained C2 geometry (no train)

| | value |
|---|---|
| strict_load_verified | True |
| feature shape | [1, 256, 100, 150] |
| channels | 256 |
| geometric_match_ok | **False** |

### microset

| self ‖flow‖ | self frac&lt;1 | Δgrad_corr (corr−shuf) | Δwarp_L1 (shuf−corr) | roll frac near expected |
|---|---|---|---|---|
| 0.018 | 1.000 | +0.0400 | +0.0449 | 0.368 |

### microset-heldout

| self ‖flow‖ | self frac&lt;1 | Δgrad_corr (corr−shuf) | Δwarp_L1 (shuf−corr) | roll frac near expected |
|---|---|---|---|---|
| 0.008 | 1.000 | +0.0121 | +0.0384 | 0.270 |

Confidence delta is reported in CSV but **not** used as proof of match correctness.

## Task C — Independent gate flags

| flag | value | meaning |
|---|---|---|
| `ref_path_works` | **True** | microset Ref path sensitivity |
| `heldout_ref_gain` | **False** | heldout Normal−Shuffle > 0.05 dB |
| `heldout_base_safety` | **False** | heldout Normal−Base > −0.1 dB |
| `allow_long_train` | **False** | all safety flags + C2 geo |

G2.2 `clear_ref_utilization = microset OR heldout` is **retracted** for authorizing further training.

## Next step (proposal only — no train this round)

- **type:** `expand_microset_single_variable`
- Propose only: keep tex_h4 (or identity) path; expand correct-only microset beyond n=8 to test whether heldout_ref_gain / heldout_base_safety can turn positive — DO NOT TRAIN in Gate 2.3

## Verdict

**Gate 2.3 diagnostic complete.** `train_authorization=DENIED`.

## Repro

```bash
cd /root/projects/TTSR-lowlight
/root/miniconda3/envs/ttsr/bin/python scripts/gate2_3_c2_and_attn_audit.py --device cuda
```
