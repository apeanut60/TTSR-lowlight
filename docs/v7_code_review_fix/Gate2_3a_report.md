# Gate 2.3a — C2 Geometry Controls (corrected)

- **HEAD:** `7155d20af37cad7f3ec1d0cddba869bce00865be`
- **C2 SHA256:** `09609b61283680829ffcd59707b080abd901d4623f917ddc3ce35b37fe927b00`
- **Retraction:** Gate 2.3 absolute Flow expectation on rolled Nano Ref is **invalid** (Nano ↛ pixel-aligned to Y0).

## Flow convention

- Matcher: `flow = (sx−tx, sy−ty)` (source−target)
- Warp: sample source at `(x+dx, y+dy)`
- `torch.roll(key, +dy,+dx)` ⇒ expected interior flow `≈ (+dx_feat, +dy_feat)` for **Y0↔rolled Y0** only

## Task 1 — Y0 positive control

### microset

| acc≤2px | err median | mean (dx,dy) | expected | pass |
|---|---|---|---|---|
| 1.000 | 0.000 | (2.99, 1.99) | (3.00, 2.00) | **True** |

### microset-heldout

| acc≤2px | err median | mean (dx,dy) | expected | pass |
|---|---|---|---|---|
| 1.000 | 0.000 | (2.99, 2.00) | (3.00, 2.00) | **True** |

## Task 2 — Nano relative source-index consistency

### microset

| idx exact acc | idx L1 | Δflow within 2px | pass |
|---|---|---|---|
| 0.901 | 9.230 | 0.902 | **True** |

### microset-heldout

| idx exact acc | idx L1 | Δflow within 2px | pass |
|---|---|---|---|
| 0.875 | 12.216 | 0.876 | **True** |

## Task 3 — Identity vs C2 warp (no GT as model input)

### microset

| Correct ΔL1(id−c2) | Correct Δgc(c2−id) | Shuffle Δgc(c2−id) | c2_useful |
|---|---|---|---|
| -0.0900 | -0.8245 | -0.4866 | **False** |

### microset-heldout

| Correct ΔL1(id−c2) | Correct Δgc(c2−id) | Shuffle Δgc(c2−id) | c2_useful |
|---|---|---|---|
| -0.0858 | -0.8901 | -0.5466 | **False** |

## Task 4 — Expand-train leakage audit

- `train[:64]` leak into blocked microset/heldout: **True** (n=24)
  - leaked IDs: `['low00001.png', 'low00002.png', 'low00003.png', 'low00004.png', 'low00005.png', 'low00006.png', 'low00007.png', 'low00009.png', 'low00010.png', 'low00011.png', 'low00012.png', 'low00014.png', 'low00015.png', 'low00016.png', 'low00017.png', 'low00018.png', 'low00019.png', 'low00020.png', 'low00021.png', 'low00022.png', 'low00023.png', 'low00024.png', 'low00025.png', 'low00026.png']`
- exclusive expand pool size: **551** (train − micro8 − heldout16)
- formal `split.dev` size: **64**
- lists: `/root/data/experiments/V7_CODE_REVIEW_FIX/gate2_3a/exclusive_id_lists.json`

### Suggested single-variable budget (proposal only)

```json
{
  "updates": 400,
  "n_images": 32,
  "states": [
    "correct"
  ],
  "d1_enabled": false,
  "a2_mode": "tex_h4",
  "freeze_llformer": true,
  "freeze_align": true,
  "setting": "A",
  "seed": 42,
  "lr": 0.0001,
  "eval_on": [
    "expand_train_subset",
    "microset_heldout_16",
    "formal_dev"
  ],
  "note": "Gate 2.4 bounded pilot proposal only: increase n with exclusive IDs; keep tex_h4; do NOT start here"
}
```

## Revised C2 readiness

| check | value |
|---|---|
| Y0 positive control | True |
| Nano relative consistency | True |
| C2 warp > identity on Correct | False |
| ready for C2-warp pilot | **False** |

## Gate 2.4 bounded pilot (proposal — do not start)

- **type:** `expand_microset_tex_h4_bounded_pilot`
- C2 Y0-shift **PASS** and Nano relative consistency **PASS**, but **C2-warp hurts vs identity** on Nano Correct (Δgc ≈ −0.82). Do **not** wire C2-warp yet.
- Next: exclusive **n=32** `tex_h4` pilot, updates≤400, eval on microset-heldout + formal `split.dev` (IDs in `exclusive_id_lists.json`).

`train_authorization=DENIED` this round.

## Repro

```bash
cd /root/projects/TTSR-lowlight && /root/miniconda3/envs/ttsr/bin/python scripts/gate2_3a_c2_geometry_fix.py --device cuda
```
