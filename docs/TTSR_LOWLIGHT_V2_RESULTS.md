# TTSR-lowlight v2.0 — Results

**Decision: Situation B — H3 (V3-B.0 Implicit Residual) is the strongest Huawei candidate; H1 LocalRefine does not beat P2 on mean PSNR.**

Retain P2 as the production default until H3 is further stress-tested. Do **not** claim H3’s gain equals “removing the explicit gate” — training recipe differs (MSE vs L1+perceptual+smooth+color; frozen H1 features).

---

## 1. Identity

| Item | Value |
|---|---|
| Branch | `experiment/ttsr-lowlight-v2` |
| Git HEAD (at branch create) | `7e1f0ab` |
| Historical modules | `apeanut60/TTSR-lowlight` **v5** @ `d3c4bd9` |
| Frozen TTSR | `model_00004.pt` sha256 `1811c4f9…1fe0` |
| Base config | **P2** (M1 α=0.75 β=0.50 λ=0.25 + lv2 `branch_bypass`) |
| Train data | Huawei Nano **500** only (Nikon unused) |
| Eval | Huawei Nano Eval30 |
| Experiment root | `/root/data/experiments/TTSR-lowlight-v2/` |

Plan / reuse: `docs/TTSR_LOWLIGHT_V2_PLAN.md`, `diagnostics/v2/code_reuse_sources.txt`.

---

## 2. Data

| Split | N | Status |
|---|---|---|
| Huawei Train500 | **500** | All decode OK; Low/GT/Nano basename-paired |
| Huawei Eval30 | 30 | Same as v1r13 |

Manifests: `diagnostics/v2/huawei_train_manifest.csv`,  
`/root/data/experiments/TTSR-lowlight-v2/manifests/`.

No GT-as-Nano fallback. No Nikon fill.

---

## 3. Modules

| Module | Params | Notes |
|---|---|---|
| `Refiner` (LocalRefine) | **78,724** | ch=32, radius=4, τ=0.1, C_out zero-init |
| `V3B0ResidualFusion` | head only (encoder/match frozen from H1) | in_ch=96, out zero-init |

Dense↔chunked LocalSoftMatch: unit-tested equal.

---

## 4. P2 regression (H0)

Eval30 from float32 P2 cache:

| Metric | Target | Got |
|---|---|---|
| RGB PSNR | ≈20.264 | **20.265** |
| SSIM | ≈0.6341 | **0.6340** |

Protocol: `ref_degrade=False`, no mean-align, tile 256/96.

---

## 5. Training

Shared: seed=42, batch=8, crop=128, 3000 Adam updates, lr 1e-4→5e-5 @2000, Huawei Train500, identical P2 Y0 cache.

| Run | Arm | Loss | Log |
|---|---|---|---|
| H1 | Nano | L1+0.1 Per+1.0 Smooth+0.5 Color | `logs/train_H1_nano.log` |
| H2 | Self | same | `logs/train_H2_self.log` |
| H3 | Nano features | **MSE** on Y0+ΔRGB; frozen H1 encoder+match | `logs/train_H3_v3b0.log` |

Shared Refiner init: `checkpoints/refiner_init_s42.pt`.

### Checkpoints (selected)

| Run | Path | sha256 (prefix) |
|---|---|---|
| H1 final | `…/H1_nano_s42/checkpoint_03000.pt` | `1c894513…` |
| H2 final | `…/H2_self_s42/checkpoint_03000.pt` | `bd5c072d…` |
| **H3 best** | `…/H3_v3b0_s42/checkpoint_02000.pt` | `742d366b…` |
| H3 final | `…/H3_v3b0_s42/checkpoint_03000.pt` | `c5b938e6…` (20.437) |

---

## 6. Main Huawei Eval30 results

| Run | PSNR | SSIM | ΔPSNR vs H0 |
|---|---|---|---|
| **H0 P2** | **20.265** | 0.6340 | — |
| H1 LocalRefine Nano | 20.265 | 0.6314 | **−0.000** |
| H2 LocalRefine Self | 20.109 | 0.6293 | −0.156 |
| **H3 V3-B.0 (step 2000)** | **20.466** | 0.6331 | **+0.201** |
| H3 step 3000 | 20.437 | 0.6335 | +0.172 |

Challenge target 20.35: **H3 meets it**; H1 does not.

---

## 7. Paired / robustness (Eval30)

### H1 − H0
- mean −0.000, median +0.045, 16W/14L  
- Large per-image swing (σ≈0.81); worst −2.63 (2053), best +1.75 (2044)

### H1 vs H2
- H1 beats H2 mean +0.155 (16W/13L) — Nano second path helps vs trained-Self arm

### H1 reference ablation (trained Nano weights)
| Condition | PSNR |
|---|---|
| Correct Nano | 20.265 |
| Self input | 20.094 |
| Mismatch Nano | 18.573 |
| Bypass (=P2) | 20.265 |

Correct ≫ Mismatch (+1.69 mean) → second-path Nano is used, but net mean does not exceed P2.

### H3 − H0 (best ckpt)
- mean **+0.201**, median **+0.418**, 20W/10L  
- Worst still severe: 2053 −2.86 dB

Caveat: H3 vs H1 is **not** a pure gate ablation (different loss + frozen features).

---

## 8. Residual frequency (H1 focus set)

On focus images, gated ΔRGB energy is **~99% low-frequency** (`ratio_low` ≈ 0.99).  
**Do not** describe H1 as recovering true high-frequency texture.

---

## 9. Go / No-Go

| Criterion | Result |
|---|---|
| H1 > H0 (20.264) | **No** (flat mean) |
| H1 > H2 | Yes |
| Correct Nano > Mismatch | Yes |
| H3 > H0 | **Yes (+0.20 dB)** |
| H3 > H1 | Yes |

**Recommendation**

1. Keep **P2** as default inference until H3 is validated more broadly.  
2. Promote **H3 step-2000** as the v2.0 research candidate for Huawei Eval30.  
3. Do **not** auto-start GateNet / full backbone retrain / Nikon mix.  
4. Next optional work (not started): H3 mismatch/self ablation; frequency analysis of H3 ΔRGB; early-stop / lower LR for H1.

---

## 10. Reproduction

```bash
conda activate ttsr
cd /root/projects/TTSR-lowlight-v1-data1SOTA
git checkout experiment/ttsr-lowlight-v2

# tests
python tests/test_v2_local_refine.py

# caches (already built under /root/data/experiments/TTSR-lowlight-v2/)
python scripts/prepare_v2_p2_cache.py

# train
python scripts/train_v2_local_refine.py --arm nano   # H1
python scripts/train_v2_local_refine.py --arm self   # H2
python scripts/train_v2_v3b0.py                      # H3 after H1

# formal eval
python scripts/eval_v2_formal.py \
  --h3_ckpt /root/data/experiments/TTSR-lowlight-v2/checkpoints/H3_v3b0_s42/checkpoint_02000.pt
```

Artifacts: `metrics/formal_summary.json`, `metrics/per_image_metrics.csv`,  
`cache_p2_{train,eval}/`, `checkpoints/`, `logs/`.

---

*v2.0 execution complete. No automatic follow-on version started.*
