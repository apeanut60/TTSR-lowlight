# TTSR-lowlight v2.6 — Random Crop & P2 LR Results

**Verdict: Case A + Case B (under L1+MS-SSIM joint FT).**  
1. **Random 192 crop (C1) clearly beats fixed-crop v2.5 W0** on Eval30 @2000 (20.459 vs 20.381).  
2. **Lower P2 LR (C2, 2e-7) beats C1** on Eval PSNR at every matched step after 0, with much smaller P2 drift and better P2-only.  
3. Absolute PSNR still **below v2.4 A2@250 MSE (20.494)**; C2@2000 ≈ H3@2000 (20.464). Eval30 is a reused development set.

---

## 1. Crop audit

v2.5 used **`md5(seed|id)` → one fixed crop per image for all visits**  
(`scripts/train_v25_loss_balance.py::stable_crop_xy`).  
Example `Huawei/00001.jpg`: visits 1–4 all `(497, 696)`.

v2.6 schedule: **each step resampled**; 500 images × 4 visits × **4 unique crops**.  
Manifest: `/root/data/experiments/TTSR-lowlight-v26/manifests/random_crop_s42.csv`.  
Details: `TTSR_LOWLIGHT_V26_CROP_AUDIT.md`.

W0 vs C1 differ in crop policy (fixed vs random) with otherwise matched init/loss/LRs → attributing Δ to crop is supported.

## 2. Protocol

| Item | Value |
|---|---|
| Loss | \(L=\mathrm{L1}+1.0(1-\mathrm{MS\text{-}SSIM})\) |
| Crop | Random 192, precomputed paired schedule |
| C1 | P2 LR **1e-6** |
| C2 | P2 LR **2e-7** (only difference) |
| H3 LRs | Enc 1e-6 / Match 2e-6 / Head 1e-5 |
| Init | P2 `model_00004` + H3@2000 (not from v2.5 finals) |
| Steps | 2000, seed 42, batch 1, Adam |

## 3. Curves

| Arm | Step | Train80 crop PSNR | Train80 full | Eval PSNR | Eval SSIM | P2-only | n&lt;−1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | 0 | 20.728 | 19.986 | 20.466 | 0.6331 | 20.265 | 6 |
| C1 | 250 | 20.885 | — | 20.436 | 0.6366 | 20.163 | 3 |
| C1 | 500 | 20.864 | — | 20.431 | 0.6369 | 20.165 | 4 |
| C1 | 1000 | 20.890 | 20.058 | 20.420 | 0.6371 | 20.159 | 3 |
| C1 | 1500 | 20.865 | — | 20.426 | 0.6375 | 20.159 | 4 |
| **C1** | **2000** | 20.774 | 20.045 | **20.459** | **0.6374** | **20.212** | 6 |
| C2 | 0 | 20.728 | 19.986 | 20.466 | 0.6331 | 20.265 | 6 |
| C2 | 250 | 20.839 | — | 20.457 | 0.6355 | 20.242 | 4 |
| C2 | 500 | 20.860 | — | 20.450 | 0.6363 | 20.226 | 3 |
| C2 | 1000 | 20.878 | 20.046 | 20.429 | 0.6367 | 20.209 | 3 |
| C2 | 1500 | 20.867 | — | 20.439 | 0.6369 | 20.206 | 3 |
| **C2** | **2000** | 20.817 | 20.031 | **20.464** | **0.6370** | **20.218** | 4 |

### vs history @2000 (or noted)

| Model | Eval PSNR | Eval SSIM | P2-only |
|---|---:|---:|---:|
| H3@2000 | 20.466 | 0.6331 | 20.265 |
| v2.5 W0@2000 (fixed crop, P2 1e-6) | 20.381 | 0.6373 | 20.102 |
| **C1@2000** (random, P2 1e-6) | **20.459** | 0.6374 | 20.212 |
| **C2@2000** (random, P2 2e-7) | **20.464** | 0.6370 | 20.218 |
| v2.4 A2@250 MSE | 20.494 | 0.6355 | — |

**C1 − W0 @2000 ≈ +0.078 dB** (random crop benefit under matched λ=1 / P2 LR).  
**C2 − C1 @2000 ≈ +0.005 dB**; mid-run C2−C1 up to **+0.021** (@250).

## 4. P2 LR effect (C2 − C1)

| Step | Δ Eval PSNR | Δ P2-only |
|---|---:|---:|
| 250 | +0.021 | +0.079 |
| 500 | +0.019 | +0.061 |
| 1000 | +0.009 | +0.049 |
| 1500 | +0.013 | +0.046 |
| 2000 | +0.005 | +0.006 |

| Arm | ‖ΔP2‖ @2000 | ‖ΔEnc‖ | ‖ΔHead‖ |
|---|---:|---:|---:|
| C1 | **0.155** | 0.006 | 0.161 |
| C2 | **0.044** | 0.009 | 0.193 |

Lower P2 LR cuts P2 drift ~3.5× and keeps P2-only closer to 20.265 through mid-training.

## 5. Reference ablation @2000

| Arm | Correct | Self | Mismatch | Zero-T | Bypass | C−S |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 20.459 | 20.259 | 19.789 | 19.174 | 20.212 | 0.200 |
| C2 | 20.464 | 20.321 | 19.785 | 19.076 | 20.218 | 0.143 |

Nano dependence retained. C2 has slightly smaller Correct−Self.

### Focus @2000 (PSNR)

| ID | C1 Y / P2 | C2 Y / P2 |
|---|---|---|
| 2053 | 21.21 / 24.24 | **21.88 / 24.88** |
| 2059 | 22.24 / 23.95 | **22.83 / 24.39** |
| 2055 | 20.04 / 21.42 | 20.32 / 21.64 |
| 2052 | 19.35 / 17.58 | 19.09 / 17.49 |
| 2044 | 19.32 / 17.69 | 18.64 / 17.06 |

C2 softens some hard failures (2053/2059) via a healthier P2 base; 2053 still ≪ P2.

## 6. Answers

| Q | Answer |
|---|---|
| Random crop vs fixed? | **Yes** — C1@2000 ≫ W0@2000; Train Full also edges up early. |
| Lower P2 LR help? | **Yes** — less P2-only collapse, better Eval mid/late. |
| Stable L1+MS-SSIM joint? | **Improved, not solved** — Eval no longer crashes to ~20.37; still no new PSNR SOTA vs A2 MSE. |

## 7. Case calls

- **Case A:** Random crop vs audited fixed W0 → positive.  
- **Case B:** C2 > C1 on Eval + P2-only stability.  
- Absolute: retain **v2.4 A2@250** as best PSNR candidate; **C2@2000** as best L1+MS-SSIM joint under random crop.

Do not auto-expand LR grids or add losses this round.

## 8. Runtime / SHA256

~21–23 min / arm, ~2.2 GB VRAM.

| Ckpt | SHA256 |
|---|---|
| C1@250 | `d9c2b193757245e1af36762be9416cc7191239a21b061338c276d226e5af40cc` |
| C1@2000 | `cd90f406cd277d219cbddde6dde8574dab4a89ebbd32e354af73f5881734ccba` |
| C2@250 | `f32d89130c0b6532f074aaf74b9429823127b2b240d0dee20aa5d7293865e970` |
| C2@2000 | `067d4a684de6c9f0357232034b262732130c15b3e9140870cb9303d3bd327b38` |

```bash
export TORCH_HOME=/root/data/cache/torch PYTHONUNBUFFERED=1
cd /root/projects/TTSR-lowlight-v1-data1SOTA
# schedule already at experiments/.../manifests/random_crop_s42.csv
python -u scripts/train_v26_random_crop.py --arm C1 --steps 2000 --seed 42
python -u scripts/train_v26_random_crop.py --arm C2 --steps 2000 --seed 42
python scripts/eval_v26_formal.py
```
