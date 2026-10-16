# TTSR-lowlight Nikon Domain FT — Results

**Verdict:** Domain rebuild works. Best research ckpt = **N3 joint A2@250** at **17.217 / 0.5159** on Nikon Eval20  
(+0.23 vs Huawei A2@250 zero-shot, +0.53 vs Huawei P2 zero-shot).  
Joint gain over N2 frozen-H3 is tiny (+0.002); most lift comes from **N1 Nikon-P2**. Eval20 n=20 — do not over-read 0.01 dB.

## Protocol

| Item | Value |
|---|---|
| Branch | `experiment/ttsr-lowlight-nikon` |
| Exp root | `/root/data/experiments/TTSR-lowlight-v27-nikon/` |
| Train | Nano man500 − Eval stems **3004/3016/3017** → **n=497** |
| Eval | Nikon Eval20 (`nanobanana_ref_Nikon`, 1264×843→resize) |
| Crop | Random 192, schedule `random_crop_s42.csv` |
| Loss (N1–N3) | RGB **MSE** |
| Eval tile | P2 256 / overlap 96 |

## Init chain

| Stage | Init |
|---|---|
| N1 P2 | `v1r3-nano-Huawei/model/model_00004.pt` (P2 LR 1e-5) |
| N2 H3 | Huawei `H3_v3b0_s42@2000` + N1@1000 P2 frozen |
| N3 Joint | N1@1000 P2 + N2@500 H3 (P2 LR 1e-6) |

**Not** continued from Huawei A2@250 (zero-shot only in P0).

## P0 — Zero-shot (Huawei weights)

| Variant | PSNR_RGB | SSIM_RGB | P2-only |
|---|---:|---:|---:|
| P2 `model_00004` | 16.683 | 0.5096 | 16.683 |
| H3 (P2+H3@2000) | 16.971 | 0.5118 | 16.683 |
| A2@250 Huawei | 16.989 | 0.5134 | 16.706 |

Artifacts: `/root/data/eval/nikon-p0-{P2,H3,A2huawei250}/`

## N1 — Nikon-P2 FT

| Step | Eval20 PSNR | SSIM |
|---:|---:|---:|
| 0 | 16.683 | 0.5096 |
| 250 | 16.941 | 0.5132 |
| 500 | 16.885 | 0.5115 |
| **1000** | **17.088** | **0.5135** |
| 1500 | 16.910 | 0.5083 |
| 2000 | 16.989 | 0.5097 |

**Best @1000.** Ckpt: `checkpoints/N1_p2_only_mse_s42/checkpoint_01000.pt`

## N2 — Nikon-H3 (frozen P2) = A0_nikon

| Step | Y PSNR | SSIM | P2-only |
|---:|---:|---:|---:|
| 0 | 17.168 | 0.5145 | 17.088 |
| 250 | 17.207 | 0.5147 | 17.088 |
| **500** | **17.214** | **0.5147** | 17.088 |
| 1000 | 17.201 | 0.5147 | 17.088 |
| 2000 | 17.183 | 0.5147 | 17.088 |

**Best @500.** Ckpt: `checkpoints/N2_frozenP2_mse_s42/checkpoint_00500.pt`

## N3 — Joint A2-style MSE

| Step | Y PSNR | SSIM | P2-only |
|---:|---:|---:|---:|
| 0 | 17.214 | 0.5147 | 17.088 |
| **250** | **17.216** | **0.5159** | **17.094** |
| 500 | 17.215 | 0.5159 | 17.098 |
| 1000 | 17.195 | 0.5153 | 17.092 |
| 2000 | 17.164 | 0.5145 | 17.083 |

**Best @250** (same early-stop pattern as Huawei A2).  
Ckpt: `checkpoints/N3_jointP2_mse_s42/checkpoint_00250.pt`

2×2 @250: A2 − A0(N2@500) ≈ **+0.002** (joint ≈ no-op beyond N2); N1 domain P2 is the main lever.

## Final export — N3@250

Dir: `/root/data/eval/nikon-N3-A2-250/` (`save_results/` + GT/Low/P2 + metrics)

| Metric | Mean |
|---|---:|
| PSNR_RGB | **17.2165** |
| SSIM_RGB | **0.515913** |
| PSNR_R / G / B | 17.22 / 17.20 / 17.25 |
| SSIM_R / G / B | 0.5159 / 0.5161 / 0.5157 |
| P2-only PSNR | 17.0945 |

## Lessons transferred from Huawei

| Lesson | Applied |
|---|---|
| MSE > L1+MS-SSIM for PSNR | N1–N3 all MSE |
| Early stop ~250 for joint | N3 best @250; longer regresses |
| Random crop not fixed | `random_crop_s42.csv` |
| Domain P2 matters | N1 alone +0.40; Huawei P2 did not transfer |
| Do not start from A2huawei joint | P0 only; rebuild P2→H3→joint |
| λ / L1+MS-SSIM grids | Skipped |

## Scripts

- `scripts/build_nikon_manifests.py`
- `scripts/build_nikon_random_crop.py`
- `scripts/train_nikon_domain.py` (`--arm P2|A0|A2`)
- `scripts/eval_nikon_export.py` (`--mode p0|export`)
