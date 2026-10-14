# v1r6-FT — Short Fine-Tune for Query Gamma Alignment

**Branch:** `experiment/v1r6-query-illum-align`  
**Init:** `.../TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt` (v1r3e4, **not overwritten**)  
**Train:** 250 Huawei Nano matched pairs (manifest), crop 192, L1 + 1.0×(1−MS-SSIM), freeze LTE, Adam lr=3e-6, seed=42, 5 epochs  
**Eval:** Huawei Nano Ref, 30 images, `ref_degrade=False`, `eval_mean_align=False`

## 1. Code / config changes

| Change | Detail |
|---|---|
| `dataset/data1_nanobanana.py` | Estimate `query_gamma` on **full** Low + Nano Ref **before** crop; store in sample |
| `dataset/data1.py` `ToTensor` | Pass through `query_gamma` scalar |
| `trainer.py` | Pass `query_gamma_override` when align on; epoch avg L1/MS-SSIM; per-image JSON |
| `model/TTSREnhance.py` | Strict core load: log 266/266, raise on missing core |
| `dataset/dataloader.py` | Seeded `Generator` + `worker_init_fn` when `--seed≥0` |
| `main.py` | `cudnn.deterministic` when seeded |
| `train/train_v1r6_ft_c0.sh` | C0 control |
| `train/train_v1r6_ft_c1.sh` | C1 α=0.75 |

**Training gamma policy (verified):**

1. Full-image Low + Nano Ref → γ (never GT).  
2. Same γ applied to cropped Query via `query_gamma_override`.  
3. Low/GT stay co-cropped; Ref keeps independent crop/geom aug (unchanged).  
4. MainNet input = original Low crop; Key/Value = Ref (no gamma).  
5. α=0 / align=off path unchanged.

**Load check:** both runs `[Pretrain] Loaded 266/266 ckpt tensors … missing_in_ckpt=0 (core=0)`.

## 2. Data integrity

| Item | Status |
|---|---|
| Train pairs | 250 (`.nanobanana_sample_manifest/Huawei.txt`) |
| Eval | 30 Huawei + 30 Nano refs |
| Missing Nano → error | yes (no GT fallback when `ref_dir` set) |
| v1r3e4 preserved | yes; new dirs under `v1r6-ft-*` |

## 3. Epoch table (same-epoch C1−C0)

Epoch 0 = prior zero-shot A0 / A3.

| Epoch | C0 PSNR | C0 SSIM | C1 PSNR | C1 SSIM | C1−C0 PSNR |
|---|---|---|---|---|---|
| 0 | **20.162** | **0.6324** | 20.068 | 0.6315 | −0.094 |
| 1 | 20.002 | 0.6309 | 20.004 | 0.6323 | +0.002 |
| 2 | 20.057 | 0.6314 | 20.066 | 0.6328 | +0.008 |
| 3 | 20.082 | 0.6321 | 20.088 | 0.6336 | +0.007 |
| 4 | 19.984 | 0.6298 | 20.035 | 0.6332 | +0.051 |
| 5 | **20.097** | **0.6333** | **20.112** | **0.6346** | +0.014 |

**Best checkpoints**

- C0 best: e5 → `/root/data/experiments/retry/v1r6-ft-C0-control/model/model_00005.pt` (20.097)  
- C1 best: e5 → `/root/data/experiments/retry/v1r6-ft-C1-gamma075/model/model_00005.pt` (20.112)

## 4. vs v1r3e4 and vs zero-shot Gamma

| Comparison | ΔPSNR |
|---|---|
| C1e5 − v1r3e4 (20.162) | **−0.050** (does **not** beat baseline) |
| C1e5 − zero-shot A3 (20.068) | **+0.044** (recovers most of the α=0.75 zero-shot drop) |
| C0e5 − v1r3e4 | **−0.065** (continue-FT alone also regresses) |
| C1e5 − C0e5 (same budget) | **+0.014** (< 0.02 → **not** a meaningful gain) |

Short FT at 3e-6 for 5 epochs **does not** improve over the frozen v1r3e4 checkpoint for either arm. C1 recovers the zero-shot gamma PSNR loss relative to A3, but stays below 20.162.

## 5. Per-image C1−C0 at epoch 5

Threshold ±0.01 dB: **17 up / 13 down / 0 flat**, mean Δ = +0.014.

| idx | C0 | C1 | Δ | idx | C0 | C1 | Δ |
|---|---|---|---|---|---|---|---|
| 00 | 21.058 | 20.439 | −0.62 | 15 | 16.691 | 16.899 | +0.21 |
| 01 | 21.900 | 21.817 | −0.08 | 16 | 26.769 | 25.541 | −1.23 |
| 02 | 20.616 | 20.681 | +0.07 | 17 | 18.594 | 18.535 | −0.06 |
| 03 | 20.037 | 19.999 | −0.04 | 18 | 22.275 | 22.104 | −0.17 |
| 04 | 23.381 | 23.531 | +0.15 | 19 | 16.101 | 16.480 | +0.38 |
| 05 | 23.545 | 23.698 | +0.15 | 20 | 18.133 | 18.433 | +0.30 |
| 06 | 20.921 | 21.368 | +0.45 | 21 | 22.541 | 22.324 | −0.22 |
| 07 | 16.676 | 16.888 | +0.21 | 22 | 24.795 | 24.693 | −0.10 |
| 08 | 20.261 | 20.215 | −0.05 | 23 | 19.055 | 18.643 | −0.41 |
| 09 | 16.370 | 16.643 | +0.27 | 24 | 20.123 | 20.744 | +0.62 |
| 10 | 17.453 | 17.471 | +0.02 | 25 | 17.217 | 17.771 | +0.55 |
| 11 | 14.770 | 14.957 | +0.19 | 26 | 25.435 | 25.603 | +0.17 |
| 12 | 24.289 | 23.950 | −0.34 | 27 | 16.248 | 16.094 | −0.15 |
| 13 | 19.886 | 20.056 | +0.17 | 28 | 18.741 | 18.778 | +0.04 |
| 14 | 21.583 | 21.526 | −0.06 | 29 | 17.447 | 17.463 | +0.02 |

Large negative outliers (e.g. idx 16) offset many small gains.

## 6. Match diagnostics (formal eval, e5)

| | C0e5 | C1e5 | Zero-shot A3 |
|---|---|---|---|
| max cos | 0.355 | **0.453** | 0.451 |
| id cos | 0.218 | **0.313** | 0.314 |
| match L2 | 26.52 | **21.71** | 21.83 |
| ≤4 px frac | 0.133 | **0.192** | 0.192 |
| Query ‖·‖ | 1.23 | 1.77 | (similar) |

FT does **not** further improve matching beyond zero-shot γ; matching stays in the improved regime while PSNR remains below v1r3e4.

## 7. Train losses (avg)

| Epoch | C0 L1 | C0 MS-SSIM | C1 L1 | C1 MS-SSIM |
|---|---|---|---|---|
| 1 | 0.165 | 0.136 | 0.167 | 0.137 |
| 2 | 0.167 | 0.145 | 0.169 | 0.146 |
| 3 | 0.171 | 0.149 | 0.172 | 0.150 |
| 4 | 0.166 | 0.130 | 0.167 | 0.131 |
| 5 | 0.163 | 0.144 | 0.164 | 0.145 |

Loss tracks are nearly identical (same seed / batch order).

## 8. Second-round α sweep

**Not run.** C1 only edges C0 by ≤0.015 dB at matched epochs (e4’s +0.05 is from a C0 dip). Primary target (>20.162) is unmet for both arms. Retuning α under the same budget is not justified.

## 9. Conclusion

| Question | Answer |
|---|---|
| Does short FT adapt MainNet to gamma Query? | Partially: C1 recovers ~0.04 dB vs zero-shot A3 |
| Does C1 beat same-budget C0? | Marginally (+0.01–0.015); **not** meaningful (≥0.01 claimed only with caution; <0.02 overall) |
| Does C1 beat v1r3e4 20.162? | **No** (best 20.112) |
| Is gamma alignment helpful for Huawei PSNR? | Matching yes; **PSNR no** under this FT recipe |
| Keep gamma as default? | **No** for production Huawei eval; keep code path optional |

## 10. Next suggestions

1. **Stop** investing in query-gamma for Huawei PSNR with this backbone/loss.  
2. If revisiting matching: try **learned** query tone (tiny 1×1/MLP) with end-to-end FT, not fixed Rec.709 γ.  
3. Prefer other levers already shown to matter more than reference matching polish (tone/illum head, data quality).  
4. Retain `query_illum_align` as a diagnostic / research flag; default remains off.

## 11. Artifacts

```
/root/data/experiments/retry/v1r6-ft-C0-control/
/root/data/experiments/retry/v1r6-ft-C1-gamma075/
/root/data/experiments/retry/v1r6-ft-compare/ft_compare_summary.json
/root/data/experiments/retry/v1r6-ft-compare/match_best.json
```
