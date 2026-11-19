# TTSR-lowlight — P2-only Official Eval Summary

**Product line = P2-only.** H3 / H1 / LLFormer fusion are historical experiments and are **not** formal scores.

Metrics: uint8 RGB `calc_psnr_rgb` + mean of per-channel SSIM (same as `eval_nikon_export.channel_metrics`). Full P2_CONFIG (M1 texture fusion + lv2 `branch_bypass`).

## Canonical results

| Dataset | Split | n | Formal P2 ckpt | PSNR_RGB | SSIM_RGB | Export |
|---|---|---:|---|---:|---:|---|
| LOLv2-Real | Official Test | 100 | `/root/data/experiments/TTSR-lowlight-lolv2real/checkpoints/L1_p2_only_mse_s42/checkpoint_00100.pt` | **20.8515** | **0.799600** | `/root/data/eval/lolv2-L1-best/` |
| Nikon | Eval20 | 20 | `/root/data/experiments/TTSR-lowlight-v27-nikon/checkpoints/N1_p2_only_mse_s42/checkpoint_01000.pt` | **17.088** | **0.5135** | docs: `TTSR_LOWLIGHT_NIKON_RESULTS.md` (N1) |
| Huawei | Eval30 | 30 | P2-only from v2.4 A2 export (reliable) | **20.2674** | **0.635625** | `/root/data/eval/v24-A2-mse250/` (`P2-only` line) |

### Channel detail (where exported)

**LOLv2 L1@100:** R/G/B PSNR 20.5688 / 21.3336 / 21.0359；SSIM 0.7765 / 0.8306 / 0.7917.

**Nikon N1@1000:** see Nikon results doc (Eval20 mean above).

**Huawei Eval30 P2-only (A2 export):** same protocol as formal A2 folder; full A2 Y includes H3 and is **not** the formal P2 score (A2 Y = 20.4937).

## Init / recipe notes

| Dataset | How the formal P2 was obtained |
|---|---|
| LOLv2 | Domain FT from Huawei `model_00004.pt`, MSE, random crop 192, LR 2e-6, early-stop → best **@100** |
| Nikon | Domain FT from Huawei `model_00004.pt`, MSE, random crop 192, LR 1e-5 → best **@1000** (N1) |
| Huawei | Research P2 baseline under M1+bypass; quoted P2-only is the frozen-P2 forward inside the v2.4 A2@250 export |

## Not formal (do not cite as product)

| Run | Why excluded |
|---|---|
| LOLv2 H3 compose 21.104 | Uses H3 |
| Huawei A2 full 20.494 | Joint P2+H3 |
| Nikon N3 17.217 | Includes H3（P2-only ≈ 17.095, ≈ N1） |
| LOLv2 `P2_official_mse_mixup_scratch` 19.985 | Worse than L1@100 |
| Huawei official FT@6000 ≈ 20.430 | Higher than 20.267 but logged after a script crash; not promoted until clean re-export |

## Related docs

- LOLv2 domain: `docs/TTSR_LOWLIGHT_LOLV2REAL_RESULTS.md` (formal = L1 P2)
- Nikon: `docs/TTSR_LOWLIGHT_NIKON_RESULTS.md`
- Huawei v2.4: `docs/TTSR_LOWLIGHT_V24_RESULTS.md` (for context; formal P2-only is the export P2 line)
- Long-schedule P2 recipe trial: `docs/P2_OFFICIAL_RECIPE_LSRW.md`
