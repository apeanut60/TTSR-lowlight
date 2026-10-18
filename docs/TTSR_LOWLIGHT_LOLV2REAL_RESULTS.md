# TTSR-lowlight LOLv2-Real Domain FT — Results

**Verdict:** 延续 H3。LOLv2 上有效组合是 **L1@100 P2 + H1@1000 Enc/Match + 华为 H3@2000 头，不再 FT** → Official Test **21.104**  
(+0.25 vs L1 P2 20.852；+0.17 vs H1-only 20.931)。对本域再训 H3 头只会掉分。

## Protocol

| Item | Value |
|---|---|
| Script | `scripts/train_lolv2_domain.py` |
| Exp root | `/root/data/experiments/TTSR-lowlight-lolv2real/` |
| Train | Low∩Normal∩`nanobanana_ref_v2` → **n=639** |
| Eval | Official Test 100 + `nanobanana_ref_v3` |
| Crop | Random 192 (`manifests_h3/random_crop_s42.csv`) |
| Loss | RGB **MSE** |
| P2 config | M1 texture fusion + lv2 `branch_bypass` |
| Metrics | RGB `calc_psnr_rgb` + mean-channel SSIM |
| Not used | LLFormer, GT-degrade Stage A, exposure loss, joint A2 |

## Init chain

| Stage | Init |
|---|---|
| L1 P2 | `v1r3-nano-Huawei/model/model_00004.pt` (P2 LR **2e-6**) |
| L2 H3 | Huawei `H3_v3b0_s42@2000` + L1@100 P2 frozen |

**Not** continued from illum-only e1 or fusion joint ckpts.

## P0 — Zero-shot (full P2_CONFIG)

| Variant | PSNR_RGB | SSIM_RGB | R / G / B PSNR |
|---|---:|---:|---|
| Huawei `model_00004` | **20.7036** | 0.7941 | 20.592 / 20.937 / 20.831 |
| illum e1 (control only) | 20.5792 | 0.7997 | 20.310 / 21.005 / 20.745 |

Artifacts: `/root/data/experiments/TTSR-lowlight-lolv2real/p0_full_p2config/`

Gate for L2: L1 best ≥ P0 + 0.05 → **20.7536**.

## L1 — LOLv2 nano-P2 FT

| Step | PSNR_RGB | SSIM_RGB |
|---:|---:|---:|
| 0 | 20.7036 | 0.7941 |
| **100** | **20.8515** | **0.7996** |
| 200 | 20.4617 | 0.7936 |
| 300 | 20.1200 | 0.7935 |
| 400 | 19.5086 | 0.7900 |

Early stop @400 (patience 3). **Best @100.**  
Ckpt: `checkpoints/L1_p2_only_mse_s42/checkpoint_00100.pt`  
Δ vs P0 = **+0.1479** → pass gate.

## L2 — Frozen-P2 + H3（华为 init，已否决）

| Step | Y PSNR | SSIM | P2-only |
|---:|---:|---:|---:|
| **0** | **20.7837** | **0.7981** | 20.8515 |
| 100 | 20.7524 | 0.7975 | 20.8515 |
| 300 | 20.6657 | 0.7959 | 20.8515 |

Early stop @300. Best@0 仍低于 L1。华为 H3 不适合 LOLv2 域 — 不当 init。

## L2b — Frozen L1@100 + H3 **scratch**（正确协议）

H3 随机初始化；LR enc/match/head = **1e-4**；P2 冻 L1@100。

| Step | Y PSNR | SSIM | P2-only |
|---:|---:|---:|---:|
| **0** | **20.8515** | **0.7996** | 20.8515 |
| 100 | 20.6991 | 0.7985 | 20.8515 |
| 200 | 20.1906 | 0.7938 | 20.8515 |
| 300 | 19.8310 | 0.7896 | 20.8515 |
| 400 | 20.0622 | 0.7926 | 20.8515 |
| 500 | 20.1206 | 0.7933 | 20.8515 |

Early stop @500。Best@0 ≡ P2（残差头近零 init）；一训就掉。  
**Discard L2b.** Ckpt dir: `checkpoints/L2b_frozenP2_h3scratch_mse_s42/`

## L2c — 华为 v2 原配方（Y0 cache → H1 → H3 head）

L2/L2b 无效是因为协议错了。华为有效 H3 不是“冻 P2 在线乱训整棵 H3”，而是：

1. 冻 P2，离线 cache Y0（`prepare_v2_p2_cache` / LOLv2: `prepare_lolv2_p2_cache.py`）
2. **H1** `LocalRefine`（Nano）：L1 + 0.1·Perceptual + Smooth + Color；batch **8**，crop **128**，lr **1e-4→5e-5@2000**，3000 steps
3. **H3** 只训 `V3B0ResidualFusion` 头；encoder+match 来自 H1 且冻结；MSE；同 batch/crop/lr

本轮 P2 = L1@100；cache + 训练根目录：`h3_v2recipe/`。

### H1 nano（特征）

| Step | H0_P2 | H1 arm | mismatch |
|---:|---:|---:|---:|
| 0 | 20.8515 | 20.8515 | 20.8515 |
| **1000** | 20.8515 | **20.9312** | 19.5059 |
| 2000 | 20.8515 | 20.3808 | 18.6300 |
| 3000 | 20.8515 | 20.4655 | 18.6676 |

**H1 best @1000 = +0.080 vs L1 P2.** Ckpt: `h3_v2recipe/checkpoints/H1_nano_s42/checkpoint_01000.pt`

### H3 head（冻 H1@1000 enc/match）

| Step | P2 | H3 |
|---:|---:|---:|
| 0 | 20.8515 | 20.8515 |
| 1000 | 20.8515 | 20.6959 |
| 2000 | 20.8515 | 19.8735 |
| 3000 | 20.8515 | 20.0260 |

从头训 H3 头全程低于 P2。

### H3 延续（Nikon/华为思路）：域内特征 + 华为头，勿再 FT

| 组合 | PSNR | 说明 |
|---|---:|---|
| L1 P2 | 20.852 | 域内 P2 |
| H1@1000 only | 20.931 | LocalRefine 整棵 |
| scratch / gentle H3 head | ≤20.85 | 再训头无效 |
| **H1@1000 Enc/Match + Huawei H3 head @0** | **21.104** | **零样本组装；一 FT 就掉** |

Ckpt: `h3_v2recipe/checkpoints/H3_h1feat_huaweiHead_s42/checkpoint_00000.pt`  
导出: `/root/data/eval/lolv2-H3-h1feat-huaweiHead/`

**当前最佳：** 该 H3 组装（21.104）。

## Final export — L1@100（P2-only 对照）

Dir: `/root/data/eval/lolv2-L1-best/` (`save_results/` + GT/Low/P2 + metrics)

| Metric | Mean |
|---|---:|
| PSNR_RGB | **20.8515** |
| SSIM_RGB | **0.799600** |
| PSNR_R / G / B | 20.57 / 21.33 / 21.04 |
| SSIM_R / G / B | 0.7765 / 0.8306 / 0.7917 |

## Lessons

| Lesson | Applied |
|---|---|
| MSE + random crop + early stop | L1/L2 |
| Dense eval (every 100) | Caught peak @100; @200 already regressing |
| Lower LR than Nikon N1 | 2e-6 (N1 used 1e-5) |
| Domain P2 is the lever | +0.15; H3 hurts on this P2 |
| No GT-degrade / exposure / fusion | Skipped |
| Illum e1 not init | Confirmed weaker than clean Huawei P0 |

## Scripts

- `scripts/train_lolv2_domain.py` (`--arm P0|P2|A0`)
