# TTSR-lowlight LOLv2-Syn Domain FT — Results

**Verdict:** 当前最佳是 **S1d@1600**（冻 S1c 主干，整图训 illum 头）**20.890 / 0.8795**。  
相对 S1c +0.31，相对 P0 +3.85。仍未过 21。H1/H3 停在旧 S1 的 Y0 上，低于这条 P2。

## Protocol

| Item | Value |
|---|---|
| Script | `scripts/train_lolv2syn_domain.py` |
| Exp root | `/root/data/experiments/TTSR-lowlight-lolv2syn/` |
| Train | Low∩Normal∩`nanobanana_ref_v2`，同名配对 → **n=739** |
| Eval | Official Test 100 + `nanobanana_ref_v3` |
| Crop | Random 192 |
| Loss | RGB MSE |
| Init | Huawei `model_00004`（不从 Real L1 续） |

## P0

Huawei `00004` 全协议 RGB：**17.0368 / 0.7541**（R/G/B 17.10 / 17.14 / 17.51）。

## S1 — 域内 nano-P2（lr 2e-6）

| Step | PSNR_RGB | SSIM |
|---:|---:|---:|
| 0 | 17.037 | 0.7541 |
| 100 | 18.001 | 0.7715 |
| 500 | 19.201 | 0.8127 |
| 900 | 19.513 | 0.8369 |
| **1400** | **19.794** | **0.8503** |
| 1500–1700 | 回落 | early-stop @1700 |

R/G/B PSNR @1400：19.50 / 20.09 / 20.32。  
Ckpt: `checkpoints/S1_p2_only_mse_s42/checkpoint_01400.pt`

## S1b — 从 @1400 续训，lr 1e-5

| Step | PSNR_RGB | SSIM |
|---:|---:|---:|
| 0 | 19.794 | 0.8503 |
| 200 | 20.003 | 0.8625 |
| 500 | 20.052 | 0.8644 |
| 700 | 20.156 | 0.8699 |
| 900 | 20.193 | 0.8747 |
| **1100** | **20.514** | **0.8764** |
| 1200 | 20.368 | 0.8722 |
| 1400 | 19.828 | 0.8665 |

Early stop @1400。相对 S1 **+0.72**，未到 21，且不站住。  
Ckpt: `checkpoints/S1b_p2_lr1e5_from1400_s43/checkpoint_01100.pt`

## S1c — 从 S1b@1100 续训，nano ref，lr 5e-6

Init 即 S1b@1100（20.514 / 0.8764）。seed 44，2000 steps，eval every 100，patience 3。参考仍是 `Train/nanobanana_ref_v2`。

| Step | PSNR_RGB | SSIM |
|---:|---:|---:|
| 0 | 20.514 | 0.8764 |
| 100 | 20.422 | 0.8716 |
| 200 | 20.567 | 0.8761 |
| 300 | 20.573 | 0.8772 |
| **400** | **20.576** | **0.8780** |
| 500 | 20.174 | 0.8719 |
| 600 | 20.481 | 0.8787 |
| 700 | 20.498 | 0.8786 |

Early stop @700。相对 S1b **+0.062**。R/G/B PSNR @400：20.10 / 20.80 / 21.30；SSIM 0.8856 / 0.8977 / 0.8506。  
Ckpt: `checkpoints/S1c_nano_lr5e6_fromS1b1100_s44/checkpoint_00400.pt`

## S1d — 冻主干，整图训 GlobalIllumHead + RefIllumTransfer

诊断：S1c@400 上逐通道 oracle affine → **24.74**（+4.16）；nano 对 GT 仅 16.63。剩余误差在整图色调。训练用 crop MSE 时，两头吃的是 192 crop 统计，评测吃整图。

Init S1c@400。冻 LTE/主干，只训 `global_illum_head` + `ref_illum`（4588 params）。前向对齐评测：tile 主干（no grad）→ 整图 illum 一次。损失整图 RGB MSE，lr 1e-4，seed 45，patience 3。参考仍是 nano。

| Step | PSNR_RGB | SSIM |
|---:|---:|---:|
| 0 | 20.576 | 0.8780 |
| 100 | 20.685 | 0.8798 |
| 500 | 20.738 | 0.8804 |
| 900 | 20.815 | 0.8802 |
| 1300 | 20.833 | 0.8806 |
| **1600** | **20.890** | **0.8795** |
| 1700–1900 | 20.85–20.87 | early-stop @1900 |

相对 S1c **+0.314**。R/G/B PSNR @1600：20.49 / 21.06 / 21.54。  
Ckpt: `checkpoints/S1d_illum_full_mse_fromS1c400_s45/checkpoint_01600.pt`  
Script: `scripts/train_lolv2syn_illum.py`

## H1 — 冻 S1@1400，Y0 cache

| Step | H1 | mismatch |
|---:|---:|---:|
| 0 | 19.794 | 19.794 |
| 1000 | 19.744 | 19.380 |
| **3000** | **19.870** | 19.106 |

增益很小（+0.075）。Ckpt: `h3_v2recipe/checkpoints/H1_nano_s42/checkpoint_03000.pt`

## H3 组装（H1@3000 Enc/Match + 华为 H3@2000 头，不 FT）

同一次前向（RGB）：

| | PSNR | SSIM |
|---|---:|---:|
| P2 S1@1400 | 19.794 | 0.8503 |
| **H1@3000** | **19.870** | **0.8480** |
| H3 组装 | 19.813 | 0.8473 |

H3 R/G/B：19.57 / 20.08 / 20.36。比 H1 低 0.06，只比 P2 高 0.02。Real 上同一组装是 +0.25；Syn 上华为头几乎没带来额外增益。  
导出：`/root/data/eval/lolv2syn-H3-h1feat-huaweiHead/`

## 结论

当前 Syn 可用点是 **S1d@1600**（20.890 / 0.8795）。整图 illum 头兑现了约 0.3 dB；oracle affine 上限仍在 24.7，头从 low/nano/当前输出预测，拿不完。下一档若要涨，应扩 illum 表达力（例如更强的低频 residual），而不是再训整网 crop MSE。
