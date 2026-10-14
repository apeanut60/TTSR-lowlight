# Nano Adaptation Huawei Report (SOTA-Nano optimization)

Code: `/root/projects/TTSR-lowlight-v1-data1SOTA` (`retry/v1-nano-adapt`)  
Metric: Huawei Eval, nanobanana ref, **RGB PSNR / RGB SSIM**, no mean-align.  
Hard constraint: no new modules / architecture change.

---

## 1. Baseline

| Model | PSNR | SSIM | Notes |
|---|---:|---:|---|
| v14e40 | 19.471 | 0.6153 | HR-ref train → nano eval |
| **SOTA-Nano-v0 = v1r3e4** | **20.162** | **0.6324** | Frozen record under `/root/data/experiments/retry/SOTA-Nano-v0/` |

Gain of v0 vs v14e40: **+0.691 PSNR / +0.017 SSIM**.

v0 config: Nano Ref, Huawei, crop=192, L1+MS-SSIM (λ=1), LTE frozen, Adam lr=1e-5 StepLR, manifest 250.

---

## 2. Dataset change (Phase 1)

```
Huawei Nano train:
before: 250
after:  271
missing (low+high but no nano ref): 2179
duplicates: 0
orphan ref: 0
integrity_bad: 0
```

Full manifest: `/root/data/datasets/data1/.nanobanana_sample_manifest_full/Huawei.txt`  
Train code: no silent HR-as-Ref fallback on small crops (pad Ref instead).

---

## 3. LR schedule comparison (Phase 2)

Start: **v1r3e4** (not v14). Data: full 271. Loss: L1 + 1.0·(1−MS-SSIM). Crop 192. Cosine 1e-5→1e-7, 10 epochs, seed 42.

| Epoch | PSNR | SSIM |
|---:|---:|---:|
| 1 | 20.067 | 0.6323 |
| 2 | 20.079 | 0.6328 |
| **3** | **20.112** | **0.6322** |
| 4 | 20.027 | 0.6334 |
| 5–10 | 19.988–20.041 | ~0.632–0.633 |

Official best: **e3 → 20.112 / 0.6322**  
Dir: `retry/SOTA-Nano-P2-cosine-full-mss1`

vs v0: **−0.050 PSNR**. Cosine + full data did **not** beat StepLR peak of v1r3e4; oscillation after early epochs remains.

---

## 4. MS weight comparison (Phase 3)

`L = L1 + λ·(1−MS-SSIM)`, else identical to Phase 2 (cosine, full 271, seed 42, from v1r3e4).

| λ | Best epoch | PSNR | SSIM | Official log |
|---:|---:|---:|---:|---|
| 0.5 | 3 | 20.076 | 0.6311 | `SOTA-Nano-P3-mss0p5` |
| 1.0 | 3 | 20.112 | 0.6322 | `SOTA-Nano-P2-cosine-full-mss1` |
| **2.0** | **3** | **20.163** | **0.6334** | `SOTA-Nano-P3-mss2p0` |

Highest PSNR: **λ=2.0**. Marginal vs v0 (+0.001 PSNR, +0.001 SSIM).

---

## 4b. Phase 4 — Ref-only photo aug (triggered: max&lt;20.2)

Same as λ=2.0 + Ref brightness/saturation ±5% (Low/GT untouched).

Official best: **e3 → 20.147 / 0.6332** (`SOTA-Nano-P4-refaug-mss2`)  
Did not beat λ=2.0 without aug.

---

## 5. Final recommended checkpoint

```
Best checkpoint:
/root/data/experiments/retry/SOTA-Nano-P3-mss2p0/model/model_00003.pt

PSNR: 20.163
SSIM: 0.6334

Config:
  start=v1r3e4
  Huawei Nano full (271)
  crop=192, batch=8
  L1 + 2.0*(1-MS-SSIM)
  freeze LTE only
  Adam lr=1e-5 cosine (10ep, eta_min=1e-7), seed=42
  eval: Huawei nanobanana, RGB PSNR/SSIM

Difference from v1r3e4 / SOTA-Nano-v0:
  +full manifest 250→271
  +cosine schedule (was StepLR)
  +ms_ssim_w 1.0→2.0
  metric delta: PSNR +0.001 / SSIM +0.001  (within noise; not a clear new SOTA)
```

**Practical recommendation:** keep **SOTA-Nano-v0 (v1r3e4)** as the frozen baseline for future V3B0/LLFormer comparisons; optionally archive P3-mss2p0 e3 as a tied alternate. **Target PSNR ≥ 20.3 was not reached.**

---

## Research question

> After adapting TTSR-lowlight-v14 from GT-degraded Reference to Nano Banana Reference, how much further can Huawei LSRW performance improve only through data coverage and optimization?

**Answer:** Under the no-new-module constraint, further gains from expanding Nano coverage (250→271), cosine LR, MS-SSIM weight sweep, and mild Ref photometric aug are **~0 dB** (best new run 20.163 vs frozen 20.162). The Nano adaptation jump from v14e40 (**+0.69 dB**) already captured most of what this optimization surface can give; the Huawei Nano ceiling for this TTSREnhance recipe appears **≈20.16–20.17 RGB PSNR** without architectural or protocol changes.

---

## Artifact index

| Path | Role |
|---|---|
| `retry/SOTA-Nano-v0/` | Frozen baseline record |
| `data1/.nanobanana_sample_manifest_full/` | 271-file Huawei manifest |
| `retry/SOTA-Nano-P2-cosine-full-mss1/` | Cosine + full data, λ=1 |
| `retry/SOTA-Nano-P3-mss0p5/` | λ=0.5 |
| `retry/SOTA-Nano-P3-mss2p0/` | λ=2.0 (best of sweeps) |
| `retry/SOTA-Nano-P4-refaug-mss2/` | Ref ±5% aug |
