# TTSR-lowlight v2.2 — Full-Image DC-Aware Results

**Verdict: Case A (mean) partial / Case C (robustness).**  
V1 Full-Image MSE fine-tune reaches **20.479 dB** (+0.013 over V0 H3@2000), Correct-Nano dependence retained, but **worst-sample tail does not improve** (2053 slightly worse). λDC=2 (V2) does not help the mean. **Centered inference remains the better fix for 2053-class failures.** Do not claim a robustness win; do not start GateNet.

Branch: `experiment/ttsr-lowlight-v22` @ `c10f97c` (working tree may include uncommitted v2/v22 files)  
Experiment: `/root/data/experiments/TTSR-lowlight-v22/`  
Eval30 is a **reused development set**.

---

## 1. Setup

| Item | Value |
|---|---|
| P2 | frozen `model_00004.pt` + M1 + lv2 bypass |
| H3 init | `H3_v3b0_s42/checkpoint_02000.pt` |
| Trainable | **V3B0ResidualFusion only** |
| Data | Huawei Train500 / Eval30 only |
| Steps | 1000, Adam lr=1e-5, seed=42, **batch=1 full image** |
| Peak VRAM | ~586 MB |
| V1 wall time | ~3.8 min |
| V2 wall time | ~3.9 min |

Loss: \(L=L_{\mathrm{spatial}}+\lambda_{\mathrm{DC}}L_{\mathrm{DC}}\).  
Unit tests: λDC=1 ≡ full MSE (value + grad); dense/chunk match eq; Eval∩Train=∅.

Phase A: `docs/TTSR_LOWLIGHT_V22_PHASE_A.md` — Train500 shares ~40% harmful DC; crop≠full mostly at borders; Full-Image training justified.

---

## 2. Main Eval30 table

Selected checkpoints: V0=@2000, **V1=@1000 (best)**, V2=@250 (least-bad among V2>0).

| Run | PSNR | SSIM | Δ vs H3(V0) | Worst Δ vs P2 | `<−1dB` |
|---|---|---|---|---|---|
| P2 | 20.265 | 0.6340 | — | — | — |
| V0 Full H3 | **20.466** | 0.6331 | — | −2.860 (2053) | 6 |
| V0 Centered | 20.419 | — | −0.047 | (2053 +0.08) | — |
| **V1 Full** | **20.479** | 0.6334 | **+0.013** | **−2.986 (2053)** | 6 |
| V1 Centered | 20.432 | — | −0.034 vs V0 Full | — | — |
| V2 Full @250 | 20.462 | 0.6330 | −0.004 | −2.821 (2053) | 6 |
| V2 Centered @250 | 20.414 | — | — | — | — |

### Training curves (Full PSNR)

| Step | V1 (λ=1) | V2 (λ=2) |
|---|---|---|
| 0 | 20.466 | 20.466 |
| 250 | 20.474 | 20.462 |
| 500 | 20.471 | 20.448 |
| 750 | 20.466 | 20.429 |
| 1000 | **20.479** | 20.441 |

V2 drifts down — heavier DC weight shrinks helpful corrective means as well as harmful ones.

---

## 3. Failure / gain samples

| Sample | P2 | V0 Full | V1 Full | V2 Full | ΔV1−V0 |
|---|---|---|---|---|---|
| **2053** | 24.38 | 21.52 | 21.39 | 21.56 | **−0.13** |
| 2059 | 24.38 | 22.85 | 22.71 | 22.86 | −0.13 |
| 2055 | 21.27 | 19.84 | 19.78 | 19.84 | −0.06 |
| 2058 | 22.01 | 20.75 | 20.68 | 20.78 | −0.07 |
| 2042 | 23.62 | 22.46 | 22.38 | 22.48 | −0.08 |
| 2052 | 17.74 | 19.49 | **19.56** | 19.46 | +0.07 |
| 2057 | 19.51 | 21.19 | **21.22** | 21.17 | +0.03 |
| 2044 | 17.24 | 18.75 | **18.86** | 18.74 | +0.11 |

Interpretation: V1 slightly **amplifies** both helpful and harmful global shifts → mean ↑, tail not healed.  
**Centered** still repairs 2053 (~24.46) under all arms.

Visuals: `/root/data/experiments/TTSR-lowlight-v22/visualizations/*_compare.png`

---

## 4. Reference-path ablation (second path only)

| Condition | V0 | V1@1000 | V2@250 |
|---|---|---|---|
| Correct Nano (Full) | 20.466 | **20.479** | 20.462 |
| Self | 20.253 | 20.245 | 20.257 |
| Mismatch | 19.571 | 19.673 | 19.672 |
| Zero-T | 18.636 | 18.979 | 18.900 |

Correct Nano remains best. Full-image FT does **not** destroy Nano dependence; Zero-T gap narrows slightly on V1 but Correct still >> Zero-T.

---

## 5. Checkpoints (SHA256)

| Run | Path | sha256 |
|---|---|---|
| V0 | `…/v2/…/H3_v3b0_s42/checkpoint_02000.pt` | `742d366b…ee22` |
| V1 best | `…/v22/…/V1_full_mse_s42/checkpoint_01000.pt` | `c9a5d511…0412` |
| V2 @250 | `…/v22/…/V2_full_dc2_s42/checkpoint_00250.pt` | `c1e64518…bc2a` |

Reproduce:

```bash
conda activate ttsr
cd /root/projects/TTSR-lowlight-v1-data1SOTA
python tests/test_v22_dc_loss.py
python tests/test_v22_fullimage_equivalence.py
python scripts/diag_v22_fullimage_dc.py          # Phase A
python scripts/train_v22_fullimage.py --arm V1
python scripts/train_v22_fullimage.py --arm V2
python scripts/eval_v22_formal.py --v1_step 1000 --v2_step 250
```

---

## 6. Answers to the scientific question

> Can Full-Image + DC-aware loss keep Nano gains while cutting bad global shifts?

| Goal | Result |
|---|---|
| Beat 20.466 mean | **Yes, barely** (V1 20.479) |
| Ease 2053 / `<−1dB` tail | **No** (still 6; 2053 worse on V1) |
| Keep Correct Nano > Self/Mismatch/Zero-T | **Yes** |
| λDC=2 better than λ=1 | **No** |

**Recommendation:** Keep **V0 H3@2000** as primary research baseline; treat **V1@1000** as a marginal mean bump only. For 2053-class failures prefer **Centered post-hoc** (or a future mean-clamped inference), not λDC reweighting alone. **No GateNet / Verifier / Nikon** this round.

---

*v2.2 complete. No automatic next version.*
