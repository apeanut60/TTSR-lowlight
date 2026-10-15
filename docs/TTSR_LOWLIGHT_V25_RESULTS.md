# TTSR-lowlight v2.5 — L1 / MS-SSIM Weight Balance Results

**Verdict: Case D — lowering MS-SSIM weight does not fix Eval PSNR.**  
W0 (λ=1) ≳ W1 (0.3) ≳ W2 (0.1) on Eval30 PSNR at matched steps.  
Train crop metrics keep improving while Eval PSNR falls for **all** λ — confirms **loss–PSNR conflict + P2 drift**, not under-training.  
Keep **v2.4 A2@250 (MSE joint)** as the PSNR-oriented research candidate. Eval30 is a reused development set.

---

## 1. Fitting audit (A3 / Q1)

See `TTSR_LOWLIGHT_V25_FITTING_AUDIT.md`.

| Metric | A3@0 → @2000 |
|---|---|
| Train80 crop PSNR / SSIM | ↑ 20.73→20.91 / 0.672→0.681 |
| Train80 crop L1 / MS-loss | ↓ both |
| Eval30 PSNR / SSIM | ↓20.47→20.37 / ↑0.633→0.638 |
| Eval P2-only | ↓ 20.265→20.112 |

Head ∇L1 vs ∇(1−MS-SSIM) cosine **−0.22** (conflict). Enc/Match aligned; P2 weakly aligned.

## 2. Protocol

| Item | Value |
|---|---|
| Branch | `experiment/ttsr-lowlight-v25` |
| Loss | \(L=\mathrm{L1}+\lambda(1-\mathrm{MS\text{-}SSIM})\) only |
| Arms | W0 λ=1.0 / W1 λ=0.3 / W2 λ=0.1 |
| Init | P2 `model_00004` + H3@2000 (same as v2.4) |
| Train | crop **192**, **md5(seed\|id)** crops, seed 42, Adam, 2000 steps, joint P2+H3 |
| W0 vs A3 | **Retrained** (crop hash fixed); Eval curves match A3 shape |

## 3. Train / Eval curves

| Arm | Step | Train80 crop PSNR | Train SSIM | Eval PSNR | Eval SSIM | Eval P2-only | gap (T−E) | n&lt;−1dB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W0 | 0 | 20.728 | 0.6723 | **20.466** | 0.6331 | 20.265 | +0.26 | 6 |
| W0 | 250 | 20.811 | 0.6782 | 20.450 | 0.6365 | 20.164 | +0.36 | 5 |
| W0 | 500 | 20.910 | 0.6794 | 20.383 | 0.6367 | 20.081 | +0.53 | 3 |
| W0 | 1000 | 20.848 | 0.6805 | 20.403 | 0.6370 | 20.119 | +0.44 | 3 |
| W0 | 1500 | 20.867 | 0.6808 | 20.400 | 0.6373 | 20.122 | +0.47 | 3 |
| W0 | 2000 | 20.888 | 0.6815 | 20.381 | 0.6373 | 20.102 | +0.51 | 3 |
| W1 | 0 | 20.728 | 0.6723 | 20.466 | 0.6331 | 20.265 | +0.26 | 6 |
| W1 | 250 | 20.812 | 0.6776 | 20.448 | 0.6364 | 20.169 | +0.36 | 5 |
| W1 | 500 | 20.912 | 0.6787 | 20.365 | 0.6362 | 20.056 | +0.55 | 3 |
| W1 | 1000 | 20.847 | 0.6797 | 20.388 | 0.6364 | 20.102 | +0.46 | 3 |
| W1 | 1500 | 20.866 | 0.6799 | 20.374 | 0.6364 | 20.093 | +0.49 | 3 |
| W1 | 2000 | 20.887 | 0.6804 | 20.349 | 0.6362 | 20.066 | +0.54 | 3 |
| W2 | 0 | 20.728 | 0.6723 | 20.466 | 0.6331 | 20.265 | +0.26 | 6 |
| W2 | 250 | 20.811 | 0.6772 | 20.447 | 0.6362 | 20.172 | +0.36 | 5 |
| W2 | 500 | 20.909 | 0.6782 | 20.358 | 0.6360 | 20.048 | +0.55 | 3 |
| W2 | 1000 | 20.843 | 0.6790 | 20.382 | 0.6361 | 20.097 | +0.46 | 3 |
| W2 | 1500 | 20.861 | 0.6790 | 20.364 | 0.6359 | 20.082 | +0.50 | 3 |
| W2 | 2000 | 20.881 | 0.6794 | 20.337 | 0.6355 | 20.052 | +0.54 | 3 |

### Answers to Phase C questions

| Question | Answer |
|---|---|
| λ↓ → Train PSNR ↑? | **No meaningful gain** — crop PSNR nearly identical across W0/W1/W2 |
| Eval PSNR sync ↑? | **No** — lower λ is slightly **worse** |
| SSIM sacrificed? | Mild: W0@2000 SSIM 0.6373 > W2 0.6355 |
| P2-only late drop relieved? | **No** — all arms ~20.05–20.10 @2000 |
| Converged @2000? | Train L1/MS still inching; Eval PSNR not recovering |
| Train↑ Eval↓? | **Yes** for all λ (gap grows ~0.26→0.50+) |

## 4. Reference ablation (@250 / @2000)

| Ckpt | Correct | Self | Mismatch | Zero-T | Bypass | C−S | n&lt;−1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| W0@250 | 20.450 | 20.325 | 19.658 | 18.587 | 20.164 | 0.125 | 5 |
| W0@2000 | 20.381 | 20.273 | 19.721 | 18.880 | 20.102 | 0.107 | 3 |
| W1@250 | 20.448 | 20.318 | 19.617 | 18.522 | 20.169 | 0.129 | 5 |
| W1@2000 | 20.349 | 20.237 | 19.699 | 18.838 | 20.066 | 0.112 | 3 |
| W2@250 | 20.447 | 20.315 | 19.599 | 18.494 | 20.172 | 0.131 | 5 |
| W2@2000 | 20.337 | 20.224 | 19.695 | 18.829 | 20.052 | 0.113 | 3 |

Reference dependence preserved. Lowering λ does not worsen `<−1dB` vs W0 (same 5→3 pattern); C−S shrinks slightly by 2000 for all arms.

## 5. Case judgment

**Case D.** Scanning λ ∈ {1.0, 0.3, 0.1} under joint P2+H3 does not yield a better Eval PSNR schedule than λ=1, and none beat step-0 / H3@2000 on mean PSNR.  

Do **not** continue blind λ search. Prefer:

1. **PSNR path:** v2.4 **A2@250 MSE** joint (20.494), with early stop.
2. **SSIM path:** L1+MS-SSIM raises Eval SSIM but trades PSNR — keep as separate objective if needed.
3. Next science (if any): PSNR-aligned losses or P2 regularization / lower P2 LR — **not** more λ sweeps.

## 6. Runtime

~22–24 min / arm, peak VRAM ~2.2 GB (crop-192 joint).

## 7. Checkpoints (SHA256)

| Ckpt | SHA256 |
|---|---|
| W0@250 | `3af3cfc2a219d66f4dcba840e7d03dd176f9b7549d5b169e8045ac29a46cb239` |
| W0@2000 | `8228a862e3d138e2d99bb1f624e8749129348e021268760a88f84112d5572166` |
| W1@250 | `689e6ffc95274f605f852b168ebdbc25be7962b67425e2aa0a3c8b36c2f6ed23` |
| W1@2000 | `81857a6f07ba50dadcc6074fa0fc6549cad5ef36b2d62c0d3cb8d99c2139491e` |
| W2@250 | `41a781ae7ae15c1ffdbebd49975cdb0a1418e4016cc2cd68fa5f2b8ecf05fcef` |
| W2@2000 | `11004c5593e6c49ff234d00dc6c0f93fcc60549d1023197158a9d31ff46a3072` |

### Reproduce

```bash
export TORCH_HOME=/root/data/cache/torch PYTHONUNBUFFERED=1
cd /root/projects/TTSR-lowlight-v1-data1SOTA
python -u scripts/audit_v25_loss_balance.py --full_train_steps 0,2000
for arm in W0 W1 W2; do
  python -u scripts/train_v25_loss_balance.py --arm $arm --steps 2000 --seed 42 --crop 192
done
python scripts/eval_v25_formal.py
```
