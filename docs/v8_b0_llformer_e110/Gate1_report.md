# V8-B0 Gate 1

- **step:** 400
- **commit:** `577701a17d1ed7bd8ad541306a830d23d470cd4a`
- **init_sha:** `58199392a768a62a`
- **authorize_long_train:** **False**

## B0_Self

train-batch mean MSE (32): **0.016843**

### formal_dev64

| Base | Self | NanoN | Shuffle | ZeroT | N−Base | N−Self | N−Shuf | severe |
|---|---|---|---|---|---|---|---|---|
| 22.858 | 22.405 | 22.480 | 22.461 | 22.657 | -0.379 | +0.074 | +0.018 | 25 |

### microset-heldout16

| Base | Self | NanoN | Shuffle | ZeroT | N−Base | N−Self | N−Shuf | severe |
|---|---|---|---|---|---|---|---|---|
| 27.024 | 26.348 | 26.438 | 26.508 | 26.868 | -0.586 | +0.089 | -0.070 | 8 |

## B0_Nano

train-batch mean MSE (32): **0.016740**

### formal_dev64

| Base | Self | NanoN | Shuffle | ZeroT | N−Base | N−Self | N−Shuf | severe |
|---|---|---|---|---|---|---|---|---|
| 22.858 | 22.359 | 22.429 | 22.359 | 22.622 | -0.430 | +0.070 | +0.070 | 26 |

### microset-heldout16

| Base | Self | NanoN | Shuffle | ZeroT | N−Base | N−Self | N−Shuf | severe |
|---|---|---|---|---|---|---|---|---|
| 27.024 | 26.281 | 26.344 | 26.361 | 26.779 | -0.680 | +0.063 | -0.017 | 8 |

## Verdict

Gate 1 diagnostic complete. No Official Test tuning. No 3k/20k auto-start.

CSV: `/root/data/experiments/V8_B0_llformer_e110/gate1/per_image_psnr.csv`

### Interpretation

- Both arms **overfit train MSE** (~0.017) but **hurt vs LLFormer Base** on formal_dev and microset-heldout (N−Base negative; many severe drops).
- Nano arm shows small **N−Self** on both splits (~+0.06–0.07) but **heldout N−Shuf ≤ 0** → reference dependence not validated for go.
- `authorize_long_train=False`. Do not start 3k/20k from this Gate1.

## Repro

```bash
cd /root/projects/TTSR-lowlight
/root/miniconda3/envs/ttsr/bin/python scripts/gate0_v8_b0.py --device cuda --force
/root/miniconda3/envs/ttsr/bin/python scripts/train_v8_b0.py --arm B0_Self --device cuda
/root/miniconda3/envs/ttsr/bin/python scripts/train_v8_b0.py --arm B0_Nano --device cuda
/root/miniconda3/envs/ttsr/bin/python scripts/eval_v8_b0_gate1.py --device cuda --step 400
```
