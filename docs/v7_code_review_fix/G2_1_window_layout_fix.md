# Gate 2.1 — Window Attention Layout Fix

- **baseline commit:** `3aa86222af2af4d17d95b13beb6975427706b4d9`
- **actual commit:** `e597d47648d72b34ad2781b4140d6e9c7e02e3a0`
- **seed / lr / updates / n:** 42 / 1e-4 / 200 / 8
- **optimizer / loss:** Adam / MSE
- **protocol:** correct-only, D2-only, frozen LLFormer E110, frozen Alignment (Setting A)
- **fix:** `V7RiverFusion._window_kv`: `F.unfold` → `reshape(B,C,K*K,HW).permute(0,3,2,1).contiguous()`
- **no Setting B / joint train / Official Test**

Machine-readable twin: `G2_1_window_layout_fix.json` (also under `/root/data/experiments/V7_CODE_REVIEW_FIX/`).

## Unit / static checks

| check | result |
|---|---|
| `_window_kv` center + 3×3 neighbor exact match | PASS |
| step0 Base identity (tiled) | PASS (`max_abs=0`) |
| win3 Q/K grad nonzero; N=1 Q/K grad ≡ 0 | PASS |
| forward shapes | PASS |

## Microset (same 8 images as G2 win3)

| metric | old win3 (buggy layout) | corrected win3 |
|---|---|---|
| loss_first | 0.015103 | 0.015103 |
| loss_last | 0.004539 | 0.004452 |
| mae_swapA2 | 4.90e-4 | 6.10e-4 |
| mae_shuffleRef | 1.63e-4 | 2.03e-4 |
| ΔPSNR n−shuf | 0.000711 | 0.001158 |

## Independent heldout (n=16, outside microset)

| metric | value |
|---|---|
| loss_mean | 0.015261 |
| mae_swapA2 | 5.37e-4 |
| mae_shuffleRef | 2.01e-4 |
| ΔPSNR n−shuf | 0.002113 |

## Verdict

**Gate 2.1 FAIL** — layout bug is closed and verified, but content response remains below threshold (`mae_swapA2 > 1e-3`). Corrected run is only marginally above the buggy win3 baseline; **stop here** (no new fusion/matcher structure in this gate).

## Repro

```bash
# unit tests
cd /root/projects/TTSR-lowlight && /root/miniconda3/envs/ttsr/bin/python -m unittest tests.test_v7_river -v

# microset + heldout eval (writes under /root/data/experiments/V7_CODE_REVIEW_FIX/)
/root/miniconda3/envs/ttsr/bin/python \
  /root/projects/TTSR-lowlight/scripts/gate2_1_window_layout_fix.py --device cuda
```
