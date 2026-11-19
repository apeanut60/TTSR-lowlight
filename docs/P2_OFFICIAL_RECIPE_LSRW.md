# P2 official-recipe LSRW retrain

Branch: `experiment/p2-official-recipe-lsrw`

Bring RetinexFormer-like training hygiene to **our P2** (structure unchanged):

- Camera-separate: Huawei-only → Eval30; Nikon-only → Eval20
- Loss: RGB **MSE** only
- Crop **256**, geometric augs, **mixup** β=1.2
- Long schedule **150k** steps, grad accum 8 (effective batch ≈8)
- Peak LR **1e-5** (FT from `model_00004`), cosine after 46k toward 1e-7
- Fit/val early stop (val_ratio 0.1); Official Eval only at the end
- Metrics: uint8 RGB + R/G/B, **no GT-mean**

Script: `scripts/train_p2_official_recipe.py`

```bash
# Huawei / Nikon
python scripts/train_p2_official_recipe.py --camera Huawei
python scripts/train_p2_official_recipe.py --camera Nikon

# LOLv2-Real (full nano on train+test) — preferred domain for this recipe
python scripts/train_p2_official_recipe.py --camera LOLv2Real --from_scratch \
  --lr 2e-4 --eta_min 1e-6 --patience 20
```

Artifacts: `/root/data/experiments/TTSR-lowlight-p2-official-recipe-{huawei,nikon,lolv2real}/`

## vs formal P2-only

This recipe is a **long-schedule trial**, not the product scorecard.

| Camera | This recipe (Test) | Formal P2-only (see `TTSR_LOWLIGHT_P2_ONLY_EVAL.md`) |
|---|---:|---|
| LOLv2-Real scratch | 19.985 @49k | **L1@100 = 20.852** (better) |
| Huawei FT from `00004` | ≈20.430 @6k (stdout; run had crash) | Reliable cite: A2-export P2-only **20.267** |
| Nikon | not completed here | **N1@1000 = 17.088** |

Do not replace L1 / N1 formal ckpts with these runs unless a clean re-eval beats them.
