# v1r14 Reproduction

## Environment

```bash
conda activate ttsr   # or: /root/miniconda3/envs/ttsr/bin/python
cd /root/projects/TTSR-lowlight-v1-data1SOTA
git checkout experiment/v1r14-independent-validation
```

## Checkpoint (frozen)

`/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt`

## Unit checks

```bash
python tests/test_v1r14_independent_validation.py
```

## Full independent validation

```bash
python -u scripts/run_v1r14_independent_validation.py --out_dir diagnostics/v1r14
```

Flags:

- `--skip_regression` — skip Huawei Eval30 historical checks
- `--skip_nikon` — skip Nikon Eval20 (leaves audit-only path)

## Configurations

| Name | Gamma α | Hist β | Identity λ | Gate | lv3 | lv2 | lv1 |
|---|---|---|---|---|---|---|---|
| A0 | off | off | off | original S | normal | normal | normal |
| M1 | 0.75 | 0.50 | 0.25 | A0 S | normal | normal | normal |
| P2 | 0.75 | 0.50 | 0.25 | A0 S | normal | **branch_bypass** | normal |

Shared: `ref_degrade=False`, `eval_mean_align=False`, tile 256 / overlap 96 cosine, full-image Gamma/LUT shared across tiles.

## Expected Eval30 regression (not independent)

| Run | PSNR | SSIM |
|---|---|---|
| A0 | ≈20.162 | ≈0.6324 |
| M1 | ≈20.1805 | ≈0.6327 |
| P2 | ≈20.264 | ≈0.6341 |

## Independent aggregate

**Nikon Eval20 only** (cross-domain). Huawei Eval30 must not enter the independent mean.
