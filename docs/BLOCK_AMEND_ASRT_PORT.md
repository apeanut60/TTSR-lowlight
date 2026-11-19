# block-amend: ASRT Stage-A → TTSR P2

Port of ASRT **Stage-A** cross-scale CSFI-U-Net into `TTSR-lowlight-v1-data1SOTA`, keeping the TTSREnhance shell (LTE + SearchTransfer + optional RefCorrection / query-illum). Git is owned by the operator; this doc only describes the workspace changes.

## What moved

| Module | Path | Role |
|---|---|---|
| IlluminationEstimator, IGAB / IG_MSA / FeedForward | `model/igab.py` | Illum features + log-gain; channel attention at H/4 & H/2 |
| CSFI-U-Net Stage-A | `model/CSFIUNet.py` | SFE → down×2 → inject/RB/IGAB → CSFI → MergeTail; `Y = Y_base + residual` |
| Backbone switch | `model/TTSREnhance.py` | `args.backbone=csfi_unet\|mainnet` (default **csfi_unet**) |
| CLI | `option.py` | `backbone`, `igab_blocks`, `igab_heads`, `enable_lightup`, `use_global_illum`, `use_ref_illum` |
| Tiled runtime | `v2_p2_runtime.py` | CSFI: pad÷4, full-image stem (illum/light-up once); `build_args_csfi()` |

**Not ported:** ASRT v3 CondResBlock / IllumAdapter, v4 TextureAdapter, vNext IRP-MSA, reliability maps `C_T/C_L`, ASRT package imports / train scripts.

Legacy `MainNetEnhance` remains for old P2 ckpts via `--backbone mainnet`.

## Defaults (Stage-A)

- `num_res_blocks=8+8+4+2`, `n_feats=64`, `igab_blocks=1,1`, `enable_lightup=True`
- `use_global_illum=False`, `use_ref_illum=False` (heads still present; flags gate them)
- Inject heads: weight ×0.01, bias 0; MergeTail soft residual
- Tensor range `[-1, 1]`
- `stage_mode_lv2` default **`normal`** (inject `T_lv2` at H/2, ASRT-style)

## Relation to P2 lv2 bypass

Frozen product P2 (LOLv2 L1@100 **20.8515**) is still **MainNetEnhance + M1 fusion + `stage_mode_lv2=branch_bypass`**. That ablation does **not** transfer automatically to the new U-Net. block-amend trains/evals CSFI with lv2 inject on; re-ablate later with `--stage_mode_lv2 branch_bypass` if needed.

`v2_p2_runtime.build_args()` / `P2_CONFIG` stay on `backbone=mainnet` + lv2 bypass so existing P2-only eval / Y0 cache paths stay bit-compatible. CSFI uses `build_args_csfi()` / `CSFI_CONFIG` (`lv2=normal`).

## Warm-start

Do **not** expect MainNet tensors from L1@100 / `model_00004` to load into CSFI-U-Net (shape / topology mismatch). LTE (+ optional RefCorrection) can load from the old ckpt with `--pretrain_require_core False`. Train new blocks from soft init under a **new** `save_dir` (e.g. `TTSR-lowlight-block-amend/`), never overwrite old P2 experiment trees.

## Train / eval entry

```bash
bash train/train_block_amend_csfi_lolv2.sh
```

Protocol: LOLv2-Real Official Test, RGB PSNR/SSIM; baseline remains L1@100 **20.8515** (old backbone). Record ΔPSNR honestly; no assumed gain.

## Smoke

```bash
conda activate ttsr
python tests/test_csfi_unet_stage_a.py
```

Covers pad/crop, `ref=None`, IGAB shapes, soft-inject ≈ bypass when `σ(S)≈0`, backbone switch, and tiled CSFI with full-image stem.
