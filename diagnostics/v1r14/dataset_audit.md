# v1r14 Dataset Provenance Audit

## Layout found

| Location | Low | GT | Nano Ref | Notes |
|---|---|---|---|---|
| `Eval/Huawei` | 30 | 30 | 30 (`nanobanana_ref`) | **Eval30 used for all v1r6–v1r13 selection** |
| `Eval/Nikon` | 20 | 20 | 20 (`nanobanana_ref_Nikon`) | Formal Eval split |
| `Training data/Huawei` | 2450 | … | 487 Nano | Train pool |
| `Training data/Nikon` | 3150 | … | 250 Nano | Not used by v1r3e4 (Huawei-only) |

Manifests:

- `.nanobanana_sample_manifest/Huawei.txt` — **250** (v1r3 / v1r6 FT)
- `.nanobanana_sample_manifest_full/Huawei.txt` — **271** (SOTA nano expansion)
- Nikon manifests exist but **v1r3e4 train script is Huawei-only**

## Checkpoint training set

`train/train_v1r3_nano_continue.sh`: `--data1_camera Huawei`, manifest dir above, eval ref = Huawei Eval Nano only.

## Method-selection set (v1r6–v1r13)

Always **Huawei Eval30** Nano RGB PSNR. Never Nikon Eval.

## Priority 1 — Independent Huawei Nano holdout

**Not available as a dedicated Eval holdout.**

- The only Huawei Eval split is the 30-image set already used for selection.
- 216 Training-tree Huawei Nano triplets sit outside man271/Eval30 (see `eligible_samples.csv`), but they live under `Training data/` and are **not** a pre-declared holdout. Using them would violate “do not silently use training data as independent test data.”
- Exact SHA1(1MB) collisions vs man271 / Eval30: **0**, but scene near-dup risk remains unruled-out without a curated split.

→ **Same-domain Huawei independent validation is blocked.**

## Priority 2 — Independent Nikon Nano Eval

**Available: 20 Eval/Nikon triplets with Nano refs.**

| Independence criterion | Status |
|---|---|
| Not used to train v1r3e4 | Yes (Huawei-only train) |
| Not used for v1r6–v1r13 selection | Yes (selection was Huawei Eval30) |
| Valid Low/GT/Nano | Yes |
| Nano not from GT | Yes (nanobanana_ref_Nikon) |
| Stem IDs also appear under Training Nikon | Yes (same IDs 3001–3020); **full-file SHA differs** (not bit-identical copies). Treat as **cross-domain**, not proof of scene uniqueness. |
| Stems in Nikon nano manifest | 3004, 3017 appear in `Nikon.txt` — unused by this Huawei checkpoint |

→ Formal evaluation proceeds on **Nikon Eval20 only**, labeled **Outcome D / cross-domain**.

## What would be needed for same-domain Huawei validation

A held-out Huawei Low/GT/Nano split that:

1. Was never in man250/man271 or any FT run,
2. Was never used for method selection,
3. Lives in a declared Eval/holdout path (not Training leftovers),
4. Has documented scene de-duplication vs Eval30 and train.

## Evaluation executed

- Huawei Training leftovers (**not** evaluated as independent).
- Nikon Eval20 A0/M1/P2 completed → see `summary.json`, `paired_analysis.json`, and `docs/v1r14_independent_validation_report.md`.
- Outcome label: **D (cross-domain)**; mean P2 PSNR did not beat A0/M1 on Nikon.
