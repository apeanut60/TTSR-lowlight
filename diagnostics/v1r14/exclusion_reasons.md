# v1r14 — Explicit exclusion reasons

## Excluded from independent / generalization aggregate

| Set | Count | Reason |
|---|---|---|
| Huawei Eval30 (`Eval/Huawei`) | 30 | Used for all method selection in v1r6–v1r13. Eligible **only** for historical regression checks. |
| Huawei train manifest man250 | 250 | Checkpoint training set for v1r3 / FT. |
| Huawei train manifest man271 | 271 | Expanded Nano train pool (superset of man250). |
| Huawei Training Nano leftovers | 216 | Live under `Training data/`; not a declared Eval holdout. Excluded to avoid silently treating train-tree leftovers as independent test data (even though SHA≠man271 / Eval30). |
| Nikon Training Nano | ~250 | Outside formal Eval; not used here. |
| Any GT-sourced reference | — | Violates Nano Reference provenance rule. |

## Included

| Set | Count | Label |
|---|---|---|
| Nikon Eval20 (`Eval/Nikon` + `nanobanana_ref_Nikon`) | 20 | **Priority 2 / Outcome D — cross-domain.** Never used for v1r3e4 training (Huawei-only) or v1r6–v1r13 selection. |

## Same-domain Huawei holdout

**Not available.** Formal Priority-1 independent Huawei validation is blocked until a curated held-out Huawei Low/GT/Nano Eval split exists.
