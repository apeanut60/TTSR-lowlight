# TTSR-lowlight v2.0 — Plan & Code Reuse

Branch: `experiment/ttsr-lowlight-v2`  
Base HEAD at branch creation: `7e1f0ab`  
Historical source: `github.com/apeanut60/TTSR-lowlight` **v5** @ `d3c4bd9`

## Objective

Freeze TTSR **P2** (M1 + lv2 branch_bypass, Eval30 ≈20.264 dB). Train LocalRefine / V3-B.0 on Huawei **Train500** Nano triplets only. Compare H1/H2/H3 on Huawei Eval30.

## Code reuse inventory

| File | Source | Action |
|---|---|---|
| `model/LocalRefine.py` | v5 verbatim | Copied |
| `model/V3BResidualFusion.py` | v5 verbatim | Copied (`V3B0ResidualFusion`) |
| `dataset/data1_localrefine.py` | v5 verbatim | Copied (shared crop/flip; decode→norm→resize) |
| `local_refine_runtime.py` | v5 adapted | Cache/eval/mismatch kept; Retinex N0 loader **removed** |
| `v2_p2_runtime.py` | **new** | Frozen TTSR P2 tiled inference |
| `scripts/prepare_v2_p2_cache.py` | new (replaces prepare_local_refine_cache) | P2 Y0 float32 cache |
| `scripts/train_v2_local_refine.py` | adapted from train_local_refine.py | H1 Nano / H2 Self |
| `scripts/train_v2_v3b0.py` | **new** (not LOL v3a stack) | H3 head on frozen H1 features |
| `tests/test_v2_local_refine.py` | adapted from v5 tests | Params, identity, dense/chunk, manifest |

**Not merged:** historical `v3a5_*` / LOL lock pipelines. H3 uses the same Huawei P2 cache + H1 encoder/match.

## Data

- Train: Huawei 500 Low/GT/Nano (`diagnostics/v2/huawei_train_manifest.csv`)
- Eval: Huawei Eval30 only
- Nikon: unused this round
- Manifest: `.nanobanana_sample_manifest_500/Huawei.txt`

## Experiment matrix

| Run | Module | Second input |
|---|---|---|
| H0 | P2 only | — |
| H1 | LocalRefine | Nano |
| H2 | LocalRefine | Self (Y0) |
| H3 | V3B0ResidualFusion | Nano features via frozen H1 encoder/match |

## Experiment root

`/root/data/experiments/TTSR-lowlight-v2/`

## Execution order

A prepare → B port+cache+tests → C H1/H2 train+eval → D H3 → E report. Stop after report.
