# Short-finetune audit, 2026-11-26

Read-only check of the finished `short_ft_5ep`, `short_ft_r2`, and `short_ft_w91` runs. `short_ft_r3` is excluded.

The narrative is `AUDIT_REPORT.md`. Machine-readable conclusions are `findings.json`.

| File | What it is |
|---|---|
| `audit_short_ft.py` | Script that produced the tables. It loads checkpoints and reads images. It does not train. |
| `artifact_inventory.csv` / `.json` | Pack metadata, file SHA256, model-state fingerprint, eval pairing. |
| `training_protocol_audit.json` | Parent chain, learning rate, Adam step counts, fingerprint uniqueness. |
| `augmentation_replay_summary.json` | Reconstructed shuffle and geometry fingerprints. |
| `augmentation_trace.jsonl` | Per-id crop, rotation, and flips for local epochs 1–5. |
| `per_image_deltas.csv` | Paired PSNR/SSIM changes. |
| `per_image_delta_summary.json` | Means, quartiles, worst-10 contribution. |
| `nikon_split_summary.csv` | Per-image Nikon luminance and hashes. |
| `nikon_audit.json` | Counts, symlink check, duplicate content, quantiles. |
| `examples_real_r2_test/` | Symlinks to existing clipped PNGs for test ids 00712, 00773, 00774 at epochs 3–5. Not a metric source. |

Code audited at git `a88566e`. Dataset code matches `92b9014` (amend-4, before these runs).
