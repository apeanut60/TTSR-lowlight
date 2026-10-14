# v1r8 reproduction

Branch: `experiment/v1r8-matching-diagnostic`  
Baseline ckpt (read-only): `/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt`

## Phase A/B — matching + visual diagnostics

```bash
conda activate ttsr
cd /root/projects/TTSR-lowlight-v1-data1SOTA
python scripts/run_v1r8_matching_diagnostic.py \
  --out_dir /root/projects/TTSR-lowlight-v1-data1SOTA/diagnostics/v1r8
```

Notes:

- RGB PSNR/SSIM use production tiled SR (full resolution).
- Match cosine / displacement / Query–Ref viz use a memory-safe scale (`max_side=512`) because full-res `R_lv3` OOMs on 960-side Huawei frames.
- Absolute match scores therefore differ from v1r7 full-res diagnostics; relative A0→A3→B2 trends are valid.

## Phase C — FT regression (no training)

```bash
python scripts/analyze_v1r8_ft_regression.py
```

Writes `diagnostics/v1r8/ft_regression/ft_regression_analysis.json`.  
Trajectory plots (optional, from that JSON):

```bash
python - <<'PY'
# see docs/v1r8_matching_diagnostic_report.md §C; plots already under ft_regression/
PY
```

## Outputs

| Path | Content |
|---|---|
| `contact_sheets/` | 6 focus images |
| `match_viz/` | A0/A3/B2 match arrows @512 |
| `per_image_metrics.csv` | PSNR/SSIM + match + luma/chroma + GT-feat distances |
| `matching_statistics.csv` | Match/S/norm/GT-feat columns only |
| `plots/correlation_scatter.png` | Spearman vs ΔPSNR(B2−A0) |
| `ft_regression/` | ckpt/log analysis + traj plots |
| `index_to_filename.json` | Eval order → filename |
| `summary_methods.json` | Aggregate means |
| `correlations.json` | Spearman ρ |
| `focus_image_deltas.json` | Six-image deltas |
| `console.log` | Run log |
