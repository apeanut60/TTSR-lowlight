# v1r9 reproduction

Branch: `experiment/v1r9-identity-texture-fusion`  
Ckpt (read-only): `/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt`

## Tests

```bash
conda activate ttsr
cd /root/projects/TTSR-lowlight-v1-data1SOTA
python tests/test_v1r9_identity_fusion.py
```

## Zero-shot F0 / F1 / M0–M4

```bash
python scripts/run_v1r9_identity_fusion_eval.py \
  --out_dir /root/projects/TTSR-lowlight-v1-data1SOTA/diagnostics/v1r9
```

Flags (defaults preserve original behavior):

- `--texture_fusion False`
- `--texture_fusion_lambda 0`
- `--texture_fusion_s_source a0`
