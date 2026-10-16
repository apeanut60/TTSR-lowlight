# v1r12 reproduction

Branch: `experiment/v1r12-texture-zero-vs-bypass`  
Ckpt: v1r3e4 · Texture: v1r9 M1 · Gate: original A0 S

```bash
conda activate ttsr
cd /root/projects/TTSR-lowlight-v1-data1SOTA
python tests/test_v1r12_texture_bypass.py
python scripts/run_v1r12_texture_bypass_audit.py \
  --out_dir /root/projects/TTSR-lowlight-v1-data1SOTA/diagnostics/v1r12
```

Stage modes (default `None` = production normal):

- `--stage_mode_lv3|lv2|lv1 normal|texture_zero|branch_bypass`
