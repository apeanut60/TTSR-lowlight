# v1r11 reproduction

Branch: `experiment/v1r11-stage-texture-audit`  
Ckpt: v1r3e4 · Texture: v1r9 M1 (λ=0.25, A0 S, α=0.75, β=0.50)

```bash
conda activate ttsr
cd /root/projects/TTSR-lowlight-v1-data1SOTA
python tests/test_v1r11_stage_mask.py
python scripts/run_v1r11_stage_texture_audit.py \
  --out_dir /root/projects/TTSR-lowlight-v1-data1SOTA/diagnostics/v1r11
```

Eval-only flags (default False): `--disable_T_lv3`, `--disable_T_lv2`, `--disable_T_lv1`.
