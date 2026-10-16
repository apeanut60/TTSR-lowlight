# v1r13 reproduction

Branch: `experiment/v1r13-stage-interaction-audit`  
Ckpt: v1r3e4 · M1 texture · A0 gate · reused v1r12 `stage_mode_*`

```bash
conda activate ttsr
cd /root/projects/TTSR-lowlight-v1-data1SOTA
python tests/test_v1r13_stage_interaction.py
python scripts/run_v1r13_stage_interaction.py \
  --out_dir /root/projects/TTSR-lowlight-v1-data1SOTA/diagnostics/v1r13
```
