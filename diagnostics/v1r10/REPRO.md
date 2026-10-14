# v1r10 reproduction

Branch: `experiment/v1r10-s-gate-audit`  
Ckpt: `/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/model/model_00004.pt`

## Tests

```bash
conda activate ttsr
cd /root/projects/TTSR-lowlight-v1-data1SOTA
python tests/test_v1r10_s_gate_audit.py
```

## Full G0–G4 audit

```bash
python scripts/run_v1r10_s_gate_audit.py \
  --out_dir /root/projects/TTSR-lowlight-v1-data1SOTA/diagnostics/v1r10
```

Frozen M1: α=0.75, β=0.50, λ=0.25, `texture_fusion=True`.  
Default production path: `gate_mode=None` (unchanged).
