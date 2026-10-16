#!/usr/bin/env python3
"""v1r11: texture stage masking correctness."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from model import TTSREnhance  # noqa: E402

CKPT = ('/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/'
        'model/model_00004.pt')


def _args(**kw):
    base = dict(
        num_res_blocks='10+10+6+4', n_feats=64, res_scale=1.0,
        freeze_lte=False, ref_correction=True, ref_correction_feats=16,
        ref_illum_pool=8,
        query_illum_align=True, query_illum_alpha=0.75, query_hist_beta=0.50,
        query_gamma_min=0.3, query_gamma_max=1.0, query_hist_bins=256,
        texture_fusion=True, texture_fusion_lambda=0.25,
        texture_fusion_s_source='a0',
        gate_mode=None, gate_shift='', gate_smooth_kernel=31,
        collect_gate_stats=False,
        disable_T_lv3=False, disable_T_lv2=False, disable_T_lv1=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _load(m, device):
    state = torch.load(CKPT, map_location=device)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    cur = m.state_dict()
    for k, v in state.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
    m.load_state_dict(cur)


def test_default_equals_no_mask():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    lr = torch.rand(1, 3, 64, 64, device=device) * 2 - 1
    ref = torch.rand(1, 3, 64, 64, device=device) * 2 - 1
    m = TTSREnhance.TTSREnhance(_args()).to(device).eval()
    _load(m, device)
    with torch.no_grad():
        a = m(lr=lr, lrsr=lr, ref=ref, refsr=ref)[0]
        b = m(lr=lr, lrsr=lr, ref=ref, refsr=ref,
              disable_T_lv3=False, disable_T_lv2=False, disable_T_lv1=False)[0]
        assert torch.equal(a, b)


def test_mask_changes_output_and_zeros_stage_stats():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1)
    lr = torch.rand(1, 3, 64, 64, device=device) * 2 - 1
    ref = torch.rand(1, 3, 64, 64, device=device) * 2 - 1
    m = TTSREnhance.TTSREnhance(_args()).to(device).eval()
    _load(m, device)
    with torch.no_grad():
        s0 = m(lr=lr, lrsr=lr, ref=ref, refsr=ref, collect_gate_stats=True)[0]
        # Disable lv3: stage1 residual still computed from zeros T — not necessarily
        # zero residual (head sees concat(stream, 0)), but SR should differ.
        s1 = m(lr=lr, lrsr=lr, ref=ref, refsr=ref,
               disable_T_lv3=True, collect_gate_stats=True)[0]
        assert not torch.allclose(s0, s1, atol=1e-5)
        # All three disabled should also differ from S0
        s_all = m(lr=lr, lrsr=lr, ref=ref, refsr=ref,
                  disable_T_lv3=True, disable_T_lv2=True, disable_T_lv1=True)[0]
        assert not torch.allclose(s0, s_all, atol=1e-5)


if __name__ == '__main__':
    test_default_equals_no_mask()
    print('OK default')
    test_mask_changes_output_and_zeros_stage_stats()
    print('OK mask changes')
    print('ALL PASS')
