#!/usr/bin/env python3
"""v1r14: config wiring + historical Eval30 regression gates (smoke)."""
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


def _base(**kw):
    d = dict(
        num_res_blocks='10+10+6+4', n_feats=64, res_scale=1.0,
        freeze_lte=False, ref_correction=True, ref_correction_feats=16,
        ref_illum_pool=8,
        query_illum_align=False, query_illum_alpha=0.0, query_hist_beta=0.0,
        query_gamma_min=0.3, query_gamma_max=1.0, query_hist_bins=256,
        texture_fusion=False, texture_fusion_lambda=0.0,
        texture_fusion_s_source='a0',
        gate_mode=None, gate_shift='', gate_smooth_kernel=31,
        collect_gate_stats=False, collect_residual_decomp=False,
        disable_T_lv3=False, disable_T_lv2=False, disable_T_lv1=False,
        stage_mode_lv3=None, stage_mode_lv2=None, stage_mode_lv1=None,
    )
    d.update(kw)
    return SimpleNamespace(**d)


def _load(m, device):
    state = torch.load(CKPT, map_location=device)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    cur = m.state_dict()
    for k, v in state.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
    m.load_state_dict(cur)


def _cfg_A0(m):
    m.args.query_illum_align = False
    m.args.query_illum_alpha = 0.0
    m.args.query_hist_beta = 0.0
    m.args.texture_fusion = False
    m.args.stage_mode_lv3 = 'normal'
    m.args.stage_mode_lv2 = 'normal'
    m.args.stage_mode_lv1 = 'normal'


def _cfg_M1(m):
    m.args.query_illum_align = True
    m.args.query_illum_alpha = 0.75
    m.args.query_hist_beta = 0.50
    m.args.texture_fusion = True
    m.args.texture_fusion_lambda = 0.25
    m.args.texture_fusion_s_source = 'a0'
    m.args.stage_mode_lv3 = 'normal'
    m.args.stage_mode_lv2 = 'normal'
    m.args.stage_mode_lv1 = 'normal'


def _cfg_P2(m):
    _cfg_M1(m)
    m.args.stage_mode_lv2 = 'branch_bypass'


def test_configs_produce_finite_and_p2_skips_conv22():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    lr = torch.rand(1, 3, 64, 64, device=device) * 2 - 1
    ref = torch.rand(1, 3, 64, 64, device=device) * 2 - 1
    m = TTSREnhance.TTSREnhance(_base()).to(device).eval()
    _load(m, device)
    outs = {}
    with torch.no_grad():
        for name, cfg in [('A0', _cfg_A0), ('M1', _cfg_M1), ('P2', _cfg_P2)]:
            cfg(m)
            outs[name] = m(lr=lr, lrsr=lr, ref=ref, refsr=ref)[0]
            assert torch.isfinite(outs[name]).all()
    assert not torch.allclose(outs['A0'], outs['M1'], atol=1e-4)
    assert not torch.allclose(outs['M1'], outs['P2'], atol=1e-5)
    # P2 skips conv22
    counts = {'c22': 0}
    real = m.MainNet.conv22_head.forward

    def wrap(*a, **k):
        counts['c22'] += 1
        return real(*a, **k)

    m.MainNet.conv22_head.forward = wrap
    _cfg_P2(m)
    with torch.no_grad():
        _ = m(lr=lr, lrsr=lr, ref=ref, refsr=ref)
    assert counts['c22'] == 0
    assert m.MainNet.last_stage_modes['lv2'] == 'branch_bypass'
    assert m.MainNet.last_stage_modes['lv3'] == 'normal'
    assert m.MainNet.last_stage_modes['lv1'] == 'normal'


def test_m1_p2_identical_T_and_S():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1)
    lr = torch.rand(1, 3, 64, 64, device=device) * 2 - 1
    ref = torch.rand(1, 3, 64, 64, device=device) * 2 - 1
    m = TTSREnhance.TTSREnhance(_base()).to(device).eval()
    _load(m, device)
    with torch.no_grad():
        _cfg_M1(m)
        _ = m(lr=lr, lrsr=lr, ref=ref, refsr=ref)
        S1 = m.last_texture_fusion_info['S_a0'].detach().clone()
        T1 = [t.detach().clone() for t in m.last_texture_fusion_info['T_mix']]
        _cfg_P2(m)
        _ = m(lr=lr, lrsr=lr, ref=ref, refsr=ref)
        S2 = m.last_texture_fusion_info['S_a0'].detach().clone()
        T2 = [t.detach().clone() for t in m.last_texture_fusion_info['T_mix']]
    assert torch.equal(S1, S2)
    for a, b in zip(T1, T2):
        assert torch.equal(a, b)


if __name__ == '__main__':
    test_configs_produce_finite_and_p2_skips_conv22()
    print('OK configs')
    test_m1_p2_identical_T_and_S()
    print('OK T/S')
    print('ALL PASS')
