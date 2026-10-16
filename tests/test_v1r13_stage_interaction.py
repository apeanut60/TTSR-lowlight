#!/usr/bin/env python3
"""v1r13: P0–P3 stage-interaction correctness."""
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

# (name, mode_lv3, mode_lv2, mode_lv1)
P0 = ('normal', 'normal', 'normal')
P1 = ('texture_zero', 'normal', 'normal')
P2 = ('normal', 'branch_bypass', 'normal')
P3 = ('texture_zero', 'branch_bypass', 'normal')


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
        collect_gate_stats=False, collect_residual_decomp=False,
        disable_T_lv3=False, disable_T_lv2=False, disable_T_lv1=False,
        stage_mode_lv3=None, stage_mode_lv2=None, stage_mode_lv1=None,
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


def _batch(device, seed=0):
    torch.manual_seed(seed)
    lr = torch.rand(1, 3, 64, 64, device=device) * 2 - 1
    ref = torch.rand(1, 3, 64, 64, device=device) * 2 - 1
    return lr, lr.clone(), ref, ref.clone()


def _fwd(m, lr, lrsr, ref, refsr, modes):
    m3, m2, m1 = modes
    return m(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr,
             stage_mode_lv3=m3, stage_mode_lv2=m2, stage_mode_lv1=m1,
             collect_gate_stats=True)[0]


def test_p0_p1_p2_p3_distinct_and_modes_recorded():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr, lrsr, ref, refsr = _batch(device)
    m = TTSREnhance.TTSREnhance(_args()).to(device).eval()
    _load(m, device)
    before = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        outs = {}
        for name, modes in [('P0', P0), ('P1', P1), ('P2', P2), ('P3', P3)]:
            outs[name] = _fwd(m, lr, lrsr, ref, refsr, modes)
            assert torch.isfinite(outs[name]).all()
            assert m.MainNet.last_stage_modes == {
                'lv3': modes[0], 'lv2': modes[1], 'lv1': modes[2],
            }
        # P3 must not accidentally equal P0/P1/P2
        for other in ('P0', 'P1', 'P2'):
            assert not torch.allclose(outs['P3'], outs[other], atol=1e-5), other
        # P1 ≠ P2
        assert not torch.allclose(outs['P1'], outs['P2'], atol=1e-5)
    for k, v in m.state_dict().items():
        assert torch.equal(before[k], v.cpu())


def test_p3_lv3_zero_still_calls_conv11_lv2_bypass_skips_conv22():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr, lrsr, ref, refsr = _batch(device, seed=1)
    m = TTSREnhance.TTSREnhance(_args()).to(device).eval()
    _load(m, device)
    counts = {'c11': 0, 'c22': 0, 'c33': 0}
    r11, r22, r33 = (m.MainNet.conv11_head.forward,
                     m.MainNet.conv22_head.forward,
                     m.MainNet.conv33_head.forward)

    def w11(*a, **k):
        counts['c11'] += 1
        return r11(*a, **k)

    def w22(*a, **k):
        counts['c22'] += 1
        return r22(*a, **k)

    def w33(*a, **k):
        counts['c33'] += 1
        return r33(*a, **k)

    m.MainNet.conv11_head.forward = w11
    m.MainNet.conv22_head.forward = w22
    m.MainNet.conv33_head.forward = w33
    with torch.no_grad():
        _ = _fwd(m, lr, lrsr, ref, refsr, P3)
    assert counts['c11'] >= 1  # texture_zero still runs head
    assert counts['c22'] == 0  # bypass skips
    assert counts['c33'] >= 1  # lv1 normal


def test_p3_stage2_sees_modified_stage1():
    """Stage-2 input path depends on stage-1 output (not cached C0 features)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr, lrsr, ref, refsr = _batch(device, seed=2)
    m = TTSREnhance.TTSREnhance(_args()).to(device).eval()
    _load(m, device)
    # Capture first input to ex12 (uses x21 from stage1 chain and x22)
    # Under P0 vs P3, x21 (from modified x11) should differ when fed to ex12.
    captured = {}
    real_ex = m.MainNet.ex12.forward

    def wrap(a, b):
        if 'x21' not in captured:
            captured['x21'] = a.detach().clone()
            captured['x22'] = b.detach().clone()
        return real_ex(a, b)

    m.MainNet.ex12.forward = wrap
    with torch.no_grad():
        _ = _fwd(m, lr, lrsr, ref, refsr, P0)
        x21_p0 = captured.pop('x21')
        _ = _fwd(m, lr, lrsr, ref, refsr, P3)
        x21_p3 = captured['x21']
    assert not torch.allclose(x21_p0, x21_p3, atol=1e-5)


def test_identical_S_and_T_construction_across_P():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr, lrsr, ref, refsr = _batch(device, seed=3)
    m = TTSREnhance.TTSREnhance(_args()).to(device).eval()
    _load(m, device)
    with torch.no_grad():
        refs = {}
        for name, modes in [('P0', P0), ('P1', P1), ('P2', P2), ('P3', P3)]:
            _ = _fwd(m, lr, lrsr, ref, refsr, modes)
            info = m.last_texture_fusion_info
            refs[name] = (
                info['S_a0'].detach().clone(),
                [t.detach().clone() for t in info['T_mix']],
            )
        for name in ('P1', 'P2', 'P3'):
            assert torch.equal(refs['P0'][0], refs[name][0])
            for a, b in zip(refs['P0'][1], refs[name][1]):
                assert torch.equal(a, b)


if __name__ == '__main__':
    test_p0_p1_p2_p3_distinct_and_modes_recorded()
    print('OK distinct')
    test_p3_lv3_zero_still_calls_conv11_lv2_bypass_skips_conv22()
    print('OK heads')
    test_p3_stage2_sees_modified_stage1()
    print('OK dependency')
    test_identical_S_and_T_construction_across_P()
    print('OK S/T')
    print('ALL PASS')
