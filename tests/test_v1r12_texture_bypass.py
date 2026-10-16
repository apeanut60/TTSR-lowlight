#!/usr/bin/env python3
"""v1r12 correctness: texture_zero vs branch_bypass."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from model import TTSREnhance  # noqa: E402
from model.MainNetEnhance import MainNetEnhance  # noqa: E402

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


def _batch(device, h=64, w=64, seed=0):
    torch.manual_seed(seed)
    lr = torch.rand(1, 3, h, w, device=device) * 2 - 1
    ref = torch.rand(1, 3, h, w, device=device) * 2 - 1
    return lr, lr.clone(), ref, ref.clone()


def test_resolve_modes():
    assert MainNetEnhance._resolve_stage_mode(None, False) == 'normal'
    assert MainNetEnhance._resolve_stage_mode(None, True) == 'texture_zero'
    assert MainNetEnhance._resolve_stage_mode('branch_bypass', True) == 'branch_bypass'
    assert MainNetEnhance._resolve_stage_mode('texture_zero', False) == 'texture_zero'
    try:
        MainNetEnhance._resolve_stage_mode('nope', False)
        assert False
    except ValueError:
        pass


def test_normal_equals_default_and_finite():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr, lrsr, ref, refsr = _batch(device)
    m = TTSREnhance.TTSREnhance(_args()).to(device).eval()
    _load(m, device)
    before = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        a = m(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr)[0]
        b = m(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr,
              stage_mode_lv3='normal', stage_mode_lv2='normal',
              stage_mode_lv1='normal')[0]
        assert torch.equal(a, b)
        assert torch.isfinite(a).all()
    for k, v in m.state_dict().items():
        assert torch.equal(before[k], v.cpu())


def test_texture_zero_matches_disable_flag():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr, lrsr, ref, refsr = _batch(device, seed=1)
    m = TTSREnhance.TTSREnhance(_args()).to(device).eval()
    _load(m, device)
    with torch.no_grad():
        for which in ('lv3', 'lv2', 'lv1'):
            kw_mode = {'stage_mode_%s' % which: 'texture_zero'}
            kw_dis = {'disable_T_%s' % which: True}
            a = m(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr, **kw_mode)[0]
            b = m(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr, **kw_dis)[0]
            assert torch.allclose(a, b, rtol=0, atol=0), which


def test_texture_zero_still_runs_conv_head():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr, lrsr, ref, refsr = _batch(device, seed=2)
    m = TTSREnhance.TTSREnhance(_args()).to(device).eval()
    _load(m, device)
    called = {'n': 0}
    real = m.MainNet.conv11_head.forward

    def wrap(*a, **k):
        called['n'] += 1
        return real(*a, **k)

    m.MainNet.conv11_head.forward = wrap
    with torch.no_grad():
        _ = m(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr,
              stage_mode_lv3='texture_zero')
        assert called['n'] >= 1
        called['n'] = 0
        _ = m(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr,
              stage_mode_lv3='branch_bypass')
        assert called['n'] == 0  # bypass must not call conv11_head


def test_bypass_leaves_stream_identical_at_injection():
    """Immediately after lv3 injection point, bypass ⇒ stream == pre-inject."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Use isolated MainNet with synthetic T/S
    args = _args(texture_fusion=False, query_illum_align=False)
    # Build only MainNetEnhance
    from model.MainNetEnhance import MainNetEnhance as MNE
    net = MNE(num_res_blocks=[10, 10, 6, 4], n_feats=64, res_scale=1.0).to(device).eval()
    torch.manual_seed(3)
    h, w = 64, 64
    x = torch.rand(1, 3, h, w, device=device) * 2 - 1
    S = torch.rand(1, 1, h // 4, w // 4, device=device)
    T3 = torch.rand(1, 256, h // 4, w // 4, device=device)
    T2 = torch.rand(1, 128, h // 2, w // 2, device=device)
    T1 = torch.rand(1, 64, h, w, device=device)

    # Monkeypatch: capture stream before/after first inject via wrapping _inject path
    # Simpler: run forward with bypass and compare intermediate by hooking SFE output
    captured = {}
    real_sfe = net.SFE.forward

    def sfe_wrap(inp):
        out = real_sfe(inp)
        captured['sfe'] = out.detach().clone()
        return out

    net.SFE.forward = sfe_wrap
    # Hook conv11_head to ensure not called; after stage1 head, x11 should equal sfe
    # Under bypass, first RB gets x11 == SFE(x)
    real_rb = net.RB11[0].forward

    def rb_wrap(inp):
        if 'first_rb_in' not in captured:
            captured['first_rb_in'] = inp.detach().clone()
        return real_rb(inp)

    net.RB11[0].forward = rb_wrap
    with torch.no_grad():
        _ = net(x, S, T3, T2, T1, apply_illum=False, apply_ref_illum=False,
                stage_mode_lv3='branch_bypass')
    assert torch.equal(captured['sfe'], captured['first_rb_in'])


def test_decomp_reconstructs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr, lrsr, ref, refsr = _batch(device, seed=4)
    m = TTSREnhance.TTSREnhance(_args()).to(device).eval()
    _load(m, device)
    with torch.no_grad():
        _ = m(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr,
              collect_residual_decomp=True)
        decomp = m.MainNet.last_residual_decomp
        assert decomp and len(decomp) == 3
        for d in decomp:
            assert d['recon_rel_err'] < 1e-5
            assert 0 <= d['R_zero_over_normal_l2'] <= 2


if __name__ == '__main__':
    test_resolve_modes()
    print('OK resolve')
    test_normal_equals_default_and_finite()
    print('OK normal')
    test_texture_zero_matches_disable_flag()
    print('OK texture_zero≡disable')
    test_texture_zero_still_runs_conv_head()
    print('OK conv_head')
    test_bypass_leaves_stream_identical_at_injection()
    print('OK bypass identity')
    test_decomp_reconstructs()
    print('OK decomp')
    print('ALL PASS')
