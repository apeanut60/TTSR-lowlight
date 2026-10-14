#!/usr/bin/env python3
"""Correctness tests for v1r10 S-gate counterfactuals."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

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
        collect_gate_stats=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _batch(device, h=64, w=64):
    torch.manual_seed(0)
    lr = torch.rand(1, 3, h, w, device=device) * 2 - 1
    ref = torch.rand(1, 3, h, w, device=device) * 2 - 1
    return lr, lr.clone(), ref, ref.clone()


def _load(model, device):
    if not os.path.isfile(CKPT):
        return
    state = torch.load(CKPT, map_location=device)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    cur = model.state_dict()
    for k, v in state.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
    model.load_state_dict(cur)


def test_default_gate_path_unchanged():
    """Instrumentation must not change G0 predictions vs pre-hook formula."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr, lrsr, ref, refsr = _batch(device)
    m = TTSREnhance.TTSREnhance(_args()).to(device).eval()
    _load(m, device)
    with torch.no_grad():
        sr = m(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr)[0]
        # Manual G from last fusion S should match MainNet last_gate
        info = m.last_texture_fusion_info
        assert info is not None
        S = info['S_a0']
        H, W = lr.shape[-2:]
        G_ref = MainNetEnhance.build_gate_from_S(S, (H, W))
        # Forward again with collect stats — SR must match
        sr2 = m(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr,
                collect_gate_stats=True)[0]
        assert torch.allclose(sr, sr2, rtol=0, atol=0)
        assert torch.allclose(m.MainNet.last_gate, G_ref, rtol=1e-5, atol=1e-5)


def test_identical_T_across_s_sources():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr, lrsr, ref, refsr = _batch(device)
    m = TTSREnhance.TTSREnhance(_args()).to(device).eval()
    _load(m, device)
    with torch.no_grad():
        m.args.texture_fusion_s_source = 'a0'
        _ = m(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr)
        Ta = [t.detach().clone() for t in m.last_texture_fusion_info['T_mix']]
        Sa = m.last_texture_fusion_info['S_a0'].detach().clone()
        Sb_from_a = m.last_texture_fusion_info['S_b2'].detach().clone()

        m.args.texture_fusion_s_source = 'b2'
        _ = m(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr)
        Tb = [t.detach().clone() for t in m.last_texture_fusion_info['T_mix']]
        Sb = m.last_texture_fusion_info['S_b2'].detach().clone()
    for a, b in zip(Ta, Tb):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-5)
    assert torch.allclose(Sb_from_a, Sb, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(Sa, Sb)


def test_g2_constant_mean_zero_var():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    G = torch.rand(1, 1, 32, 40, device=device)
    G2 = MainNetEnhance.apply_gate_transform(G, gate_mode='constant_mean')
    assert G2.shape == G.shape
    assert float(G2.std(unbiased=False)) < 1e-7
    assert torch.allclose(G2.mean(), G.mean(), rtol=1e-6, atol=1e-6)
    assert (G2 >= 0).all() and (G2 <= 1).all()


def test_g3_preserves_mean():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1)
    G = torch.rand(1, 1, 64, 64, device=device)
    G3 = MainNetEnhance.apply_gate_transform(
        G, gate_mode='smooth', smooth_kernel=31)
    assert torch.allclose(G3.mean(), G.mean(), rtol=1e-4, atol=1e-4)
    assert (G3 >= 0).all() and (G3 <= 1).all()
    # Smoothed should have lower spatial variance
    assert float(G3.std()) <= float(G.std()) + 1e-6


def test_g4_preserves_distribution():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    G = torch.rand(1, 1, 48, 48, device=device)
    for shift in [(16, 0), (0, 16), (12, 12)]:
        G4 = MainNetEnhance.apply_gate_transform(
            G, gate_mode='shift', gate_shift=shift)
        assert torch.allclose(
            torch.sort(G.flatten())[0], torch.sort(G4.flatten())[0])
        assert (G4 >= 0).all() and (G4 <= 1).all()


def test_gate_finite_and_ckpt_unchanged():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr, lrsr, ref, refsr = _batch(device)
    m = TTSREnhance.TTSREnhance(_args()).to(device).eval()
    _load(m, device)
    before = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        for mode, shift in [
            (None, None),
            ('constant_mean', None),
            ('smooth', None),
            ('shift', (8, 8)),
        ]:
            _ = m(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr,
                  gate_mode=mode, gate_shift=shift, collect_gate_stats=True)
            g = m.MainNet.last_gate
            assert torch.isfinite(g).all()
            assert float(g.min()) >= -1e-6 and float(g.max()) <= 1.0 + 1e-6
    after = m.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k].cpu())


def test_mainnet_receives_low():
    """API contract: first MainNet arg is original lr (tested via fusion path)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr, lrsr, ref, refsr = _batch(device)
    m = TTSREnhance.TTSREnhance(_args()).to(device).eval()
    # Monkeypatch to capture MainNet input
    captured = {}
    real = m.MainNet.forward

    def wrap(x, *a, **kw):
        captured['x'] = x.detach().clone()
        return real(x, *a, **kw)

    m.MainNet.forward = wrap
    with torch.no_grad():
        _ = m(lr=lr, lrsr=lrsr * 0.1, ref=ref, refsr=refsr)
    assert torch.equal(captured['x'], lr)


if __name__ == '__main__':
    test_g2_constant_mean_zero_var()
    print('OK G2')
    test_g3_preserves_mean()
    print('OK G3')
    test_g4_preserves_distribution()
    print('OK G4')
    test_default_gate_path_unchanged()
    print('OK default')
    test_identical_T_across_s_sources()
    print('OK identical T')
    test_gate_finite_and_ckpt_unchanged()
    print('OK finite/ckpt')
    test_mainnet_receives_low()
    print('OK Low')
    print('ALL PASS')
