"""Unit / consistency tests for v1r6 query illumination alignment."""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from model.query_illum_align import (  # noqa: E402
    align_query_illumination, estimate_query_gamma, _solve_gamma_binary,
    _mean_luma,
)


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_gamma_no_nan():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # black, dark, normal
    lows = [
        torch.full((1, 3, 32, 32), -1.0, device=device),
        torch.full((1, 3, 32, 32), -0.9, device=device),
        torch.zeros((1, 3, 48, 64), device=device),
    ]
    refs = [
        torch.zeros((1, 3, 32, 32), device=device),
        torch.full((1, 3, 32, 32), 0.2, device=device),
        torch.full((1, 3, 40, 50), 0.5, device=device),  # size mismatch
    ]
    for low, ref in zip(lows, refs):
        out, info = align_query_illumination(low, ref, alpha=1.0)
        g = info['gamma']
        _assert(torch.isfinite(g).all(), 'gamma not finite')
        _assert(torch.isfinite(out).all(), 'output not finite')
        _assert(out.min() >= -1.0 - 1e-5 and out.max() <= 1.0 + 1e-5,
                'output out of [-1,1]')
        _assert((g >= 0.3 - 1e-6).all() and (g <= 1.0 + 1e-6).all(),
                'gamma out of range')
    print('OK test_gamma_no_nan')


def test_alpha0_identity():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    low = torch.randn(2, 3, 64, 64, device=device).clamp(-1, 1)
    ref = torch.randn(2, 3, 64, 64, device=device).clamp(-1, 1)
    out, _ = align_query_illumination(low, ref, alpha=0.0)
    _assert(torch.equal(out, low), 'alpha=0 must return identical low')
    print('OK test_alpha0_identity')


def test_binary_brighter():
    """gamma < 1 on (0,1) brightens; solving toward brighter ref → gamma < 1."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    low01 = torch.full((1, 3, 16, 16), 0.2, device=device)
    tgt = torch.tensor([0.5], device=device)
    g = _solve_gamma_binary(low01, tgt, 0.3, 1.0)
    _assert(g.item() < 0.99, 'expected gamma < 1 to brighten')
    powered = low01.pow(g.view(1, 1, 1, 1))
    m = _mean_luma(powered).item()
    _assert(abs(m - 0.5) < 0.02, 'mean luma not close to target: %f' % m)
    print('OK test_binary_brighter gamma=%.4f mean=%.4f' % (g.item(), m))


def test_model_alpha0_matches_baseline():
    """Flag off vs flag on alpha=0 must match numerically on same weights."""
    from model import TTSREnhance

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = ('/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/'
            'model/model_00004.pt')
    if not os.path.isfile(ckpt):
        print('SKIP test_model_alpha0_matches_baseline (no ckpt)')
        return

    def _make(align, alpha):
        a = argparse.Namespace(
            num_res_blocks='10+10+6+4', n_feats=64, res_scale=1.0,
            ref_illum_pool=8, freeze_lte=False, ref_correction=True,
            ref_correction_feats=16, query_illum_align=align,
            query_illum_alpha=alpha, query_gamma_min=0.3, query_gamma_max=1.0,
        )
        m = TTSREnhance.TTSREnhance(a).to(device).eval()
        state = torch.load(ckpt, map_location=device)
        if any(k.startswith('module.') for k in state):
            state = {k.replace('module.', '', 1): v for k, v in state.items()}
        cur = m.state_dict()
        for k, v in state.items():
            if k in cur and cur[k].shape == v.shape:
                cur[k] = v
        m.load_state_dict(cur)
        return m

    m0 = _make(False, 0.0)
    m1 = _make(True, 0.0)
    # weight shapes unchanged
    s0 = m0.state_dict()
    s1 = m1.state_dict()
    _assert(set(s0.keys()) == set(s1.keys()), 'state keys differ')
    for k in s0:
        _assert(s0[k].shape == s1[k].shape, 'shape mismatch %s' % k)

    torch.manual_seed(0)
    lr = torch.randn(1, 3, 128, 128, device=device).clamp(-1, 1)
    ref = torch.randn(1, 3, 128, 128, device=device).clamp(-1, 1)
    with torch.no_grad():
        sr0 = m0(lr=lr, lrsr=lr, ref=ref, refsr=ref)[0]
        sr1 = m1(lr=lr, lrsr=lr, ref=ref, refsr=ref)[0]
    max_diff = (sr0 - sr1).abs().max().item()
    _assert(max_diff < 1e-5, 'alpha=0 mismatch max_diff=%g' % max_diff)
    print('OK test_model_alpha0_matches_baseline max_diff=%g' % max_diff)


def test_estimate_gamma_batch():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    low = torch.rand(3, 3, 32, 32, device=device) * 2 - 1
    ref = torch.rand(3, 3, 32, 32, device=device) * 2 - 1
    g = estimate_query_gamma(low, ref)
    _assert(g.shape == (3,), 'gamma shape')
    _assert(torch.isfinite(g).all(), 'gamma finite')
    print('OK test_estimate_gamma_batch')


if __name__ == '__main__':
    test_gamma_no_nan()
    test_alpha0_identity()
    test_binary_brighter()
    test_estimate_gamma_batch()
    test_model_alpha0_matches_baseline()
    print('ALL PASSED')
