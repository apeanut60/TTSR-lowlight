"""Tests for v1r7 gamma + luminance histogram query alignment."""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from model.query_illum_align import (  # noqa: E402
    align_query_illumination, build_luma_hist_lut, apply_luma_hist_lut,
    estimate_query_align_maps,
)


def _assert(c, m):
    if not c:
        raise AssertionError(m)


def test_beta0_matches_gamma():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    low = torch.rand(2, 3, 64, 64, device=device) * 0.4 - 1.0  # dark
    ref = torch.rand(2, 3, 64, 64, device=device) * 0.6
    a, ia = align_query_illumination(low, ref, alpha=0.75, beta=0.0)
    b, ib = align_query_illumination(low, ref, alpha=0.75, beta=0.0)
    # beta=0 path identical to gamma-only (same call)
    _assert(torch.allclose(a, b), 'beta0 not deterministic')
    # Explicit: with lut applied at beta=0 inside apply should no-op
    g, lut, _ = estimate_query_align_maps(low, ref, alpha=0.75)
    c, _ = align_query_illumination(
        low, ref, alpha=0.75, beta=0.0, gamma_override=g, lut_override=lut)
    _assert(torch.allclose(a, c, atol=1e-6), 'beta0 with lut_override differs')
    print('OK test_beta0_matches_gamma')


def test_alpha0_beta0_identity():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    low = torch.randn(1, 3, 48, 48, device=device).clamp(-1, 1)
    ref = torch.randn(1, 3, 48, 48, device=device).clamp(-1, 1)
    out, _ = align_query_illumination(low, ref, alpha=0.0, beta=0.0)
    _assert(torch.equal(out, low), 'a0b0 must be identity')
    print('OK test_alpha0_beta0_identity')


def test_lut_monotonic_no_nan():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # black / dark / normal / saturated-ish
    cases = [
        torch.full((1, 3, 32, 32), -1.0, device=device),
        torch.full((1, 3, 32, 32), -0.85, device=device),
        torch.rand(1, 3, 40, 50, device=device) * 2 - 1,
        torch.full((1, 3, 32, 32), 0.9, device=device),
    ]
    refs = [
        torch.zeros(1, 3, 32, 32, device=device),
        torch.rand(1, 3, 32, 32, device=device) * 0.5,
        torch.rand(1, 3, 30, 40, device=device),  # size mismatch
        torch.full((1, 3, 32, 32), 0.2, device=device),
    ]
    for low, ref in zip(cases, refs):
        out, info = align_query_illumination(
            low, ref, alpha=0.75, beta=0.5)
        _assert(torch.isfinite(out).all(), 'NaN/Inf in output')
        g, lut, _ = estimate_query_align_maps(low, ref, alpha=0.75)
        _assert(torch.isfinite(lut).all(), 'NaN in lut')
        d = lut[0, 1:] - lut[0, :-1]
        _assert((d >= -1e-6).all(), 'LUT not monotonic')
        _assert(torch.isfinite(info['gamma']).all(), 'gamma bad')
    print('OK test_lut_monotonic_no_nan')


def test_hist_changes_when_beta():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    low = torch.full((1, 3, 64, 64), -0.7, device=device)
    ref = torch.full((1, 3, 64, 64), 0.3, device=device)
    o0, i0 = align_query_illumination(low, ref, alpha=0.75, beta=0.0)
    o1, i1 = align_query_illumination(low, ref, alpha=0.75, beta=0.5)
    _assert(not torch.allclose(o0, o1), 'beta should change query')
    # CDF distance should not increase after hist (usually decreases)
    _assert(float(i1['cdf_l1_after_hist']) <= float(i1['cdf_l1_before_hist']) + 1e-5,
            'hist should not worsen CDF L1')
    print('OK test_hist_changes_when_beta cdf %.4f->%.4f' % (
        float(i1['cdf_l1_before_hist']), float(i1['cdf_l1_after_hist'])))


def test_model_shapes_unchanged():
    from model import TTSREnhance
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = ('/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/'
            'model/model_00004.pt')
    if not os.path.isfile(ckpt):
        print('SKIP test_model_shapes_unchanged')
        return

    def make(align, a, b):
        ns = argparse.Namespace(
            num_res_blocks='10+10+6+4', n_feats=64, res_scale=1.0,
            ref_illum_pool=8, freeze_lte=False, ref_correction=True,
            ref_correction_feats=16, query_illum_align=align,
            query_illum_alpha=a, query_hist_beta=b,
            query_gamma_min=0.3, query_gamma_max=1.0, query_hist_bins=256)
        m = TTSREnhance.TTSREnhance(ns).to(device).eval()
        st = torch.load(ckpt, map_location=device)
        if any(k.startswith('module.') for k in st):
            st = {k.replace('module.', '', 1): v for k, v in st.items()}
        cur = m.state_dict()
        for k, v in st.items():
            if k in cur and cur[k].shape == v.shape:
                cur[k] = v
        m.load_state_dict(cur)
        return m

    m0 = make(False, 0.0, 0.0)
    m1 = make(True, 0.75, 0.25)
    _assert(set(m0.state_dict()) == set(m1.state_dict()), 'keys differ')
    for k in m0.state_dict():
        _assert(m0.state_dict()[k].shape == m1.state_dict()[k].shape, k)
    torch.manual_seed(0)
    lr = torch.randn(1, 3, 128, 128, device=device).clamp(-1, 1)
    ref = torch.randn(1, 3, 128, 128, device=device).clamp(-1, 1)
    with torch.no_grad():
        s0 = m0(lr=lr, lrsr=lr, ref=ref, refsr=ref)[0]
        # alpha=0 beta=0 with align on
        m00 = make(True, 0.0, 0.0)
        s00 = m00(lr=lr, lrsr=lr, ref=ref, refsr=ref)[0]
    _assert((s0 - s00).abs().max().item() < 1e-5, 'a0b0 mismatch')
    print('OK test_model_shapes_unchanged')


def test_full_image_maps_apply_to_crop():
    """LUT from full image applied to crop must be finite and use override."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    full_low = torch.rand(1, 3, 256, 256, device=device) * 0.3 - 1
    full_ref = torch.rand(1, 3, 256, 256, device=device) * 0.5
    g, lut, _ = estimate_query_align_maps(full_low, full_ref, alpha=0.75)
    crop_low = full_low[:, :, 32:160, 40:168]
    crop_ref = full_ref[:, :, 10:138, 20:148]  # independent crop
    out, info = align_query_illumination(
        crop_low, crop_ref, alpha=0.75, beta=0.5,
        gamma_override=g, lut_override=lut)
    _assert(torch.isfinite(out).all(), 'crop apply not finite')
    _assert(float(info['gamma']) == float(g), 'gamma override unused')
    print('OK test_full_image_maps_apply_to_crop')


if __name__ == '__main__':
    test_beta0_matches_gamma()
    test_alpha0_beta0_identity()
    test_lut_monotonic_no_nan()
    test_hist_changes_when_beta()
    test_full_image_maps_apply_to_crop()
    test_model_shapes_unchanged()
    print('ALL PASSED')
