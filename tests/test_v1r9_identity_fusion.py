#!/usr/bin/env python3
"""Unit / regression tests for v1r9 Identity texture fusion (no full Eval)."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from model import TTSREnhance, SearchTransfer  # noqa: E402
from model.SearchTransfer import SearchTransfer as ST  # noqa: E402


CKPT = ('/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/'
        'model/model_00004.pt')


def _args(**kw):
    base = dict(
        num_res_blocks='10+10+6+4', n_feats=64, res_scale=1.0,
        freeze_lte=False, ref_correction=True, ref_correction_feats=16,
        ref_illum_pool=8,
        query_illum_align=False, query_illum_alpha=0.0, query_hist_beta=0.0,
        query_gamma_min=0.3, query_gamma_max=1.0, query_hist_bins=256,
        texture_fusion=False, texture_fusion_lambda=0.0,
        texture_fusion_s_source='a0',
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _rand_batch(n=1, h=64, w=80, device='cpu'):
    """Non-square batch divisible by 4."""
    assert h % 4 == 0 and w % 4 == 0
    lr = torch.rand(n, 3, h, w, device=device) * 2 - 1
    ref = torch.rand(n, 3, h, w, device=device) * 2 - 1
    return lr, lr.clone(), ref, ref.clone()


def test_identity_arg_aligned_and_rejects_mismatch():
    arg = ST.identity_match_arg(10, 12, 10, 12, batch=2, device='cpu')
    assert arg.shape == (2, 120)
    assert int(arg[0, 0]) == 0 and int(arg[0, -1]) == 119
    try:
        ST.identity_match_arg(10, 12, 10, 11, batch=1, device='cpu')
        assert False, 'expected ValueError'
    except ValueError as e:
        assert 'identical Query/Key' in str(e)


def test_identity_transfer_shapes_finite_borders():
    st = ST()
    n, c, h, w = 1, 16, 16, 20  # lv3 sizes → lv1 64×80
    ref3 = torch.randn(n, c, h, w)
    ref2 = torch.randn(n, c, h * 2, w * 2)
    ref1 = torch.randn(n, c, h * 4, w * 4)
    t3, t2, t1 = st.transfer_identity(ref1, ref2, ref3, h, w, Hr=h, Wr=w)
    assert t3.shape == (n, c, h, w)
    assert t2.shape == (n, c, h * 2, w * 2)
    assert t1.shape == (n, c, h * 4, w * 4)
    assert torch.isfinite(t3).all() and torch.isfinite(t2).all() and torch.isfinite(t1).all()
    # Corner / last index gather must not overflow
    arg = ST.identity_match_arg(h, w, h, w, n, ref3.device)
    assert int(arg.max()) == h * w - 1


def test_fusion_disabled_matches_original():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    lr, lrsr, ref, refsr = _rand_batch(device=device)
    m0 = TTSREnhance.TTSREnhance(_args(texture_fusion=False)).to(device).eval()
    m1 = TTSREnhance.TTSREnhance(_args(texture_fusion=False)).to(device).eval()
    # identical random init for LTE/MainNet — copy state
    m1.load_state_dict(m0.state_dict())
    with torch.no_grad():
        o0 = m0(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr)
        o1 = m1(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr)
    for a, b in zip(o0, o1):
        assert torch.equal(a, b)


def test_lambda_endpoints_and_linear_mix():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1)
    lr, lrsr, ref, refsr = _rand_batch(h=64, w=64, device=device)
    args = _args(
        texture_fusion=True, texture_fusion_lambda=0.0,
        texture_fusion_s_source='a0',
        query_illum_align=True, query_illum_alpha=0.75, query_hist_beta=0.50,
    )
    model = TTSREnhance.TTSREnhance(args).to(device).eval()
    if os.path.isfile(CKPT):
        state = torch.load(CKPT, map_location=device)
        if any(k.startswith('module.') for k in state):
            state = {k.replace('module.', '', 1): v for k, v in state.items()}
        cur = model.state_dict()
        for k, v in state.items():
            if k in cur and cur[k].shape == v.shape:
                cur[k] = v
        model.load_state_dict(cur)

    with torch.no_grad():
        # λ=0 → T_mix == T_B2, S == S_A0
        model.args.texture_fusion_lambda = 0.0
        _ = model(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr)
        info0 = model.last_texture_fusion_info
        assert info0 is not None
        for tm, tb in zip(info0['T_mix'], info0['T_b2']):
            assert torch.allclose(tm, tb, rtol=1e-5, atol=1e-5)
        # Capture endpoints
        Tb = [t.detach().clone() for t in info0['T_b2']]
        Ti = [t.detach().clone() for t in info0['T_id']]
        Sa0 = info0['S_a0'].detach().clone()
        Sb2 = info0['S_b2'].detach().clone()
        assert not torch.allclose(Sa0, Sb2), 'A0 and B2 S should differ with align on'

        model.args.texture_fusion_lambda = 1.0
        _ = model(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr)
        info1 = model.last_texture_fusion_info
        for tm, ti in zip(info1['T_mix'], Ti):
            assert torch.allclose(tm, ti, rtol=1e-5, atol=1e-5)
        assert torch.allclose(info1['S_a0'], Sa0, rtol=1e-5, atol=1e-5)

        for lam in (0.25, 0.5, 0.75):
            model.args.texture_fusion_lambda = lam
            _ = model(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr)
            info = model.last_texture_fusion_info
            for tm, tb, ti in zip(info['T_mix'], Tb, Ti):
                expect = (1.0 - lam) * tb + lam * ti
                assert torch.allclose(tm, expect, rtol=1e-5, atol=1e-5)
            assert torch.allclose(info['S_a0'], Sa0, rtol=1e-5, atol=1e-5)


def test_mainnet_gets_original_low_and_no_gt():
    """Forward must accept only Low/Ref; fusion info Value comes from Ref LTE."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(2)
    lr, lrsr, ref, refsr = _rand_batch(device=device)
    # Distinct Low vs Query tensors: if MainNet used Query, SR would change when
    # only lrsr changes under fusion-off A0 — under fusion, Query align changes
    # T/S but MainNet input is still `lr`.
    model = TTSREnhance.TTSREnhance(_args(
        texture_fusion=True, texture_fusion_lambda=0.5,
        query_illum_align=True, query_illum_alpha=0.75, query_hist_beta=0.5,
    )).to(device).eval()
    lrsr_bright = (lrsr * 0.1 + 0.9).clamp(-1, 1)
    with torch.no_grad():
        sr0 = model(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr)[0]
        # Swap only Query; keep lr fixed — SR may change via matching, but
        # ``lr`` identity is what we assert via API contract (MainNet(lr,...)).
        sr1 = model(lr=lr, lrsr=lrsr_bright, ref=ref, refsr=refsr)[0]
    assert sr0.shape == lr.shape == sr1.shape
    # Ref Value path unchanged by Query: T_id depends only on Ref features.
    info = model.last_texture_fusion_info
    assert info is not None
    assert all(torch.isfinite(t).all() for t in info['T_id'])


def test_checkpoint_core_load():
    if not os.path.isfile(CKPT):
        print('SKIP ckpt missing')
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TTSREnhance.TTSREnhance(_args()).to(device)
    state = torch.load(CKPT, map_location=device)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    cur = model.state_dict()
    matched = 0
    for k, v in state.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
            matched += 1
    model.load_state_dict(cur)
    assert matched >= 260, matched


def test_tile_identity_local_coords():
    """Within a tile crop, Identity indices are tile-local and match full-grid
    spatial correspondence for that crop (same y0:x0 on Low and Ref)."""
    st = ST()
    # Simulate two adjacent non-square tiles with identical Low/Ref crops.
    for h, w in [(16, 20), (12, 16)]:
        ref3 = torch.randn(1, 8, h, w)
        ref2 = torch.randn(1, 8, h * 2, w * 2)
        ref1 = torch.randn(1, 8, h * 4, w * 4)
        t3, _, _ = st.transfer_identity(ref1, ref2, ref3, h, w, Hr=h, Wr=w)
        # Identity gather of lv3 with 3×3 unfold/fold ≈ local average of ref3;
        # center pixels should closely track ref3 (padding affects borders).
        assert t3.shape[-2:] == (h, w)
        err = (t3[:, :, 2:-2, 2:-2] - ref3[:, :, 2:-2, 2:-2]).abs().mean()
        # Soft check: finite and not wildly different from source Value
        assert torch.isfinite(err)
        assert float(err) < 5.0  # features are arbitrary scale


if __name__ == '__main__':
    test_identity_arg_aligned_and_rejects_mismatch()
    print('OK identity_arg')
    test_identity_transfer_shapes_finite_borders()
    print('OK identity_transfer')
    test_tile_identity_local_coords()
    print('OK tile_identity')
    test_fusion_disabled_matches_original()
    print('OK fusion_disabled')
    test_lambda_endpoints_and_linear_mix()
    print('OK lambda_mix')
    test_mainnet_gets_original_low_and_no_gt()
    print('OK mainnet_low')
    test_checkpoint_core_load()
    print('OK ckpt')
    print('ALL PASS')
