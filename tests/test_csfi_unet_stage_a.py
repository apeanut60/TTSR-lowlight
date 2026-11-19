"""Smoke tests for ASRT Stage-A CSFI-U-Net port (block-amend)."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from model.CSFIUNet import CSFIUNet  # noqa: E402
from model.igab import IGAB, IlluminationEstimator  # noqa: E402
from model.TTSREnhance import TTSREnhance  # noqa: E402
from types import SimpleNamespace


def _csfi_args(**kw):
    d = dict(
        backbone='csfi_unet',
        num_res_blocks='2+2+1+1',  # tiny for CPU smoke
        n_feats=32,
        res_scale=1.0,
        igab_blocks='1,1',
        igab_heads=1,
        enable_lightup=True,
        use_global_illum=False,
        use_ref_illum=False,
        freeze_lte=True,
        ref_correction=False,
        ref_illum_pool=8,
        query_illum_align=False,
        query_illum_alpha=0.0,
        query_hist_beta=0.0,
        query_gamma_min=0.3,
        query_gamma_max=1.0,
        query_hist_bins=64,
        texture_fusion=False,
        texture_fusion_lambda=0.0,
        texture_fusion_s_source='a0',
        gate_mode=None,
        gate_shift='',
        gate_smooth_kernel=31,
        collect_gate_stats=False,
        collect_residual_decomp=False,
        disable_T_lv3=False,
        disable_T_lv2=False,
        disable_T_lv1=False,
        stage_mode_lv3=None,
        stage_mode_lv2=None,
        stage_mode_lv1=None,
    )
    d.update(kw)
    return SimpleNamespace(**d)


def test_pad_crop_odd_hw():
    net = CSFIUNet(num_res_blocks='2+2+1+1', n_feats=32, enable_lightup=True)
    x = torch.randn(1, 3, 37, 41)
    y = net(x, S=None, T_lv3=None, T_lv2=None, T_lv1=None)
    assert y.shape == x.shape, (y.shape, x.shape)


def test_ref_none_runs():
    net = CSFIUNet(num_res_blocks='2+2+1+1', n_feats=32)
    x = torch.randn(1, 3, 64, 64)
    y = net(x)
    assert y.shape == (1, 3, 64, 64)
    assert torch.isfinite(y).all()


def test_igab_shape():
    igab = IGAB(32, num_blocks=1, heads=1)
    feat = torch.randn(2, 32, 16, 20)
    illu = torch.randn(2, 32, 16, 20)
    out = igab(feat, illu)
    assert out.shape == feat.shape


def test_estimator_lightup_soft():
    est = IlluminationEstimator(32)
    low01 = torch.rand(1, 3, 48, 48)
    f, log_gain = est(low01)
    assert f.shape == (1, 32, 48, 48)
    assert log_gain.shape == (1, 3, 48, 48)
    # zero-init conv2 → log_gain ≈ 0 at init
    assert float(log_gain.abs().max()) < 1e-6


def test_soft_inject_near_identity():
    """With soft-init inject heads and S≈-20 (σ≈0), output ≈ light-up base path."""
    torch.manual_seed(0)
    net = CSFIUNet(
        num_res_blocks='2+2+1+1', n_feats=32, enable_lightup=True,
        use_global_illum=False, use_ref_illum=False)
    x = torch.randn(1, 3, 64, 64)
    # Fake SearchTransfer tensors at LTE scales: H, H/2, H/4 channels 64/128/256
    S = torch.full((1, 1, 16, 16), -20.0)  # sigmoid(S)≈0
    T3 = torch.randn(1, 256, 16, 16)
    T2 = torch.randn(1, 128, 32, 32)
    T1 = torch.randn(1, 64, 64, 64)
    y_ref = net(x, S=S, T_lv3=T3, T_lv2=T2, T_lv1=T1)
    y_bypass = net(
        x, S=S, T_lv3=T3, T_lv2=T2, T_lv1=T1,
        stage_mode_lv3='branch_bypass',
        stage_mode_lv2='branch_bypass',
        stage_mode_lv1='branch_bypass')
    # Soft inject with σ(S)≈0 should match full bypass
    err = float((y_ref - y_bypass).abs().max())
    assert err < 1e-4, err


def test_ttsr_backbone_switch():
    args_csfi = _csfi_args()
    m1 = TTSREnhance(args_csfi)
    assert isinstance(m1.MainNet, CSFIUNet)

    args_main = _csfi_args(backbone='mainnet', num_res_blocks='2+2+1+1')
    # MainNetEnhance expects list of 4 ints via split — ok
    from model.MainNetEnhance import MainNetEnhance
    m2 = TTSREnhance(args_main)
    assert isinstance(m2.MainNet, MainNetEnhance)


def test_ttsr_forward_shapes_cpu():
    args = _csfi_args()
    model = TTSREnhance(args).eval()
    lr = torch.randn(1, 3, 48, 52)  # not multiple of 4
    ref = torch.randn(1, 3, 48, 52)
    with torch.no_grad():
        sr, S, T3, T2, T1 = model(lr=lr, lrsr=lr, ref=ref, refsr=ref)
    assert sr.shape == lr.shape
    assert torch.isfinite(sr).all()


def test_tiled_csfi_smoke():
    from v2_p2_runtime import build_args_csfi, forward_p2_tiled_backbone

    args = build_args_csfi()
    args.num_res_blocks = '2+2+1+1'
    args.n_feats = 32
    args.ref_correction = False
    args.tile_size = 64
    args.tile_overlap = 16
    args.texture_fusion = False
    args.query_illum_align = False
    model = TTSREnhance(args).eval()
    assert isinstance(model.MainNet, CSFIUNet)

    lr = torch.randn(1, 3, 100, 120)
    ref = torch.randn(1, 3, 100, 120)
    with torch.no_grad():
        sr = forward_p2_tiled_backbone(
            model, args, lr, lr, ref, ref, enable_grad=False)
    assert sr.shape == lr.shape
    assert torch.isfinite(sr).all()


if __name__ == '__main__':
    test_pad_crop_odd_hw()
    test_ref_none_runs()
    test_igab_shape()
    test_estimator_lightup_soft()
    test_soft_inject_near_identity()
    test_ttsr_backbone_switch()
    test_ttsr_forward_shapes_cpu()
    test_tiled_csfi_smoke()
    print('all csfi stage-a smoke tests passed')
