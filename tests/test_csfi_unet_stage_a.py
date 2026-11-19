"""Smoke tests for the CSFI-U-Net backbone (block-amend, csfi_lfca_v1; real VGG LTE)."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from model.CSFIUNet import CSFIUNet  # noqa: E402
from model.frequency_mixer import LFCA  # noqa: E402
from model.illumination import IlluminationEstimator  # noqa: E402
from model.TTSREnhance import TTSREnhance  # noqa: E402
from types import SimpleNamespace


def _csfi_args(**kw):
    d = dict(
        backbone='csfi_unet',
        num_res_blocks='2+2+1+1',  # tiny for CPU smoke
        n_feats=32,
        res_scale=1.0,
        restoration_blocks='1,1',
        restoration_heads=1,
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


def test_lfca_shape():
    block = LFCA(32, num_blocks=1, heads=1)
    feat = torch.randn(2, 32, 16, 20)
    out = block(feat)
    assert out.shape == feat.shape


def test_illumination_compat_import():
    """Old import location still resolves to the same (moved) class."""
    from model.igab import IlluminationEstimator as Old
    assert Old is IlluminationEstimator


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


def test_fullsize_real_vgg_train_step():
    """Baseline-size csfi_lfca_v1 + real VGG19 LTE: one L1+MS-SSIM step on
    random tensors (CPU). A finite-gradient smoke test, not a quality check."""
    from block_amend_ckpt import check_grad_finite
    from block_amend_main import BASELINE_FIXED, build_optimizer
    from loss.loss_enhance import get_loss_dict_enhance
    from model.igab import IGAB, IG_MSA

    torch.manual_seed(0)
    args = SimpleNamespace(
        **BASELINE_FIXED, lr_rate=1e-4, lr_rate_lte=1e-6, decay=50, gamma=0.5,
        cpu=True, num_gpu=1)
    model = TTSREnhance(args).train()
    assert model.model_variant == 'csfi_lfca_v1'
    assert not any(isinstance(m, (IGAB, IG_MSA)) for m in model.modules())
    opt, _ = build_optimizer(model, args)
    loss_all = get_loss_dict_enhance(args, None)
    assert set(loss_all) == {'rec_loss', 'ms_ssim_loss'}
    lr = torch.rand(2, 3, 160, 160) * 0.4 - 1.0
    hr = torch.rand(2, 3, 160, 160) * 2 - 1
    ref = torch.rand(2, 3, 160, 160) * 2 - 1
    opt.zero_grad(set_to_none=True)
    sr = model(lr=lr, lrsr=lr, ref=ref, refsr=ref)[0]
    loss = args.rec_w * loss_all['rec_loss'](sr, hr) \
        + args.ms_ssim_w * loss_all['ms_ssim_loss'](sr, hr)
    assert torch.isfinite(loss) and torch.isfinite(sr).all()
    loss.backward()
    check_grad_finite(model, 1, 1, 'synthetic', loss, sr)
    for pos in ('coarse_freq_block', 'middle_freq_block'):
        for n, p in getattr(model.MainNet, pos).named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (pos, n)
        blk = getattr(model.MainNet, pos).blocks[0]
        for w in (blk.attn.to_q.weight, blk.attn.to_k.weight,
                  blk.attn.to_v.weight, blk.attn.proj.weight):
            assert float(w.grad.abs().sum()) > 0
    for p in model.LTE.parameters():
        assert p.requires_grad is False and p.grad is None
    for n, p in model.RefCorrection.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    opt.step()
    for p in model.MainNet.parameters():
        assert torch.isfinite(p).all()


if __name__ == '__main__':
    test_fullsize_real_vgg_train_step()
    test_pad_crop_odd_hw()
    test_ref_none_runs()
    test_lfca_shape()
    test_illumination_compat_import()
    test_estimator_lightup_soft()
    test_soft_inject_near_identity()
    test_ttsr_backbone_switch()
    test_ttsr_forward_shapes_cpu()
    test_tiled_csfi_smoke()
    print('all csfi stage-a smoke tests passed')
