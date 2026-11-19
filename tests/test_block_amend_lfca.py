"""LFCA (csfi_lfca_v1) tests: filter, attention math, HF branch, capacity,
integration, save/restore. CPU + synthetic data only.

The LTE is replaced by a small conv stub in this file (no VGG). These are
engineering/consistency checks, not real-model validation; the real-VGG smoke
lives in tests/test_csfi_unet_stage_a.py.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from model.frequency_mixer import (  # noqa: E402
    ConvFeedForward, FixedBinomialLowPass, HighFreqLocalBranch, LFCA, LFCABlock,
    LowFreqRelationChannelAttention,
)
from model.restoration_config import (  # noqa: E402
    MODEL_VARIANT, RESTORATION_CONFIG_KEYS, RESTORATION_DEFAULTS,
    reject_legacy_igab_state, require_model_variant,
    resolve_restoration_aliases, restoration_config_from_args,
)
from tests.test_block_amend_round2 import (  # noqa: E402
    _patch_lte, _synth_tree, _tiny_args,
)


def _raises(exc, fn, contains=None):
    try:
        fn()
    except exc as e:
        if contains is not None:
            assert contains in str(e), (contains, str(e))
        return e
    raise AssertionError('expected %s' % exc.__name__)


def _n_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _tokens(x):
    return x.flatten(2).transpose(1, 2)


# =====================================================================
# A. fixed low-pass
# =====================================================================
def test_a_fixed_lowpass():
    lp = FixedBinomialLowPass()
    k1 = torch.tensor([1., 4., 6., 4., 1.]) / 16.
    assert lp.kernel.shape == (1, 1, 5, 5)
    assert torch.allclose(lp.kernel[0, 0], torch.outer(k1, k1))
    assert abs(float(lp.kernel.sum()) - 1.0) < 1e-6
    # buffer, not Parameter; not trainable; defined by code (not in state_dict)
    assert list(lp.parameters()) == []
    assert 'kernel' in dict(lp.named_buffers())
    assert lp.kernel.requires_grad is False
    assert len(lp.state_dict()) == 0

    # size preserving, including degenerate sizes (replicate fallback)
    for shape in [(2, 8, 16, 20), (1, 4, 1, 1), (1, 4, 1, 7), (1, 4, 7, 1),
                  (1, 4, 2, 2), (1, 4, 2, 9), (1, 4, 3, 3), (1, 3, 5, 4)]:
        x = torch.randn(*shape)
        y = lp(x)
        assert y.shape == x.shape, (shape, y.shape)
        assert torch.isfinite(y).all()

    # constant invariance -> F_high ~ 0 (all border modes)
    for shape in [(1, 3, 1, 1), (1, 3, 2, 2), (1, 3, 1, 6), (2, 3, 12, 9)]:
        c = torch.tensor([0.3, -1.2, 5.0]).view(1, 3, 1, 1).expand(*shape)
        low = lp(c)
        assert torch.allclose(low, c, atol=1e-6)
        assert float((c - low).abs().max()) < 1e-6

    # F_low + F_high == F
    x = torch.randn(2, 6, 11, 13)
    low = lp(x)
    high = x - low
    assert torch.allclose(low + high, x, atol=1e-6)
    # not an identity map
    assert float((low - x).abs().max()) > 1e-2
    # impulse response is the kernel (interior), depthwise (no channel mixing)
    imp = torch.zeros(1, 2, 9, 9)
    imp[0, 0, 4, 4] = 1.0
    out = lp(imp)
    assert torch.allclose(out[0, 0, 2:7, 2:7], lp.kernel[0, 0], atol=1e-7)
    assert float(out[0, 1].abs().max()) == 0.0
    # reflect borders on normal sizes
    ref = F.conv2d(F.pad(x, (2, 2, 2, 2), mode='reflect'),
                   lp.kernel.expand(6, 1, 5, 5), groups=6)
    assert torch.allclose(low, ref, atol=1e-6)

    # dtype follows the input / module cast
    x64 = torch.randn(1, 2, 6, 6, dtype=torch.float64)
    assert lp(x64).dtype == torch.float64
    assert FixedBinomialLowPass().double()(x64).dtype == torch.float64
    # gradients pass to the input; the kernel itself has none
    xg = torch.randn(1, 2, 6, 6, requires_grad=True)
    lp(xg).sum().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()
    assert lp.kernel.grad is None
    print('A fixed_lowpass ok')


# =====================================================================
# B. attention math
# =====================================================================
def _reference_attention(mod, feat, feat_low, transpose_direction=False):
    """Independent per-sample / per-head loop implementation.

    Channel-major layout [C, N]; head g owns channels [g*d, (g+1)*d).
    A_g = softmax(t_g * (Qn_g @ Kn_g^T), dim=-1);  mixed_g = A_g @ V_g.
    """
    b, c, h, w = feat.shape
    heads, d, n = mod.num_heads, mod.dim_head, h * w
    wq, wk, wv = mod.to_q.weight, mod.to_k.weight, mod.to_v.weight
    wo, bo = mod.proj.weight, mod.proj.bias
    out = torch.zeros_like(feat)
    for i in range(b):
        fl = feat_low[i].reshape(c, n)
        ff = feat[i].reshape(c, n)
        q, k, v = wq @ fl, wk @ fl, wv @ ff
        mixed = torch.zeros(c, n, dtype=feat.dtype)
        for g in range(heads):
            sl = slice(g * d, (g + 1) * d)
            qg, kg, vg = q[sl], k[sl], v[sl]
            qn = qg / qg.norm(dim=1, keepdim=True).clamp_min(mod.eps)
            kn = kg / kg.norm(dim=1, keepdim=True).clamp_min(mod.eps)
            rel = (kn @ qn.t()) if transpose_direction else (qn @ kn.t())
            a = torch.softmax(mod.temperature[g, 0, 0] * rel, dim=-1)
            mixed[sl] = a @ vg
        out[i] = (wo @ mixed + bo[:, None]).reshape(c, h, w)
    return out


def test_b_attention_math():
    torch.manual_seed(0)
    lp = FixedBinomialLowPass()
    for heads in (1, 4):
        c = 16
        mod = LowFreqRelationChannelAttention(c, heads=heads).double()
        assert mod.temperature.shape == (heads, 1, 1)
        assert mod.temperature.requires_grad
        assert torch.equal(mod.temperature.detach(),
                           torch.ones(heads, 1, 1, dtype=torch.float64))
        with torch.no_grad():   # non-trivial per-head temperature
            mod.temperature.copy_(
                torch.linspace(0.5, 3.0, heads, dtype=torch.float64).view(heads, 1, 1))
        feat = torch.randn(2, c, 5, 7, dtype=torch.float64)
        low = lp(feat)

        attn = mod.attention_matrix(low)
        assert attn.shape == (2, heads, c // heads, c // heads), attn.shape
        assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)))

        out = mod(feat, low)
        ref = _reference_attention(mod, feat, low)
        assert torch.allclose(out, ref, atol=1e-10), float((out - ref).abs().max())
        # the reference discriminates the matrix direction (Qn@Kn^T vs Kn@Qn^T)
        wrong = _reference_attention(mod, feat, low, transpose_direction=True)
        assert float((out - wrong).abs().max()) > 1e-6

        # Q/K read F_low, V reads full-band F (forward hooks, nothing stored
        # on the production module)
        seen = {}
        hooks = [getattr(mod, n).register_forward_hook(
            lambda m, inp, o, n=n: seen.__setitem__(n, inp[0].detach().clone()))
            for n in ('to_q', 'to_k', 'to_v')]
        mod(feat, low)
        for hk in hooks:
            hk.remove()
        assert torch.equal(seen['to_q'], _tokens(low))
        assert torch.equal(seen['to_k'], _tokens(low))
        assert torch.equal(seen['to_v'], _tokens(feat))
        assert not torch.allclose(seen['to_v'], _tokens(low))
        assert [k for k, v in mod.__dict__.items() if torch.is_tensor(v)] == []

    # channels not divisible by heads
    _raises(ValueError, lambda: LowFreqRelationChannelAttention(10, heads=4))
    _raises(ValueError, lambda: LFCA(10, num_blocks=1, heads=4))
    from model.CSFIUNet import CSFIUNet
    _raises(ValueError, lambda: CSFIUNet(
        num_res_blocks='1+1+1+1', n_feats=16, restoration_heads=5))

    # zero and very low amplitude inputs: finite forward / backward
    for heads in (1, 2):
        block = LFCABlock(8, heads=heads)
        for x in (torch.zeros(2, 8, 6, 6), 1e-12 * torch.randn(2, 8, 6, 6),
                  torch.zeros(1, 8, 1, 1)):
            block.zero_grad()
            x = x.clone().requires_grad_(True)
            y = block(x)
            assert torch.isfinite(y).all()
            y.pow(2).sum().backward()
            assert torch.isfinite(x.grad).all()
            for name, p in block.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
        a0 = block.attn.attention_matrix(torch.zeros(1, 8, 4, 4))
        assert torch.allclose(a0, torch.full_like(a0, 1.0 / (8 // heads)))
    print('B attention_math ok')


# =====================================================================
# C. high-frequency branch + block formula
# =====================================================================
def test_c_highfreq_branch_and_block():
    torch.manual_seed(1)
    c = 12
    br = HighFreqLocalBranch(c)
    convs = [m for m in br.modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 2
    for cv in convs:
        assert cv.bias is None and cv.groups == c and cv.kernel_size == (3, 3)
    z = br(torch.zeros(2, c, 7, 5))
    assert float(z.abs().max()) == 0.0          # 0 in -> exactly 0 out

    block = LFCABlock(c, heads=3)
    from model.igab import IGAB, IG_MSA
    assert not any(isinstance(m, (IGAB, IG_MSA)) for m in block.modules())
    assert not hasattr(block.attn, 'pos_emb')    # old pos_emb(V) not kept
    dw_c = [m for m in block.modules()
            if isinstance(m, nn.Conv2d) and m.groups == c]
    assert len(dw_c) == 2                        # only the HF branch
    assert not any('alpha' in n or 'beta' in n or 'gate' in n
                   for n, _ in block.named_parameters())

    feat = torch.randn(2, c, 9, 10)
    seen = {}
    hooks = [
        block.lowpass.register_forward_hook(
            lambda m, i, o: seen.__setitem__('low', o.detach().clone())),
        block.local.register_forward_hook(
            lambda m, i, o: seen.__setitem__('local_in', i[0].detach().clone())),
        block.attn.register_forward_hook(
            lambda m, i, o: seen.__setitem__(
                'attn_in', (i[0].detach().clone(), i[1].detach().clone()))),
    ]
    y = block(feat)
    for hk in hooks:
        hk.remove()
    low = block.lowpass(feat)
    assert torch.equal(seen['low'], low)
    assert torch.equal(seen['local_in'], feat - low)          # F_high
    assert torch.equal(seen['attn_in'][0], feat)              # full band
    assert torch.equal(seen['attn_in'][1], low)               # F_low
    # X = F + O_attn + O_local ; Y = X + FFN(LN(X)), coefficients 1
    x = feat + block.attn(feat, low) + block.local(feat - low)
    y_ref = x + block.ffn(block.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
    assert torch.allclose(y, y_ref, atol=1e-6)

    # gradient reaches the HF branch
    block.zero_grad()
    block(feat).pow(2).sum().backward()
    for cv in dw_c:
        assert cv.weight.grad is not None and float(cv.weight.grad.abs().sum()) > 0

    # stacked blocks: block i recomputes P(F) from its own input
    stack = LFCA(c, num_blocks=2, heads=1)
    ins, outs = [], []
    hooks = [b.lowpass.register_forward_hook(
        lambda m, i, o: ins.append(i[0].detach().clone())) for b in stack.blocks]
    hooks += [b.register_forward_hook(
        lambda m, i, o: outs.append(o.detach().clone())) for b in stack.blocks]
    stack(feat)
    for hk in hooks:
        hk.remove()
    assert torch.equal(ins[0], feat) and torch.equal(ins[1], outs[0])
    print('C highfreq_branch_and_block ok')


# =====================================================================
# D. capacity / optimizer
# =====================================================================
def _lfca_to_igab_key(k):
    """Map an LFCA state key to the historical IGAB key layout."""
    k = k.replace('coarse_freq_block.', 'coarse_igab.')
    k = k.replace('middle_freq_block.', 'middle_igab.')
    k = re.sub(r'(blocks\.\d+)\.attn\.temperature', r'\1.0.rescale', k)
    k = re.sub(r'(blocks\.\d+)\.attn\.', r'\1.0.', k)
    k = re.sub(r'(blocks\.\d+)\.local\.net\.', r'\1.0.pos_emb.', k)
    k = re.sub(r'(blocks\.\d+)\.norm\.', r'\1.1.', k)
    k = re.sub(r'(blocks\.\d+)\.ffn\.net\.', r'\1.2.net.', k)
    return k


def test_d_capacity_and_optimizer():
    from model.igab import IGAB, FeedForward
    print('D param counts (trainable), same C / heads / depth / FFN x4:')
    print('   C heads depth |     IGAB     LFCA | LFCA buffers(non-trainable)')
    for c, heads, depth in [(64, 1, 1), (64, 4, 2), (16, 1, 1)]:
        torch.manual_seed(7)
        old = IGAB(c, num_blocks=depth, heads=heads)
        torch.manual_seed(7)
        new = LFCA(c, num_blocks=depth, heads=heads)
        n_old, n_new = _n_trainable(old), _n_trainable(new)
        n_buf = sum(b.numel() for b in new.buffers())
        print('  %3d %5d %5d | %8d %8d | %d' % (c, heads, depth, n_old, n_new, n_buf))
        assert n_new == n_old
        assert n_buf == 25 * depth
        assert n_new == sum(p.numel() for p in new.parameters())   # buffers excluded
        assert all(not b.requires_grad for b in new.buffers())
        # tensor-for-tensor same layout as IGAB (names differ only)
        old_sd = old.state_dict()
        mapped = {_lfca_to_igab_key(k): v for k, v in new.state_dict().items()}
        assert set(mapped) == set(old_sd)
        for k in old_sd:
            assert mapped[k].shape == old_sd[k].shape, k
            # same seed -> same initial values (construction order kept)
            assert torch.equal(mapped[k], old_sd[k]), k
        # FFN capacity not reduced
        ffn = new.blocks[0].ffn
        assert ffn.net[0].out_channels == 4 * c and ffn.net[2].groups == 4 * c
        assert _n_trainable(ffn) == _n_trainable(FeedForward(c))

    g = dict(
        qkv=3 * 64 * 64, temperature=1, proj=64 * 64 + 64, local=2 * 64 * 9,
        norm=2 * 64, ffn=64 * 256 + 256 * 9 + 256 * 64)
    assert sum(g.values()) == _n_trainable(LFCA(64, 1, 1)) == 52801
    print('  per-block breakdown @C=64,heads=1: %s' % g)

    from model.CSFIUNet import CSFIUNet
    net = CSFIUNet()   # baseline config 8+8+4+2, 64 ch, depth 1,1, heads 1
    n_main = _n_trainable(net)
    n_blocks = _n_trainable(net.coarse_freq_block) + _n_trainable(net.middle_freq_block)
    n_igab = 2 * _n_trainable(IGAB(64, 1, 1))
    print('  CSFIUNet baseline: total=%d  two LFCA=%d  (two IGAB would be %d '
          '-> IGAB-variant total %d)' % (n_main, n_blocks, n_igab,
                                         n_main - n_blocks + n_igab))
    assert net.restoration_identity() == dict(
        model_variant='csfi_lfca_v1', restoration_block='lfca',
        coarse_depth=1, middle_depth=1, heads=1, ffn_expansion=4,
        lowpass_kernel='binomial', lowpass_kernel_size=5)

    # optimizer: every new parameter exactly once, nothing IGAB left
    from block_amend_main import build_optimizer
    _patch_lte()
    from model.TTSREnhance import TTSREnhance
    args = _tiny_args()
    model = TTSREnhance(args)
    opt, _ = build_optimizer(model, args)
    counts = {}
    for grp in opt.param_groups:
        for p in grp['params']:
            counts[id(p)] = counts.get(id(p), 0) + 1
    assert max(counts.values()) == 1
    expected_suffix = {
        'attn.to_q.weight', 'attn.to_k.weight', 'attn.to_v.weight',
        'attn.temperature', 'attn.proj.weight', 'attn.proj.bias',
        'local.net.0.weight', 'local.net.2.weight', 'norm.weight', 'norm.bias',
        'ffn.net.0.weight', 'ffn.net.2.weight', 'ffn.net.4.weight'}
    for pos in ('coarse_freq_block', 'middle_freq_block'):
        got = set()
        for n, p in getattr(model.MainNet, pos).named_parameters():
            assert p.requires_grad and counts.get(id(p)) == 1, (pos, n)
            got.add(n.split('.', 2)[2])
        assert got == expected_suffix, got ^ expected_suffix
    for n, p in model.named_parameters():
        assert 'igab' not in n.lower(), n
        if n.startswith('MainNet.') and p.requires_grad:
            assert id(p) in counts, n
    assert not any('igab' in k.lower() for k in model.state_dict())
    for b in model.MainNet.coarse_freq_block.buffers():
        assert id(b) not in counts
    print('D capacity_and_optimizer ok')


# =====================================================================
# E. integration (stub LTE)
# =====================================================================
def _stub_model(**kw):
    _patch_lte()
    from model.TTSREnhance import TTSREnhance
    return TTSREnhance(_tiny_args(**kw))


def test_e_integration_forward():
    from model.tiled_infer import tiled_forward
    import model.igab as igab_mod

    # backbone module must not even import the historical IGAB module
    code = ("import sys; sys.path.insert(0, %r); import model.CSFIUNet; "
            "assert 'model.igab' not in sys.modules" % ROOT)
    subprocess.run([sys.executable, '-c', code], check=True, cwd=ROOT)

    # any use of the old IGAB classes on the default path would blow up
    saved = (igab_mod.IGAB.__init__, igab_mod.IGAB.forward, igab_mod.IG_MSA.forward)

    def _boom(*a, **k):
        raise AssertionError('historical IGAB used on the default path')
    igab_mod.IGAB.__init__ = _boom
    igab_mod.IGAB.forward = _boom
    igab_mod.IG_MSA.forward = _boom
    try:
        torch.manual_seed(3)
        model = _stub_model().eval()
        main = model.MainNet
        calls = []
        hooks = [
            main.coarse_freq_block.register_forward_hook(
                lambda m, i, o: calls.append(('coarse', len(i), tuple(i[0].shape)))),
            main.middle_freq_block.register_forward_hook(
                lambda m, i, o: calls.append(('middle', len(i), tuple(i[0].shape)))),
        ]
        lr = torch.randn(1, 3, 48, 64)
        ref = torch.randn(1, 3, 48, 64)
        with torch.no_grad():
            sr, S, T3, T2, T1 = model(lr=lr, lrsr=lr, ref=ref, refsr=ref)
        assert sr.shape == lr.shape and torch.isfinite(sr).all()
        assert S is not None and T3 is not None
        # both new blocks ran once, with the feature as their only input
        assert calls == [('coarse', 1, (1, 16, 12, 16)),
                         ('middle', 1, (1, 16, 24, 32))], calls
        for hk in hooks:
            hk.remove()

        with torch.no_grad():
            # no-ref
            sr0, S0, *_ = model(lr=lr, lrsr=lr, ref=None, refsr=None)
            assert sr0.shape == lr.shape and S0 is None and torch.isfinite(sr0).all()
            # odd H/W, with and without ref
            lro = torch.randn(1, 3, 37, 41)
            refo = torch.randn(1, 3, 37, 41)
            for r in (refo, None):
                o = model(lr=lro, lrsr=lro, ref=r, refsr=r)[0]
                assert o.shape == lro.shape and torch.isfinite(o).all()
            # single tile == direct forward
            direct = model(lr=lr, lrsr=lr, ref=ref, refsr=ref,
                           apply_illum=False, apply_ref_illum=False)[0]
            tiled = tiled_forward(model, lr, lr, ref, ref, tile_size=256,
                                  overlap=0, apply_illum=False,
                                  apply_ref_illum=False)
            assert torch.allclose(direct, tiled, atol=1e-5, rtol=1e-5)
            # multi tile: shape / finite, odd canvas too
            for h, w in ((100, 120), (53, 91)):
                a = torch.randn(1, 3, h, w)
                b = torch.randn(1, 3, h, w)
                t = tiled_forward(model, a, a, b, b, tile_size=32, overlap=8)
                assert t.shape == a.shape and torch.isfinite(t).all()
    finally:
        igab_mod.IGAB.__init__, igab_mod.IGAB.forward, igab_mod.IG_MSA.forward = saved
    print('E integration_forward ok')


def test_e_lightup_unchanged():
    from model.CSFIUNet import CSFIUNet
    torch.manual_seed(4)
    net = CSFIUNet(num_res_blocks='1+1+1+1', n_feats=16).eval()
    est = net.illumination_estimator
    assert set(est.state_dict()) == {
        'conv1.weight', 'conv1.bias', 'depth_conv.weight', 'depth_conv.bias',
        'conv2.weight', 'conv2.bias'}
    assert float(est.conv2.weight.abs().max()) == 0.0
    assert float(est.conv2.bias.abs().max()) == 0.0

    x = torch.rand(1, 3, 37, 41) * 2 - 1
    with torch.no_grad():
        stem = net.encode_low_stem(x)
        assert not any(k.startswith('illu') for k in stem), sorted(stem)
        assert torch.equal(stem['base'], stem['low'])       # zero log_gain at init
        out_a = net(x, collect_residual_decomp=True)
        dec = net.last_residual_decomp
        assert torch.allclose(out_a, dec['base'] + dec['rgb_residual'], atol=1e-6)
        res_a = dec['rgb_residual'].clone()

        # Restoration path does not read the estimator feature: with the
        # log_gain readout still zero, changing the feature changes nothing.
        est.conv1.weight.add_(torch.randn_like(est.conv1.weight))
        est.depth_conv.weight.add_(torch.randn_like(est.depth_conv.weight))
        assert torch.equal(net(x), out_a)

        # Non-zero readout: Y_base = 2*(low01*exp(clamp(log_gain,-1.5,1.5)))-1,
        # and the estimator influences the output only through that base.
        est.conv2.weight.normal_(0, 0.5)
        est.conv2.bias.normal_(0, 0.5)
        stem = net.encode_low_stem(x)
        low01 = (stem['low'] + 1.) * 0.5
        mean_c = low01.mean(dim=1, keepdim=True)
        lg = est.conv2(est.depth_conv(est.conv1(torch.cat([low01, mean_c], 1))))
        assert float(lg.abs().max()) > 1.5                  # clamp is exercised
        base_ref = 2. * (low01 * torch.exp(lg.clamp(-1.5, 1.5))) - 1.
        assert torch.allclose(stem['base'], base_ref, atol=1e-6)
        out_b = net(x, collect_residual_decomp=True)
        dec = net.last_residual_decomp
        assert not torch.allclose(dec['base'], out_a, atol=1e-3)
        assert torch.allclose(dec['rgb_residual'], res_a, atol=1e-6)
        assert torch.allclose(out_b, dec['base'] + dec['rgb_residual'], atol=1e-6)
    print('E lightup_unchanged ok')


def test_e_train_steps_l1_msssim():
    from block_amend_ckpt import check_grad_finite
    from block_amend_main import build_optimizer
    from loss.loss_enhance import get_loss_dict_enhance
    torch.manual_seed(5)
    model = _stub_model(ms_ssim_w=1.0, rec_w=1.0, freeze_lte=True).train()
    args = model.args
    loss_all = get_loss_dict_enhance(args, None)
    assert set(loss_all) == {'rec_loss', 'ms_ssim_loss'}
    opt, _ = build_optimizer(model, args)
    rc_before = {k: v.detach().clone()
                 for k, v in model.RefCorrection.state_dict().items()}
    lr = torch.rand(2, 3, 32, 32) * 0.4 - 1.0
    hr = torch.rand(2, 3, 32, 32) * 2 - 1
    ref = torch.rand(2, 3, 32, 32) * 2 - 1
    nz_first = {}
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        sr = model(lr=lr, lrsr=lr, ref=ref, refsr=ref)[0]
        loss = args.rec_w * loss_all['rec_loss'](sr, hr) \
            + args.ms_ssim_w * loss_all['ms_ssim_loss'](sr, hr)
        assert torch.isfinite(loss) and torch.isfinite(sr).all()
        loss.backward()
        check_grad_finite(model, 1, step + 1, 'synthetic', loss, sr)
        for pos in ('coarse_freq_block', 'middle_freq_block'):
            for n, p in getattr(model.MainNet, pos).named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), (pos, n)
                if step == 0:
                    nz_first['%s.%s' % (pos, n)] = float(p.grad.abs().sum()) > 0
        for p in model.LTE.parameters():
            assert p.requires_grad is False and p.grad is None
        for n, p in model.RefCorrection.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
        opt.step()
    # key new projections receive non-zero gradient already at step 1
    for pos in ('coarse_freq_block', 'middle_freq_block'):
        for key in ('attn.to_q.weight', 'attn.to_k.weight', 'attn.to_v.weight',
                    'attn.proj.weight', 'attn.temperature',
                    'local.net.0.weight', 'local.net.2.weight',
                    'ffn.net.0.weight', 'ffn.net.4.weight'):
            assert nz_first['%s.blocks.0.%s' % (pos, key)], (pos, key)
    # RefCorrection gradient path is alive (its zero-init readouts moved)
    moved = [k for k, v in model.RefCorrection.state_dict().items()
             if not torch.equal(v, rc_before[k])]
    assert moved, 'RefCorrection received no update'
    print('E train_steps_l1_msssim ok (RefCorrection moved: %s)' % moved)


# =====================================================================
# config aliases
# =====================================================================
def test_config_aliases():
    from types import SimpleNamespace as NS
    assert restoration_config_from_args(NS()) == RESTORATION_DEFAULTS
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        cfg = restoration_config_from_args(NS(igab_blocks='2', igab_heads=2))
        assert any(issubclass(x.category, FutureWarning) for x in w)
    assert cfg['restoration_blocks'] == '2,2' and cfg['restoration_heads'] == 2
    assert not any(k.startswith('igab') for k in cfg)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # alias agreeing with the neutral name is fine; disagreeing is an error
        restoration_config_from_args(NS(igab_heads=2, restoration_heads=2))
        _raises(ValueError, lambda: restoration_config_from_args(
            NS(igab_heads=2, restoration_heads=1)), 'Conflicting')
        _raises(ValueError, lambda: restoration_config_from_args(
            NS(igab_blocks='2,1', restoration_blocks='1,1')), 'Conflicting')
        ns = NS(igab_blocks='1,2', igab_heads=None, restoration_blocks=None,
                restoration_heads=None, model_variant=None)
        explicit = {'igab_blocks', 'mode'}
        resolve_restoration_aliases(ns, explicit)
        assert ns.restoration_blocks == '1,2' and not hasattr(ns, 'igab_blocks')
        assert not hasattr(ns, 'igab_heads')
        assert explicit == {'restoration_blocks', 'mode'}
        assert ns.model_variant == MODEL_VARIANT
    _raises(ValueError, lambda: restoration_config_from_args(
        NS(model_variant='csfi_igab_v0')), 'not buildable')
    _raises(ValueError, lambda: restoration_config_from_args(
        NS(restoration_block='igab')))
    _raises(ValueError, lambda: restoration_config_from_args(
        NS(restoration_ffn_expansion=2)))
    _raises(ValueError, lambda: restoration_config_from_args(
        NS(lowpass_kernel_size=3)))

    # legacy option.py CLI: alias maps (with warning), conflict exits non-zero
    def run(argv):
        code = ("import sys; sys.path.insert(0, %r); sys.argv=['x']+%r; "
                "import option; a=option.args; "
                "print(a.model_variant, a.restoration_block, a.restoration_blocks,"
                " a.restoration_heads, hasattr(a,'igab_heads'))" % (ROOT, argv))
        return subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                              capture_output=True, text=True)
    r = run([])
    assert r.returncode == 0 and r.stdout.split() == [
        'csfi_lfca_v1', 'lfca', '1,1', '1', 'False'], (r.stdout, r.stderr)
    r = run(['--igab_heads', '2', '--igab_blocks', '2,1'])
    assert r.returncode == 0 and r.stdout.split() == [
        'csfi_lfca_v1', 'lfca', '2,1', '2', 'False'], (r.stdout, r.stderr)
    assert 'deprecated alias' in r.stderr
    r = run(['--igab_heads', '2', '--restoration_heads', '4'])
    assert r.returncode != 0 and 'Conflicting' in r.stderr
    print('config_aliases ok')


# =====================================================================
# F. save / restore / variant lock (through block_amend_main.main)
# =====================================================================
def _fake_igab_pack(pack):
    """Re-label an LFCA pack as a schema-2 IGAB-variant pack.

    Tensor shapes are identical to a real IGAB checkpoint (see test D), which
    is exactly the case that must not be accepted.
    """
    old = dict(pack)
    old.pop('model_variant')
    old['checkpoint_schema_version'] = 2
    old['model_state_dict'] = {
        _lfca_to_igab_key(k): v for k, v in pack['model_state_dict'].items()}
    cfg = dict(pack['config'])
    for k in RESTORATION_CONFIG_KEYS:
        cfg.pop(k)
    cfg['igab_blocks'], cfg['igab_heads'] = '1,1', 1
    old['config'] = cfg
    return old


def test_f_save_restore_and_variant_lock():
    import block_amend_main as bam
    from block_amend_ckpt import (
        CHECKPOINT_SCHEMA_VERSION, LOCKED_CONFIG_KEYS, require_resume_fields,
    )
    from model.TTSREnhance import load_pretrained_weights
    from model.weight_init import load_lte_only, load_strict_state

    assert CHECKPOINT_SCHEMA_VERSION == 3
    for k in RESTORATION_CONFIG_KEYS:
        assert k in LOCKED_CONFIG_KEYS and k in bam.BASELINE_FIXED
    assert not any(k.startswith('igab') for k in LOCKED_CONFIG_KEYS)
    assert not any(k.startswith('igab') for k in bam.BASELINE_FIXED)
    assert bam.BASELINE_FIXED['model_variant'] == 'csfi_lfca_v1'
    # training recipe untouched by this change
    assert (bam.BASELINE_FIXED['rec_w'], bam.BASELINE_FIXED['ms_ssim_w'],
            bam.BASELINE_FIXED['rec_loss_type']) == (1.0, 1.0, 'l1')

    _patch_lte()
    root = _synth_tree(n_train=8, h=48, w=48)
    save = os.path.join(tempfile.mkdtemp(), 'run')
    saved_baseline = dict(bam.BASELINE_FIXED)
    # Tiny backbone for CPU; loss (L1 + MS-SSIM, weights 1/1) stays baseline.
    bam.BASELINE_FIXED.update(
        num_res_blocks='1+1+1+1', n_feats=16, ref_correction_feats=4,
        tile_size=32, tile_overlap=8)
    common = ['--cpu', '--num_workers', '0']
    try:
        # ---- fresh train, 1 epoch, non-default batch/crop/seed ----
        bam.main(['--mode', 'train', '--model_variant', 'csfi_lfca_v1',
                  '--dataset_dir', root, '--save_dir', save,
                  '--batch_size', '2', '--train_crop_size', '32',
                  '--num_epochs', '1', '--val_ratio', '0.25', '--seed', '5',
                  '--print_every', '100'] + common)
        last_path = os.path.join(save, 'checkpoint_last.pt')
        pack = torch.load(last_path)
        require_resume_fields(pack, last_path)
        assert pack['checkpoint_schema_version'] == 3
        assert pack['model_variant'] == 'csfi_lfca_v1'
        cfg = pack['config']
        for k, v in RESTORATION_DEFAULTS.items():
            assert cfg[k] == v, (k, cfg[k])
        assert not any(k.startswith('igab') for k in cfg)
        assert cfg['batch_size'] == 2 and cfg['train_crop_size'] == 32
        assert any(k.startswith('MainNet.coarse_freq_block.')
                   for k in pack['model_state_dict'])
        assert not any('igab' in k for k in pack['model_state_dict'])
        with open(os.path.join(save, 'run_meta.json')) as f:
            meta = json.load(f)
        assert meta['model_variant'] == 'csfi_lfca_v1'
        assert meta['restoration'] == dict(
            model_variant='csfi_lfca_v1', restoration_block='lfca',
            coarse_depth=1, middle_depth=1, heads=1, ffn_expansion=4,
            lowpass_kernel='binomial', lowpass_kernel_size=5)
        assert meta['checkpoint_schema_version'] == 3

        # ---- strict save + load reproduces the model ----
        m_b = _stub_model()
        load_strict_state(m_b, last_path, mode='inference')
        sd_b = {k: v for k, v in m_b.state_dict().items() if '_copy' not in k}
        assert set(sd_b) == set(pack['model_state_dict'])
        for k, v in pack['model_state_dict'].items():
            assert torch.equal(sd_b[k], v), k

        # ---- eval rebuilds from checkpoint config (not from code defaults) ----
        bam.BASELINE_FIXED['n_feats'] = 32      # would break a strict load
        eval_dir = os.path.join(save, 'eval_val')
        bam.main(['--mode', 'eval', '--model_path', last_path,
                  '--save_dir', eval_dir, '--eval_split', 'val'] + common)
        with open(os.path.join(eval_dir, 'eval_val.json')) as f:
            ev = json.load(f)
        assert ev['n'] == 2 and ev['protocol'] == 'rgb_float_clipped_v1'
        bam.BASELINE_FIXED['n_feats'] = 16

        # ---- resume through main: epoch-boundary continue to epoch 2 ----
        bam.main(['--mode', 'resume', '--resume_checkpoint', last_path,
                  '--save_dir', save, '--num_epochs', '2'] + common)
        pack2 = torch.load(last_path)
        assert pack2['epoch'] == 2 and pack2['model_variant'] == 'csfi_lfca_v1'
        assert pack2['config']['batch_size'] == 2
        assert pack2['scheduler_state_dict']['last_epoch'] == 2

        resume = ['--mode', 'resume', '--resume_checkpoint', last_path,
                  '--save_dir', save, '--num_epochs', '3'] + common
        # locked restoration config cannot be overridden on resume
        _raises(RuntimeError, lambda: bam.main(resume + ['--restoration_heads', '2']),
                'restoration_heads')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # deprecated alias is held to the same lock
            _raises(RuntimeError, lambda: bam.main(resume + ['--igab_heads', '2']),
                    'restoration_heads')
            _raises(SystemExit, lambda: bam.main(
                resume + ['--igab_heads', '2', '--restoration_heads', '1']))
        _raises(SystemExit, lambda: bam.main(
            resume + ['--model_variant', 'csfi_igab_v0']))
        assert torch.load(last_path)['epoch'] == 2      # nothing was trained

        # ---- an IGAB-variant checkpoint is refused everywhere ----
        tmp = tempfile.mkdtemp()
        old_pack = _fake_igab_pack(pack2)
        old_path = os.path.join(tmp, 'igab_pack.pt')
        torch.save(old_pack, old_path)
        flat_path = os.path.join(tmp, 'igab_flat.pt')
        torch.save(old_pack['model_state_dict'], flat_path)
        posing = dict(old_pack)
        posing['model_variant'] = 'csfi_lfca_v1'
        posing['checkpoint_schema_version'] = 3
        posing['config'] = dict(pack2['config'])
        posing_path = os.path.join(tmp, 'igab_posing_as_lfca.pt')
        torch.save(posing, posing_path)

        for p in (old_path, posing_path):
            e = _raises(RuntimeError, lambda: bam.main(
                ['--mode', 'resume', '--resume_checkpoint', p,
                 '--save_dir', save, '--num_epochs', '3'] + common),
                'model_variant=csfi_igab_v0')
            assert 'csfi_lfca_v1' in str(e)
            _raises(RuntimeError, lambda: bam.main(
                ['--mode', 'eval', '--model_path', p,
                 '--save_dir', os.path.join(tmp, 'ev')] + common),
                'model_variant=csfi_igab_v0')
            _raises(RuntimeError, lambda: require_model_variant(torch.load(p), p))
        _raises(RuntimeError, lambda: bam.main(
            ['--mode', 'eval', '--model_path', flat_path,
             '--save_dir', os.path.join(tmp, 'ev2')] + common),
            'model_variant=csfi_igab_v0')
        assert torch.load(last_path)['epoch'] == 2

        # no strict / shape-filtered / legacy loader accepts it either
        model = _stub_model()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for p in (old_path, flat_path, posing_path):
            _raises(RuntimeError, lambda: load_strict_state(model, p, mode='resume'),
                    'csfi_igab_v0')
        _raises(RuntimeError, lambda: load_pretrained_weights(
            model, flat_path, device='cpu'), 'csfi_igab_v0')
        # legacy shape-filtered Trainer.load (called unbound on a minimal self)
        from trainer import Trainer
        from types import SimpleNamespace as NS
        fake_self = NS(model=model, device='cpu',
                       logger=NS(info=lambda *a, **k: None))
        _raises(RuntimeError, lambda: Trainer.load(fake_self, model_path=flat_path),
                'csfi_igab_v0')
        _raises(RuntimeError, lambda: reject_legacy_igab_state(
            old_pack['model_state_dict'].keys(), flat_path))
        _raises(RuntimeError, lambda: reject_legacy_igab_state(
            ['module.' + k for k in old_pack['model_state_dict']], flat_path))
        for k, v in model.state_dict().items():
            assert torch.equal(before[k], v), k            # nothing leaked in

        # ---- LTE-only stays LTE-only (even from an IGAB-variant file) ----
        main_before = {k: v.detach().clone()
                       for k, v in model.MainNet.state_dict().items()}
        n = load_lte_only(model, flat_path, device='cpu')
        assert n == len(model.LTE.state_dict())
        for k, v in model.MainNet.state_dict().items():
            assert torch.equal(main_before[k], v), k
        for k, v in model.LTE.state_dict().items():
            assert torch.equal(v, old_pack['model_state_dict']['LTE.' + k]), k
    finally:
        bam.BASELINE_FIXED.clear()
        bam.BASELINE_FIXED.update(saved_baseline)
    print('F save_restore_and_variant_lock ok')


if __name__ == '__main__':
    print('Python', sys.version.split()[0], 'PyTorch', torch.__version__,
          '| LTE: conv stub (no VGG)')
    test_a_fixed_lowpass()
    test_b_attention_math()
    test_c_highfreq_branch_and_block()
    test_d_capacity_and_optimizer()
    test_e_integration_forward()
    test_e_lightup_unchanged()
    test_e_train_steps_l1_msssim()
    test_config_aliases()
    test_f_save_restore_and_variant_lock()
    print('all block-amend LFCA tests passed')
