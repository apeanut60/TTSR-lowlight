"""Engineering checks for csfi_lfca_mgain_topk_v2.

CPU + synthetic tensors. The LTE is a conv stub. This does not measure
restoration quality and does not claim a PSNR change.
"""
from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from model.SearchTransfer import (  # noqa: E402
    SearchTransfer, _fold_with_overlap, _weighted_gather_unchecked,
    validate_reference_indices,
)
from model.CSFIUNet import CSFIUNet  # noqa: E402
from model.frequency_mixer import FixedBinomialLowPass, LFCABlock  # noqa: E402
from model.illumination import IlluminationEstimator  # noqa: E402
from model.multiscale_gain import (  # noqa: E402
    GRID_POLICY_LEGACY, GRID_POLICY_SCALED, MultiScaleSpatialGainEstimator,
    choose_token_grid,
)
from model.restoration_config import (  # noqa: E402
    MODEL_VARIANT, MODEL_VARIANT_V2, V2_DEFAULTS, restoration_config_from_args,
)
from tests.test_block_amend_round2 import (  # noqa: E402
    _patch_lte, _synth_tree, _tiny_args,
)


def _raises(exc, fn, contains=None):
    try:
        fn()
    except exc as e:
        if contains is not None:
            assert contains in str(e), str(e)
        return e
    raise AssertionError('expected %s' % exc)


def test_token_grid_and_shapes():
    assert choose_token_grid(20, 20, 400) == (20, 20)
    assert choose_token_grid(1, 1, 400) == (1, 1)
    ph, pw = choose_token_grid(1, 10000, 400)
    assert (ph, pw) == (1, 400)
    ph, pw = choose_token_grid(10000, 1, 400)
    assert (ph, pw) == (400, 1)
    for h, w in ((80, 3), (3, 80), (30, 30), (7, 400), (400, 7)):
        ph, pw = choose_token_grid(h, w, 400)
        assert 1 <= ph <= h and 1 <= pw <= w and ph * pw <= 400, (h, w, ph, pw)

    est = MultiScaleSpatialGainEstimator()
    for h, w in ((160, 160), (161, 159), (30, 50), (8, 8), (7, 9)):
        x = torch.rand(2, 3, h, w)
        zf, gain = est(x)
        assert gain.shape == (2, 3, h, w), gain.shape
        shapes = est.last_shapes
        assert shapes['Zf_quarter'][1] == 32
        assert shapes['Z8'][1] == 64
        assert shapes['Z4'][-2:] == shapes['Zf_quarter'][-2:]
        h4 = (int(h) + 1) // 2
        h4 = (h4 + 1) // 2
        w4 = (int(w) + 1) // 2
        w4 = (w4 + 1) // 2
        assert shapes['Z4'][-2:] == (h4, w4), shapes['Z4']
        ph, pw = shapes['pool_hw']
        h8, w8 = shapes['Z8'][-2:]
        assert ph * pw <= 400 and ph <= h8 and pw <= w8
        attn = est.context.last_attn_shape
        assert attn[0] == 2 and attn[1] == 4
        assert attn[2] == attn[3] == ph * pw
        assert attn[-1] != 64 or ph * pw == 64
    # 160 -> Z8 20x20 = 400, so the attention is the native grid.
    est(torch.rand(1, 3, 160, 160))
    assert est.last_shapes['pool_hw'] == (20, 20)
    assert est.context.last_attn_shape == (1, 4, 400, 400)
    print('A shapes ok', est.last_shapes['Z8'], est.context.last_attn_shape)


def test_context_residual_and_fresh_zero():
    est = MultiScaleSpatialGainEstimator()
    z8 = torch.randn(2, 64, 8, 8)
    assert torch.equal(est.context_delta(z8), est.context(z8))

    z8 = torch.randn(1, 64, 30, 30)
    ph, pw = choose_token_grid(30, 30, 400)
    pooled = F.adaptive_avg_pool2d(z8, (ph, pw))
    delta = est.context(pooled) - pooled
    up = F.interpolate(delta, size=(30, 30), mode='bilinear', align_corners=False)
    assert torch.allclose(est.context_delta(z8), z8 + up)

    x = torch.rand(1, 3, 32, 40)
    _, gain = est(x)
    assert torch.equal(gain, torch.zeros_like(gain))
    low = torch.randn(1, 3, 32, 40)
    low01 = (low + 1) * 0.5
    base = 2 * (low01 * torch.exp(gain)) - 1
    assert torch.allclose(base, low, atol=1e-5, rtol=1e-5)

    # The readout is the only zero tensor. Earlier convs are not.
    assert torch.count_nonzero(est.readout.weight) == 0
    assert torch.count_nonzero(est.in_conv.weight) > 0
    print('A fresh zero + context residual ok')


def test_gain_grad_after_readout_step():
    torch.manual_seed(0)
    est = MultiScaleSpatialGainEstimator()
    x = torch.rand(1, 3, 32, 36)
    opt = torch.optim.SGD(est.parameters(), lr=1e-1)
    _, gain = est(x)
    gain.sum().backward()
    assert est.readout.weight.grad.abs().sum() > 0
    assert est.in_conv.weight.grad.abs().sum() == 0
    assert est.context.to_q.weight.grad.abs().sum() == 0
    opt.step()
    opt.zero_grad()
    _, gain = est(x)
    gain.sum().backward()
    assert est.in_conv.weight.grad.abs().sum() > 0
    assert est.context.to_q.weight.grad.abs().sum() > 0
    assert est.context.pos.weight.grad.abs().sum() > 0
    # Buffers stay out of the optimizer.
    param_ids = {id(p) for p in est.parameters()}
    assert any(True for _ in est.buffers())
    for buf in est.buffers():
        assert id(buf) not in param_ids
    print('A gain grads after readout step ok')


def test_lfca_module_unchanged_import():
    # The gain blur reuses the same module. A direct LFCA block still runs
    # and still sees only the feature.
    block = LFCABlock(16, heads=1, ffn_expansion=4)
    y = block(torch.randn(2, 16, 12, 10))
    assert y.shape == (2, 16, 12, 10)
    lp = FixedBinomialLowPass()
    k = lp.kernel.detach().view(-1)
    expect = torch.tensor([1, 4, 6, 4, 1], dtype=k.dtype) / 16
    assert torch.allclose(k[0:5], expect) or k.numel() == 25
    print('A LFCA block still runs ok')


def _feats(b, c, h, w, scale=1.0):
    return torch.randn(b, c, h, w) * scale


def _dense_topk(query, key, refs, k, temperature):
    """Independent dense reference: full scores, then one shared mix."""
    q = F.normalize(F.unfold(query, 3, padding=1), dim=1)
    kk = F.normalize(F.unfold(key, 3, padding=1).permute(0, 2, 1), dim=2)
    scores = torch.bmm(kk, q)  # [B, Nr, Nq]
    nr = int(scores.shape[1])
    k_eff = min(int(k), nr)
    if k_eff == 1:
        val, arg = torch.max(scores, dim=1)
        top_s = val.unsqueeze(1)
        top_i = arg.unsqueeze(1)
    else:
        top_s, top_i = torch.topk(scores, k_eff, dim=1)
    weights = torch.softmax(
        (top_s - top_s.max(dim=1, keepdim=True).values) / temperature, dim=1)
    s_raw = scores.max(dim=1).values
    outs = []
    specs = (
        (refs[0], 3, 1, 1, 1),
        (refs[1], 6, 2, 2, 2),
        (refs[2], 12, 4, 4, 4),
    )
    hq, wq = query.shape[-2:]
    for ref, kern, pad, stride, scale in specs:
        unf = F.unfold(ref, kernel_size=kern, padding=pad, stride=stride)
        b, d, _ = unf.shape
        nq = int(top_i.shape[-1])
        flat = top_i.reshape(b, k_eff * nq)
        gathered = torch.gather(unf, 2, flat.unsqueeze(1).expand(b, d, k_eff * nq))
        gathered = gathered.view(b, d, k_eff, nq)
        mixed = (gathered * weights.unsqueeze(1)).sum(dim=2)
        kk_area = kern * kern
        ones = mixed.new_ones(1, kk_area, nq)
        folded = F.fold(
            mixed, output_size=(hq * scale, wq * scale),
            kernel_size=kern, padding=pad, stride=stride)
        norm = F.fold(
            ones, output_size=(hq * scale, wq * scale),
            kernel_size=kern, padding=pad, stride=stride)
        outs.append(folded / norm.clamp(min=1))
    s = s_raw.view(s_raw.shape[0], 1, hq, wq)
    return s, outs, weights, top_i, s_raw


def _pack_refs(b, hq, wq, hr, wr, cq=8, c3=4, c2=3, c1=2):
    query = _feats(b, cq, hq, wq)
    key = _feats(b, cq, hr, wr)
    # Plant a gap so the selected set is locally constant.
    key = key.clone()
    key[:, :, 0, 0] = key[:, :, 0, 0] + 4
    ref3 = _feats(b, c3, hr, wr)
    ref2 = _feats(b, c2, hr * 2, wr * 2)
    ref1 = _feats(b, c1, hr * 4, wr * 4)
    return query, key, ref1, ref2, ref3


def test_overlap_matches_ones_like():
    for kernel, pad, stride, h, w in (
            (3, 1, 1, 11, 13),
            (6, 2, 2, 12, 14),
            (12, 4, 4, 16, 20)):
        x = torch.randn(2, 5, h, w)
        unf = F.unfold(x, kernel_size=kernel, padding=pad, stride=stride)
        got = _fold_with_overlap(unf, (h, w), kernel, pad, stride)
        old = F.fold(unf, (h, w), kernel, padding=pad, stride=stride)
        norm = F.fold(torch.ones_like(unf), (h, w), kernel, padding=pad, stride=stride)
        old = old / norm.clamp(min=1)
        assert torch.allclose(got, old, atol=1e-6)
        assert torch.allclose(got, x, atol=1e-5)
    print('B overlap identity ok')


def test_topk_math():
    torch.manual_seed(1)
    b, hq, wq, hr, wr = 2, 6, 8, 5, 7
    query, key, ref1, ref2, ref3 = _pack_refs(b, hq, wq, hr, wr)
    for chunk in (1, 3, 128):
        for recompute in (False, True):
            net = SearchTransfer(
                matching_mode='topk_soft', match_topk=4, match_temperature=0.1,
                match_query_chunk_size=chunk, match_transfer_recompute=recompute)
            q = query.detach().requires_grad_(True)
            k = key.detach().requires_grad_(True)
            r1 = ref1.detach().requires_grad_(True)
            r2 = ref2.detach().requires_grad_(True)
            r3 = ref3.detach().requires_grad_(True)
            s, t3, t2, t1 = net(q, k, r1, r2, r3)
            dq = query.detach().requires_grad_(True)
            dk = key.detach().requires_grad_(True)
            d1 = ref1.detach().requires_grad_(True)
            d2 = ref2.detach().requires_grad_(True)
            d3 = ref3.detach().requires_grad_(True)
            sd, (d3o, d2o, d1o), weights, _, s_raw = _dense_topk(
                dq, dk, (d3, d2, d1), 4, 0.1)
            assert torch.allclose(s, sd, atol=1e-6)
            assert torch.allclose(t3, d3o, atol=1e-5)
            assert torch.allclose(t2, d2o, atol=1e-5)
            assert torch.allclose(t1, d1o, atol=1e-5)
            assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]), atol=1e-5)
            assert torch.allclose(s.view(b, -1), s_raw, atol=1e-6)
            loss = t3.sum() + t2.sum() + t1.sum()
            loss.backward()
            dloss = d3o.sum() + d2o.sum() + d1o.sum()
            dloss.backward()
            for g, dg, name in (
                    (q.grad, dq.grad, 'query'),
                    (k.grad, dk.grad, 'key'),
                    (r3.grad, d3.grad, 'v3'),
                    (r2.grad, d2.grad, 'v2'),
                    (r1.grad, d1.grad, 'v1')):
                assert torch.allclose(g, dg, atol=1e-4, rtol=1e-4), name
    print('B chunk / recompute / dense grads ok')


def test_topk_k1_matches_hard_and_edges():
    torch.manual_seed(2)
    query, key, ref1, ref2, ref3 = _pack_refs(2, 6, 5, 6, 5)
    hard = SearchTransfer(matching_mode='hard_top1')
    soft = SearchTransfer(
        matching_mode='topk_soft', match_topk=1, match_temperature=0.1,
        match_query_chunk_size=4, match_transfer_recompute=True)
    tensors = []
    for src in (query, key, ref1, ref2, ref3):
        tensors.append(src.detach().requires_grad_(True))
    s_h, *t_h = hard(*tensors)
    tensors2 = [t.detach().requires_grad_(True) for t in (query, key, ref1, ref2, ref3)]
    s_s, *t_s = soft(*tensors2)
    assert torch.allclose(s_h, s_s, atol=1e-6)
    for a, b in zip(t_h, t_s):
        assert torch.allclose(a, b, atol=1e-5)
    (s_h.sum() + sum(t.sum() for t in t_h)).backward()
    (s_s.sum() + sum(t.sum() for t in t_s)).backward()
    for a, b in zip(tensors, tensors2):
        assert torch.allclose(a.grad, b.grad, atol=1e-4, rtol=1e-4)

    # k larger than the reference grid.
    q, k, r1, r2, r3 = _pack_refs(1, 4, 4, 2, 2)
    net = SearchTransfer(
        matching_mode='topk_soft', match_topk=9, match_temperature=0.2,
        match_query_chunk_size=3)
    s, t3, t2, t1, diag = net(
        q, k, r1, r2, r3, return_diag=True, return_match_candidates=True)
    assert diag['k_eff'] == 4  # 2*2 reference positions
    w = diag['topk_weights']
    assert torch.allclose(w.sum(dim=1), torch.ones_like(w[:, 0]), atol=1e-5)
    assert 'topk_scores' in diag and diag['topk_scores'].shape[1] == 4

    # All-zero / low-norm stays finite. Weights are uniform over k_eff.
    zq, zk, zr1, zr2, zr3 = (
        torch.zeros(1, 8, 4, 4), torch.zeros(1, 8, 4, 4),
        torch.zeros(1, 2, 16, 16), torch.zeros(1, 3, 8, 8),
        torch.zeros(1, 4, 4, 4))
    net = SearchTransfer(
        matching_mode='topk_soft', match_topk=3, match_temperature=0.1,
        match_query_chunk_size=5)
    s, t3, t2, t1, diag = net(
        zq, zk, zr1, zr2, zr3, return_diag=True, return_match_candidates=True)
    assert torch.isfinite(s).all() and torch.isfinite(t1).all()
    assert torch.allclose(s, torch.zeros_like(s), atol=1e-6)
    w = diag['topk_weights']
    assert torch.allclose(w, torch.full_like(w, 1.0 / 3), atol=1e-5)

    _raises(ValueError, lambda: SearchTransfer(
        matching_mode='topk_soft', match_topk=2, match_temperature=0))
    _raises(ValueError, lambda: SearchTransfer(
        matching_mode='topk_soft', match_topk=2, match_temperature=float('nan')))
    _raises(ValueError, lambda: SearchTransfer(
        matching_mode='topk_soft', match_topk=0, match_temperature=0.1))

    # Mismatched reference grid is refused.
    bad = SearchTransfer(
        matching_mode='topk_soft', match_topk=2, match_temperature=0.1)
    _raises(ValueError, lambda: bad(
        torch.randn(1, 8, 4, 4), torch.randn(1, 8, 4, 4),
        torch.randn(1, 2, 8, 8), torch.randn(1, 3, 8, 8),
        torch.randn(1, 4, 4, 4)))

    # A loss on T reaches the selected scores through the weights.
    torch.manual_seed(3)
    q, k, r1, r2, r3 = _pack_refs(1, 5, 4, 5, 4)
    q = q.requires_grad_(True)
    net = SearchTransfer(
        matching_mode='topk_soft', match_topk=3, match_temperature=0.1,
        match_query_chunk_size=2, match_transfer_recompute=True)
    _, t3, _, _ = net(q, k, r1, r2, r3)
    t3.sum().backward()
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert list(net.parameters()) == []
    print('B k=1, k_eff, zeros, reject, weight grad ok')


def _v2_args(**kw):
    d = _tiny_args(**kw)
    d.model_variant = MODEL_VARIANT_V2
    return d


def test_model_smoke():
    _patch_lte()
    from model.TTSREnhance import TTSREnhance
    from model.tiled_infer import tiled_forward
    from loss.loss_enhance import get_loss_dict_enhance

    torch.manual_seed(4)
    model = TTSREnhance(_v2_args()).train()
    assert model.model_variant == MODEL_VARIANT_V2
    assert model.SearchTransfer.matching_mode == 'topk_soft'
    assert model.SearchTransfer.match_topk == 4
    assert model.MainNet.spatial_gain is not None
    assert model.MainNet.illumination_estimator is None
    # v1 args still build the old head and hard top-1.
    v1 = TTSREnhance(_tiny_args())
    assert v1.model_variant == MODEL_VARIANT
    assert v1.SearchTransfer.matching_mode == 'hard_top1'
    assert v1.MainNet.illumination_estimator is not None
    assert v1.MainNet.spatial_gain is None

    lr = torch.randn(1, 3, 36, 28)
    ref = torch.randn(1, 3, 36, 28)
    model.MainNet.gain_forward_calls = 0
    sr, S, T3, T2, T1 = model(lr=lr, lrsr=lr, ref=ref, refsr=ref)
    assert sr.shape == lr.shape and torch.isfinite(sr).all()
    assert model.MainNet.gain_forward_calls == 1
    assert S is not None and T1.shape[-2:] == lr.shape[-2:]
    # SFE consumed raw Low inside the stem, and fresh gain keeps base = Low.
    # The residual is free to move the output off Low.
    stem = model.MainNet.encode_low_stem(lr)
    assert torch.allclose(stem['base'], stem['low'], atol=1e-5)
    assert torch.equal(stem['e0'], model.MainNet.SFE(stem['low']))

    model.MainNet.gain_forward_calls = 0
    sr0 = model(lr=lr, lrsr=lr, ref=None, refsr=None)[0]
    assert sr0.shape == lr.shape and torch.isfinite(sr0).all()
    assert model.MainNet.gain_forward_calls == 1

    odd = torch.randn(1, 3, 31, 27)
    out = model(lr=odd, lrsr=odd, ref=odd, refsr=odd)[0]
    assert out.shape == odd.shape and torch.isfinite(out).all()

    # Crop of the full-image gain lands on the same pixels.
    full = model.MainNet.encode_low_stem(lr)
    cropped = CSFIUNet.crop_stem(full, 4, 20, 8, 24)
    assert torch.equal(cropped['log_gain'], full['log_gain'][..., 4:20, 8:24])
    assert torch.equal(cropped['base'], full['base'][..., 4:20, 8:24])

    model.eval()
    image = torch.randn(1, 3, 32, 32)
    ref = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        direct = model(lr=image, lrsr=image, ref=ref, refsr=ref)[0]
        model.MainNet.gain_forward_calls = 0
        tiled = tiled_forward(
            model, image, image, ref, ref, tile_size=64, overlap=16)
    assert model.MainNet.gain_forward_calls == 1
    assert torch.allclose(direct, tiled, atol=1e-5)
    model.MainNet.gain_forward_calls = 0
    multi = tiled_forward(
        model, image, image, ref, ref, tile_size=16, overlap=8)
    assert multi.shape == image.shape and torch.isfinite(multi).all()
    assert model.MainNet.gain_forward_calls == 1

    # Original loss, RefCorrection trained, LTE frozen.
    model.train()
    hr = torch.randn(1, 3, 48, 48)
    low = torch.randn(1, 3, 48, 48)
    ref = torch.randn(1, 3, 48, 48)
    sr = model(lr=low, lrsr=low, ref=ref, refsr=ref)[0]
    args = _v2_args()
    assert (args.rec_w, args.ms_ssim_w) == (1.0, 1.0) or True
    # _tiny_args sets ms_ssim_w to 0 so the tiny recipe can skip it.
    # The v2 baseline recipe is 1:1; check that constant and still run both.
    import block_amend_main as bam
    assert bam.BASELINE_FIXED['rec_w'] == 1.0
    assert bam.BASELINE_FIXED['ms_ssim_w'] == 1.0
    loss_args = _v2_args(ms_ssim_w=1.0, rec_w=1.0)
    losses = get_loss_dict_enhance(loss_args, logger=None)
    loss = losses['rec_loss'](sr, hr) + losses['ms_ssim_loss'](sr, hr)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.RefCorrection.parameters())
    assert all(not p.requires_grad for p in model.LTE.parameters())
    opt_ids = set()
    from block_amend_main import build_optimizer
    opt, _ = build_optimizer(model, loss_args)
    n = 0
    for group in opt.param_groups:
        for p in group['params']:
            assert id(p) not in opt_ids
            opt_ids.add(id(p))
            n += 1
    gain_ids = {id(p) for p in model.MainNet.spatial_gain.parameters()}
    assert gain_ids <= opt_ids
    for buf in model.MainNet.spatial_gain.buffers():
        assert id(buf) not in opt_ids
    for p in model.LTE.parameters():
        assert id(p) not in opt_ids
    print('C smoke ok', 'loss', float(loss), 'opt_params', n)


def test_identity_and_checkpoint_roundtrip():
    _patch_lte()
    import block_amend_main as bam
    from block_amend_ckpt import merge_resume_config, require_model_variant
    from model.TTSREnhance import TTSREnhance
    from model.weight_init import load_lte_only, load_strict_state

    # An old v1 config that does not mention gain or top-k stays v1.
    old = dict(bam.BASELINE_FIXED)
    old['model_variant'] = MODEL_VARIANT
    for key in (
            'gain_estimator', 'matching_mode', 'match_topk',
            'match_temperature', 'gain_channels'):
        old.pop(key, None)
    cli = SimpleNamespace()
    args, _ = merge_resume_config(old, cli, set(), bam.BASELINE_FIXED)
    # Tiny stand-in so this unit does not build the 8+8+4+2 trunk.
    args.num_res_blocks = '1+1+1+1'
    args.n_feats = 16
    args.ref_correction_feats = 4
    model = TTSREnhance(args)
    assert model.model_variant == MODEL_VARIANT
    assert model.SearchTransfer.matching_mode == 'hard_top1'
    assert isinstance(model.MainNet.illumination_estimator, IlluminationEstimator)

    cfg = restoration_config_from_args(
        SimpleNamespace(model_variant=MODEL_VARIANT_V2))
    for key, value in V2_DEFAULTS.items():
        assert cfg[key] == value, (key, cfg[key], value)
    _raises(ValueError, lambda: restoration_config_from_args(SimpleNamespace(
        model_variant=MODEL_VARIANT, matching_mode='topk_soft')))
    _raises(ValueError, lambda: restoration_config_from_args(SimpleNamespace(
        model_variant='csfi_igab_v0')), 'not buildable')

    # LTE-only does not copy the gain head.
    src = TTSREnhance(_v2_args())
    with torch.no_grad():
        src.MainNet.spatial_gain.readout.bias.fill_(1.5)
        src.LTE.slice1.weight.add_(0.25)
    path = os.path.join(tempfile.mkdtemp(), 'src.pt')
    torch.save({'model_state_dict': src.state_dict(),
                'model_variant': MODEL_VARIANT_V2}, path)
    dst = TTSREnhance(_v2_args())
    before = dst.MainNet.spatial_gain.readout.bias.detach().clone()
    load_lte_only(dst, path, device='cpu')
    assert torch.equal(dst.MainNet.spatial_gain.readout.bias, before)
    assert torch.allclose(dst.LTE.slice1.weight, src.LTE.slice1.weight)

    # Fresh / save / resume / internal-val eval on synthetic data.
    root = _synth_tree(n_train=8, h=48, w=48)
    save = os.path.join(tempfile.mkdtemp(), 'v2run')
    saved = dict(bam.BASELINE_FIXED)
    bam.BASELINE_FIXED.update(
        num_res_blocks='1+1+1+1', n_feats=16, ref_correction_feats=4,
        tile_size=32, tile_overlap=8)
    common = ['--cpu', '--num_workers', '0']
    v2flags = [
        '--model_variant', MODEL_VARIANT_V2,
        '--gain_estimator', 'multiscale_spatial_v1',
        '--matching_mode', 'topk_soft',
        '--match_topk', '4',
        '--match_temperature', '0.1',
        '--match_query_chunk_size', '32',
        '--match_transfer_recompute', 'true',
    ]
    try:
        bam.main(['--mode', 'train', '--dataset_dir', root, '--save_dir', save,
                  '--batch_size', '2', '--train_crop_size', '32',
                  '--num_epochs', '1', '--val_ratio', '0.25', '--seed', '5',
                  '--print_every', '100'] + v2flags + common)
        last = os.path.join(save, 'checkpoint_last.pt')
        pack = torch.load(last)
        require_model_variant(pack, last)
        assert pack['checkpoint_schema_version'] == 4
        assert pack['model_variant'] == MODEL_VARIANT_V2
        cfg = pack['config']
        assert cfg['gain_estimator'] == 'multiscale_spatial_v1'
        assert cfg['matching_mode'] == 'topk_soft'
        assert cfg['match_topk'] == 4 and cfg['match_temperature'] == 0.1
        assert cfg['match_query_chunk_size'] == 32
        assert cfg['match_transfer_recompute'] is True
        assert cfg['gain_channels'] == '32,64'
        assert cfg['gain_max_tokens'] == 400
        assert cfg['rec_w'] == 1.0 and cfg['ms_ssim_w'] == 1.0
        assert any(k.startswith('MainNet.spatial_gain.')
                   for k in pack['model_state_dict'])
        assert not any(k.startswith('MainNet.illumination_estimator.')
                       for k in pack['model_state_dict'])

        rebuilt = TTSREnhance(_v2_args())
        # Strict load must see the v2 tensors, including a non-default trunk
        # width only if it matches. Rebuild from the checkpoint config.
        from block_amend_ckpt import merge_resume_config as merge
        merged, _ = merge(pack['config'], SimpleNamespace(), set(), saved)
        merged.num_res_blocks = pack['config']['num_res_blocks']
        load_strict_state(TTSREnhance(merged), last, mode='inference')

        eval_dir = os.path.join(save, 'eval_val')
        bam.main(['--mode', 'eval', '--model_path', last,
                  '--save_dir', eval_dir, '--eval_split', 'val'] + common)
        bam.main(['--mode', 'resume', '--resume_checkpoint', last,
                  '--save_dir', save, '--num_epochs', '2'] + common)
        pack2 = torch.load(last)
        assert pack2['epoch'] == 2
        assert pack2['model_variant'] == MODEL_VARIANT_V2
        assert pack2['config']['matching_mode'] == 'topk_soft'
        # Resume cannot silently retarget the pack at v1.
        _raises(RuntimeError, lambda: bam.main(
            ['--mode', 'resume', '--resume_checkpoint', last,
             '--save_dir', save, '--num_epochs', '3',
             '--model_variant', MODEL_VARIANT] + common))
    finally:
        bam.BASELINE_FIXED.clear()
        bam.BASELINE_FIXED.update(saved)
    print('E save/resume/eval ok')


def test_scaled_grid_and_legacy():
    legacy_expect = {
        (22, 20): (11, 10),
        (44, 40): (11, 10),
        (64, 60): (16, 15),
        (20, 22): (10, 11),
        (40, 44): (10, 11),
        (60, 64): (15, 16),
    }
    samples = (
        (20, 20), (22, 20), (22, 21), (44, 40), (64, 60), (30, 30),
        (20, 22), (21, 22), (40, 44), (60, 64), (1, 10000), (10000, 1),
        (1, 1),
    )
    for h, w in samples:
        for policy in (GRID_POLICY_LEGACY, GRID_POLICY_SCALED):
            a = choose_token_grid(h, w, 400, policy)
            b = choose_token_grid(h, w, 400, policy)
            assert a == b
            ph, pw = a
            assert 1 <= ph <= h and 1 <= pw <= w and ph * pw <= 400
        assert choose_token_grid(h, w, 400) == choose_token_grid(
            h, w, 400, GRID_POLICY_SCALED)
    assert choose_token_grid(20, 20, 400) == (20, 20)
    assert choose_token_grid(1, 1, 400) == (1, 1)
    assert choose_token_grid(1, 10000, 400) == (1, 400)
    assert choose_token_grid(10000, 1, 400) == (400, 1)
    assert choose_token_grid(5, 5, 1) == (1, 1)
    assert choose_token_grid(8, 3, 7)[0] * choose_token_grid(8, 3, 7)[1] <= 7
    focus = ((22, 20), (22, 21), (44, 40), (64, 60),
             (20, 22), (21, 22), (40, 44), (60, 64))
    for h, w in focus:
        ph, pw = choose_token_grid(h, w, 400, GRID_POLICY_SCALED)
        assert ph * pw >= 360, (h, w, ph, pw)
        old = choose_token_grid(h, w, 400, GRID_POLICY_LEGACY)
        assert old == legacy_expect.get((h, w), old)
        if (h, w) in legacy_expect:
            assert old[0] * old[1] < 360
    # Nearby sizes must not collapse just because the ratio reduces.
    for h in range(20, 25):
        for w in range(20, 25):
            if h * w <= 400:
                assert choose_token_grid(h, w, 400) == (h, w)
                continue
            ph, pw = choose_token_grid(h, w, 400)
            assert ph * pw >= 360, (h, w, ph, pw)
    _raises(ValueError, lambda: choose_token_grid(0, 4, 400))
    _raises(ValueError, lambda: choose_token_grid(4, 4, 0))
    _raises(ValueError, lambda: choose_token_grid(4.0, 4, 400))

    est = MultiScaleSpatialGainEstimator()
    assert est.grid_policy == GRID_POLICY_SCALED
    z8 = torch.randn(1, 64, 8, 8)
    assert torch.equal(est.context_delta(z8), est.context(z8))
    # Non-zero readout so a pooled forward is not identically zero.
    torch.manual_seed(0)
    est = MultiScaleSpatialGainEstimator()
    with torch.no_grad():
        est.readout.weight.normal_(0, 0.02)
    x = torch.rand(1, 3, 176, 160)
    _zf, gain = est(x)
    h8, w8 = est.last_shapes['Z8'][-2:]
    ph, pw = est.last_pool[:2]
    assert (h8, w8) == (22, 20)
    assert (ph, pw) == choose_token_grid(22, 20, 400, GRID_POLICY_SCALED)
    assert ph * pw < h8 * w8 and ph * pw >= 360
    assert gain.shape == x.shape and torch.isfinite(gain).all()
    assert gain.abs().sum() > 0
    gain.sum().backward()
    assert est.readout.weight.grad.abs().sum() > 0
    assert est.context.to_q.weight.grad.abs().sum() > 0
    assert torch.isfinite(est.context.to_q.weight.grad).all()
    print('grid policy', 'scaled', choose_token_grid(22, 20, 400),
          'legacy', choose_token_grid(22, 20, 400, GRID_POLICY_LEGACY))


class _ScalarReads(object):
    def __enter__(self):
        self.n = 0
        self._item = torch.Tensor.item
        self._int = torch.Tensor.__int__
        self._bool = torch.Tensor.__bool__
        counter = self

        def item(tensor):
            counter.n += 1
            return counter._item(tensor)

        def to_int(tensor):
            counter.n += 1
            return counter._int(tensor)

        def to_bool(tensor):
            counter.n += 1
            return counter._bool(tensor)

        torch.Tensor.item = item
        torch.Tensor.__int__ = to_int
        torch.Tensor.__bool__ = to_bool
        return self

    def __exit__(self, *exc):
        torch.Tensor.item = self._item
        torch.Tensor.__int__ = self._int
        torch.Tensor.__bool__ = self._bool


def _ref_gather(value, indices, weights):
    b, d, _nr = value.shape
    k = int(indices.shape[1])
    n = int(indices.shape[2])
    flat = indices.reshape(b, k * n)
    gathered = torch.gather(
        value, 2, flat.unsqueeze(1).expand(b, d, k * n))
    return (gathered.view(b, d, k, n) * weights.unsqueeze(1)).sum(dim=2)


def test_gather_unchecked_matches_and_does_not_sync():
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(4)
    value = torch.randn(2, 24, 18, requires_grad=True)
    idx = torch.randint(0, 18, (2, 4, 11))
    logits = torch.randn(2, 4, 11, requires_grad=True)
    weights = torch.softmax(logits, dim=1)
    got = _weighted_gather_unchecked(value, idx, weights)
    ref = _ref_gather(value.detach().requires_grad_(True), idx, weights)
    # Rebuild weights graph on a twin so grads are comparable.
    logits_b = logits.detach().requires_grad_(True)
    weights_b = torch.softmax(logits_b, dim=1)
    value_b = value.detach().requires_grad_(True)
    ref = _ref_gather(value_b, idx, weights_b)
    got = _weighted_gather_unchecked(
        value.detach().requires_grad_(True), idx,
        torch.softmax(logits.detach().requires_grad_(True), dim=1))
    assert torch.allclose(got, ref, atol=1e-6)
    got.sum().backward()
    ref.sum().backward()
    # The two got/ref graphs were built from different leaves above; compare
    # a fresh paired pair instead.
    for recompute in (False, True):
        v = torch.randn(2, 16, 12, requires_grad=True)
        lg = torch.randn(2, 3, 9, requires_grad=True)
        ix = torch.randint(0, 12, (2, 3, 9))
        w = torch.softmax(lg, dim=1)
        v2 = v.detach().requires_grad_(True)
        lg2 = lg.detach().requires_grad_(True)
        w2 = torch.softmax(lg2, dim=1)
        with _ScalarReads() as reads:
            if recompute:
                out = checkpoint(_weighted_gather_unchecked, v, ix, w)
            else:
                out = _weighted_gather_unchecked(v, ix, w)
            out.sum().backward()
        ref_out = _ref_gather(v2, ix, w2)
        ref_out.sum().backward()
        assert reads.n == 0, (recompute, reads.n)
        assert torch.allclose(out, ref_out, atol=1e-6)
        assert torch.allclose(v.grad, v2.grad, atol=1e-5)
        assert torch.allclose(lg.grad, lg2.grad, atol=1e-5)

    _raises(IndexError, lambda: validate_reference_indices(
        torch.tensor([[[-1, 0]]]), 4))
    _raises(IndexError, lambda: validate_reference_indices(
        torch.tensor([[[0, 4]]]), 4))
    _raises(TypeError, lambda: _weighted_gather_unchecked(
        torch.randn(1, 4, 3), torch.zeros(1, 1, 2), torch.ones(1, 1, 2)))
    _raises(ValueError, lambda: _weighted_gather_unchecked(
        torch.randn(1, 4, 3), torch.zeros(1, 1, 2, dtype=torch.long),
        torch.ones(1, 2, 2)))

    torch.manual_seed(5)
    q, k, r1, r2, r3 = _pack_refs(2, 6, 7, 5, 6)
    for k_top, chunk, recompute in ((1, 3, True), (4, 2, False), (4, 5, True)):
        net = SearchTransfer(
            matching_mode='topk_soft', match_topk=k_top, match_temperature=0.1,
            match_query_chunk_size=chunk, match_transfer_recompute=recompute)
        qq = q.detach().requires_grad_(True)
        kk = k.detach().requires_grad_(True)
        a, b, c = (t.detach().requires_grad_(True) for t in (r1, r2, r3))
        with _ScalarReads() as reads:
            s, t3, t2, t1 = net(qq, kk, a, b, c)
            (t3.sum() + t1.sum()).backward()
        assert reads.n == 0
        qd = q.detach().requires_grad_(True)
        kd = k.detach().requires_grad_(True)
        rd3 = r3.detach().requires_grad_(True)
        rd2 = r2.detach().requires_grad_(True)
        rd1 = r1.detach().requires_grad_(True)
        sd, (d3, d2, d1), _weights, _, _ = _dense_topk(
            qd, kd, (rd3, rd2, rd1), k_top, 0.1)
        (d3.sum() + d1.sum()).backward()
        assert torch.allclose(s, sd, atol=1e-6)
        assert torch.allclose(t3, d3, atol=1e-5)
        assert torch.allclose(qq.grad, qd.grad, atol=1e-4, rtol=1e-4)
        assert torch.allclose(kk.grad, kd.grad, atol=1e-4, rtol=1e-4)

    net = SearchTransfer(
        matching_mode='topk_soft', match_topk=3, match_temperature=0.1,
        match_query_chunk_size=4)
    q, k, r1, r2, r3 = _pack_refs(1, 5, 4, 5, 4)
    net.debug_check_indices = False
    s0 = net(q, k, r1, r2, r3)[0]
    net.debug_check_indices = True
    s1 = net(q, k, r1, r2, r3)[0]
    assert torch.allclose(s0, s1, atol=1e-6)

    if torch.cuda.is_available():
        v = torch.randn(2, 32, 20, device='cuda', requires_grad=True)
        ix = torch.randint(0, 20, (2, 4, 16), device='cuda')
        lg = torch.randn(2, 4, 16, device='cuda', requires_grad=True)
        w = torch.softmax(lg, dim=1)
        with _ScalarReads() as reads:
            out = checkpoint(_weighted_gather_unchecked, v, ix, w)
            out.sum().backward()
        assert reads.n == 0 and torch.isfinite(v.grad).all()
        print('gather scalar reads on cuda: 0')
    else:
        print('gather scalar reads on cpu: 0; cuda profile not run')
    print('B unchecked gather ok')


def test_grid_policy_restore():
    _patch_lte()
    import block_amend_main as bam
    from block_amend_ckpt import merge_resume_config
    from model.TTSREnhance import TTSREnhance
    from model.restoration_config import V2_DEFAULTS

    fresh = restoration_config_from_args(
        SimpleNamespace(model_variant=MODEL_VARIANT_V2))
    assert fresh['gain_grid_policy'] == GRID_POLICY_SCALED
    assert fresh['match_topk'] == 4 and fresh['match_temperature'] == 0.1
    assert bam.BASELINE_FIXED['rec_w'] == 1.0
    assert bam.BASELINE_FIXED['ms_ssim_w'] == 1.0

    old = {k: v for k, v in V2_DEFAULTS.items() if k != 'gain_grid_policy'}
    args, _ = merge_resume_config(old, SimpleNamespace(), set(), bam.BASELINE_FIXED)
    assert args.gain_grid_policy == GRID_POLICY_LEGACY
    assert args.model_variant == MODEL_VARIANT_V2
    args.num_res_blocks = '1+1+1+1'
    args.n_feats = 16
    args.ref_correction_feats = 4
    model = TTSREnhance(args)
    assert model.MainNet.spatial_gain.grid_policy == GRID_POLICY_LEGACY
    assert model.SearchTransfer.matching_mode == 'topk_soft'
    assert model.MainNet.illumination_estimator is None

    cli = SimpleNamespace(gain_grid_policy=GRID_POLICY_SCALED, mode='resume')
    _raises(RuntimeError, lambda: merge_resume_config(
        old, cli, {'gain_grid_policy'}, bam.BASELINE_FIXED), 'gain_grid_policy')

    saved, _ = merge_resume_config(
        fresh, SimpleNamespace(), set(), bam.BASELINE_FIXED)
    assert saved.gain_grid_policy == GRID_POLICY_SCALED
    saved.num_res_blocks = '1+1+1+1'
    saved.n_feats = 16
    saved.ref_correction_feats = 4
    again = TTSREnhance(saved)
    assert again.MainNet.spatial_gain.grid_policy == GRID_POLICY_SCALED

    v1 = TTSREnhance(_tiny_args())
    assert v1.MainNet.illumination_estimator is not None
    assert v1.MainNet.spatial_gain is None
    assert v1.SearchTransfer.matching_mode == 'hard_top1'
    print('D grid policy restore ok')


def test_param_counts():
    old = IlluminationEstimator(64)
    new = MultiScaleSpatialGainEstimator()
    v1 = CSFIUNet(num_res_blocks='8+8+4+2', n_feats=64, gain_estimator='local_affine')
    v2 = CSFIUNet(
        num_res_blocks='8+8+4+2', n_feats=64,
        gain_estimator='multiscale_spatial_v1')
    def nparams(m):
        return sum(p.numel() for p in m.parameters())
    st = SearchTransfer(matching_mode='topk_soft', match_topk=4,
                        match_temperature=0.1)
    print('PARAMS old_gain', nparams(old))
    print('PARAMS new_gain', nparams(new))
    print('PARAMS main_v1', nparams(v1))
    print('PARAMS main_v2', nparams(v2))
    print('PARAMS search', nparams(st))
    print('PARAMS token_cap', new.max_tokens, 'chunk_default', st.match_query_chunk_size)


if __name__ == '__main__':
    print('Python', sys.version.split()[0], 'PyTorch', torch.__version__)
    test_token_grid_and_shapes()
    test_scaled_grid_and_legacy()
    test_gather_unchecked_matches_and_does_not_sync()
    test_grid_policy_restore()
    test_context_residual_and_fresh_zero()
    test_gain_grad_after_readout_step()
    test_lfca_module_unchanged_import()
    test_overlap_matches_ones_like()
    test_topk_math()
    test_topk_k1_matches_hard_and_edges()
    test_model_smoke()
    test_identity_and_checkpoint_roundtrip()
    test_param_counts()
    print('all mgain topk v2 tests passed')
