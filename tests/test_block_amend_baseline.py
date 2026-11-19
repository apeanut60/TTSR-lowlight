"""Unit / smoke tests for block-amend baseline engineering fixes."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)


def _tiny_args(**kw):
    d = dict(
        backbone='csfi_unet',
        num_res_blocks='2+2+1+1',
        n_feats=32,
        res_scale=1.0,
        igab_blocks='1,1',
        igab_heads=1,
        enable_lightup=True,
        use_global_illum=False,
        use_ref_illum=False,
        freeze_lte=True,
        ref_correction=True,
        ref_correction_feats=8,
        ref_illum_pool=8,
        query_illum_align=False,
        query_illum_alpha=0.0,
        query_hist_beta=0.0,
        query_gamma_min=0.3,
        query_gamma_max=1.0,
        query_hist_bins=32,
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
        stage_mode_lv3='normal',
        stage_mode_lv2='normal',
        stage_mode_lv1='normal',
        tile_size=64,
        tile_overlap=16,
    )
    d.update(kw)
    return SimpleNamespace(**d)


class _StubLTE(nn.Module):
    """Deterministic LTE stand-in (no VGG download)."""

    def __init__(self, requires_grad=True, rgb_range=1):
        super().__init__()
        self.slice1 = nn.Conv2d(3, 64, 3, padding=1)
        self.slice2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.slice3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        for p in self.parameters():
            p.requires_grad = requires_grad

    def forward(self, x):
        x1 = self.slice1(x)
        x2 = self.slice2(x1)
        x3 = self.slice3(x2)
        return x1, x2, x3


def _patch_lte():
    import model.LTE as LTE_mod
    import model.TTSREnhance as TE
    LTE_mod.LTE = _StubLTE
    TE.LTE.LTE = _StubLTE


def _build_model(**kw):
    _patch_lte()
    from model.TTSREnhance import TTSREnhance
    return TTSREnhance(_tiny_args(**kw))


# ---- 1. LTE-only load leaves MainNet/RefCorrection unchanged ----
def test_lte_only_preserves_mainnet_refcorrection():
    from model.weight_init import load_lte_only

    model = _build_model()
    wrap = model
    # Fake ckpt with LTE + polluted MainNet
    ckpt = {}
    for k, v in wrap.LTE.state_dict().items():
        ckpt['LTE.' + k] = v.clone() + 0.123
    for k, v in wrap.MainNet.state_dict().items():
        ckpt['MainNet.' + k] = torch.randn_like(v)
    if wrap.RefCorrection is not None:
        for k, v in wrap.RefCorrection.state_dict().items():
            ckpt['RefCorrection.' + k] = torch.randn_like(v)

    before_main = {k: v.detach().clone() for k, v in wrap.MainNet.state_dict().items()}
    before_ref = {k: v.detach().clone()
                  for k, v in wrap.RefCorrection.state_dict().items()}

    path = os.path.join(tempfile.mkdtemp(), 'lte.pt')
    torch.save(ckpt, path)
    load_lte_only(model, path, device='cpu')

    for k, v in wrap.MainNet.state_dict().items():
        assert torch.equal(before_main[k], v), k
    for k, v in wrap.RefCorrection.state_dict().items():
        assert torch.equal(before_ref[k], v), k
    # LTE should have changed
    assert not torch.equal(
        list(wrap.LTE.parameters())[0],
        list(_StubLTE(requires_grad=False).parameters())[0] * 0 + list(wrap.LTE.parameters())[0] * 0)
    # Stronger: LTE weights equal ckpt
    for k, v in wrap.LTE.state_dict().items():
        assert torch.allclose(v, ckpt['LTE.' + k])


# ---- 2. Inference does not rewrite config ----
def test_tiled_does_not_mutate_args():
    from model.tiled_infer import tiled_forward

    model = _build_model()
    model.eval()
    model.args.query_illum_align = False
    model.args.texture_fusion = False
    model.args.stage_mode_lv2 = 'normal'
    before = dict(vars(model.args))
    lr = torch.randn(1, 3, 80, 100)
    ref = torch.randn(1, 3, 80, 100)
    with torch.no_grad():
        _ = tiled_forward(model, lr, lr, ref, ref, tile_size=64, overlap=16)
    assert dict(vars(model.args)) == before


# ---- 3. Identity / constant tiled stitching ----
class _ConstNet(nn.Module):
    def __init__(self, value=0.25):
        super().__init__()
        self.args = SimpleNamespace(
            query_illum_align=False, query_illum_alpha=0.0, query_hist_beta=0.0,
            stage_mode_lv3='normal', stage_mode_lv2='normal', stage_mode_lv1='normal',
            tile_size=64, tile_overlap=16)
        self.value = value
        self.MainNet = SimpleNamespace()  # not CSFI

    def forward(self, lr=None, **kw):
        return torch.full_like(lr, self.value), None, None, None, None

    def get_effective_reference(self, ref):
        return ref

    def apply_ref_illum(self, x, low, ref):
        return x

    def apply_illum_head(self, x, low):
        return x


def test_tiled_constant_shapes():
    from model.tiled_infer import tiled_forward, tile_spans, tile_blend_weight

    cases = [
        (128, 128),   # square multi-tile
        (40, 200),    # short H, long W
        (200, 40),    # long H, short W
        (37, 53),     # odd
        (30, 30),     # smaller than tile
    ]
    net = _ConstNet(0.3)
    for h, w in cases:
        lr = torch.zeros(1, 3, h, w)
        sr = tiled_forward(net, lr, lr, lr, lr, tile_size=64, overlap=16,
                           apply_illum=False, apply_ref_illum=False)
        assert sr.shape == lr.shape, (h, w, sr.shape)
        assert torch.allclose(sr, torch.full_like(sr, 0.3), atol=1e-5)

    # weight coverage: no zero weights on a multi-tile canvas
    H, W, tile, ov = 120, 120, 64, 16
    wt = torch.zeros(1, 1, H, W)
    for y0, y1 in tile_spans(H, tile, ov):
        for x0, x1 in tile_spans(W, tile, ov):
            w = tile_blend_weight(y0, y1, x0, x1, H, W, ov, 'cpu', torch.float32)
            wt[:, :, y0:y1, x0:x1] += w
    assert (wt > 0).all()


# ---- 4. Single-tile matches direct forward (stub) ----
def test_single_tile_matches_direct():
    from model.tiled_infer import tiled_forward

    model = _build_model()
    model.eval()
    lr = torch.randn(1, 3, 48, 48)
    ref = torch.randn(1, 3, 48, 48)
    with torch.no_grad():
        direct = model(lr=lr, lrsr=lr, ref=ref, refsr=ref,
                       apply_illum=False, apply_ref_illum=False)[0]
        tiled = tiled_forward(
            model, lr, lr, ref, ref, tile_size=256, overlap=0,
            apply_illum=False, apply_ref_illum=False)
    assert torch.allclose(direct, tiled, atol=1e-5, rtol=1e-5)


# ---- 5. No-reference forward ----
def test_no_ref_forward():
    model = _build_model()
    model.eval()
    lr = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        sr, S, T3, T2, T1 = model(lr=lr, lrsr=lr, ref=None, refsr=None)
    assert sr.shape == lr.shape
    assert S is None and T3 is None
    assert torch.isfinite(sr).all()
    # half-specified must error
    try:
        model(lr=lr, lrsr=lr, ref=lr, refsr=None)
        assert False, 'expected ValueError'
    except ValueError:
        pass


# ---- 6. Missing generated ref + small-image pad does not use GT ----
def test_missing_ref_and_small_pad_no_gt():
    from dataset.block_amend_lolv2 import (
        collect_triplets, _paired_pad_crop, make_coord_triplet,
    )
    tmp = tempfile.mkdtemp()
    train = os.path.join(tmp, 'Train')
    for sub in ('Low', 'Normal', 'nanobanana_ref_v2'):
        os.makedirs(os.path.join(train, sub), exist_ok=True)
    # Only Low+Normal, no ref
    from imageio import imsave
    im = np.zeros((32, 32, 3), np.uint8)
    imsave(os.path.join(train, 'Low', 'low00001.png'), im)
    imsave(os.path.join(train, 'Normal', 'normal00001.png'), im + 1)
    try:
        collect_triplets(tmp, 'Train', ref_subdir='nanobanana_ref_v2',
                         require_all=True)
        assert False, 'expected RuntimeError'
    except RuntimeError as e:
        assert 'missing Nano' in str(e) or 'missing' in str(e).lower()

    # Pad/crop must not replace Ref with GT
    low, gt, ref = make_coord_triplet(20, 20)
    import random
    rng = random.Random(0)
    lr2, hr2, ref2 = _paired_pad_crop(low, gt, ref, crop_size=32, rng=rng)
    assert not np.array_equal(ref2, hr2), 'Ref must not become GT after pad'


# ---- 7. Paired geometry on coord grids ----
def test_paired_geometry():
    from dataset.block_amend_lolv2 import _paired_flip_rot, make_coord_triplet
    import random
    low, gt, ref = make_coord_triplet(40, 40)
    rng = random.Random(123)
    # Force known transform by controlling rng sequence: call once
    a, b, c = _paired_flip_rot(low.copy(), gt.copy(), ref.copy(), rng)
    # Relative spatial relationship: Low[y,x,0]==x, after same rot/flip all
    # three must share the same permutation of coordinates.
    # Check a corner marker uniquely placed.
    low2 = low.copy(); gt2 = gt.copy(); ref2 = ref.copy()
    low2[5, 7] = [9, 9, 9]
    gt2[5, 7] = [8, 8, 8]
    ref2[5, 7] = [7, 7, 7]
    rng = random.Random(99)
    # Apply identical ops by using same rng state copies
    import copy
    r1 = random.Random(99)
    r2 = random.Random(99)
    r3 = random.Random(99)
    # Better: one call already pairs them
    rng = random.Random(99)
    L, G, R = _paired_flip_rot(low2, gt2, ref2, rng)
    # Find where the marker went in L and require same coords in G,R
    pos = np.argwhere((L == [9, 9, 9]).all(-1))
    assert len(pos) == 1
    y, x = pos[0]
    assert (G[y, x] == [8, 8, 8]).all()
    assert (R[y, x] == [7, 7, 7]).all()


# ---- 8. Changing GT does not change Ref construction; forward has no GT ----
def test_gt_does_not_affect_ref_inputs():
    from dataset.block_amend_lolv2 import BlockAmendTrainSet
    tmp = tempfile.mkdtemp()
    # Build two items differing only in GT content
    items = []
    from imageio import imsave
    for i, gt_val in enumerate([10, 200]):
        d = os.path.join(tmp, 's%d' % i)
        os.makedirs(d, exist_ok=True)
        low = np.full((48, 48, 3), 30, np.uint8)
        gt = np.full((48, 48, 3), gt_val, np.uint8)
        ref = np.full((48, 48, 3), 90, np.uint8)
        imsave(os.path.join(d, 'low.png'), low)
        imsave(os.path.join(d, 'gt.png'), gt)
        imsave(os.path.join(d, 'ref.png'), ref)
        items.append(dict(
            id='s%d' % i,
            low=os.path.join(d, 'low.png'),
            normal=os.path.join(d, 'gt.png'),
            ref=os.path.join(d, 'ref.png')))
    ds0 = BlockAmendTrainSet([items[0]], crop_size=32, ref_degrade=False, seed=0)
    ds1 = BlockAmendTrainSet([items[1]], crop_size=32, ref_degrade=False, seed=0)
    s0, s1 = ds0[0], ds1[0]
    assert torch.equal(s0['Ref'], s1['Ref'])
    assert torch.equal(s0['LR'], s1['LR'])
    assert not torch.equal(s0['HR'], s1['HR'])
    # Model forward signature does not take GT
    model = _build_model()
    model.eval()
    with torch.no_grad():
        out = model(lr=s0['LR'][None], lrsr=s0['LR_sr'][None],
                    ref=s0['Ref'][None], refsr=s0['Ref_sr'][None])
    assert len(out) >= 1


# ---- 9. Tiny forward/backward: finite, LTE frozen ----
def test_tiny_train_step_grads():
    model = _build_model(freeze_lte=True)
    model.train()
    for p in model.LTE.parameters():
        assert p.requires_grad is False
    lr = torch.randn(1, 3, 64, 64, requires_grad=False)
    ref = torch.randn(1, 3, 64, 64)
    hr = torch.randn(1, 3, 64, 64)
    sr = model(lr=lr, lrsr=lr, ref=ref, refsr=ref)[0]
    loss = F_l1(sr, hr)
    assert torch.isfinite(loss)
    loss.backward()
    # Some MainNet params should have grad; LTE none
    main_grads = [p.grad for p in model.MainNet.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
               for g in main_grads)
    for p in model.LTE.parameters():
        assert p.grad is None


def F_l1(a, b):
    return torch.nn.functional.l1_loss(a, b)


# ---- 10. Over-range PNG no wrap; metrics protocol ----
def test_png_no_wrap_and_metrics():
    from utils_metrics_block_amend import (
        metrics_rgb_float_clipped_v1, to_uint8_png, PROTOCOL_NAME,
    )
    sr = torch.full((1, 3, 32, 32), 2.5)   # >> 1 in [-1,1] space → >1 in [0,1]
    hr = torch.zeros(1, 3, 32, 32)
    png = to_uint8_png(sr)
    assert png.dtype == np.uint8 and png.max() == 255 and png.min() == 255
    m = metrics_rgb_float_clipped_v1(sr, hr)
    assert m['protocol'] == PROTOCOL_NAME
    assert m['frac_gt1'] > 0


# ---- 11. Checkpoint save/resume + fresh rejects non-empty dir ----
def test_ckpt_and_fresh_dir():
    from block_amend_main import (
        atomic_torch_save, ensure_save_dir_fresh, BASELINE_FIXED,
    )
    from model.weight_init import load_strict_state

    model = _build_model()
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
    # one step so scheduler state is non-trivial after step
    dummy = sum(p.sum() for p in model.parameters() if p.requires_grad) * 0
    dummy.backward()
    opt.step()
    sched.step()
    sched_state = deepcopy(sched.state_dict())

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'checkpoint_last.pt')
    pack = {
        'model_state_dict': {
            k: v for k, v in model.state_dict().items() if '_copy' not in k
        },
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': sched.state_dict(),
        'epoch': 3,
        'global_step': 10,
        'best_val_psnr': 12.0,
        'config': dict(BASELINE_FIXED),
    }
    atomic_torch_save(pack, path)

    model2 = _build_model()
    load_strict_state(model2, path, mode='resume')
    opt2 = torch.optim.Adam(
        [p for p in model2.parameters() if p.requires_grad], lr=1e-4)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=50, gamma=0.5)
    opt2.load_state_dict(pack['optimizer_state_dict'])
    sched2.load_state_dict(pack['scheduler_state_dict'])
    assert sched2.state_dict()['last_epoch'] == sched_state['last_epoch']

    # fresh rejects non-empty
    open(os.path.join(tmp, 'marker'), 'w').write('x')
    try:
        ensure_save_dir_fresh(tmp)
        assert False, 'expected SystemExit'
    except SystemExit:
        pass


def test_stage_mode_rejects_unknown():
    from model.CSFIUNet import CSFIUNet
    net = CSFIUNet(num_res_blocks='2+2+1+1', n_feats=32)
    x = torch.randn(1, 3, 32, 32)
    try:
        net(x, stage_mode_lv2='not_a_mode')
        assert False
    except ValueError:
        pass


def test_v2_runtime_no_longer_rewrites_cfg():
    """forward_p2_tiled_backbone must not call apply_*_cfg."""
    from v2_p2_runtime import forward_p2_tiled_backbone

    model = _build_model()
    model.eval()
    model.args.stage_mode_lv2 = 'texture_zero'
    model.args.query_illum_align = False
    before = model.args.stage_mode_lv2
    lr = torch.randn(1, 3, 64, 64)
    ref = torch.randn(1, 3, 64, 64)
    args = model.args
    with torch.no_grad():
        _ = forward_p2_tiled_backbone(model, args, lr, lr, ref, ref)
    assert model.args.stage_mode_lv2 == before


if __name__ == '__main__':
    # Run without pytest
    test_lte_only_preserves_mainnet_refcorrection()
    print('1 ok')
    test_tiled_does_not_mutate_args()
    print('2 ok')
    test_tiled_constant_shapes()
    print('3 ok')
    test_single_tile_matches_direct()
    print('4 ok')
    test_no_ref_forward()
    print('5 ok')
    test_missing_ref_and_small_pad_no_gt()
    print('6 ok')
    test_paired_geometry()
    print('7 ok')
    test_gt_does_not_affect_ref_inputs()
    print('8 ok')
    test_tiny_train_step_grads()
    print('9 ok')
    test_png_no_wrap_and_metrics()
    print('10 ok')
    test_ckpt_and_fresh_dir()
    print('11 ok')
    test_stage_mode_rejects_unknown()
    print('12 ok')
    test_v2_runtime_no_longer_rewrites_cfg()
    print('13 ok')
    print('all block-amend baseline tests passed')
