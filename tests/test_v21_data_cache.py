#!/usr/bin/env python
"""Regression tests for the V2.1 data / cache / tiling fixes.

Runs with pytest, or standalone:  python tests/test_v21_data_cache.py

These are the checks the audit (§5 of V2_1_PREFLIGHT_AUDIT.md) said were
missing: they pin the *ordering* of decode->normalise->resize, the data-pass
augmentation, and the tiled blend coverage. They are CPU-only and need no
checkpoints except where noted.
"""

import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.data1_localrefine import (LocalRefineTrainSet,       # noqa: E402
                                       raster_to_model_tensor,
                                       read_rgb, read_rgb_model_tensor,
                                       resize_tensor)
from trainer import tile_starts, Trainer                          # noqa: E402

ROOT = '/root/data/experiments/retinex_v2_localref'
TRAIN_MANIFEST = os.path.join(ROOT, 'manifests_train.csv')
EVAL_MANIFEST = os.path.join(ROOT, 'manifests_eval.csv')
TRAIN_CACHE = os.path.join(ROOT, 'cache_n0_train')


# ── 1. value mapping must come from the source dtype, not from "is it float" ──

def test_constant_gray_scale_is_stable_under_resize():
    """A constant-128 uint8 image must map to ~0.0039216 either way."""
    arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    t = raster_to_model_tensor(arr)
    assert abs(float(t.mean()) - (128 / 127.5 - 1.0)) < 1e-6

    # same raster, forced through the resize path
    x = torch.from_numpy(arr.astype(np.float32).transpose(2, 0, 1))[None] / 255.0
    resized = resize_tensor(raster_to_model_tensor(arr), 32, 48)
    assert abs(float(resized.mean()) - float(t.mean())) < 1e-6, \
        'resize changed the value scale'
    assert float(resized.abs().max()) < 0.01, 'resized image left [-1,1] scale'
    del x


def test_unknown_dtype_is_rejected():
    arr = np.zeros((8, 8, 3), dtype=np.float32)
    try:
        raster_to_model_tensor(arr, what='synthetic')
    except ValueError as e:
        assert 'unsupported dtype' in str(e)
    else:
        raise AssertionError('float32 raster was accepted; range would be guessed')


# ── 2. training read must equal the eval read on a real resized reference ──

def test_train_and_eval_readers_agree_on_real_resized_reference():
    import csv
    if not os.path.isfile(EVAL_MANIFEST):
        return 'skip: eval manifest missing'
    from local_refine_runtime import _read_eval_tensor
    rows = list(csv.DictReader(open(EVAL_MANIFEST, encoding='utf-8')))
    row = rows[0]
    low = read_rgb(row['low_path'])
    h, w = low.shape[:2]
    mine = read_rgb_model_tensor(row['nano_path'], size=(h, w))
    theirs = _read_eval_tensor(row['nano_path'], 'cpu', size=(h, w))[0]
    assert mine.shape == theirs.shape
    d = float((mine - theirs).abs().max())
    assert d < 1e-6, 'train/eval reference reads differ by %.3e' % d
    assert float(mine.abs().max()) <= 1.001, 'reference left model scale'


# ── 3. data pass must actually change the augmentation draw ──

def test_data_pass_changes_geometry_and_is_reproducible():
    import csv
    if not os.path.isfile(TRAIN_MANIFEST) or not os.path.isdir(TRAIN_CACHE):
        return 'skip: train manifest/cache missing'
    rows = list(csv.DictReader(open(TRAIN_MANIFEST, encoding='utf-8')))[:8]
    ds = LocalRefineTrainSet(rows, TRAIN_CACHE, crop_size=128, seed=42)
    idx = 3
    draws = []
    for p in range(6):
        ds.set_data_pass(p)
        draws.append(ds[idx]['y0'].clone())

    uniq = {round(float(d.mean()), 6) for d in draws}
    assert len(uniq) > 1, 'every data pass produced the same crop/geometry'

    ds.set_data_pass(2)
    a = ds[idx]['y0'].clone()
    ds.set_data_pass(2)
    b = ds[idx]['y0'].clone()
    assert torch.equal(a, b), 'same (seed, pass, idx) was not reproducible'


# ── 4. tiled inference: full coverage, identity round-trip, no grey frame ──

class _IdentityModel(torch.nn.Module):
    def forward(self, lr, lrsr, ref, refsr, **kw):
        return lr.clone(), None, None, None, None


def _stub(tile_size=256, overlap=96, window='cosine'):
    stub = types.SimpleNamespace(
        model=_IdentityModel(),
        args=types.SimpleNamespace(tile_size=tile_size, tile_overlap=overlap,
                                   tile_window=window))
    return Trainer._tiled_forward.__get__(stub)


def test_tile_starts_cover_without_negative_origin():
    for length, tile, stride in [(128, 256, 160), (400, 128, 64),
                                 (600, 256, 160), (255, 255, 128),
                                 (257, 256, 160), (1000, 256, 160)]:
        tile = min(tile, length)          # caller clamps per axis
        starts = tile_starts(length, tile, stride)
        assert starts[0] == 0
        assert starts[-1] + tile == length, (starts[-1], tile, length)
        assert all(s >= 0 and s + tile <= length for s in starts)
        covered = np.zeros(length, bool)
        for s in starts:
            covered[s:s + tile] = True
        assert covered.all(), 'gaps in %s' % (starts,)


def test_tiled_forward_identity_and_coverage():
    for (h, w) in [(128, 128), (256, 256), (400, 600), (128, 400),
                   (400, 128), (255, 257)]:
        for ov in (0, 1, 96):
            fn = _stub(overlap=ov)
            x = torch.rand(1, 3, h, w) * 2 - 1
            sr = fn(lr=x, lr_sr=x, ref=x, ref_sr=x)[0]
            assert sr.shape == x.shape, (h, w, ov, sr.shape)
            err = float((sr - x).abs().max())
            assert err <= 1e-6, 'identity drift %.3e at %dx%d overlap=%d' % (
                err, h, w, ov)
            border = sr[:, :, 0, :]
            assert float(border.abs().max()) > 0 or float(x[:, :, 0, :].abs().max()) == 0
            assert torch.isfinite(sr).all()


def test_grey_frame_is_gone_on_a_real_size():
    """The old window made the outer 1px ring exactly 0; it must not any more."""
    fn = _stub(tile_size=256, overlap=96)
    x = torch.full((1, 3, 720, 960), 0.5)
    sr = fn(lr=x, lr_sr=x, ref=x, ref_sr=x)[0]
    assert float((sr - 0.5).abs().max()) < 1e-6, 'border was painted grey again'


def test_overlap_one_does_not_zero_the_seam():
    fn = _stub(tile_size=256, overlap=1)
    x = torch.full((1, 3, 300, 300), -0.25)
    sr = fn(lr=x, lr_sr=x, ref=x, ref_sr=x)[0]
    assert float((sr - (-0.25)).abs().max()) < 1e-6


# ── 5. metric labels ──

def test_ssim_is_labelled_as_y_channel():
    from local_refine_runtime import _metric_triple
    g = torch.Generator().manual_seed(0)
    sr = torch.rand(1, 3, 32, 32, generator=g)
    hr = torch.rand(1, 3, 32, 32, generator=g)
    d = _metric_triple(sr, hr)
    assert 'ssim_y' in d, 'metric triple still labels Y-channel SSIM as RGB'
    assert 'psnr_rgb' in d and 'mse' in d


def main():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith('test_') and callable(f)]
    bad = 0
    for name, fn in fns:
        try:
            msg = fn()
            print('PASS %-58s %s' % (name, msg or ''))
        except Exception as e:                                   # noqa: BLE001
            bad += 1
            print('FAIL %-58s %s: %s' % (name, type(e).__name__, e))
    print('\n%d/%d passed' % (len(fns) - bad, len(fns)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
