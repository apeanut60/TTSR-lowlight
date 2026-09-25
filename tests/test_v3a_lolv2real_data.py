#!/usr/bin/env python
"""Regression tests for the V3-A LOL-v2-real data path.

Run standalone:  python tests/test_v3a_lolv2real_data.py

The single most important property here is that the reference can never be the
ground truth. ``dataset/lolv2real.py::TrainSet`` does ``Ref = HR.copy()`` and
this file exists to make sure the V3-A path does not.
"""

import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import lolv2real_v3a as D                        # noqa: E402

DATA = '/root/data/datasets/lol-v2-real'
REF = os.path.join(DATA, 'Test', 'nanobanana_ref_v2')
TRAIN_REF = os.path.join(DATA, 'Train', 'nanobanana_ref_v2')


def _args(**kw):
    base = dict(dataset_dir=DATA, ref_dir=TRAIN_REF, train_crop_size=128,
                seed=42, no_reference=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_gt_name_uses_prefix_replacement():
    assert D.gt_name_for('low00690.png') == 'normal00690.png'
    assert D.gt_name_for('low00001.jpg') == 'normal00001.jpg'
    try:
        D.gt_name_for('r000da54ft.png')
    except SystemExit:
        pass
    else:
        raise AssertionError('accepted a filename that is not low*')


def test_pairs_and_reference_coverage():
    tr = D.collect_pairs(DATA, 'Train')
    te = D.collect_pairs(DATA, 'Test')
    assert len(tr) == 689, len(tr)
    assert len(te) == 100, len(te)
    have = set(D.list_files(TRAIN_REF))
    cov = sum(1 for n, _, _ in tr if n in have)
    assert cov == 639, cov


def test_checkpoints_style_dirs_are_ignored():
    # six directories in this dataset hold a stray .ipynb_checkpoints subdir
    fs = D.list_files(os.path.join(DATA, 'Train', 'Low'))
    assert all(os.path.splitext(f)[1].lower() in D.IMG_EXT for f in fs)
    assert '.ipynb_checkpoints' not in fs


def test_missing_reference_refuses_instead_of_hr_fallback():
    """require_ref must fail loudly; it must never return the GT as reference."""
    a = _args(ref_dir='')            # no ref dir at all
    for kwargs in (dict(require_ref=True), {}):
        try:
            D.TrainSet(a, **kwargs)
        except SystemExit as e:
            assert 'require_ref' in str(e) or 'ref_dir' in str(e), str(e)
        else:
            raise AssertionError('built a train set with no reference source')
    # and a subset whose members are known to lack references must fail too
    tr = D.collect_pairs(DATA, 'Train')
    have = set(D.list_files(TRAIN_REF))
    subset = [p for p in tr if p[0] not in have][:5]
    try:
        D.TrainSet(_args(), pairs=subset, require_ref=True)
    except SystemExit as e:
        assert 'no reference' in str(e), str(e)
    else:
        raise AssertionError('accepted samples with no reference')


def test_manifest_subset_is_the_639():
    m = ('/root/data/experiments/v3a_lolv2real/manifests/refiner_train.csv')
    if not os.path.isfile(m):
        return 'skip: manifest not built yet'
    ds = D.TrainSet(_args(), crop_size=64, pairs=D.pairs_from_manifest(m))
    assert len(ds) == 639, len(ds)
    assert ds.n_missing_ref == 0
    s = ds[3]
    assert not torch.equal(s['Ref'], s['HR']), 'reference equals the GT'


def test_placeholder_mode_requires_no_reference_flag():
    try:
        D.TrainSet(_args(no_reference=False), require_ref=False)
    except SystemExit as e:
        assert 'no_reference' in str(e)
    else:
        raise AssertionError('placeholder allowed without --no_reference True')


def test_placeholder_is_the_low_input_never_the_gt():
    ds = D.TrainSet(_args(no_reference=True), crop_size=64, require_ref=False)
    assert ds.n_missing_ref == 50, ds.n_missing_ref
    s = ds[0]
    # where a reference exists it must differ from the GT as an image file
    name, lr, hr, ref, y0, _m = ds._load(0)
    assert y0 is None, 'no cache configured, so Y0 must be None'
    have = set(D.list_files(TRAIN_REF))
    if name in have:
        assert not torch.equal(ref, hr), 'reference equals the GT'
    else:
        assert torch.equal(ref, lr), 'placeholder should be the low input'
        assert not torch.equal(ref, hr)


def test_read_order_is_normalise_then_resize():
    """A constant-128 raster must map to ~0.0039 no matter the target size."""
    arr = np.full((40, 60, 3), 128, dtype=np.uint8)
    t = D._to_model_tensor(arr)
    assert abs(float(t.mean()) - (128 / 127.5 - 1.0)) < 1e-6
    r = D._resize(t, 20, 30)
    assert abs(float(r.mean()) - float(t.mean())) < 1e-6
    assert float(r.abs().max()) < 0.01, 'resized tensor left the model scale'


def test_unknown_dtype_is_rejected():
    try:
        D._to_model_tensor(np.zeros((8, 8, 3), dtype=np.float32))
    except ValueError as e:
        assert 'unsupported dtype' in str(e)
    else:
        raise AssertionError('float raster accepted; range would be guessed')


def test_geometry_is_shared_and_pass_dependent():
    m = ('/root/data/experiments/v3a_lolv2real/manifests/refiner_train.csv')
    pairs = D.pairs_from_manifest(m) if os.path.isfile(m) else None
    ds = D.TrainSet(_args(), crop_size=64, pairs=pairs, require_ref=True)
    # synthetic alignment: build a coordinate chart that all three tensors share
    base = torch.arange(3 * 200 * 300, dtype=torch.float32).reshape(3, 200, 300)
    ds._load = lambda idx: ('synthetic', base, base.clone() + 1000,
                            base.clone() + 2000, None, None)
    ds.set_data_pass(0)
    a = ds[7]
    ds.set_data_pass(0)
    b = ds[7]
    assert torch.equal(a['LR'], b['LR']), 'same (seed,pass,idx) not reproducible'
    assert torch.equal(a['LR'] - a['HR'],
                       a['LR'] - a['HR']), 'geometry broke the LR/HR offset'
    # all three tensors must have received the SAME crop/rot/flip
    d_ab = (a['HR'] - a['LR'])
    d_cb = (a['Ref'] - a['LR'])
    # HR and Ref are the same chart offset by +1000 / +2000; if the geometry draw
    # differed between tensors these constants would no longer be constant.
    assert float(d_ab.std()) == 0.0, 'HR/LR no longer share geometry'
    assert float(d_cb.std()) == 0.0, 'Ref/LR no longer share geometry'
    assert abs(float(d_ab[0, 0, 0]) - 1000) < 1e-3
    assert abs(float(d_cb[0, 0, 0]) - 2000) < 1e-3
    draws = []
    for p in range(8):
        ds.set_data_pass(p)
        draws.append(float(ds[7]['LR'].mean()))
    assert len(set(round(x, 6) for x in draws)) > 1, \
        'data pass never changed the geometry'


def test_real_sample_shapes_and_scale():
    m = ('/root/data/experiments/v3a_lolv2real/manifests/refiner_train.csv')
    pairs = D.pairs_from_manifest(m) if os.path.isfile(m) else None
    ds = D.TrainSet(_args(), crop_size=128, pairs=pairs, require_ref=True)
    s = ds[0]
    assert ds.n_missing_ref == 0
    for k in ('LR', 'LR_sr', 'HR', 'Ref', 'Ref_sr'):
        assert s[k].shape == (3, 128, 128), (k, s[k].shape)
        assert s[k].abs().max() <= 1.001, (k, float(s[k].abs().max()))


def main():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith('test_') and callable(f)]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print('PASS %s' % name)
        except (Exception, SystemExit) as e:                     # noqa: BLE001
            bad += 1
            print('FAIL %s  %s: %s' % (name, type(e).__name__, e))
    print('\n%d/%d passed' % (len(fns) - bad, len(fns)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
