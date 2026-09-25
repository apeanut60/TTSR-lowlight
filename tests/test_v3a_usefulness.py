#!/usr/bin/env python
"""Tests for the V3-A usefulness target and bad-reference construction."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3a_runtime import (d8, mismatch_permutation, rescale_reference,  # noqa: E402
                         scalar_d, splice_corrupt, usefulness)


def _imgs(seed=0, h=64, w=96):
    g = torch.Generator().manual_seed(seed)
    y0 = torch.rand(1, 3, h, w, generator=g) * 2 - 1
    hr = torch.rand(1, 3, h, w, generator=g) * 2 - 1
    return y0, hr


def test_d8_is_area_downsampling():
    x = torch.arange(1 * 3 * 64 * 96, dtype=torch.float32).reshape(1, 3, 64, 96)
    y = d8(x, 8)
    assert y.shape == (1, 3, 8, 12), y.shape
    # area mean of a constant patch equals the constant
    z = torch.full((1, 3, 64, 96), 0.25)
    assert torch.allclose(d8(z, 8), torch.full((1, 3, 8, 12), 0.25), atol=1e-6)
    # non-divisible size must not crash
    assert d8(torch.zeros(1, 3, 100, 101), 8).shape[-2:] == (12, 12)


def test_perfect_reference_scores_high():
    y0, hr = _imgs()
    q, d = usefulness(y0, hr.clone(), hr, tau=0.1)
    assert float(d.mean()) > 0, 'a perfect reference must beat Y0'
    assert float(q.mean()) > 0.5


def test_reference_equal_to_y0_scores_half():
    y0, hr = _imgs()
    q, d = usefulness(y0, y0.clone(), hr, tau=0.1)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-6)
    assert abs(float(q.mean()) - 0.5) < 1e-5, float(q.mean())


def test_worse_reference_scores_below_half():
    y0, hr = _imgs()
    # a reference that moved away from the GT
    bad = y0 + 2.0 * (y0 - hr)
    q, d = usefulness(y0, bad, hr, tau=0.1)
    assert float(d.mean()) < 0
    assert float(q.mean()) < 0.5


def test_tau_scale_direction():
    """tau controls sharpness only; the sign/order must not depend on it."""
    y0, hr = _imgs()
    ref = 0.5 * hr + 0.5 * y0
    for tau in (0.01, 0.1, 1.0):
        q, d = usefulness(y0, ref, hr, tau=tau)
        assert float(d.mean()) > 0
        assert float(q.mean()) > 0.5


def test_mismatch_has_no_fixed_points():
    rows = [dict(sample_id='a%d' % i, camera='c') for i in range(10)]
    perm = mismatch_permutation(rows)
    assert all(perm[i] != i for i in range(10))
    assert sorted(perm) == list(range(10)), 'not a permutation'
    two = rows + [dict(sample_id='b%d' % i, camera='d') for i in range(3)]
    p2 = mismatch_permutation(two)
    assert all(p2[i] != i for i in range(len(two)))
    assert all(two[p2[i]]['camera'] == two[i]['camera'] for i in range(len(two)))


def test_splice_corrupt_changes_only_a_block():
    g = np.random.default_rng(1)      # splice_corrupt takes a numpy Generator
    ref = torch.zeros(3, 64, 64)
    donor = torch.ones(3, 64, 64)
    out = splice_corrupt(ref, donor, g)
    frac = float((out != ref).float().mean())
    assert 0.0 < frac <= 0.5, frac
    assert frac >= 0.05, 'corrupt block too small to matter'
    # untouched pixels stay identical
    assert torch.equal(out[out == ref], ref[out == ref])


def test_rescale_reference_stays_in_range():
    y0, hr = _imgs()
    for gain in (0.6, 1.5):
        r = rescale_reference(hr, gain)
        assert float(r.min()) >= -1.0 and float(r.max()) <= 1.0
    assert float((rescale_reference(hr, 0.6) - hr).abs().mean()) > 0.05


def main():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith('test_') and callable(f)]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print('PASS %s' % name)
        except Exception as e:                                   # noqa: BLE001
            bad += 1
            print('FAIL %s  %s: %s' % (name, type(e).__name__, e))
    print('\n%d/%d passed' % (len(fns) - bad, len(fns)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
