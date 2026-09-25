#!/usr/bin/env python
"""V3-A.1 protocol tests: R1 must be reconstruction-only, low must be passed.

These exist because the V3-A run got both wrong:
  * R1 shared the loss function with R2, so it carried trust/reject;
  * no caller passed ``low``, so the verifier's FX collapsed to F0.

Run standalone:  python tests/test_v3a1_protocol.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.V3A1Refiner import V3A1Refiner                            # noqa: E402
from v3a1_runtime import (loss_r1_naive_from_out, loss_r2_state,      # noqa: E402
                          usefulness_at)


class _Counter:
    """Records which supervision terms were touched."""

    def __init__(self):
        self.trust = 0
        self.safe = 0
        self.mismatch_forward = 0
        self.corrupt_forward = 0


def _fns(counter):
    def f_identity(x, y):
        return (x - y).abs().mean()

    def f_zero(x):
        return torch.zeros((), device=x.device if torch.is_tensor(x) else None)

    return (lambda a, b: (a - b).abs().mean(),      # rec
            lambda a, b: (a - b).abs().mean(),      # per
            lambda a: a.abs().mean() * 0,           # smooth
            lambda a: a.abs().mean() * 0)           # color


def test_r1_loss_has_no_trust_or_safe_terms():
    """loss_r1_naive_from_out must not compute trust/safety at all."""
    rec, per, sm, co = _fns(None)
    g = torch.Generator().manual_seed(0)
    out = torch.rand(1, 3, 32, 32, generator=g)
    hr = torch.rand(1, 3, 32, 32, generator=g)
    total, info = loss_r1_naive_from_out(out, hr, rec, per, sm, co)
    assert info['trust'] == 0.0, 'R1 computed a trust term'
    assert info['safe'] == 0.0, 'R1 computed a safety term'
    assert set(info) == {'rec', 'per', 'smooth', 'color', 'trust', 'safe'}


def test_r1_backward_never_touches_a_verifier():
    """R1's graph must contain no verifier parameters."""
    m = V3A1Refiner()
    verifier_ids = {p.data_ptr() for p in m.verifier.parameters()}
    g = torch.Generator().manual_seed(1)
    y0 = torch.rand(1, 3, 64, 64, generator=g)
    ref = torch.rand(1, 3, 64, 64, generator=g)
    hr = torch.rand(1, 3, 64, 64, generator=g)
    _sr, aux = m.proposal(y0, ref)          # proposal path only -- what R1 uses
    out = y0 + aux['gate'] * aux['delta']
    rec, per, sm, co = _fns(None)
    total, info = loss_r1_naive_from_out(out, hr, rec, per, sm, co)
    total.backward()
    touched = {p.data_ptr() for p in m.parameters() if p.grad is not None}
    assert not (touched & verifier_ids), 'R1 backprop reached the verifier'
    assert info['trust'] == 0.0 and info['safe'] == 0.0


def test_r2_does_compute_trust_and_safe():
    """The mirror image: R2 must have both, or the control is meaningless."""
    g = torch.Generator().manual_seed(2)
    out = torch.rand(1, 3, 32, 32, generator=g)
    hr = torch.rand(1, 3, 32, 32, generator=g)
    y0 = torch.rand(1, 3, 32, 32, generator=g)
    q = torch.full((1, 1, 8, 8), 0.5)
    qs = torch.full((1, 1, 8, 8), 0.5)
    rec, per, sm, co = _fns(None)
    _t, info = loss_r2_state(out, hr, y0, q, qs, rec, per, sm, co, bad=False)
    assert info['trust'] > 0.0, 'R2 did not compute trust'
    assert info['safe'] == 0.0, 'safe must be off for the correct reference'
    _t, info_bad = loss_r2_state(out, hr, y0, q, qs, rec, per, sm, co, bad=True)
    assert info_bad['trust'] > 0.0 and info_bad['safe'] > 0.0


def test_low_is_required_and_raises():
    """P0-2 as a hard failure instead of a silent default."""
    m = V3A1Refiner().eval()
    g = torch.Generator().manual_seed(3)
    y0 = torch.rand(1, 3, 64, 64, generator=g)
    ref = torch.rand(1, 3, 64, 64, generator=g)
    try:
        m(y0, ref)                      # no low
    except ValueError as e:
        assert 'low-light input' in str(e), str(e)
    else:
        raise AssertionError('forward() accepted a missing low input')
    out, aux = m(y0, ref, low=y0)       # explicit is fine
    assert not aux['bypass']


def test_verifier_sees_the_true_low_not_y0():
    """Hook the verifier encoder: it must receive X, and X != Y0 here."""
    m = V3A1Refiner().eval()
    g = torch.Generator().manual_seed(4)
    y0 = torch.rand(1, 3, 64, 64, generator=g)
    low = torch.rand(1, 3, 64, 64, generator=g)     # deliberately != y0
    ref = torch.rand(1, 3, 64, 64, generator=g)
    seen = []
    h = m.verifier.conv0.register_forward_hook(lambda mod, inp, out: seen.append(inp[0].detach().clone()))
    with torch.no_grad():
        m(y0, ref, low=low)
    h.remove()
    assert len(seen) == 3, 'verifier encoder should be called exactly 3x (X, Y0, R)'
    assert torch.equal(seen[0], low), 'first verifier call is not the low input'
    assert torch.equal(seen[1], y0)
    assert torch.equal(seen[2], ref)
    assert not torch.equal(seen[0], seen[1]), 'X and Y0 must differ in this test'


def test_qstar_is_at_h4_resolution():
    g = torch.Generator().manual_seed(5)
    y0 = torch.rand(1, 3, 64, 96, generator=g)
    hr = torch.rand(1, 3, 64, 96, generator=g)
    ref = hr.clone()
    q, _d = usefulness_at(y0, ref, hr, tau=0.1)
    assert q.shape[-2:] == (16, 24), q.shape      # H/4, matching the verifier


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
