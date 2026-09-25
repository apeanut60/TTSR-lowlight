#!/usr/bin/env python
"""Structural and gradient tests for the V3-A refiner.

Run standalone:  python tests/test_v3a_refiner.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.V3ARefiner import V3ARefiner, count_params          # noqa: E402


def _refs(B, H, W, seed=0):
    g = torch.Generator().manual_seed(seed)
    y0 = torch.rand(B, 3, H, W, generator=g) * 2 - 1
    low = torch.rand(B, 3, H, W, generator=g) * 2 - 1
    cor = torch.rand(B, 3, H, W, generator=g) * 2 - 1
    mis = torch.rand(B, 3, H, W, generator=g) * 2 - 1
    cor_h = cor.clone()
    cor_h[:, :, H // 4:H // 2, W // 4:W // 2] = mis[:, :, H // 4:H // 2, W // 4:W // 2]
    return y0, low, cor, mis, cor_h


def test_parameter_count_is_reported():
    m = V3ARefiner()
    n = count_params(m)
    print('    params = %d' % n)
    assert n > 0


def test_bypass_is_exact():
    m = V3ARefiner().eval()
    y0 = torch.rand(2, 3, 64, 64) * 2 - 1
    out, aux = m(y0, None)
    assert aux['bypass'] is True and aux['gate'] is None
    assert torch.equal(out, y0), 'bypass is not bit-exact'


def test_step0_identity_for_every_reference_state():
    """Zero-init proposal => Y_hat == Y0 exactly, per the plan's §8."""
    m = V3ARefiner().eval()
    y0, low, cor, mis, cor_h = _refs(2, 128, 128)
    with torch.no_grad():
        for name, r in (('correct', cor), ('mismatch', mis), ('corrupt', cor_h)):
            out, aux = m(y0, r, low)
            assert torch.equal(out, y0), 'step0 identity broken for %s' % name
            assert float(aux['delta'].abs().max()) == 0.0
            # the verifier must still be producing a non-degenerate gate
            assert 0.0 < float(aux['gate'].mean()) < 1.0


def test_shapes_and_odd_sizes():
    m = V3ARefiner().eval()
    for (h, w) in [(128, 128), (400, 600), (255, 257), (64, 96)]:
        y0, low, cor, _, _ = _refs(1, h, w)
        with torch.no_grad():
            out, aux = m(y0, cor, low)
        assert out.shape == y0.shape, (h, w, out.shape)
        # gate is a single channel, delta is RGB
        assert aux['gate'].shape == (1, 1, h, w), (h, w, 'gate', aux['gate'].shape)
        assert aux['delta'].shape == y0.shape, (h, w, 'delta', aux['delta'].shape)
        assert bool(torch.isfinite(out).all()), (h, w, 'non-finite output')
        assert float(out.abs().max()) < 10.0, (h, w, 'output exploded')


def test_correction_is_low_frequency_by_construction():
    """delta is produced at H/8, so it cannot carry fine detail."""
    m = V3ARefiner().eval()
    # break the zero-init so delta is non-trivial
    with torch.no_grad():
        m.p_out.weight.normal_(0, 0.1)
    y0, low, cor, _, _ = _refs(1, 128, 128)
    with torch.no_grad():
        _, aux = m(y0, cor, low)
    d = aux['delta']
    # second difference in x should be ~0: a piecewise-smooth upsample
    dx = (d[:, :, :, 2:] - 2 * d[:, :, :, 1:-1] + d[:, :, :, :-2]).abs().mean()
    scale = d.abs().mean() + 1e-12
    assert float(dx / scale) < 0.2, 'delta carries high-frequency energy'


def test_gradient_staging():
    """step0: rec grad reaches the proposal; trust grad reaches the verifier."""
    m = V3ARefiner().train()
    y0, low, cor, _, _ = _refs(2, 128, 128)
    hr = torch.rand(2, 3, 128, 128) * 2 - 1

    # (a) reconstruction alone: delta == 0 at step 0, so the only gradient that
    #     matters is into p_out; the encoder must NOT receive reconstruction
    #     gradient yet.
    out, aux = m(y0, cor, low)
    (out - hr).abs().mean().backward()
    assert m.p_out.weight.grad is not None, 'no rec grad into the proposal output'
    assert float(m.p_out.weight.grad.abs().max()) > 0
    eg = m.encoder.conv0.weight.grad
    assert eg is None or float(eg.abs().max()) == 0.0, \
        'encoder received reconstruction gradient while delta is still zero'

    # (b) trust alone must reach the verifier even at step 0
    m.zero_grad(set_to_none=True)
    _, aux = m(y0, cor, low)
    l_trust = torch.nn.functional.binary_cross_entropy(
        aux['gate8'].clamp(1e-6, 1 - 1e-6), torch.full_like(aux['gate8'], 0.7))
    l_trust.backward()
    assert m.v_out.weight.grad is not None, 'no trust grad into the verifier'
    assert float(m.v_out.weight.grad.abs().max()) > 0


def test_encoder_gets_gradient_after_proposal_moves():
    m = V3ARefiner().train()
    with torch.no_grad():
        m.p_out.weight.normal_(0, 0.05)
    y0, low, cor, _, _ = _refs(2, 128, 128)
    hr = torch.rand(2, 3, 128, 128) * 2 - 1
    out, _ = m(y0, cor, low)
    (out - hr).abs().mean().backward()
    g = m.encoder.conv0.weight.grad
    assert g is not None and float(g.abs().max()) > 0, \
        'encoder never receives gradient once the proposal is non-zero'


def test_deterministic_in_eval_mode():
    m = V3ARefiner().eval()
    y0, low, cor, _, _ = _refs(1, 64, 64)
    with torch.no_grad():
        a, _ = m(y0, cor, low)
        b, _ = m(y0, cor, low)
    assert torch.equal(a, b)


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
