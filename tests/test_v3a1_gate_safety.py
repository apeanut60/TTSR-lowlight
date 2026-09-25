#!/usr/bin/env python
"""V3-A.1 gate-safety and proposal-regression tests (plan §11).

  q_v = 0  ->  output == Y0        (suppression is exact)
  q_v = 1  ->  output == the V2 proposal output, bit-exact

The second one is what makes "q_v can only suppress, never amplify" a checked
property rather than a claim about the code.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.LocalRefine import Refiner as V2Proposal                   # noqa: E402
from model.V3A1Refiner import V3A1Refiner, LowAnchorVerifier          # noqa: E402


def _inputs(seed=0, h=64, w=96, b=2):
    g = torch.Generator().manual_seed(seed)
    y0 = torch.rand(b, 3, h, w, generator=g) * 2 - 1
    low = torch.rand(b, 3, h, w, generator=g) * 2 - 1
    ref = torch.rand(b, 3, h, w, generator=g) * 2 - 1
    return y0, low, ref


def test_qv_zero_returns_y0_exactly():
    m = V3A1Refiner().eval()
    with torch.no_grad():
        m.proposal.c_out.weight.normal_(0, 0.05)      # make delta non-trivial
    y0, low, ref = _inputs(1)
    with torch.no_grad():
        out, aux = m(y0, ref, low=low, force_qv=0.0)
    assert torch.equal(out, y0), 'q_v=0 did not return Y0 exactly'
    assert float(aux['gate_final'].abs().max()) == 0.0


def test_qv_one_matches_the_v2_proposal_bit_exact():
    m = V3A1Refiner().eval()
    with torch.no_grad():
        m.proposal.c_out.weight.normal_(0, 0.05)
    y0, low, ref = _inputs(2)
    with torch.no_grad():
        out, aux = m(y0, ref, low=low, force_qv=1.0)
        v2_sr, v2_aux = m.proposal(y0, ref)
    assert torch.equal(out, v2_sr), 'q_v=1 is not the V2 proposal output'
    assert torch.equal(aux['delta'], v2_aux['delta'])
    assert torch.equal(aux['gate_v2'], v2_aux['gate'])


def test_proposal_is_literally_the_v2_module():
    """Same class -> the same parameter layout the V2 checkpoints use."""
    m = V3A1Refiner()
    assert isinstance(m.proposal, V2Proposal)
    v2 = V2Proposal()
    a = dict(m.proposal.named_parameters())
    b = dict(v2.named_parameters())
    assert set(a) == set(b), 'proposal parameter names diverged from V2'
    for k in a:
        assert a[k].shape == b[k].shape


def test_qv_one_equals_v2_for_a_loaded_v2_checkpoint():
    """Load an actual V2 checkpoint and confirm the q_v=1 path reproduces it."""
    ck = '/root/data/experiments/retinex_v21_verified/V2_Nano_s42/checkpoint_03000.pt'
    if not os.path.isfile(ck):
        return 'skip: V2 checkpoint not present'
    m = V3A1Refiner().eval()
    sd = torch.load(ck, map_location='cpu')['refiner']
    missing, unexpected = m.proposal.load_state_dict(sd, strict=True), None
    del unexpected, missing
    y0, low, ref = _inputs(3, h=400, w=600, b=1)
    with torch.no_grad():
        out, _ = m(y0, ref, low=low, force_qv=1.0)
        v2_sr, _ = m.proposal(y0, ref)
    d = float((out - v2_sr).abs().max())
    assert d == 0.0, 'q_v=1 differs from the loaded V2 model by %.3e' % d


def test_step0_identity_with_fresh_init_and_real_verifier():
    """C_out zero-init => all three reference states return Y0 bit-exactly."""
    m = V3A1Refiner().eval()
    y0, low, ref = _inputs(4)
    other = torch.flip(ref, dims=[3])
    corrupt = ref.clone()
    corrupt[:, :, 16:40, 20:60] = other[:, :, 16:40, 20:60]
    with torch.no_grad():
        for name, r in (('correct', ref), ('mismatch', other), ('corrupt', corrupt)):
            out, aux = m(y0, r, low=low)
            assert torch.equal(out, y0), 'step0 identity broken for %s' % name
            assert float(aux['delta'].abs().max()) == 0.0
            assert 0.0 < float(aux['q_v'].mean()) < 1.0, 'verifier output degenerate'


def test_gate_ranges_are_bounded():
    m = V3A1Refiner().eval()
    y0, low, ref = _inputs(5)
    with torch.no_grad():
        _o, aux = m(y0, ref, low=low)
    for k in ('gate_v2', 'q_v', 'gate_final'):
        v = aux[k]
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0, '%s out of [0,1]' % k
    assert float((aux['gate_final'] - aux['gate_v2'] * aux['q_v']).abs().max()) == 0.0


def test_verifier_param_count_is_small_relative_to_proposal():
    m = V3A1Refiner()
    nv = sum(p.numel() for p in m.verifier.parameters())
    np_ = sum(p.numel() for p in m.proposal.parameters())
    print('    verifier %d params, proposal %d params' % (nv, np_))
    assert 0 < nv < np_, 'verifier should be the smaller add-on'


def main():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith('test_') and callable(f)]
    bad = 0
    for name, fn in fns:
        try:
            msg = fn()
            print('PASS %s %s' % (name, msg or ''))
        except Exception as e:                                   # noqa: BLE001
            bad += 1
            print('FAIL %s  %s: %s' % (name, type(e).__name__, e))
    print('\n%d/%d passed' % (len(fns) - bad, len(fns)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
