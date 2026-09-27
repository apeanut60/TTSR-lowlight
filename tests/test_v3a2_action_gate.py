#!/usr/bin/env python
"""V3-A.2 tests: q_opt mathematics, masking, and the frozen-proposal contract."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.V3A1Refiner import V3A1Refiner                            # noqa: E402
from v3a2_runtime import (action_optimal_gate, energy_threshold,      # noqa: E402
                          masked_smooth_l1)
from v3a_runtime import mismatch_permutation                         # noqa: E402

R1_CK = ('/root/data/experiments/v3a1_lolv2real/R1_v2stable_naive_s42/'
         'checkpoint_03000.pt')


def _rand(seed=0, h=64, w=64, b=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, 3, h, w, generator=g) * 2 - 1


def test_qopt_one_when_correction_equals_the_error():
    y0, hr = _rand(1), _rand(2)
    d = hr - y0
    q, _e = action_optimal_gate(y0, hr, d)
    assert float(q.mean()) > 0.99, float(q.mean())


def test_qopt_half_when_correction_is_double():
    y0, hr = _rand(3), _rand(4)
    d = 2.0 * (hr - y0)
    q, _e = action_optimal_gate(y0, hr, d)
    assert abs(float(q.mean()) - 0.5) < 0.02, float(q.mean())


def test_qopt_zero_when_correction_is_backwards():
    y0, hr = _rand(5), _rand(6)
    d = -(hr - y0)
    q, _e = action_optimal_gate(y0, hr, d)
    assert float(q.max()) == 0.0, float(q.max())


def test_qopt_masks_zero_correction():
    y0, hr = _rand(7), _rand(8)
    d = torch.zeros_like(y0)
    _q, e = action_optimal_gate(y0, hr, d)
    assert float(e.max()) < 1e-12, 'zero correction should have zero energy'


def test_masked_loss_ignores_low_energy_positions():
    q_v = torch.zeros(1, 1, 4, 4)
    q_opt = torch.ones(1, 1, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., :2, :2] = 1.0
    v = masked_smooth_l1(q_v, q_opt, mask)
    assert v > 0
    # changing q_v only where the mask is 0 must not change the loss
    q_v2 = q_v.clone()
    q_v2[..., 2:, 2:] = 0.9
    assert torch.allclose(v, masked_smooth_l1(q_v2, q_opt, mask))


def test_energy_threshold_is_a_p10():
    e = [torch.linspace(0, 1, 101).reshape(1, 1, 101, 1)]
    t = energy_threshold(e)
    assert 0.05 < t < 0.15, t


def test_proposal_is_frozen_but_verifier_updates():
    if not os.path.isfile(R1_CK):
        return 'skip: R1 checkpoint missing'
    m = V3A1Refiner().eval()
    m.load_state_dict(torch.load(R1_CK, map_location='cpu')['model'])
    for p in m.proposal.parameters():
        p.requires_grad_(False)
    before = {k: v.clone() for k, v in m.proposal.state_dict().items()}
    vb = {k: v.clone() for k, v in m.verifier.state_dict().items()}

    opt = torch.optim.Adam(m.verifier.parameters(), lr=1e-3)
    y0, low, ref = _rand(9), _rand(10), _rand(11)
    hr = _rand(12)
    m.train()
    out, aux = m(y0, ref, low=low)
    q_opt, e = action_optimal_gate(y0, hr, aux['gate_v2'] * aux['delta'])
    mask = (e > 0.0).float()
    loss = masked_smooth_l1(aux['q_v4'], q_opt, mask) + 0.1 * (out - hr).abs().mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    for k, v in m.proposal.state_dict().items():
        assert torch.equal(v, before[k]), 'proposal tensor %s changed' % k
    changed = any(not torch.equal(v, vb[k]) for k, v in m.verifier.state_dict().items())
    assert changed, 'verifier did not update'


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
