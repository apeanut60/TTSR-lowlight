#!/usr/bin/env python
"""V3-A.3 tests (plan §15): exposure semantics, action input, shared init, gate."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.V3A3Verifier import (ACTION_CH, V3A3Refiner, action_features,  # noqa: E402
                                build_shared_init)
from v3a_runtime import contrast_compress, exposure_gain                  # noqa: E402

R1_CK = ('/root/data/experiments/v3a1_lolv2real/R1_v2stable_naive_s42/'
         'checkpoint_03000.pt')


def _inp(seed=0, h=64, w=96, b=2):
    g = torch.Generator().manual_seed(seed)
    y0 = torch.rand(b, 3, h, w, generator=g) * 2 - 1
    low = torch.rand(b, 3, h, w, generator=g) * 2 - 1
    ref = torch.rand(b, 3, h, w, generator=g) * 2 - 1
    return y0, low, ref


def test_exposure_transform_semantics():
    x01 = torch.tensor([0.0, 0.5, 1.0])
    x = x01 * 2 - 1
    out = (exposure_gain(x, 0.5) + 1) / 2
    assert torch.allclose(out, torch.tensor([0.0, 0.25, 0.5]), atol=1e-6), out
    # black stays black, unlike the contrast-compress version
    assert abs(float(exposure_gain(torch.tensor([-1.0]), 0.5)) + 1.0) < 1e-6
    assert abs(float(contrast_compress(torch.tensor([-1.0]), 0.5)) + 0.5) < 1e-6


def test_action_features_are_area_downsample_and_energy():
    g = torch.Generator().manual_seed(1)
    d = torch.rand(2, 3, 64, 96, generator=g) * 2 - 1
    d4, e4 = action_features(d, factor=4)
    assert d4.shape == (2, 3, 16, 24)
    assert e4.shape == (2, 1, 16, 24)
    ref = F.interpolate(d, size=(16, 24), mode='area')
    assert torch.allclose(d4, ref, atol=1e-6)
    assert torch.allclose(e4, (d4 ** 2).mean(dim=1, keepdim=True), atol=1e-6)


def test_verifier_receives_exactly_D4_and_E4():
    m = V3A3Refiner(use_action=True).eval()
    y0, low, ref = _inp(2)
    seen = {}

    def hook(mod, inp, out):
        seen['x'] = inp[0].detach().clone()

    h = m.verifier.head0.register_forward_hook(hook)
    with torch.no_grad():
        out, aux = m(y0, ref, low=low)
    h.remove()
    D4, E4 = action_features(aux['D'])
    expect = torch.cat([m.verifier.encode(low), m.verifier.encode(y0),
                        m.verifier.encode(ref),
                        (m.verifier.encode(y0) - m.verifier.encode(ref)).abs(),
                        (m.verifier.encode(low) - m.verifier.encode(ref)).abs(),
                        D4, E4], dim=1)
    assert seen['x'].shape[1] == 5 * m.verifier.out_ch + ACTION_CH
    assert torch.allclose(seen['x'], expect, atol=1e-6), 'action channels differ'


def test_gate_endpoints():
    """FIXED (V3-A.4 §2.4).

    The original version loaded ``ck['model']`` -- a dict whose keys are
    prefixed with ``proposal.``/``verifier.`` -- into ``m.proposal`` with
    ``strict=False``. Every key missed, the proposal kept its zero-init, and
    the test then passed *trivially* because q=0 and q=1 both return Y0.
    It now loads strictly and asserts the proposal is non-degenerate.
    """
    m = V3A3Refiner(use_action=True).eval()
    with torch.no_grad():                     # no external checkpoint needed
        m.proposal.c_out.weight.normal_(0, 0.3)
        m.proposal.c_out.bias.normal_(0, 0.05)
    assert float(m.proposal.c_out.weight.abs().max()) > 0.0
    y0, low, ref = _inp(3, h=64, w=96, b=1)
    with torch.no_grad():
        o0, _ = m(y0, ref, low=low, force_qv=0.0)
        o1, _ = m(y0, ref, low=low, force_qv=1.0)
        sr, _aux = m.proposal(y0, ref)
    assert torch.equal(o0, y0), 'q=0 must return Base exactly'
    assert torch.equal(o1, sr), 'q=1 must return the frozen R1 output exactly'
    assert not torch.equal(o1, y0), 'proposal output == Y0: endpoint check is vacuous'


def test_real_r1_checkpoint_loads_strictly():
    """Integration test. Missing checkpoint is reported, not counted as a pass."""
    if not os.path.isfile(R1_CK):
        return 'SKIP: R1 checkpoint missing'
    from v3a4_runtime import load_r1_proposal_strict
    m = V3A3Refiner(use_action=True)
    load_r1_proposal_strict(m, R1_CK)
    assert float(m.proposal.c_out.weight.abs().max()) > 0.0


def test_shared_init_common_weights_and_zero_action_channels():
    s0, s1 = build_shared_init(seed=42)
    assert s1['verifier.head0.weight'].shape[1] == 5 * 32 + ACTION_CH
    assert s0['verifier.head0.weight'].shape[1] == 5 * 32
    for k in s0:
        if k == 'verifier.head0.weight':
            assert torch.equal(s0[k], s1[k][:, :s0[k].shape[1]])
        else:
            assert torch.equal(s0[k], s1[k]), 'common weight %s differs' % k
    act = s1['verifier.head0.weight'][:, 5 * 32:]
    assert float(act.abs().max()) == 0.0, 'action channels must start at zero'


def test_step0_c0_and_c1_agree():
    """With zero action channels, C1 must behave exactly like C0 at step 0."""
    s0, s1 = build_shared_init(seed=7)
    c0 = V3A3Refiner(use_action=False).eval()
    c1 = V3A3Refiner(use_action=True).eval()
    c0.load_state_dict(s0)
    c1.load_state_dict(s1)
    y0, low, ref = _inp(4)
    with torch.no_grad():
        a, _ = c0(y0, ref, low=low)
        b, _ = c1(y0, ref, low=low)
    assert torch.allclose(a, b, atol=1e-6), 'C0/C1 differ at step 0'


def test_proposal_frozen_but_verifier_learns():
    m = V3A3Refiner(use_action=True)
    for p in m.proposal.parameters():
        p.requires_grad_(False)
    before = {k: v.clone() for k, v in m.proposal.state_dict().items()}
    opt = torch.optim.Adam(m.verifier.parameters(), lr=1e-3)
    y0, low, ref = _inp(5)
    m.train()
    out, _aux = m(y0, ref, low=low)
    loss = out.abs().mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    for k, v in m.proposal.state_dict().items():
        assert torch.equal(v, before[k]), 'proposal %s changed' % k


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
