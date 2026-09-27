#!/usr/bin/env python
"""V3-A.4 protocol tests (§20): proposal loading, mismatch isolation,
normalization, connectivity, shared init, correct gap, real dev metrics."""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.V3A4Verifier import ACTION_CH, V3A4Refiner, build_shared_init  # noqa: E402
from v3a4_runtime import (assert_action_connectivity, action_features_norm,  # noqa: E402
                          action_features_raw, build_mismatch_map, gap,
                          load_r1_proposal_strict, masked_mae, masked_rmse,
                          masked_smooth_l1, pixel_corr)

R1_CK = ('/root/data/experiments/v3a1_lolv2real/R1_v2stable_naive_s42/'
         'checkpoint_03000.pt')


def _inp(seed=0, h=64, w=64, b=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(b, 3, h, w, generator=g) * 2 - 1,
            torch.rand(b, 3, h, w, generator=g) * 2 - 1,
            torch.rand(b, 3, h, w, generator=g) * 2 - 1)


# ── 2.4 proposal loading: no trivial pass ───────────────────────────────────

def test_q_endpoints_with_a_nonzero_synthetic_proposal():
    """The old test passed trivially because the proposal stayed zero-init."""
    m = V3A4Refiner('none').eval()
    with torch.no_grad():                      # make the proposal non-degenerate
        m.proposal.c_out.weight.normal_(0, 0.3)
        m.proposal.c_out.bias.normal_(0, 0.05)
    assert float(m.proposal.c_out.weight.abs().max()) > 0.0
    y0, low, ref = _inp(1)
    with torch.no_grad():
        o0, _ = m(y0, ref, low=low, force_qv=0.0)
        o1, _ = m(y0, ref, low=low, force_qv=1.0)
        sr, _aux = m.proposal(y0, ref)
    assert torch.equal(o0, y0), 'q=0 must be Base exactly'
    assert torch.equal(o1, sr), 'q=1 must be the proposal exactly'
    assert not torch.equal(o1, y0), \
        'proposal output equals Y0 -- the endpoint check proves nothing'


def test_strict_loader_rejects_the_prefix_trap():
    if not os.path.isfile(R1_CK):
        return 'SKIP: R1 checkpoint missing'
    m = V3A4Refiner('none')
    load_r1_proposal_strict(m, R1_CK)
    assert float(m.proposal.c_out.weight.abs().max()) > 0.0
    # the buggy form must be detectable, not silent
    m2 = V3A4Refiner('none')
    missing, _unexpected = m2.proposal.load_state_dict(
        torch.load(R1_CK, map_location='cpu')['model'], strict=False)
    assert len(missing) == len(m2.proposal.state_dict())
    assert float(m2.proposal.c_out.weight.abs().max()) == 0.0


# ── 2.3 mismatch isolation ──────────────────────────────────────────────────

def test_mismatch_maps_never_cross_splits():
    tr = ['low%05d' % i for i in range(1, 576)]        # 575
    dv = ['low%05d' % i for i in range(700, 764)]      # 64
    m_tr = build_mismatch_map(tr)
    m_dv = build_mismatch_map(dv)
    assert set(m_tr) == set(tr) and set(m_dv) == set(dv)
    assert all(k != v for k, v in m_tr.items())
    assert all(k != v for k, v in m_dv.items())
    assert not (set(m_tr.values()) & set(dv)), 'train donor leaked into dev'
    assert not (set(m_dv.values()) & set(tr)), 'dev donor leaked into train'


# ── 5. normalization ───────────────────────────────────────────────────────

def test_action_rms_normalization_gives_order_one_features():
    g = torch.Generator().manual_seed(2)
    d = torch.randn(2, 3, 64, 64, generator=g) * 0.02      # small, like R1's D
    d4, e4 = action_features_raw(d)
    # rms must be measured on D4 itself (plan 5.2), not on the pre-pool D
    rms = [float(d4[:, c].pow(2).mean().sqrt()) for c in range(3)]
    d4n, e4n = action_features_norm(d, rms=rms)
    assert float(d4n.abs().mean()) > 5 * float(d4.abs().mean())
    # by construction D4/rms has unit RMS per channel, so E4 ~ 1 (3 channels)
    assert 0.5 < float(e4n.mean()) < 2.0, float(e4n.mean())
    assert float(e4.mean()) < 0.01
    # a zero correction must stay exactly zero (no mean subtraction)
    z0, z1 = action_features_norm(torch.zeros_like(d), rms=rms)
    assert float(z0.abs().max()) == 0.0 and float(z1.abs().max()) == 0.0


# ── 8. action connectivity ──────────────────────────────────────────────────

def test_action_forward_and_backward_connectivity():
    for mode in ('raw', 'norm'):
        m = V3A4Refiner(mode, action_rms=[0.02, 0.02, 0.02])
        fwd, bwd = assert_action_connectivity(m, ACTION_CH)
        assert fwd > 0.0, '%s: changing D did not change the verifier output' % mode
        assert bwd > 0.0, '%s: action weights received no gradient' % mode


# ── 7. shared init ──────────────────────────────────────────────────────────

def test_shared_init_common_weights_identical_and_step0_agree():
    init = build_shared_init(seed=42, action_rms=[0.02, 0.02, 0.02])
    m = {k: V3A4Refiner(k).eval() for k in ('none', 'raw', 'norm')}
    for k in m:
        m[k].load_state_dict(init[k])
    s = {k: m[k].state_dict() for k in m}
    for key in s['none']:
        if key == 'verifier.head0.weight':
            assert torch.equal(s['none'][key], s['raw'][key][:, :160])
            assert torch.equal(s['norm'][key], s['raw'][key])
        else:
            assert torch.equal(s['none'][key], s['raw'][key])
            assert torch.equal(s['norm'][key], s['raw'][key])
    assert float(s['raw']['verifier.head0.weight'][:, -ACTION_CH:].abs().max()) == 0.0
    y0, low, ref = _inp(3)
    with torch.no_grad():
        outs = [m[k](y0, ref, low=low)[0] for k in ('none', 'raw', 'norm')]
    assert torch.allclose(outs[0], outs[1], atol=1e-6)
    assert torch.allclose(outs[0], outs[2], atol=1e-6)


# ── 2.1 correct gap / 2.2 real metrics ──────────────────────────────────────

def test_gap_formula_is_correct():
    qv = {'correct': torch.full((1, 1, 4, 4), 0.731),
          'a': torch.full((1, 1, 4, 4), 0.447),
          'b': torch.full((1, 1, 4, 4), 0.722)}
    qo = {'correct': torch.full((1, 1, 4, 4), 0.374),
          'a': torch.full((1, 1, 4, 4), 0.319),
          'b': torch.full((1, 1, 4, 4), 0.495)}
    gp, gt, ge = gap(qv, qo, ['a', 'b'])
    assert abs(gp - (0.731 - (0.447 + 0.722) / 2)) < 1e-6, gp
    assert abs(gt - (0.374 - (0.319 + 0.495) / 2)) < 1e-6, gt
    assert abs(ge - (gp - gt)) < 1e-6
    assert gp > 0, 'the old formula gave -0.146 here; the sign must be positive'


def test_masked_metrics_are_not_all_the_same_number():
    g = torch.Generator().manual_seed(4)
    q_v = torch.rand(1, 1, 16, 16, generator=g)
    q_opt = torch.rand(1, 1, 16, 16, generator=g)
    mask = torch.ones(1, 1, 16, 16)
    mae, rmse = masked_mae(q_v, q_opt, mask), masked_rmse(q_v, q_opt, mask)
    sl1 = float(masked_smooth_l1(q_v, q_opt, mask))
    assert mae > 0 and rmse > 0
    assert abs(mae - sl1) > 1e-6, 'MAE must not be the SmoothL1 value'
    assert rmse > mae, 'RMSE >= MAE for the same errors'
    assert 0.9 < pixel_corr(q_v, q_v, mask) <= 1.0


def test_proposal_stays_frozen_and_in_eval_mode():
    from v3a4_runtime import freeze_proposal
    m = V3A4Refiner('norm', action_rms=[0.02, 0.02, 0.02])
    before = freeze_proposal(m)
    opt = torch.optim.Adam(m.verifier.parameters(), lr=1e-3)
    y0, low, ref = _inp(5)
    m.train()
    assert m.proposal.training is False, 'm.train() must not unfreeze the proposal'
    out, _ = m(y0, ref, low=low)
    opt.zero_grad(set_to_none=True)
    out.abs().mean().backward()
    opt.step()
    for k, v in m.proposal.state_dict().items():
        assert torch.equal(v, before[k]), '%s drifted' % k


def main():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith('test_') and callable(f)]
    bad = 0
    for name, fn in fns:
        try:
            msg = fn()
            print('PASS %s %s' % (name, msg or ''))
        except (Exception, SystemExit) as e:                     # noqa: BLE001
            bad += 1
            print('FAIL %s  %s: %s' % (name, type(e).__name__, e))
    print('\n%d/%d passed' % (len(fns) - bad, len(fns)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
