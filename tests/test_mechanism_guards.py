#!/usr/bin/env python
"""Mechanism guards: assert that the intended signal actually flows.

Every bug of the "silent failure" class found in this project would have been
caught by one of these, and none of them is caught by the output-shape tests
that were already in place:

  R1 shared R2's loss            -> G4 (a path that must not train has no grad)
  verifier never received low    -> G3 (input receipt)
  GT gate missing/doubled g_v2    -> G1 (counterfactual equality) + G7 (scale)
  proposal state dict not loaded  -> G2 (load fidelity, non-degenerate source)
  action channels 100x too small  -> G7 (input-scale parity)
  target degenerate (D == 0)      -> G5 (target sanity)

Run standalone:  python tests/test_mechanism_guards.py
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

R1_CK = ('/root/data/experiments/v3a1_lolv2real/R1_v2stable_naive_s42/'
         'checkpoint_03000.pt')


# ── reusable helpers (import these from training scripts) ────────────────────

def assert_load_fidelity(module, tensor_names, source_sd, min_abs=1e-6):
    """G2: a load that silently matches nothing is the worst failure mode."""
    got = module.state_dict()
    for n in tensor_names:
        if n not in got or n not in source_sd:
            raise AssertionError('missing key %r when checking load fidelity' % n)
        if not torch.equal(got[n], source_sd[n]):
            raise AssertionError('%s did not load bit-exactly' % n)
        if float(source_sd[n].abs().max()) < min_abs:
            raise AssertionError('%s is all-zero in the source: the checkpoint is '
                                 'degenerate, so "loaded" cannot be verified' % n)


def assert_frozen(before_sd, module, where='module'):
    """G6: freezing must be bit-exact, not approximately true."""
    now = module.state_dict()
    if set(now) != set(before_sd):
        raise AssertionError('%s key set changed' % where)
    for k in now:
        if not torch.equal(now[k], before_sd[k]):
            raise AssertionError('%s tensor %s changed while frozen' % (where, k))


def assert_input_scale_parity(groups, ratio_limit=5.0, names=None):
    """G7: channel groups entering one layer must be comparable in magnitude.

    ``groups`` maps a name to a tensor of activations. A group an order of
    magnitude smaller than the rest cannot influence the layer, so any claim
    that it "was added as an input" is false in practice.
    """
    scales = {k: float(v.abs().mean()) for k, v in groups.items()}
    med = sorted(scales.values())[len(scales) // 2]
    bad = {k: s for k, s in scales.items() if s * ratio_limit < med or s > med * ratio_limit}
    if bad:
        raise AssertionError(
            'input scale imbalance: median %.3e, offenders %s (limit x%.0f); '
            'normalise the small groups before claiming they are inputs'
            % (med, {k: '%.2e' % v for k, v in bad.items()}, ratio_limit))
    return scales


def record_hook(store, key):
    """G3 helper. NOTE: the returned hook must return None.

    A hook that returns a value *replaces the module's output* -- a mistake that
    once produced a phantom "wrong number of channels" error.
    """
    def hook(mod, inp, out):
        store[key] = inp[0].detach().clone()
        return None
    return hook


# ── guards exercised against the real artifacts ──────────────────────────────

def test_g2_load_fidelity_catches_a_prefix_mismatch():
    """The exact V3-A.3 bug: prefixed keys match nothing and load silently."""
    from model.V3A1Refiner import V2Proposal
    if not os.path.isfile(R1_CK):
        return 'skip: R1 checkpoint missing'
    full_sd = torch.load(R1_CK, map_location='cpu')['model']
    prop = V2Proposal()
    # the buggy form: full dict into the submodule, nothing matches
    missing, unexpected = prop.load_state_dict(full_sd, strict=False)
    assert len(missing) == len(prop.state_dict()), 'expected zero matches'
    assert float(prop.c_out.weight.abs().max()) == 0.0, \
        'zero-init proposal is the symptom we rely on to detect this'
    # the correct form, plus the fidelity assertion
    pref = {k[len('proposal.'):]: v for k, v in full_sd.items()
            if k.startswith('proposal.')}
    prop.load_state_dict(pref, strict=True)
    assert_load_fidelity(prop, ['c_out.weight', 'c_out.bias'], pref)


def test_g3_verifier_receives_low_and_action_channels():
    from model.V3A3Verifier import V3A3Refiner, action_features
    m = V3A3Refiner(use_action=True).eval()
    g = torch.Generator().manual_seed(0)
    y0 = torch.rand(1, 3, 64, 96, generator=g) * 2 - 1
    low = torch.rand(1, 3, 64, 96, generator=g) * 2 - 1
    ref = torch.rand(1, 3, 64, 96, generator=g) * 2 - 1
    store = {}
    h = m.verifier.head0.register_forward_hook(record_hook(store, 'head0_in'))
    with torch.no_grad():
        _o, aux = m(y0, ref, low=low)
    h.remove()
    x = store['head0_in']
    c = m.verifier.out_ch
    assert torch.equal(x[:, :c], m.verifier.encode(low)), 'first group is not X'
    d4, e4 = action_features(aux['D'])
    assert torch.allclose(x[:, 5 * c:5 * c + 3], d4, atol=1e-6), \
        'action channels are not area_downsample(g_v2*delta)'
    assert torch.allclose(x[:, 5 * c + 3:], e4, atol=1e-6)


def test_g7_scale_parity_catches_the_v3a3_failure():
    """Reproduce the V3-A.3 numbers and confirm the guard fires."""
    groups = {'FX': torch.full((1, 32, 8, 8), 0.510),
              'F0': torch.full((1, 32, 8, 8), 0.210),
              'FR': torch.full((1, 32, 8, 8), 0.411),
              '|F0-FR|': torch.full((1, 32, 8, 8), 0.235),
              '|FX-FR|': torch.full((1, 32, 8, 8), 0.130),
              'D4': torch.full((1, 3, 8, 8), 0.0234),
              'E4': torch.full((1, 1, 8, 8), 0.000885)}
    try:
        assert_input_scale_parity(groups)
    except AssertionError as e:
        assert 'E4' in str(e) and 'D4' in str(e), str(e)
    else:
        raise AssertionError('guard did not flag the V3-A.3 scale imbalance')
    # after normalising to a comparable range the guard must pass
    groups['D4'] = groups['D4'] / 0.0234 * 0.3
    groups['E4'] = groups['E4'] / 0.000885 * 0.3
    assert_input_scale_parity(groups)


def test_g4_a_path_that_must_not_train_has_no_gradient():
    """The V3-A bug: R1's loss carried trust/reject because it shared a function."""
    from model.V3A1Refiner import V3A1Refiner
    m = V3A1Refiner()
    vids = {p.data_ptr() for p in m.verifier.parameters()}
    g = torch.Generator().manual_seed(1)
    y0 = torch.rand(1, 3, 64, 64, generator=g)
    ref = torch.rand(1, 3, 64, 64, generator=g)
    hr = torch.rand(1, 3, 64, 64, generator=g)
    _sr, aux = m.proposal(y0, ref)                  # proposal path only
    out = y0 + aux['gate'] * aux['delta']
    (out - hr).abs().mean().backward()
    touched = {p.data_ptr() for p in m.parameters() if p.grad is not None}
    assert not (touched & vids), 'a path that must not train reached the verifier'


def test_g5_target_is_not_degenerate():
    from v3a2_runtime import action_optimal_gate
    g = torch.Generator().manual_seed(2)
    y0 = torch.rand(1, 3, 64, 64, generator=g)
    hr = torch.rand(1, 3, 64, 64, generator=g)
    d = hr - y0
    q, e = action_optimal_gate(y0, hr, d)
    assert float(e.mean()) > 0.0, 'energy is zero -> the target is meaningless'
    assert float(q.std()) > 0.0 or float(q.mean()) > 0.9, 'q_opt has no content'
    # a zero correction must be detected instead of silently supervised
    _q0, e0 = action_optimal_gate(y0, hr, torch.zeros_like(d))
    assert float(e0.max()) == 0.0


def test_g1_counterfactual_equality():
    """Neutralising the one declared difference must make the arms bit-identical.

    This is what V3-A.3 needed: C1 with zero action channels must equal C0.
    """
    from model.V3A3Verifier import V3A3Refiner, build_shared_init
    s0, s1 = build_shared_init(seed=42)
    c0 = V3A3Refiner(use_action=False).eval()
    c1 = V3A3Refiner(use_action=True).eval()
    c0.load_state_dict(s0)
    c1.load_state_dict(s1)
    assert float(c1.verifier.head0.weight[:, 5 * 32:].abs().max()) == 0.0
    g = torch.Generator().manual_seed(3)
    y0 = torch.rand(1, 3, 64, 64, generator=g)
    low = torch.rand(1, 3, 64, 64, generator=g)
    ref = torch.rand(1, 3, 64, 64, generator=g)
    with torch.no_grad():
        a, _ = c0(y0, ref, low=low)
        b, _ = c1(y0, ref, low=low)
    assert torch.allclose(a, b, atol=1e-6), \
        'arms differ although the declared difference is neutralised'


def test_g6_frozen_module_is_bit_exact_after_a_step():
    from model.V3A3Verifier import V3A3Refiner
    m = V3A3Refiner(use_action=True)
    for p in m.proposal.parameters():
        p.requires_grad_(False)
    before = {k: v.clone() for k, v in m.proposal.state_dict().items()}
    opt = torch.optim.Adam(m.verifier.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(4)
    y0 = torch.rand(1, 3, 64, 64, generator=g)
    low = torch.rand(1, 3, 64, 64, generator=g)
    ref = torch.rand(1, 3, 64, 64, generator=g)
    out, _ = m(y0, ref, low=low)
    opt.zero_grad(set_to_none=True)
    out.abs().mean().backward()
    opt.step()
    assert_frozen(before, m.proposal, where='proposal')


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
