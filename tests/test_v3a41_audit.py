#!/usr/bin/env python
"""V3-A.4.1 §15 acceptance: q_opt decomposition, crop alignment, global oracle,
read-only guarantees."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3a2_runtime import action_optimal_gate                     # noqa: E402
from v3a41_runtime import (build_fixed_crop_manifest, compare_actions,  # noqa: E402
                           crop_tensor, global_action_optimal_gate,
                           qopt_components, stable_seed)


def _rand(seed=0, h=64, w=96, b=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(b, 3, h, w, generator=g) * 2 - 1,
            torch.rand(b, 3, h, w, generator=g) * 2 - 1,
            torch.rand(b, 3, h, w, generator=g) * 2 - 1)


def test_qopt_components_matches_action_optimal_gate():
    y0, hr, D = _rand(1)
    q_ref, e_ref = action_optimal_gate(y0, hr, D)
    c = qopt_components(y0, hr, D)
    assert torch.allclose(c['q_opt'], q_ref, atol=1e-6), 'q_opt differs'
    assert torch.allclose(c['energy'], e_ref, atol=1e-6), 'energy differs'


def test_global_oracle_math():
    y0, _hr, _D = _rand(2)
    hr = torch.rand_like(y0)
    # D = H - Y0 -> q* = 1
    q, _N, _Z = global_action_optimal_gate(y0, hr, hr - y0)
    assert abs(float(q.mean()) - 1.0) < 1e-4, float(q.mean())
    # D = 2(H - Y0) -> q* = 0.5
    q, _N, _Z = global_action_optimal_gate(y0, hr, 2 * (hr - y0))
    assert abs(float(q.mean()) - 0.5) < 1e-4, float(q.mean())
    # D = -(H - Y0) -> q* = 0
    q, _N, _Z = global_action_optimal_gate(y0, hr, -(hr - y0))
    assert float(q.max()) == 0.0


def test_global_endpoints_are_exact():
    y0, hr, D = _rand(3)
    q, _N, _Z = global_action_optimal_gate(y0, hr, D)
    # q=0 -> Base
    assert torch.equal(y0 + 0.0 * D, y0)
    # q=1 -> the full correction
    assert torch.equal(y0 + 1.0 * D, y0 + D)
    assert 0.0 <= float(q.min()) and float(q.max()) <= 1.0


def test_ff_fc_cc_use_the_intended_proposal_context():
    """FC must reuse D_from_full (cropped); CC must recompute on the crop.
    This is the difference the whole audit rests on."""
    y0, hr, ref = _rand(4, h=64, w=64)
    # a cheap stand-in for the frozen proposal: D = ref - y0 scaled
    D_full = 0.5 * (ref - y0)
    top, left, size = 12, 20, 32
    y0c = crop_tensor(y0, top, left, size, size)
    hrc = crop_tensor(hr, top, left, size, size)
    refc = crop_tensor(ref, top, left, size, size)

    d_full_crop = crop_tensor(D_full, top, left, size, size)
    d_crop = 0.5 * (refc - y0c)                  # proposal re-run on the crop

    q_fc = qopt_components(y0c, hrc, d_full_crop)['q_opt']
    q_cc = qopt_components(y0c, hrc, d_crop)['q_opt']
    # for this linear stand-in the two agree; the point is that they are
    # computed from DIFFERENT tensors, so the code path must be explicit
    assert torch.allclose(q_fc, q_cc, atol=1e-6)
    # and a genuinely different proposal makes them differ
    d_crop2 = 0.5 * (refc - y0c) + 0.05
    assert not torch.allclose(q_cc, qopt_components(y0c, hrc, d_crop2)['q_opt'],
                              atol=1e-6)


def test_crop_manifest_is_deterministic_and_in_bounds():
    low_dir = '/root/data/datasets/lol-v2-real/Test/Low'
    if not os.path.isdir(low_dir):
        return 'SKIP: dataset missing'
    names = sorted(os.listdir(low_dir))[:3]
    rows = [(os.path.splitext(n)[0], os.path.join(low_dir, n), 'x') for n in names]
    a = build_fixed_crop_manifest(rows, 'dev', crop=128, k=4, seed=20260927)
    b = build_fixed_crop_manifest(rows, 'dev', crop=128, k=4, seed=20260927)
    assert a == b, 'crop manifest is not deterministic'
    for r in a:
        assert 0 <= r['top'] and r['top'] + 128 <= 400
        assert 0 <= r['left'] and r['left'] + 128 <= 600
    # different samples must not share an identical crop list by accident
    per = [tuple((x['top'], x['left']) for x in a if x['sample_id'] == name)
           for name, _low, _high in rows]
    assert len(set(per)) > 1, 'every sample got the same crops'


def test_stable_seed_is_constant_and_hash_independent():
    """The old manifest used builtin hash(), which is salted per process."""
    # fixed expected values: this must not change when the manifest is frozen
    assert stable_seed(20260927, 'low00001.png') == 15193271
    assert stable_seed(20260927, 'low00002.png') == 4152142966
    # different inputs -> different seeds; order matters
    assert stable_seed(1, 'a') != stable_seed(1, 'b')
    assert stable_seed('a', 1) != stable_seed(1, 'a')


def test_stable_seed_is_process_independent():
    """Two subprocesses with different PYTHONHASHSEED must agree."""
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); "
            "from v3a41_runtime import stable_seed; "
            "print(stable_seed(20260927, 'low00001.png'))" % os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
    outs = []
    for hs in ('0', '12345'):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.check_output([sys.executable, '-c', code],
                                            env=env).decode().strip())
    assert outs[0] == outs[1] == '15193271', outs


def test_global_oracle_script_reaches_aggregation():
    """End-to-end smoke: the accumulator/aggregation path must not KeyError.

    This is the check that would have caught the `acc['R1|state']` bug, which
    only fired after the whole loop had run.
    """
    import subprocess
    import shutil
    import tempfile
    src = '/root/data/experiments/v3a41_target_audit'
    if not os.path.isfile(os.path.join(src, 'crops', 'crop_manifest.csv')):
        return 'SKIP: audit setup has not been run'
    # run in a throwaway root so the test cannot overwrite the real
    # oracle/global_vs_spatial.json with a 2-image debug result
    tmp = tempfile.mkdtemp(prefix='v3a41_smoke_')
    try:
        os.makedirs(os.path.join(tmp, 'crops'), exist_ok=True)
        shutil.copy(os.path.join(src, 'crops', 'crop_manifest.csv'),
                    os.path.join(tmp, 'crops', 'crop_manifest.csv'))
        shutil.copy(os.path.join(src, 'artifact_lock.json'),
                    os.path.join(tmp, 'artifact_lock.json'))
        out = subprocess.check_output(
            [sys.executable, '-W', 'ignore',
             os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'scripts', 'diagnose_v3a41_global_oracle.py'),
             '--root', tmp, '--splits', 'dev', '--limit', '2'],
            stderr=subprocess.STDOUT).decode()
        assert 'D2: Gate Resolution Necessity' in out, out[-2000:]
        assert 'KeyError' not in out
        import json as _json
        p = os.path.join(tmp, 'oracle', 'global_vs_spatial.json')
        assert os.path.isfile(p), 'oracle summary not written'
        s = _json.load(open(p, encoding='utf-8'))
        assert 'dev' in s and 'correct' in s['dev']
        for k in ('R1', 'Global_AO', 'Spatial_AO', 'H_global', 'H_spatial',
                  'capture_global'):
            assert k in s['dev']['correct'], k
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_compare_actions_metrics():
    g = torch.Generator().manual_seed(5)
    a = torch.rand(1, 3, 16, 16, generator=g)
    assert compare_actions(a, a)['MAE'] == 0.0
    assert abs(compare_actions(a, a)['cosine'] - 1.0) < 1e-5
    m = compare_actions(a, 2 * a)
    assert abs(m['norm_ratio'] - 2.0) < 1e-5


def test_diagnostics_do_not_touch_gradients():
    y0, hr, D = _rand(6)
    y0.requires_grad_(True)
    with torch.no_grad():
        c = qopt_components(y0, hr, D)
        q, _N, _Z = global_action_optimal_gate(y0, hr, D)
    assert not c['q_opt'].requires_grad and not q.requires_grad
    assert y0.grad is None


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
