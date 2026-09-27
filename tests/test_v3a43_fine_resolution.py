#!/usr/bin/env python
"""V3-A.4.3 acceptance: fine-resolution saturation sweep.

Plan §13/§15/§16/§21/§23:
  * G16 ⊂ G32 ⊂ G64 ⊂ Block_H4 on the REAL 400x600 / 100x150 geometry
  * the chain survives the hierarchical fallback when round(linspace) is not
    nested (constructive test on a lattice where it is not)
  * G16's edges are bit-identical to the frozen V3-A.4.2 definition
  * continuous MSE is non-increasing along the ladder
  * the legacy anchor never enters the resolution ladder
  * the V3-A.4.2 reproduction check (G16 / Block_H4 / Legacy / cap16)
  * the lock hard-fails on a moved V3-A.4.2 anchor
plus a §23 end-to-end smoke that also asserts no checkpoint is written.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from v3a42_runtime import (block_index_map, blockwise_action_optimal_gate,  # noqa: E402
                           coarse_edges_from_base)
from v3a43_runtime import (LEVELS, ORACLE_DEF_VERSION, build_level_chain,   # noqa: E402
                           check_v3a42_reproduction, check_worktree,
                           is_refinement,
                           level_geometry_report, linspace_index_edges,
                           mse_chain_violations, nested_index_chain,
                           refine_index_edges, saturation_metrics,
                           verify_v3a43_artifact_lock)
from v3a4_runtime import verify_v3a4_artifact_lock                       # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V42 = '/root/data/experiments/v3a42_blockwise_oracle'
GRIDS = (1, 2, 4, 8, 16, 32, 64)
H, W = 400, 600
BASE = (100, 150)


def _rand(seed=0, h=H, w=W, b=1):
    g = torch.Generator().manual_seed(seed)
    y0 = torch.rand(b, 3, h, w, generator=g) * 2 - 1
    hr = torch.rand(b, 3, h, w, generator=g) * 2 - 1
    d = torch.rand(b, 3, h, w, generator=g) * 2 - 1
    return y0, hr, d


def _expect_fail(fn, what):
    try:
        fn()
    except SystemExit as e:
        assert str(e).strip(), '%s raised an empty message' % what
        return
    raise AssertionError('%s did not hard-fail' % what)


# ── §15 the real-geometry nesting test ──────────────────────────────────────

def test_g16_g32_g64_are_nested_on_real_geometry():
    chain = build_level_chain(H, W, BASE, GRIDS)
    assert chain['all_nested'], chain['containment']
    for c in chain['containment']:
        assert c['y'] and c['x'], c
    _, nby_b, nbx_b = block_index_map(H, W, BASE)
    for level in GRIDS:
        ey, ex = chain['y'][level], chain['x'][level]
        # every level is built from whole base cells: its pixel edges are base
        # cell edges, and the base edge set is the multiples of 4
        assert set(int(v) for v in ey) <= set(int(v) for v in
                                              coarse_edges_from_base(H, 100, 100))
        assert set(int(v) for v in ex) <= set(int(v) for v in
                                              coarse_edges_from_base(W, 150, 150))
        assert ey[0] == 0 and ey[-1] == H and ex[0] == 0 and ex[-1] == W
        assert len(ey) - 1 <= level and len(ex) - 1 <= level
    # actual block counts are NOT nominally G x G (100 and 150 are not
    # divisible by 16/32/64) -- that is exactly why they are recorded
    geo = level_geometry_report(H, W, BASE, chain)
    assert geo['levels']['16']['nby'] == 16 and geo['levels']['16']['nbx'] == 16
    assert geo['levels']['32']['nby'] == 32
    assert geo['levels']['64']['nby'] == 64
    assert geo['scheme']['y'] == 'linspace_nested'
    assert geo['scheme']['x'] == 'linspace_nested'
    assert geo['levels']['16']['block_h'] in (24, 28)


def test_level0_is_bit_identical_to_the_frozen_v3a42_definition():
    """G16 must keep the V3-A.4.2 edges byte-for-byte, or the reproduction
    anchor would compare two different partitions."""
    chain = build_level_chain(H, W, BASE, GRIDS)
    np.testing.assert_array_equal(chain['y'][16], coarse_edges_from_base(H, 100, 16))
    np.testing.assert_array_equal(chain['x'][16], coarse_edges_from_base(W, 150, 16))
    # and the oracle must give the same answer through both call styles
    y0, hr, D = _rand(1, h=64, w=96)
    base = (16, 24)
    c = build_level_chain(64, 96, base, (4, 8, 16))
    a = blockwise_action_optimal_gate(y0, hr, D, 8, base_grid=base)
    b = blockwise_action_optimal_gate(y0, hr, D, 8, edges=(c['y'][8], c['x'][8]))
    assert torch.equal(a['q_grid'], b['q_grid']), 'edges= path diverged'
    assert torch.equal(a['Y_oracle'], b['Y_oracle'])


def test_hierarchical_fallback_keeps_containment():
    """Constructive test: on a lattice where round(linspace) is NOT nested, the
    chain must still be strictly nested (plan §16)."""
    n_base = 7
    levels = (2, 3, 5, 7)
    # demonstrate the raw linspace rule really is not nested here
    lin = {g: linspace_index_edges(n_base, g) for g in levels}
    broken = any(not is_refinement(lin[a], lin[b])
                 for a, b in zip(levels[:-1], levels[1:]))
    assert broken, [list(v) for v in lin.values()]
    chain, notes = nested_index_chain(n_base, levels)
    for a, b in zip(levels[:-1], levels[1:]):
        assert is_refinement(chain[a], chain[b]), (a, b, chain[a], chain[b])
    assert any(n.startswith('hierarchical') for n in notes.values()), notes
    # every level keeps the endpoints and never produces an empty block
    for g in levels:
        assert chain[g][0] == 0 and chain[g][-1] == n_base
        assert np.all(np.diff(chain[g]) >= 1)
    # a single-cell block cannot be split; refinement must not delete it
    tiny = refine_index_edges(np.array([0, 1, 3]), 2)
    assert is_refinement(np.array([0, 1, 3]), tiny), tiny


def test_continuous_mse_is_non_increasing_along_the_ladder():
    y0, hr, D = _rand(2, h=128, w=128)
    base = (32, 32)
    chain = build_level_chain(128, 128, base, (4, 8, 16))
    mses = {}
    for g in (4, 8, 16):
        o = blockwise_action_optimal_gate(y0, hr, D, g,
                                          edges=(chain['y'][g], chain['x'][g]))
        mses['G%d' % g] = float((o['Y_oracle'] - hr).pow(2).mean())
    ob = blockwise_action_optimal_gate(y0, hr, D, base)
    mses['Block_H4'] = float((ob['Y_oracle'] - hr).pow(2).mean())
    mses['R1'] = float((y0 + D - hr).pow(2).mean())
    # NOTE: on 128x128 with a 32x32 base grid, G32 IS Block_H4 (same partition),
    # so the ladder stops at G16 -- otherwise the last step is not strict.
    ladder = ['R1', 'G4', 'G8', 'G16', 'Block_H4']
    assert mse_chain_violations(mses, ladder) == [], mses
    # and it is a strict ladder on random data, not an accidental flat line
    assert mses['Block_H4'] < mses['G16'] < mses['G8'] < mses['G4']


# ── §10-§12 saturation metrics ──────────────────────────────────────────────

def test_saturation_metrics_bands_and_ladder():
    m = {'R1': 20.0, 'G1': 20.01, 'G2': 20.04, 'G4': 20.06, 'G8': 20.08,
         'G16': 20.1, 'G32': 20.13, 'G64': 20.16,
         'Block_H4': 20.90, 'Spatial_H4_legacy': 20.5}
    sat = saturation_metrics(m, LEVELS, block_arm='Block_H4')
    assert list(sat) == ['G16->G32', 'G32->G64', 'G64->Block_H4'], list(sat)
    assert abs(sat['G16->G32']['delta'] - 0.03) < 1e-12
    assert sat['G16->G32']['band'] == 'marginal'
    assert sat['G32->G64']['band'] == 'marginal'
    assert sat['G64->Block_H4']['band'] == 'structural'
    # the legacy anchor must never appear as a resolution step
    assert not any('Legacy' in k for k in sat)
    full = saturation_metrics(m, GRIDS, block_arm='Block_H4', start_arm='R1')
    assert list(full)[0] == 'R1->G1'
    assert list(full)[-1] == 'G64->Block_H4'
    assert not any('Legacy' in k for k in full)
    # band boundaries
    assert saturation_metrics({'G16': 0.0, 'G32': 0.02, 'G64': 0.05,
                               'Block_H4': 0.05}, (16, 32, 64))[
        'G16->G32']['band'] == 'marginal'
    assert saturation_metrics({'G16': 0.0, 'G32': 0.0199, 'G64': 0.05,
                               'Block_H4': 0.05}, (16, 32, 64))[
        'G16->G32']['band'] == 'negligible'


# ── §21 reproduction against V3-A.4.2 ───────────────────────────────────────

def _repro_case(d_g16=0.0, d_block=0.0, d_legacy=0.0, d_cap=0.0):
    got = dict(mean_psnr={'G16': 20.0 + d_g16, 'Block_H4': 20.3,
                          'Spatial_H4_legacy': 20.29},
               capture={'G16': dict(capture=0.867 + d_cap)})
    ref = dict(mean_psnr={'G16': 20.0, 'Block_H4': 20.3 + d_block,
                          'Spatial_H4_legacy': 20.29 + d_legacy},
               capture={'G16': dict(capture=0.867)})
    return got, ref


def test_v3a42_reproduction_check():
    got, ref = _repro_case()
    row = check_v3a42_reproduction(got, ref)
    assert row['ok'] and row['mismatches'] == [], row
    assert set(row['psnr']) == {'G16', 'Block_H4', 'Spatial_H4_legacy'}
    assert row['capture_G16']['d'] == 0.0
    # each anchor is checked independently and named
    for kwargs, name in ((dict(d_g16=5e-3), 'psnr_G16'),
                         (dict(d_block=5e-3), 'psnr_Block_H4'),
                         (dict(d_legacy=5e-3), 'psnr_Spatial_H4_legacy'),
                         (dict(d_cap=5e-3), 'capture_G16')):
        row = check_v3a42_reproduction(*_repro_case(**kwargs))
        assert not row['ok'] and row['mismatches'] == [name], (kwargs, row)
    # tolerances are the plan's (§21): 2e-3 dB / 1e-3 capture
    got, ref = _repro_case(d_g16=2.5e-3, d_cap=1.5e-3)
    row = check_v3a42_reproduction(got, ref)
    assert not row['ok'] and row['mismatches'] == ['psnr_G16', 'capture_G16']
    got, ref = _repro_case(d_g16=1.9e-3, d_cap=0.9e-3)
    assert check_v3a42_reproduction(got, ref)['ok']


def test_reproduction_anchor_must_be_usable():
    got, ref = _repro_case()
    got['capture']['G16'] = dict(capture=None)
    _expect_fail(lambda: check_v3a42_reproduction(got, ref),
                 'undefined cap16 in our own run')


# ── P1: provenance guards ───────────────────────────────────────────────────

def test_formal_run_refuses_a_dirty_tree():
    """A lock pins repo_commit, so a dirty tree must not produce formal numbers.
    Deterministic: the decision is a pure function of (limit, head, dirty)."""
    head, dirty = 'a' * 40, ['scripts/x.py', 'v3a43_runtime.py']
    _expect_fail(lambda: check_worktree(0, head=head, dirty=dirty),
                 'formal run on a dirty tree')
    # a smoke run is allowed, but must say so out loud
    h, d, warn = check_worktree(2, head=head, dirty=dirty)
    assert h == head and d == dirty and warn and 'smoke run' in warn
    assert 'NOT attributable' in warn
    # a clean tree is silent and never raises, formal or not
    for limit in (0, 2, 64):
        assert check_worktree(limit, head=head, dirty=[]) == (head, [], None)


# ── §20 lock ────────────────────────────────────────────────────────────────

def _write_lock(tmp, commit=None, grids='1,2,4,8,16,32,64',
                levels='16,32,64'):
    v4 = verify_v3a4_artifact_lock(V4, SRC)
    if commit is None:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                         cwd=ROOT).decode().strip()
    chain = build_level_chain(H, W, BASE, [int(g) for g in grids.split(',')])
    lock = dict(
        repo_commit=commit,
        proposal_sha256=v4['proposal_sha256'],
        cache_metadata_sha256=v4['cache_metadata_sha256'],
        manifest_sha256=v4['manifest_sha256'],
        split_sha256=v4['split_sha256'],
        mismatch_train_sha256=v4['mismatch_train_sha256'],
        mismatch_dev_sha256=v4['mismatch_dev_sha256'],
        energy_stats_sha256=v4['energy_stats_sha256'],
        grids=grids, resolution_levels=levels, block_h4_factor=4,
        states='correct+true_dark_g0.5+mismatch',
        oracle_def_version=ORACLE_DEF_VERSION,
        nesting_scheme='%s/%s' % (chain['scheme_y'], chain['scheme_x']),
        nesting_all_nested=bool(chain['all_nested']),
        v3a42_summary_path=os.path.join(V42, 'oracle', 'summary.json'),
        v3a42_summary_sha256=_sha(os.path.join(V42, 'oracle', 'summary.json')))
    path = os.path.join(tmp, 'artifact_lock.json')
    json.dump(lock, open(path, 'w', encoding='utf-8'), indent=2, sort_keys=True)
    return path, lock


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def test_lock_hard_fails_on_a_moved_v3a42_anchor():
    tmp = tempfile.mkdtemp(prefix='v3a43_lock_')
    try:
        _expect_fail(lambda: verify_v3a43_artifact_lock(tmp, SRC, v4_root=V4),
                     'missing lock')
        path, lock = _write_lock(tmp)
        assert verify_v3a43_artifact_lock(tmp, SRC, v4_root=V4)['grids'] == lock['grids']
        # the reproduction anchor is an input: a moved summary must refuse
        bad = dict(lock, v3a42_summary_sha256='0' * 64)
        json.dump(bad, open(path, 'w', encoding='utf-8'))
        _expect_fail(lambda: verify_v3a43_artifact_lock(tmp, SRC, v4_root=V4),
                     'tampered v3a42 summary')
        # a missing anchor file must refuse too
        bad = dict(lock, v3a42_summary_path=os.path.join(tmp, 'nope.json'))
        json.dump(bad, open(path, 'w', encoding='utf-8'))
        _expect_fail(lambda: verify_v3a43_artifact_lock(tmp, SRC, v4_root=V4),
                     'missing v3a42 summary')
        # stale commit / changed oracle definition
        bad = dict(lock, repo_commit='deadbeef' * 5)
        json.dump(bad, open(path, 'w', encoding='utf-8'))
        _expect_fail(lambda: verify_v3a43_artifact_lock(tmp, SRC, v4_root=V4),
                     'stale commit')
        bad = dict(lock, oracle_def_version='v3a43:something-else')
        json.dump(bad, open(path, 'w', encoding='utf-8'))
        _expect_fail(lambda: verify_v3a43_artifact_lock(tmp, SRC, v4_root=V4),
                     'changed oracle definition')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_setup_script_locks_the_real_nesting_scheme():
    tmp = tempfile.mkdtemp(prefix='v3a43_setup_')
    try:
        setup = os.path.join(ROOT, 'scripts', 'setup_v3a43_fine_resolution.py')
        out = subprocess.check_output([sys.executable, setup, '--root', tmp],
                                      stderr=subprocess.STDOUT).decode()
        assert 'linspace_nested' in out, out[-800:]
        lock = json.load(open(os.path.join(tmp, 'artifact_lock.json'),
                              encoding='utf-8'))
        assert lock['nesting_all_nested'] is True
        assert lock['block_h4_factor'] == 4
        assert lock['resolution_levels'] == '16,32,64'
        assert lock['grids'] == '1,2,4,8,16,32,64'
        assert lock['nesting_levels']['16'] == dict(nby=16, nbx=16,
                                                    note_y='linspace_on_base_lattice',
                                                    note_x='linspace_on_base_lattice')
        assert lock['v3a42_summary_sha256'] == _sha(
            os.path.join(V42, 'oracle', 'summary.json'))
        # levels must be inside grids, and G32/G64 optional
        for argv, needle in ((['--levels', '16,128'], 'subset'),
                             (['--levels', '32,64'], 'contain 16')):
            try:
                subprocess.check_output([sys.executable, setup, '--root', tmp]
                                        + argv, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                assert needle.encode() in (e.output or b''), (argv, e.output[-400:])
            else:
                raise AssertionError('setup accepted %r' % (argv,))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── §23 smoke ───────────────────────────────────────────────────────────────

def test_smoke_reaches_aggregation_and_repro_path():
    tmp = tempfile.mkdtemp(prefix='v3a43_smoke_')
    try:
        _path, lock = _write_lock(tmp)
        script = os.path.join(ROOT, 'scripts',
                              'diagnose_v3a43_fine_resolution.py')
        out = subprocess.check_output(
            [sys.executable, '-W', 'ignore', script, '--root', tmp,
             '--splits', 'dev', '--limit', '2'], stderr=subprocess.STDOUT).decode()
        assert 'Traceback' not in out and 'KeyError' not in out, out[-3000:]
        assert 'dev done' in out and 'continuous-MSE chain verified' in out, out[-2000:]
        assert '[repro] --limit 2: skipped' in out, out[-2000:]
        o = os.path.join(tmp, 'oracle')
        for fn in ('summary.json', 'per_image.csv', 'fine_capture_curve.csv',
                   'nesting.json'):
            assert os.path.isfile(os.path.join(o, fn)), fn
        s = json.load(open(os.path.join(o, 'summary.json'), encoding='utf-8'))
        assert s['protocol']['levels'] == [16, 32, 64]
        assert s['protocol']['grids'] == list(GRIDS)
        assert s['protocol']['primary_denominator'] == 'Block_H4'
        assert s['protocol']['grid_scheme'] == 'verified_nested_from_block_h4'
        # P1: the anchor actually used is the one the lock pinned, and the real
        # worktree state is recorded instead of a hard-coded zero
        assert s['protocol']['repro_anchor'] == lock['v3a42_summary_path']
        assert s['protocol']['repro_anchor_source'] == 'artifact_lock.json'
        assert s['protocol']['git_head'] == lock['repo_commit']
        assert len(s['protocol']['git_dirty']) == \
            min(s['protocol']['git_dirty_count'], 20)
        # The warning line must appear IFF the subprocess really was dirty. This
        # test runs from whatever tree state the caller has -- the runner is
        # expected to be CLEAN (commit first), so it must never assume dirtiness.
        # The dirty-warning path itself is pinned deterministically by
        # test_formal_run_refuses_a_dirty_tree().
        dirty = s['protocol']['git_dirty_count'] > 0
        assert ('git WARNING' in out) == dirty, out[-1500:]
        assert s['nesting_summary']['all_nested'] is True
        assert s['nesting_summary']['mse_chain_violations'] == 0
        for arm in ('R1', 'G16', 'G32', 'G64', 'Block_H4', 'Spatial_H4_legacy'):
            assert arm in s['splits']['dev']['correct']['mean_psnr'], arm
        assert list(s['splits']['dev']['correct']['saturation']) == \
            ['G16->G32', 'G32->G64', 'G64->Block_H4']
        rows = list(__import__('csv').DictReader(
            open(os.path.join(o, 'per_image.csv'), encoding='utf-8')))
        assert len(rows) == 2 * 3, len(rows)
        for col in ('G16', 'G32', 'G64', 'Block_H4', 'Spatial_H4_legacy',
                    'delta_G16_vs_R1', 'capture_G16', 'capture_G32', 'capture_G64',
                    'mse_G16', 'mse_Block_H4'):
            assert col in rows[0], col
        # nesting.json must record the real geometry + actual block counts
        n = json.load(open(os.path.join(o, 'nesting.json'), encoding='utf-8'))
        assert n['base_grid'] == [100, 150]
        assert n['levels']['64']['nby'] == 64 and n['levels']['64']['nbx'] == 64
        assert n['scheme'] == dict(y='linspace_nested', x='linspace_nested')
        assert n['mse_chain']['n_violations'] == 0
        for dirpath, _d, files in os.walk(tmp):
            for fn in files:
                assert not fn.endswith(('.pt', '.pth', '.ckpt')), \
                    'checkpoint written: %s' % os.path.join(dirpath, fn)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
