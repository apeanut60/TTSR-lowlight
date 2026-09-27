#!/usr/bin/env python
"""V3-A.4.2 §22 acceptance: the true blockwise gate-resolution oracle.

Covers every box in §22:
  * grid=1 ≡ V3-A.4.1 Global-AO
  * Spatial-H/4 ≡ V3-A.4.1 Spatial-AO
  * D=H-Y0 -> q=1 ; D=2(H-Y0) -> q=0.5 ; D=-(H-Y0) -> q=0   (all grids)
  * synthetic 2x2 quadrant target recovered by G=2
  * block partition: no gap, no overlap, non-divisible sizes
  * the coarse oracle is NOT AvgPool(q_spatial)
  * parameters read-only, no checkpoint written
  * the artifact lock hard-fails at the entry point
plus §23: --splits dev --limit 2 reaches aggregation and writes outputs.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from v3a41_runtime import (global_action_optimal_gate,          # noqa: E402
                           qopt_components)
from v3a42_runtime import (ORACLE_DEF_VERSION, block_edges,     # noqa: E402
                           block_index_map, blockwise_action_optimal_gate,
                           global_action_optimal_gate_reference,
                           spatial_action_optimal_gate, spatial_oracle_q4,
                           verify_v3a42_artifact_lock)
from v3a4_runtime import verify_v3a4_artifact_lock               # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
GRIDS = (1, 2, 4, 8, 16)


def _rand(seed=0, h=64, w=96, b=1):
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


# ── §8.1 global equivalence ─────────────────────────────────────────────────

def test_grid1_equals_v3a41_global_ao():
    y0, hr, D = _rand(1)
    o = blockwise_action_optimal_gate(y0, hr, D, 1)
    q41, _N, _Z = global_action_optimal_gate(y0, hr, D)
    q_ref, N, Z = global_action_optimal_gate_reference(y0, hr, D)
    assert o['nby'] == 1 and o['nbx'] == 1, (o['nby'], o['nbx'])
    assert torch.allclose(o['q_grid'], q41, atol=1e-6), 'G1 != V3-A.4.1 global'
    assert torch.allclose(o['q_grid'], q_ref, atol=1e-7)
    assert o['N_grid'].shape == o['Z_grid'].shape == (1, 1)
    # the reconstructed output must be identical, not merely close in q
    out41 = y0 + q41 * D
    assert torch.allclose(o['Y_oracle'], out41, atol=1e-6)
    for grid in GRIDS:
        # q_full must be the exact nearest-neighbour expansion of q_grid
        og = blockwise_action_optimal_gate(y0, hr, D, grid)
        h, w = y0.shape[-2:]
        idx, nby, nbx = block_index_map(h, w, grid)
        assert (nby, nbx) == (og['nby'], og['nbx'])
        expect = og['q_grid'][0][torch.as_tensor(idx)].reshape(h, w)
        assert torch.equal(og['q_full'][0, 0], expect), grid


def test_spatial_equals_v3a41_spatial_ao():
    y0, hr, D = _rand(2)
    c = qopt_components(y0, hr, D)
    ref = F.interpolate(c['q_opt'], size=y0.shape[-2:], mode='bilinear',
                        align_corners=False)
    q = spatial_action_optimal_gate(y0, hr, D)
    assert torch.allclose(q, ref, atol=1e-7), 'Spatial-H/4 != V3-A.4.1'
    assert torch.allclose(spatial_oracle_q4(y0, hr, D), c['q_opt'], atol=1e-7)
    # H/4 really is H/4 (and never below 1 pixel)
    q4 = spatial_oracle_q4(y0, hr, D)
    assert q4.shape[-2:] == (y0.shape[-2] // 4, y0.shape[-1] // 4)


# ── §8.2-§8.4 algebra, at every grid ────────────────────────────────────────

def test_perfect_correction_is_one_at_every_grid():
    y0, hr, _D = _rand(3)
    for grid in GRIDS:
        o = blockwise_action_optimal_gate(y0, hr, hr - y0, grid)
        assert abs(float(o['q_grid'].mean()) - 1.0) < 1e-4, (grid, float(o['q_grid'].mean()))
        assert torch.allclose(o['Y_oracle'], hr, atol=1e-3), grid


def test_double_correction_is_half_at_every_grid():
    y0, hr, _D = _rand(4)
    for grid in GRIDS:
        o = blockwise_action_optimal_gate(y0, hr, 2 * (hr - y0), grid)
        assert abs(float(o['q_grid'].mean()) - 0.5) < 1e-4, (grid, float(o['q_grid'].mean()))


def test_wrong_direction_is_zero_at_every_grid():
    y0, hr, _D = _rand(5)
    for grid in GRIDS:
        o = blockwise_action_optimal_gate(y0, hr, -(hr - y0), grid)
        assert float(o['q_grid'].max()) == 0.0, grid
        assert torch.equal(o['Y_oracle'], y0), grid


def test_gate_is_always_in_unit_interval():
    y0, hr, D = _rand(6)
    for grid in GRIDS:
        o = blockwise_action_optimal_gate(y0, hr, D, grid)
        assert 0.0 <= float(o['q_grid'].min()) and float(o['q_grid'].max()) <= 1.0
        assert 0.0 <= float(o['q_full'].min()) and float(o['q_full'].max()) <= 1.0


# ── §8.5 synthetic regional recovery ────────────────────────────────────────

def _quadrant_case(h=40, w=60):
    """TL q=1, TR q=0, BL q=0.5, BR q=0 with T = H-Y0 = 1 everywhere.

    With T constant, q*_raw = T/D, so a constant D per quadrant selects the
    quadrant's optimum exactly: D=1 -> 1, D=-1 -> 0, D=2 -> 0.5, D=-2 -> 0.
    """
    y0 = torch.zeros(1, 3, h, w)
    hr = torch.ones(1, 3, h, w)
    D = torch.zeros_like(y0)
    D[:, :, :h // 2, :w // 2] = 1.0
    D[:, :, :h // 2, w // 2:] = -1.0
    D[:, :, h // 2:, :w // 2] = 2.0
    D[:, :, h // 2:, w // 2:] = -2.0
    return y0, hr, D


def test_synthetic_quadrants_recovered_by_grid2():
    y0, hr, D = _quadrant_case()
    o = blockwise_action_optimal_gate(y0, hr, D, 2)
    assert (o['nby'], o['nbx']) == (2, 2)
    want = torch.tensor([[1.0, 0.0], [0.5, 0.0]])
    got = o['q_grid'][0].reshape(2, 2)
    assert torch.allclose(got, want, atol=1e-6), got
    # nearest expansion: whole quadrants share one value
    qf = o['q_full'][0, 0]
    assert float(qf[:20, :30].min()) == float(qf[:20, :30].max()) == 1.0
    assert float(qf[20:, :30].min()) == float(qf[20:, :30].max()) == 0.5
    assert float(qf[20:, 30:].max()) == 0.0
    # and G=1 must NOT be able to do this
    one = blockwise_action_optimal_gate(y0, hr, D, 1)
    assert float(one['q_grid'][0, 0]) < 1.0


# ── §6 partition rules ──────────────────────────────────────────────────────

def test_block_edges_are_gap_free_and_inclusive():
    for length in (37, 53, 5, 7, 400, 600, 17):
        for grid in (1, 2, 3, 4, 8, 16):
            e = block_edges(length, grid)
            assert e[0] == 0 and e[-1] == length, (length, grid, e)
            assert np.all(np.diff(e) >= 1), (length, grid, e)
            assert len(e) <= grid + 1
    assert list(block_edges(37, 4)) == [0, 9, 18, 28, 37]
    assert list(block_edges(1, 1)) == [0, 1]
    assert list(block_edges(400, 1)) == [0, 400]
    # a grid finer than the axis collapses instead of producing empty blocks
    assert len(block_edges(5, 8)) == 6


def test_partition_covers_every_pixel_exactly_once():
    for (h, w) in ((64, 96), (37, 53), (40, 60), (5, 7), (17, 19), (400, 600)):
        for grid in (1, 2, 4, 8, 16):
            idx, nby, nbx = block_index_map(h, w, grid)
            assert idx.shape == (h * w,)
            assert sorted(set(idx.tolist())) == list(range(nby * nbx))
            counts = np.bincount(idx, minlength=nby * nbx)
            assert counts.min() >= 1, (h, w, grid)
            assert int(counts.sum()) == h * w, (h, w, grid)
            ey, ex = block_edges(h, grid), block_edges(w, grid)
            m = idx.reshape(h, w)
            for i in range(nby):
                for j in range(nbx):
                    blk = m[ey[i]:ey[i + 1], ex[j]:ex[j + 1]]
                    assert blk.size == counts[i * nbx + j], (h, w, grid, i, j)
                    assert np.all(blk == i * nbx + j), (h, w, grid, i, j)


# ── §4.1 the coarse oracle is not a smoothed spatial gate ───────────────────

def test_coarse_oracle_is_not_avgpool_of_q_spatial():
    """A case where the blockwise optimum and AvgPool(q_spatial) provably differ.

    h=w=32, T = H-Y0 = 1, factor=4 so the H/4 map is 8x8 and one G=2 block is a
    4x4 patch of H/4 cells. D = +0.5 on x 0..7 and -0.5 on x 8..15, so inside
    that block the H/4 oracle clamps two cells to 1 and two to 0.

    Blockwise optimum: N = 8*(+0.5) + 8*(-0.5) = 0 -> q = 0.
    AvgPool(q_spatial) over the same block: (1+1+0+0)/4 = 0.5.
    Different gate -- and the averaged one is strictly worse in MSE, which is
    exactly why §4.1 forbids it as a coarse oracle.
    """
    y0 = torch.zeros(1, 3, 32, 32)
    hr = torch.ones(1, 3, 32, 32)
    D = torch.zeros_like(y0)
    D[:, :, :, :8] = 0.5
    D[:, :, :, 8:16] = -0.5
    o = blockwise_action_optimal_gate(y0, hr, D, 2)
    q_block = o['q_grid'][0].reshape(2, 2)
    q4 = spatial_oracle_q4(y0, hr, D)                 # [1,1,8,8]
    assert float(q4[0, 0, 0, 0]) == 1.0
    assert float(q4[0, 0, 0, 3]) == 0.0
    # AvgPool(q_spatial): one G=2 block is 4x4 H/4 cells -> kernel/stride 4
    q_avg = F.avg_pool2d(q4, kernel_size=4, stride=4)           # [1,1,2,2]
    assert float(q_avg[0, 0, 0, 0]) == 0.5, float(q_avg[0, 0, 0, 0])
    assert float(q_block[0, 0]) == 0.0
    assert not torch.allclose(q_block.reshape(1, 1, 2, 2), q_avg)
    # the averaged gate is feasible for the blockwise problem but suboptimal
    q_avg_full = q_avg.repeat_interleave(16, 2).repeat_interleave(16, 3)
    mse_block = float((hr - o['Y_oracle']).pow(2).mean())
    mse_avg = float((hr - (y0 + q_avg_full * D)).pow(2).mean())
    assert mse_block < mse_avg, (mse_block, mse_avg)


def test_runtime_and_script_never_average_the_spatial_gate():
    """Static guard: no avg-pool / adaptive-pool anywhere on the oracle path."""
    banned = ('avg_pool', 'AvgPool2d', 'adaptive_avg_pool')
    for rel in ('v3a42_runtime.py', 'scripts/diagnose_v3a42_blockwise_oracle.py'):
        src = open(os.path.join(ROOT, rel), encoding='utf-8').read()
        for token in banned:
            assert token not in src, '%s uses %s' % (rel, token)


# ── read-only guarantees ────────────────────────────────────────────────────

def test_oracle_does_not_touch_gradients():
    y0, hr, D = _rand(7)
    y0.requires_grad_(True)
    hr.requires_grad_(True)
    D.requires_grad_(True)
    with torch.no_grad():
        o = blockwise_action_optimal_gate(y0, hr, D, 4)
        s = spatial_action_optimal_gate(y0, hr, D)
    for t in (o['q_grid'], o['q_full'], o['Y_oracle'], s):
        assert not t.requires_grad
    assert y0.grad is None and hr.grad is None and D.grad is None


def test_block_index_map_returns_a_private_copy():
    a, _nb, _nx = block_index_map(16, 16, 2)
    a[:] = -1                        # must not corrupt the cached map
    b, _nb, _nx = block_index_map(16, 16, 2)
    assert b.min() == 0 and b.max() == 3


# ── §20/§22 artifact lock ───────────────────────────────────────────────────

def _write_lock(tmp, grids='1,2,4,8,16', commit=None):
    v4 = verify_v3a4_artifact_lock(V4, SRC)
    if commit is None:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                         cwd=ROOT).decode().strip()
    lock = dict(
        repo_commit=commit,
        proposal_sha256=v4['proposal_sha256'],
        cache_metadata_sha256=v4['cache_metadata_sha256'],
        manifest_sha256=v4['manifest_sha256'],
        split_sha256=v4['split_sha256'],
        mismatch_train_sha256=v4['mismatch_train_sha256'],
        mismatch_dev_sha256=v4['mismatch_dev_sha256'],
        energy_stats_sha256=v4['energy_stats_sha256'],
        grids=grids, states='correct+true_dark_g0.5+mismatch',
        oracle_def_version=ORACLE_DEF_VERSION)
    path = os.path.join(tmp, 'artifact_lock.json')
    json.dump(lock, open(path, 'w', encoding='utf-8'), indent=2, sort_keys=True)
    return path, lock


def test_artifact_lock_hard_checks():
    tmp = tempfile.mkdtemp(prefix='v3a42_lock_')
    try:
        _expect_fail(lambda: verify_v3a42_artifact_lock(tmp, SRC, v4_root=V4),
                     'missing lock')
        path, lock = _write_lock(tmp)
        assert verify_v3a42_artifact_lock(tmp, SRC, v4_root=V4)['grids'] == lock['grids']

        bad = dict(lock, proposal_sha256='0' * 64)
        json.dump(bad, open(path, 'w', encoding='utf-8'))
        _expect_fail(lambda: verify_v3a42_artifact_lock(tmp, SRC, v4_root=V4),
                     'tampered proposal sha')

        bad = dict(lock, split_sha256='0' * 64)
        json.dump(bad, open(path, 'w', encoding='utf-8'))
        _expect_fail(lambda: verify_v3a42_artifact_lock(tmp, SRC, v4_root=V4),
                     'tampered split sha')

        bad = dict(lock, oracle_def_version='v3a42:something-else')
        json.dump(bad, open(path, 'w', encoding='utf-8'))
        _expect_fail(lambda: verify_v3a42_artifact_lock(tmp, SRC, v4_root=V4),
                     'changed oracle definition')

        bad = dict(lock, repo_commit='deadbeef' * 5)
        json.dump(bad, open(path, 'w', encoding='utf-8'))
        _expect_fail(lambda: verify_v3a42_artifact_lock(tmp, SRC, v4_root=V4),
                     'stale repo commit')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_setup_script_refuses_to_redefine_a_frozen_protocol():
    tmp = tempfile.mkdtemp(prefix='v3a42_setup_')
    try:
        setup = os.path.join(ROOT, 'scripts', 'setup_v3a42_oracle.py')
        subprocess.check_output([sys.executable, setup, '--root', tmp],
                                stderr=subprocess.STDOUT)
        # same protocol -> fine (commit may move)
        subprocess.check_output([sys.executable, setup, '--root', tmp],
                                stderr=subprocess.STDOUT)
        # a different grid list must refuse instead of silently overwriting
        try:
            subprocess.check_output([sys.executable, setup, '--root', tmp,
                                     '--grids', '1,2,4'], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            assert b'refusing to overwrite' in e.output, e.output[-500:]
        else:
            raise AssertionError('setup accepted a changed grid list')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── §23 smoke ───────────────────────────────────────────────────────────────

def test_smoke_script_reaches_aggregation():
    """End-to-end on 2 dev images: the aggregation path must not KeyError, the
    outputs must exist, and no checkpoint may appear anywhere under the root."""
    tmp = tempfile.mkdtemp(prefix='v3a42_smoke_')
    try:
        _write_lock(tmp)
        script = os.path.join(ROOT, 'scripts',
                              'diagnose_v3a42_blockwise_oracle.py')
        out = subprocess.check_output(
            [sys.executable, '-W', 'ignore', script, '--root', tmp,
             '--splits', 'dev', '--grids', '1,2,4,8,16', '--limit', '2'],
            stderr=subprocess.STDOUT).decode()
        assert 'Traceback' not in out, out[-3000:]
        assert 'KeyError' not in out, out[-3000:]
        assert 'dev done' in out, out[-3000:]
        s = json.load(open(os.path.join(tmp, 'oracle', 'summary.json'),
                           encoding='utf-8'))
        assert s['protocol']['grids'] == [1, 2, 4, 8, 16]
        for arm in ('G1', 'G2', 'G4', 'G8', 'G16', 'Spatial_H4'):
            assert arm in s['splits']['dev']['correct']['mean_psnr'], arm
        import csv as _csv
        rows = list(_csv.DictReader(open(os.path.join(tmp, 'oracle',
                                                      'per_image.csv'),
                                        encoding='utf-8')))
        assert len(rows) == 2 * 3, len(rows)
        for state in ('correct', 'true_dark_g0.5', 'mismatch'):
            r = [x for x in rows if x['state'] == state]
            assert len(r) == 2, state
            for arm in ('Base', 'R1', 'G1', 'G2', 'G4', 'G8', 'G16',
                        'Spatial_H4', 'delta_G1_vs_R1', 'capture_G1'):
                assert arm in r[0], '%s missing from %s' % (arm, state)
        curve = list(_csv.DictReader(open(os.path.join(tmp, 'oracle',
                                                       'capture_curve.csv'),
                                         encoding='utf-8')))
        arms = {x['arm'] for x in curve}
        assert arms == {'R1', 'G1', 'G2', 'G4', 'G8', 'G16', 'Spatial_H4'}, arms
        # every grid must be monotonically >= R1 on the same aggregation
        for state in ('correct', 'true_dark_g0.5', 'mismatch'):
            r1 = [x for x in curve if x['arm'] == 'R1' and x['state'] == state][0]
            for g in (1, 2, 4, 8, 16):
                row = [x for x in curve if x['arm'] == 'G%d' % g
                       and x['state'] == state][0]
                # a q==1 gate is feasible for every block grid, so its optimum
                # can never be worse in mean PSNR (1e-6 = float wiggle only)
                assert float(row['mean_psnr']) >= float(r1['mean_psnr']) - 1e-6, \
                    (state, g)
        assert '[repro] --limit 2: skipped' in out, out[-3000:]
        # §22: no checkpoint anywhere under the diagnostic root
        for dirpath, _dirs, files in os.walk(tmp):
            for fn in files:
                assert not fn.endswith(('.pt', '.pth', '.ckpt')), \
                    'checkpoint written: %s' % os.path.join(dirpath, fn)
        assert os.path.isfile(os.path.join(tmp, 'logs',
                                           'diagnose_blockwise_oracle.log'))
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
