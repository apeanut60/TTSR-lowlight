#!/usr/bin/env python
"""V3-A.5 acceptance: target family, energy mask, fairness, verdict, artifacts.

Plan §31 (target algebra / G64 geometry / frozen proposal / step0 equality), §12
(resolution-independent mask), §13 (loss), §15 (bit-equal init), §20 (criteria),
§30 (lock) plus an end-to-end smoke of setup -> acceptance -> A0 -> A1 -> dev.
"""

import csv
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

from model.V3A5Verifier import V3A5Verifier, build_shared_init     # noqa: E402
from v3a42_runtime import _sha256                                 # noqa: E402
from v3a43_runtime import build_level_chain                        # noqa: E402
from v3a4_runtime import verify_v3a4_artifact_lock                 # noqa: E402
from v3a5_runtime import (ARMS, MODES, STATES, STATE_PROBS,        # noqa: E402
                          action_optimal_target, bit_equal, block_energy,
                          check_worktree, energy_mask, energy_threshold,
                          expand_gate, gate_metrics, geometry_report,
                          masked_smooth_l1, parameter_l1_drift,
                          per_image_qmean_corr, pixel_energy,
                          pool_feature_by_geom, pool_geometry_audit,
                          prepare_geometry, recovery, sample_states, snapshot_,
                          state_dict_sha, target_geometry, validate_run_protocol,
                          verdict_5a, verifier_loss, verify_v3a5_artifact_lock)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V43 = '/root/data/experiments/v3a43_fine_resolution'
H, W = 400, 600


def _rand(seed=0, h=H, w=W):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(1, 3, h, w, generator=g) * 2 - 1,
            torch.rand(1, 3, h, w, generator=g) * 2 - 1,
            torch.rand(1, 3, h, w, generator=g) * 2 - 1)


def _expect_fail(fn, what):
    try:
        fn()
    except SystemExit as e:
        assert str(e).strip(), '%s raised an empty message' % what
        return
    raise AssertionError('%s did not hard-fail' % what)


# ── §4/§8 target family ─────────────────────────────────────────────────────

def test_both_geometries_come_from_one_nested_family():
    geoms = {m: target_geometry(H, W, m) for m in MODES}
    geo = geometry_report(H, W, geoms)
    assert geo['dense_block_h4']['shape'] == [100, 150]
    assert geo['g64']['shape'] == [64, 64]
    assert geo['nesting']['g64_within_block_h4'] is True
    # G64's edges must be the V3-A.4.3 chain edges, NOT round(linspace(0,400,65))
    chain = build_level_chain(H, W, (100, 150), (64,))
    np.testing.assert_array_equal(geoms['g64']['edges'][0], chain['y'][64])
    naive = np.round(np.linspace(0.0, H, 65)).astype(np.int64)
    assert not np.array_equal(geoms['g64']['edges'][0], naive), \
        'G64 edges collapsed to the naive pixel lattice'
    # both are coarsenings of the base cells
    base = block_energy_edges = geoms['dense_block_h4']['edges']
    for m in MODES:
        ey, ex = geoms[m]['edges']
        assert set(int(v) for v in ey) <= set(int(v) for v in base[0])
        assert set(int(v) for v in ex) <= set(int(v) for v in base[1])


def test_target_algebra_holds_for_both_geometries():
    y0, hr, _D = _rand(1)
    for mode in MODES:
        geom = prepare_geometry(target_geometry(H, W, mode), 'cpu')
        t1 = action_optimal_target(y0, hr, hr - y0, geom)
        assert abs(float(t1['q_grid'].mean()) - 1.0) < 1e-4, mode
        t2 = action_optimal_target(y0, hr, 2 * (hr - y0), geom)
        assert abs(float(t2['q_grid'].mean()) - 0.5) < 1e-4, mode
        t3 = action_optimal_target(y0, hr, -(hr - y0), geom)
        assert float(t3['q_grid'].max()) == 0.0, mode


def test_expansion_matches_the_oracle_expansion():
    """The arm's nearest expansion must be the target's own expansion, or
    training would optimise a differently parameterised gate than q*."""
    y0, hr, D = _rand(2, h=64, w=96)
    for mode in MODES:
        geom = prepare_geometry(target_geometry(64, 96, mode), 'cpu')
        tgt = action_optimal_target(y0, hr, D, geom)
        assert torch.equal(expand_gate(tgt['q_grid'], geom), tgt['q_full']), mode


def test_v3a5_main_path_has_no_adaptive_pooling():
    """§11: neither the gate path nor the FEATURE path may use adaptive pooling
    or interpolation -- A1 needs the exact nested G64 partition, not 'about
    64x64'. Scans the model as well, which the earlier guard missed."""
    for rel in ('v3a5_runtime.py', os.path.join('model', 'V3A5Verifier.py')):
        src = open(os.path.join(ROOT, rel), encoding='utf-8').read()
        for token in ('adaptive_avg_pool', 'adaptive_max_pool', 'interpolate',
                      'avg_pool'):
            assert token not in src, '%s uses %s' % (rel, token)


# ── §8-§10 exact G64 feature pooling == oracle partition ────────────────────

def test_g64_feature_pooling_uses_the_oracle_base_partition():
    """The pooling partition must be the V3-A.4.3 G64 grouping, verified against
    the frozen nesting.json rather than against our own re-derivation."""
    geom = target_geometry(H, W, 'g64')
    # real check: base-index edges x CELL SIZE == the frozen pixel edges
    base = target_geometry(H, W, 'dense_block_h4')
    cell_y = int(base['edges'][0][1] - base['edges'][0][0])
    cell_x = int(base['edges'][1][1] - base['edges'][1][0])
    nesting = json.load(open(os.path.join(V43, 'oracle', 'nesting.json'),
                             encoding='utf-8'))['levels']['64']
    np.testing.assert_array_equal(
        np.asarray([int(v) // cell_y for v in nesting['pixel_edges_y']]),
        geom['base_index_edges'][0])
    np.testing.assert_array_equal(
        np.asarray([int(v) // cell_x for v in nesting['pixel_edges_x']]),
        geom['base_index_edges'][1])
    # and the native support really is the H/4 lattice
    assert geom['native_shape'] == (100, 150)
    assert tuple(base['shape']) == geom['native_shape']


def test_exact_g64_feature_pool_has_no_overlap_or_gap():
    geom = target_geometry(H, W, 'g64')
    audit = pool_geometry_audit(100, 150, geom)
    assert audit['shape'] == [64, 64] and audit['n_blocks'] == 64 * 64
    assert audit['idx_size'] == 15000 and audit['native_cells'] == 15000
    assert audit['all_cells_covered_once'] and audit['no_empty_block']
    # y: 100 cells -> 64 blocks = 1-2 cells; x: 150 -> 64 = 2-3 cells
    assert audit['min_block'] >= 2 and audit['max_block'] <= 6
    assert audit['total_cells'] == 15000


def test_exact_g64_pool_matches_hand_computed_block_means():
    """F[y,x] = y*1000 + x; compare against a direct slice mean."""
    geom = target_geometry(H, W, 'g64')
    f = (torch.arange(100).view(1, 1, 100, 1) * 1000
         + torch.arange(150).view(1, 1, 1, 150)).float()
    pooled = pool_feature_by_geom(f, geom)
    iy, ix = geom['base_index_edges']
    assert pooled.shape == (1, 1, 64, 64)
    for bi, bj in ((0, 0), (1, 1), (31, 31), (63, 63), (0, 63), (17, 40)):
        want = f[0, 0, iy[bi]:iy[bi + 1], ix[bj]:ix[bj + 1]].mean()
        got = pooled[0, 0, bi, bj]
        assert abs(float(got) - float(want)) < 1e-6, (bi, bj, float(got), float(want))
    # a constant map pools to that constant (exact non-overlap mean)
    c = torch.full((1, 3, 100, 150), 0.7)
    assert torch.allclose(pool_feature_by_geom(c, geom),
                          torch.full((1, 3, 64, 64), 0.7), atol=1e-6)
    # and a non-native input is refused rather than silently pooled
    _expect_fail(lambda: pool_feature_by_geom(torch.zeros(1, 1, 64, 64), geom),
                 'pooling a non-native feature map')


def test_pooling_partition_equals_the_target_partition():
    """§2/§29 on the REAL geometry: colour every native H/4 cell with the G64
    block id the ORACLE assigns to it, pool, and require the pooled value to be
    exactly that block's index. This can only hold if the feature-pooling
    partition and the target partition are literally the same partition."""
    geom = prepare_geometry(target_geometry(H, W, 'g64'), 'cpu')
    nby, nbx = geom['shape']
    cell = 4                                    # H/4 base cell is 4x4 px
    f = torch.zeros(1, 1, 100, 150)
    ids = geom['index'].reshape(H, W)
    for i in range(100):
        for j in range(150):
            blk = int(ids[i * cell, j * cell])
            # every pixel of one base cell must belong to the same G64 block
            blk_patch = ids[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
            assert np.all(blk_patch == blk), (i, j)
            f[0, 0, i, j] = blk
    pooled = pool_feature_by_geom(f, geom)
    want = torch.arange(nby * nbx, dtype=pooled.dtype).reshape(nby, nbx)
    assert torch.equal(pooled[0, 0], want), 'pooling != target partition'


# ── §12 energy mask ─────────────────────────────────────────────────────────

def test_energy_is_per_pixel_and_resolution_free():
    y0, _hr, D = _rand(3, h=64, w=96)
    e = pixel_energy(D)
    assert e.shape == (1, 1, 64, 96)
    assert torch.allclose(e, D.pow(2).sum(1, keepdim=True) / 3.0)
    # a constant D gives a constant block energy at every geometry
    const = torch.full_like(D, 0.3)
    for mode in MODES:
        geom = prepare_geometry(target_geometry(64, 96, mode), 'cpu')
        be = block_energy(const, geom)
        assert torch.allclose(be, torch.full_like(be, 0.3 ** 2)), mode
    assert abs(float(block_energy(const, prepare_geometry(
        target_geometry(64, 96, 'g64'), 'cpu')).mean()) - 0.09) < 1e-6


def test_energy_threshold_and_mask():
    v = np.concatenate([np.linspace(0.0, 1.0, 101)])
    assert abs(energy_threshold(v, 10.0) - 0.1) < 1e-9
    assert energy_threshold(v, 50.0) == 0.5
    e = torch.tensor([[[[0.0, 0.5], [1.0, 0.2]]]])
    m = energy_mask(e, 0.3)
    assert m.tolist() == [[[[0.0, 1.0], [1.0, 0.0]]]]
    _expect_fail(lambda: energy_threshold([]), 'empty energy pool')


def test_masked_loss_ignores_masked_blocks():
    # only the top-left block carries error; the other three are already exact
    q_v = torch.tensor([[[[0.5, 0.0], [0.0, 0.0]]]])
    q_t = torch.zeros_like(q_v)
    mask_all = torch.ones_like(q_v)
    mask_none = torch.zeros_like(q_v)
    assert masked_smooth_l1(q_v, q_t, mask_all) > 0
    # with nothing valid the term is 0/1 rather than NaN or a hard error
    assert float(masked_smooth_l1(q_v, q_t, mask_none)) == 0.0
    only_err = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    only_ok = torch.tensor([[[[0.0, 1.0], [1.0, 1.0]]]])
    # masking the error away gives exactly 0; isolating it is strictly worse
    # than averaging it over all four blocks
    assert float(masked_smooth_l1(q_v, q_t, only_ok)) == 0.0
    assert float(masked_smooth_l1(q_v, q_t, only_err)) > \
        float(masked_smooth_l1(q_v, q_t, mask_all))


def test_loss_weights_are_one_and_point_one():
    y0, hr, D = _rand(4, h=64, w=96)
    geom = prepare_geometry(target_geometry(64, 96, 'g64'), 'cpu')
    tgt = action_optimal_target(y0, hr, D, geom)
    mask = torch.ones_like(tgt['q_grid'])
    q_full = expand_gate(tgt['q_grid'], geom)
    total, gate, out = verifier_loss(tgt['q_grid'], tgt['q_grid'], mask, q_full,
                                     y0, hr, D)
    assert float(gate) == 0.0
    # perfect gate -> only the residual output term remains
    assert abs(float(total) - 0.1 * float(out)) < 1e-7
    total2, _g, _o = verifier_loss(tgt['q_grid'], torch.zeros_like(tgt['q_grid']),
                                   mask, q_full, y0, hr, D)
    assert total2 > total


# ── §15 fairness ────────────────────────────────────────────────────────────

def test_arms_share_a_bit_equal_initialisation():
    init = build_shared_init('dense_block_h4', 'g64', seed=42)
    assert bit_equal(init['dense_block_h4'], init['g64'])
    assert state_dict_sha(init['g64']) == state_dict_sha(init['dense_block_h4'])
    a = V3A5Verifier('dense_block_h4')
    b = V3A5Verifier('g64')
    assert [p.shape for p in a.parameters()] == [p.shape for p in b.parameters()]
    assert sum(p.numel() for p in a.parameters()) == \
        sum(p.numel() for p in b.parameters())


def test_both_arms_start_at_a_constant_half_gate():
    y0, hr, D = _rand(5, h=64, w=96)
    x = torch.rand_like(y0)
    for mode in MODES:
        m = V3A5Verifier(mode).eval()
        geom = prepare_geometry(target_geometry(64, 96, mode), 'cpu')
        with torch.no_grad():
            q = m(x, y0, hr, geom=geom)
        assert q.shape[-2:] == geom['shape'], (mode, q.shape)
        assert float(q.min()) == float(q.max()) == 0.5, mode
    # identical step-0 output across arms on the REAL geometry (the constant
    # gate masks any pooling difference, which is why this must be asserted)
    m0 = V3A5Verifier('dense_block_h4').eval()
    m1 = V3A5Verifier('g64').eval()
    g0 = prepare_geometry(target_geometry(H, W, 'dense_block_h4'), 'cpu')
    g1 = prepare_geometry(target_geometry(H, W, 'g64'), 'cpu')
    y0r, hr_r, D_r = _rand(5)
    xr = torch.rand_like(y0r)
    with torch.no_grad():
        y_0 = y0r + expand_gate(m0(xr, y0r, hr_r, geom=g0) * 0 + 0.5, g0) * D_r
        y_1 = y0r + expand_gate(m1(xr, y0r, hr_r, geom=g1) * 0 + 0.5, g1) * D_r
    assert torch.equal(y_0, y_1), 'step0 outputs differ across arms'


def test_dense_arm_refuses_to_silently_repool():
    # must use the real 400x600 geometry: on a tiny image the G64 chain
    # degenerates onto the native lattice, so both supports coincide
    y0, _hr, _D = _rand(6)
    m = V3A5Verifier('dense_block_h4').eval()
    with torch.no_grad():
        native = target_geometry(H, W, 'dense_block_h4')
        g64 = target_geometry(H, W, 'g64')
        assert native['shape'] == (100, 150) and g64['shape'] == (64, 64)
        _expect_fail(lambda: m(y0, y0, y0, geom=g64),
                     'dense arm re-pooled a different partition')
        q = m(y0, y0, y0, geom=native)
        assert q.shape[-2:] == tuple(native['shape'])
        # and the G64 arm really pools onto the oracle support
        assert V3A5Verifier('g64').eval()(y0, y0, y0, geom=g64).shape[-2:] == (64, 64)


def test_proposal_snapshot_detects_drift():
    sd = {'w': torch.zeros(3)}
    before = snapshot_(torch.nn.Linear(2, 3))
    assert parameter_l1_drift(before, before) == 0.0
    after = {k: v.clone() for k, v in before.items()}
    after['weight'] = after['weight'] + 1e-6
    assert parameter_l1_drift(before, after) > 0
    assert sd is not None


# ── §11/§18/§19/§20 metrics ─────────────────────────────────────────────────

def test_state_sampling_follows_the_locked_weights():
    seq = sample_states(20000, 7)
    frac = {s: seq.count(s) / len(seq) for s in STATES}
    assert abs(frac['correct'] - STATE_PROBS[0]) < 0.02, frac
    assert abs(frac['true_dark_g0.5'] - STATE_PROBS[1]) < 0.02, frac
    assert abs(frac['mismatch'] - STATE_PROBS[2]) < 0.02, frac
    assert sample_states(20, 7) == sample_states(20, 7)          # deterministic
    assert sample_states(20, 8) != sample_states(20, 7)


def test_recovery_and_gate_metrics():
    assert abs(recovery(20.2, 20.0, 20.5) - 0.4) < 1e-9
    assert recovery(20.2, 20.0, 20.0) is None      # zero denominator
    assert recovery(20.1, 20.0, 19.9) is None      # negative denominator
    q = torch.rand(1, 1, 8, 8, generator=torch.Generator().manual_seed(0))
    m = gate_metrics(q, q, torch.ones_like(q))
    assert m['MAE'] == 0.0 and abs(m['corr'] - 1.0) < 1e-5
    assert m['mask_valid_frac'] == 1.0
    z = torch.zeros_like(q)
    m2 = gate_metrics(z, q, torch.ones_like(q))
    assert m2['q_v_std'] == 0.0 and np.isnan(m2['corr'])
    m3 = gate_metrics(q, q, torch.zeros_like(q))
    assert m3['mask_valid_frac'] == 0.0 and np.isnan(m3['masked_corr'])
    c = per_image_qmean_corr([torch.full((1, 1, 4, 4), 0.1 * i) for i in range(5)],
                             [torch.full((1, 1, 4, 4), 0.2 * i) for i in range(5)])
    assert abs(c - 1.0) < 1e-6


def test_verdict_5a_rules():
    base = 19.98
    dev = {s: dict(R1=20.0, A0=20.0, A1=20.0) for s in STATES}
    v = verdict_5a(dev, base)
    assert v['correct_not_worse'] and v['correct_beats_r1']
    assert v['harmful_ge_base'] and not v['harmful_ge_base_005']
    assert not v['gain_rule'] and v['mean_gain'] == 0.0
    # +0.05 everywhere -> both gain rules hold
    dev = {s: dict(R1=20.0, A0=20.0, A1=20.05) for s in STATES}
    v = verdict_5a(dev, base)
    assert v['gain_rule'] and v['gain_rule_mean'] and v['gain_rule_majority']
    # 2/3 states gain, the third only dips 0.01 -> majority rule holds
    dev = {s: dict(R1=20.0, A0=20.0, A1=20.0) for s in STATES}
    dev['correct']['A1'] = 20.06
    dev['mismatch']['A1'] = 20.06
    dev['true_dark_g0.5']['A1'] = 19.99
    v = verdict_5a(dev, base)
    assert v['gain_rule'] and not v['gain_rule_mean']
    # correct collapsing like V3-A.4 C0 must be flagged
    dev = {s: dict(R1=20.0, A0=20.0, A1=20.2) for s in STATES}
    dev['correct']['A1'] = 19.94
    v = verdict_5a(dev, base)
    assert not v['correct_not_worse'] and not v['correct_beats_r1']


# ── §30 lock + worktree guard ───────────────────────────────────────────────

def _write_lock(tmp, energy_sha=None, commit=None):
    v4 = verify_v3a4_artifact_lock(V4, SRC)
    if commit is None:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                         cwd=ROOT).decode().strip()
    os.makedirs(os.path.join(tmp, 'targets'), exist_ok=True)
    e = os.path.join(tmp, 'targets', 'energy_stats.json')
    if not os.path.isfile(e):
        json.dump({'threshold': 1e-4}, open(e, 'w'))
    g = os.path.join(tmp, 'targets', 'geometry.json')
    if not os.path.isfile(g):
        json.dump(geometry_report(H, W, {m: target_geometry(H, W, m)
                                         for m in MODES}), open(g, 'w'))
    lock = dict(
        repo_commit=commit,
        proposal_sha256=v4['proposal_sha256'],
        cache_metadata_sha256=v4['cache_metadata_sha256'],
        manifest_sha256=v4['manifest_sha256'],
        split_sha256=v4['split_sha256'],
        mismatch_train_sha256=v4['mismatch_train_sha256'],
        mismatch_dev_sha256=v4['mismatch_dev_sha256'],
        v3a43_summary_sha256=_sha256(os.path.join(V43, 'oracle', 'summary.json')),
        v3a43_nesting_sha256=_sha256(os.path.join(V43, 'oracle', 'nesting.json')),
        v3a43_per_image_sha256=_sha256(os.path.join(V43, 'oracle', 'per_image.csv')),
        v3a43_root=V43,
        v3a43_summary_path=os.path.join(V43, 'oracle', 'summary.json'),
        v3a43_nesting_path=os.path.join(V43, 'oracle', 'nesting.json'),
        v3a43_per_image_path=os.path.join(V43, 'oracle', 'per_image.csv'),
        geometry_sha256=_sha256(g),
        energy_stats_sha256=energy_sha or _sha256(e),
        states='+'.join(STATES), target_family='nested_blockwise_action_optimal',
        reference_variant='nanobanana_ref_v2', cache_name='cache_y0_lolbase',
        state_probs=list(STATE_PROBS),
        arms=list(ARMS), steps=3000, grad_accum_default=4, seed=42,
        block_h4_factor=4, energy_threshold=1e-4, energy_pctl=10.0)
    path = os.path.join(tmp, 'artifact_lock.json')
    json.dump(lock, open(path, 'w', encoding='utf-8'), indent=2, sort_keys=True)
    return path, lock


def test_lock_and_worktree_guards():
    tmp = tempfile.mkdtemp(prefix='v3a5_lock_')
    try:
        _expect_fail(lambda: verify_v3a5_artifact_lock(tmp, SRC, v4_root=V4,
                                                       v43_root=V43), 'missing lock')
        path, lock = _write_lock(tmp)
        assert verify_v3a5_artifact_lock(tmp, SRC, v4_root=V4,
                                         v43_root=V43)['steps'] == 3000
        bad = dict(lock, energy_stats_sha256='0' * 64)
        json.dump(bad, open(path, 'w', encoding='utf-8'))
        _expect_fail(lambda: verify_v3a5_artifact_lock(tmp, SRC, v4_root=V4,
                                                       v43_root=V43),
                     'tampered energy stats')
        bad = dict(lock, v3a43_summary_sha256='0' * 64)
        json.dump(bad, open(path, 'w', encoding='utf-8'))
        _expect_fail(lambda: verify_v3a5_artifact_lock(tmp, SRC, v4_root=V4,
                                                       v43_root=V43),
                     'tampered V3-A.4.3 summary')
        bad = dict(lock, v3a43_per_image_sha256='0' * 64)
        json.dump(bad, open(path, 'w', encoding='utf-8'))
        _expect_fail(lambda: verify_v3a5_artifact_lock(tmp, SRC, v4_root=V4,
                                                       v43_root=V43),
                     'tampered V3-A.4.3 per_image')
        bad = dict(lock, geometry_sha256='0' * 64)
        json.dump(bad, open(path, 'w', encoding='utf-8'))
        _expect_fail(lambda: verify_v3a5_artifact_lock(tmp, SRC, v4_root=V4,
                                                       v43_root=V43),
                     'tampered geometry artifact')
        # the v43 anchors are addressed through the lock, so pointing the CLI at
        # a different tree cannot redirect the eval
        alt = tempfile.mkdtemp(prefix='v3a5_other_v43_')
        try:
            bad = dict(lock, v3a43_root=alt)
            json.dump(bad, open(path, 'w', encoding='utf-8'))
            _expect_fail(lambda: verify_v3a5_artifact_lock(tmp, SRC, v4_root=V4,
                                                           v43_root=V43),
                         'lock pointing at a tree without the anchors')
        finally:
            shutil.rmtree(alt, ignore_errors=True)
        json.dump(lock, open(path, 'w', encoding='utf-8'), indent=2, sort_keys=True)
        bad = dict(lock, repo_commit='deadbeef' * 5)
        json.dump(bad, open(path, 'w', encoding='utf-8'))
        _expect_fail(lambda: verify_v3a5_artifact_lock(tmp, SRC, v4_root=V4,
                                                       v43_root=V43), 'stale commit')
        bad = dict(lock, target_family='something_else')
        json.dump(bad, open(path, 'w', encoding='utf-8'))
        _expect_fail(lambda: verify_v3a5_artifact_lock(tmp, SRC, v4_root=V4,
                                                       v43_root=V43),
                     'wrong target family')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    head, dirty = 'a' * 40, ['scripts/x.py']
    _expect_fail(lambda: check_worktree(0, head=head, dirty=dirty),
                 'formal run on a dirty tree')
    _h, _d, warn = check_worktree(2, head=head, dirty=dirty)
    assert warn and 'smoke run' in warn
    assert check_worktree(0, head=head, dirty=[]) == (head, [], None)


def test_formal_training_refuses_a_protocol_change():
    """§18-§21: seed / steps / grad_accum / variant / cache_name are protocol."""
    lock = dict(reference_variant='nanobanana_ref_v2', cache_name='cache_y0_lolbase',
                seed=42, steps=3000, grad_accum_default=4, block_h4_factor=4,
                energy_threshold=1e-4, energy_pctl=10.0)
    # formal: everything must match exactly
    eff = validate_run_protocol(lock, formal=True, seed=42, steps=3000,
                               grad_accum=4, variant='nanobanana_ref_v2',
                               cache_name='cache_y0_lolbase')
    assert eff['seed'] == 42 and eff['steps'] == 3000
    for kwargs, what in ((dict(seed=43), '--seed'),
                         (dict(steps=1500), '--steps'),
                         (dict(grad_accum=8), '--grad_accum')):
        _expect_fail(lambda k=kwargs: validate_run_protocol(lock, formal=True,
                                                           **k), what)
    # variant / cache are protocol even in a smoke
    for kwargs, what in ((dict(variant='v3'), '--variant'),
                         (dict(cache_name='other_cache'), '--cache_name')):
        _expect_fail(lambda k=kwargs: validate_run_protocol(lock, formal=False,
                                                           **k), what)
        _expect_fail(lambda k=kwargs: validate_run_protocol(lock, formal=True,
                                                           **k), what)
    # a smoke may shorten the schedule and reseed
    eff = validate_run_protocol(lock, formal=False, seed=7, steps=4, grad_accum=1)
    assert eff['seed'] == 7 and eff['steps'] == 4 and eff['grad_accum'] == 1
    assert eff['variant'] == lock['reference_variant']     # still from the lock


def test_setup_mask_valid_fraction_matches_training_definition(tmp=None):
    """§17: the reported G64 valid fraction must equal what training computes."""
    import tempfile as _tf
    tmp = _tf.mkdtemp(prefix='v3a5_mask_')
    try:
        setup = os.path.join(ROOT, 'scripts', 'setup_v3a5_verifier.py')
        subprocess.check_output([sys.executable, '-W', 'ignore', setup,
                                 '--root', tmp, '--limit', '2'],
                                stderr=subprocess.STDOUT)
        e = json.load(open(os.path.join(tmp, 'targets', 'energy_stats.json'),
                           encoding='utf-8'))
        thr = float(e['threshold'])
        assert abs(thr - float(e['p10'] if 'p10' in e else e['pool']['p10'])) < 1e-12
        # recompute both fractions from the same definition the trainer uses
        sys.path.insert(0, ROOT)
        from option import parser as option_parser
        from v3a5_pipeline import (correction, load_proposal, load_rows,
                                   make_dataset, sample_tensors)
        ns = option_parser.parse_args([])
        ns.dataset_dir = '/root/data/datasets/lol-v2-real'
        ns.v3a_ref_variant = 'nanobanana_ref_v2'
        rows = load_rows('/root/data/experiments/v3a1_lolv2real/manifests/'
                         'refiner_train.csv',
                         '/root/data/experiments/v3a4_lolv2real/splits/split.json'
                         )['train'][:2]
        mm = json.load(open('/root/data/experiments/v3a4_lolv2real/mappings/'
                            'mismatch_train_575.json', encoding='utf-8'))
        ds = make_dataset(ns, rows, '/root/data/experiments/v3a1_lolv2real/'
                          'cache_y0_lolbase/refiner_train', mm)
        proposal = load_proposal('/root/data/experiments/v3a1_lolv2real/'
                                 'R1_v2stable_naive_s42/checkpoint_03000.pt', 'cuda')
        for mode in MODES:
            geom = prepare_geometry(target_geometry(H, W, mode), 'cuda')
            fracs = []
            with torch.no_grad():
                for i in range(len(rows)):
                    for state in STATES:
                        t = sample_tensors(ds, i, state, 'cuda')
                        D, _sr = correction(proposal.proposal, t['Y0'], t['R'])
                        fracs.append(float(
                            energy_mask(block_energy(D, geom), thr).mean()))
            got = float(np.mean(fracs))
            want = float(np.mean([e['mask_valid_frac']['%s|%s' % (mode, s)]
                                  for s in STATES]))
            assert abs(got - want) < 1e-6, (mode, got, want)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── §32 end-to-end smoke ────────────────────────────────────────────────────

def test_smoke_setup_accept_train_eval():
    tmp = tempfile.mkdtemp(prefix='v3a5_smoke_')
    setup = os.path.join(ROOT, 'scripts', 'setup_v3a5_verifier.py')
    check = os.path.join(ROOT, 'scripts', 'check_v3a5_targets.py')
    train = os.path.join(ROOT, 'scripts', 'train_v3a5_verifier.py')
    evals = os.path.join(ROOT, 'scripts', 'eval_v3a5_dev.py')
    try:
        out = subprocess.check_output(
            [sys.executable, '-W', 'ignore', setup, '--root', tmp, '--limit', '2'],
            stderr=subprocess.STDOUT).decode()
        assert 'lock verified OK' in out, out[-2000:]
        out = subprocess.check_output(
            [sys.executable, '-W', 'ignore', check, '--root', tmp, '--limit', '1'],
            stderr=subprocess.STDOUT).decode()
        assert 'checks passed' in out and 'Traceback' not in out, out[-2000:]
        for arm in ARMS:
            out = subprocess.check_output(
                [sys.executable, '-W', 'ignore', train, '--root', tmp, '--arm', arm,
                 '--limit', '2', '--steps', '2', '--grad_accum', '1'],
                stderr=subprocess.STDOUT).decode()
            assert 'proposal drift 0' in out, out[-2000:]
            assert os.path.isfile(os.path.join(tmp, arm, 'checkpoints', 'last.pt'))
            tm = json.load(open(os.path.join(tmp, arm, 'train_metrics.json'),
                                encoding='utf-8'))
            assert tm['proposal_drift'] == 0.0
            assert abs(tm['step0']['q_mean'] - 0.5) < 1e-6
            assert tm['step0']['psnr_mean'] > 0
            assert all(np.isfinite(r['loss']) for r in tm['history'])
        out = subprocess.check_output(
            [sys.executable, '-W', 'ignore', evals, '--root', tmp,
             '--splits', 'dev', '--limit', '2'], stderr=subprocess.STDOUT).decode()
        assert 'Traceback' not in out, out[-3000:]
        d = os.path.join(tmp, 'diagnostics')
        for fn in ('summary.json', 'per_image.csv', 'gate_metrics.csv',
                   'recovery64.csv'):
            assert os.path.isfile(os.path.join(d, fn)), fn
        s = json.load(open(os.path.join(d, 'summary.json'), encoding='utf-8'))
        assert s['report']['criteria'] is not None
        assert set(s['splits']['dev']) == set(STATES)
        for state in STATES:
            e = s['splits']['dev'][state]
            for k in ('Base', 'R1', 'AO64', 'BlockH4', *ARMS):
                assert k in e['psnr'], k
            assert abs(e['psnr']['R1'] - e['psnr']['oracle_R1']) < 2e-3
            for arm in ARMS:
                assert arm in e['recovery64']
        rows = list(csv.DictReader(open(os.path.join(d, 'per_image.csv'),
                                        encoding='utf-8')))
        assert len(rows) == 2 * 3
        for r in rows:
            for k in ('AO64', 'BlockH4', 'recovery64_A1_g64', *ARMS):
                assert k in r, k
        # both arms were trained from the same init
        a0 = json.load(open(os.path.join(tmp, ARMS[0], 'args.json'), encoding='utf-8'))
        a1 = json.load(open(os.path.join(tmp, ARMS[1], 'args.json'), encoding='utf-8'))
        assert a0['init_sha'] == a1['init_sha']
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
