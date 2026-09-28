#!/usr/bin/env python
"""V3-A.5 §12/§29/§30: target + energy-mask artifacts and the artifact lock.

One pass over the frozen proposal on train575 x 3 states produces everything the
training arms need to be comparable:

  * targets/energy_stats.json -- the resolution-INDEPENDENT cell-energy pool and
    the single p10 threshold both arms use,
  * targets/target_stats.json -- q* statistics for Block_H4 and G64 plus each
    arm's mask-valid fraction,
  * artifact_lock.json -- the hashes that make the run reproducible.

Read-only w.r.t. the frozen proposal; no training, no checkpoint.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from option import parser as option_parser                        # noqa: E402
from v3a42_runtime import _sha256                                 # noqa: E402
from v3a43_runtime import (build_level_chain, level_geometry_report)  # noqa: E402
from v3a4_runtime import verify_v3a4_artifact_lock                # noqa: E402
from v3a5_pipeline import (correction, load_proposal, load_rows,  # noqa: E402
                           make_dataset, sample_tensors)
from v3a5_runtime import (ARMS, ARM_MODE, ENERGY_PCTL, G64, MODES,  # noqa: E402
                          STATES, action_optimal_target, block_energy,
                          energy_threshold, geometry_report,
                          pixel_energy, prepare_geometry, STATE_PROBS,
                          target_geometry,
                          verify_v3a5_artifact_lock)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V43 = '/root/data/experiments/v3a43_fine_resolution'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
REF_H, REF_W = 400, 600


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--v43_root', default=V43)
    ap.add_argument('--root', default='/root/data/experiments/v3a5_g64_verifier')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0,
                    help='debug/smoke: pool only the first N train images')
    a = ap.parse_args(_CLI)

    for d in ('targets', 'logs'):
        os.makedirs(os.path.join(a.root, d), exist_ok=True)

    # geometry is frozen for the reference resolution and re-derived per image
    geoms = {m: target_geometry(REF_H, REF_W, m) for m in MODES}
    geo = geometry_report(REF_H, REF_W, geoms)
    print('target geometry (400x600):')
    for m in MODES:
        e = geo[m]
        print('  %-16s %4d x %-4d blocks  y %d-%d px  x %d-%d px  [%s]'
              % (m, e['shape'][0], e['shape'][1], e['block_h'], e['block_h_max'],
                 e['block_w'], e['block_w_max'], e['provenance']))
    print('  nesting(g64 inside block_h4): %s' % geo['nesting']['g64_within_block_h4'])

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    rows = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                     os.path.join(a.v4_root, 'splits', 'split.json'))['train']
    mmap = json.load(open(os.path.join(a.v4_root, 'mappings',
                                       'mismatch_train_575.json'), encoding='utf-8'))
    ds = make_dataset(ns, rows, os.path.join(a.src_root, a.cache_name,
                                             'refiner_train'), mmap)
    model = load_proposal(os.path.join(a.src_root, R1_CK), a.device)
    if a.limit:
        rows = rows[:a.limit]
    # attach device tensors once per geometry (index maps are size-dependent)
    dev_geoms = {m: prepare_geometry(geoms[m], a.device) for m in MODES}

    cells, per_state, qstats, mfrac = [], {}, {}, {}
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(rows)):
            for state in STATES:
                t = sample_tensors(ds, i, state, a.device)
                D, _sr = correction(model.proposal, t['Y0'], t['R'])
                cell = block_energy(D, dev_geoms['dense_block_h4'])  # resolution-free
                cells.append(cell)
                per_state.setdefault(state, []).append(cell)
                for m in MODES:
                    g = dev_geoms[m]
                    tgt = action_optimal_target(t['Y0'], t['H'], D, g)
                    q = tgt['q_grid']
                    st = qstats.setdefault('%s|%s' % (m, state), [])
                    st.append(dict(mean=float(q.mean()), std=float(q.std()),
                                   frac0=float((q <= 0.01).float().mean()),
                                   frac1=float((q >= 0.99).float().mean())))
                    # the reported valid fraction must be the mask the arm will
                    # really use: THIS mode's own block energies (§15/§16), not
                    # the dense cell energy recycled for every mode
                    mfrac.setdefault('%s|%s' % (m, state), []).append(
                        block_energy(D, g))
            if (i + 1) % 25 == 0:
                print('  pooled %d/%d images (%.0fs)'
                      % (i + 1, len(rows), time.time() - t0))

    all_e = torch.cat([c.reshape(-1) for c in cells]).cpu().numpy()
    thr = energy_threshold(all_e, ENERGY_PCTL)
    energy_stats = dict(
        pctl=ENERGY_PCTL, threshold=thr, limit=a.limit,
        pooled_images=len(rows), pooled_states=list(STATES),
        definition='e_B = (1/(3|B|)) sum_{B,c} D^2 (Block_H4 cells)',
        pool=dict(mean=float(np.mean(all_e)), std=float(np.std(all_e)),
                  p10=float(np.percentile(all_e, 10)), p25=float(np.percentile(all_e, 25)),
                  median=float(np.median(all_e)), p75=float(np.percentile(all_e, 75)),
                  p90=float(np.percentile(all_e, 90)), n=int(all_e.size)),
        per_state={s: dict(mean=float(np.mean(torch.cat([c.reshape(-1) for c in v]).cpu().numpy())),
                           p10=float(np.percentile(torch.cat([c.reshape(-1) for c in v]).cpu().numpy(), 10)))
                   for s, v in per_state.items()},
        base_grid=list(geoms['dense_block_h4']['base_grid']),
        proposal_sha256=_sha256(os.path.join(a.src_root, R1_CK)),
        split_sha256=_sha256(os.path.join(a.v4_root, 'splits', 'split.json')))
    for m in MODES:
        for s in STATES:
            key = '%s|%s' % (m, s)
            v = torch.cat([c.reshape(-1) for c in mfrac[key]]).cpu().numpy()
            energy_stats.setdefault('mask_valid_frac', {})[key] = float((v >= thr).mean())

    tstats = {}
    for key, vals in qstats.items():
        tstats[key] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    target_stats = dict(geometry=geo, q_opt=tstats,
                        energy_threshold=thr, energy_pctl=ENERGY_PCTL,
                        pooled_images=len(rows), limit=a.limit)
    # §28/§29: ONE geometry artifact that target, feature pooling and gate
    # expansion all reference, derived from the frozen V3-A.4.3 nesting
    nesting = json.load(open(os.path.join(a.v43_root, 'oracle', 'nesting.json'),
                             encoding='utf-8'))
    ref_level = nesting['levels'][str(G64)]
    base_y = geo['dense_block_h4']['base_index_edges_y']
    base_x = geo['dense_block_h4']['base_index_edges_x']
    if geo['g64']['pixel_edges_y'] != [int(v) for v in ref_level['pixel_edges_y']] or \
            geo['g64']['pixel_edges_x'] != [int(v) for v in ref_level['pixel_edges_x']]:
        raise SystemExit('our G64 pixel edges differ from the frozen V3-A.4.3 '
                         'nesting.json -- refusing to write a geometry artifact')
    # structural check on the PIXEL lattice: every G64 pixel edge must be a
    # Block_H4 cell edge (the base-index edges are 0,1,2,... so testing those
    # would be vacuous -- the base cells are not 1 px wide)
    for name, edges, dense_edges in (
            ('y', geo['g64']['pixel_edges_y'],
             geo['dense_block_h4']['pixel_edges_y']),
            ('x', geo['g64']['pixel_edges_x'],
             geo['dense_block_h4']['pixel_edges_x'])):
        if not set(int(v) for v in edges) <= set(int(v) for v in dense_edges):
            raise SystemExit('G64 %s pixel edges are not Block_H4 cell edges' % name)
    geometry = dict(
        native_feature_shape=geo['native_feature_shape'],
        dense=dict(shape=geo['dense_block_h4']['shape'],
                   base_index_edges_y=base_y,
                   base_index_edges_x=base_x,
                   pixel_edges_y=geo['dense_block_h4']['pixel_edges_y'],
                   pixel_edges_x=geo['dense_block_h4']['pixel_edges_x']),
        g64=dict(shape=geo['g64']['shape'],
                 base_index_edges_y=geo['g64']['base_index_edges_y'],
                 base_index_edges_x=geo['g64']['base_index_edges_x'],
                 pixel_edges_y=geo['g64']['pixel_edges_y'],
                 pixel_edges_x=geo['g64']['pixel_edges_x'],
                 v3a43_nesting_level=str(G64),
                 v3a43_pixel_edges_y=[int(v) for v in ref_level['pixel_edges_y']]),
        source='V3-A.4.3 oracle/nesting.json',
        note='target q*, feature pooling and gate expansion all use these edges')
    json.dump(geometry, open(os.path.join(a.root, 'targets', 'geometry.json'), 'w',
                             encoding='utf-8'), indent=2, sort_keys=True)
    json.dump(energy_stats, open(os.path.join(a.root, 'targets',
                                              'energy_stats.json'), 'w',
                                 encoding='utf-8'), indent=2, sort_keys=True)
    json.dump(target_stats, open(os.path.join(a.root, 'targets',
                                              'target_stats.json'), 'w',
                                 encoding='utf-8'), indent=2, sort_keys=True)
    print('energy threshold (p%.0f) = %.6e' % (ENERGY_PCTL, thr))
    for m in MODES:
        print('  mask valid frac %-16s %s' % (m, {s: round(
            energy_stats['mask_valid_frac']['%s|%s' % (m, s)], 3) for s in STATES}))
    print('geometry -> %s/targets/geometry.json (native %s, dense %s, g64 %s)'
          % (a.root, geo['native_feature_shape'], geo['dense_block_h4']['shape'],
             geo['g64']['shape']))

    v4lock = verify_v3a4_artifact_lock(a.v4_root, a.src_root)
    lock = dict(
        repo_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ).decode().strip(),
        proposal_sha256=v4lock['proposal_sha256'],
        cache_metadata_sha256=v4lock['cache_metadata_sha256'],
        manifest_sha256=v4lock['manifest_sha256'],
        split_sha256=v4lock['split_sha256'],
        mismatch_train_sha256=v4lock['mismatch_train_sha256'],
        mismatch_dev_sha256=v4lock['mismatch_dev_sha256'],
        v3a43_summary_sha256=_sha256(os.path.join(a.v43_root, 'oracle', 'summary.json')),
        v3a43_nesting_sha256=_sha256(os.path.join(a.v43_root, 'oracle', 'nesting.json')),
        v3a43_per_image_sha256=_sha256(os.path.join(a.v43_root, 'oracle',
                                                    'per_image.csv')),
        v3a43_root=a.v43_root,
        v3a43_summary_path=os.path.join(a.v43_root, 'oracle', 'summary.json'),
        v3a43_nesting_path=os.path.join(a.v43_root, 'oracle', 'nesting.json'),
        v3a43_per_image_path=os.path.join(a.v43_root, 'oracle', 'per_image.csv'),
        geometry_sha256=_sha256(os.path.join(a.root, 'targets', 'geometry.json')),
        energy_stats_sha256=_sha256(os.path.join(a.root, 'targets', 'energy_stats.json')),
        target_stats_sha256=_sha256(os.path.join(a.root, 'targets', 'target_stats.json')),
        states='+'.join(STATES), target_family='nested_blockwise_action_optimal',
        reference_variant=a.variant, cache_name=a.cache_name,
        state_probs=list(STATE_PROBS), block_h4_factor=4,
        arms=list(ARMS), modes=[ARM_MODE[x] for x in ARMS],
        g64_shape=geo['g64']['shape'], dense_shape=geo['dense_block_h4']['shape'],
        g64_base_index_edges_y=geo['g64']['base_index_edges_y'],
        g64_base_index_edges_x=geo['g64']['base_index_edges_x'],
        g64_edges_y=geo['g64']['pixel_edges_y'], g64_edges_x=geo['g64']['pixel_edges_x'],
        energy_threshold=thr, energy_pctl=ENERGY_PCTL,
        optimizer='adam', lr='1e-4 (0-2000) -> 5e-5 (2000-3000)', steps=3000,
        grad_accum_default=4, seed=42,
        energy_pool_images=len(rows), energy_pool_limit=a.limit,
        reference_geometry=[REF_H, REF_W])

    path = os.path.join(a.root, 'artifact_lock.json')
    if os.path.isfile(path):
        old = json.load(open(path, encoding='utf-8'))
        diff = [k for k in lock if k != 'repo_commit' and old.get(k) != lock[k]]
        if diff:
            raise SystemExit('existing V3-A.5 lock differs on %s -- a lock pins the '
                             'whole protocol; use a new --root (or delete the lock '
                             'if this root has no checkpoints yet)'
                             % ', '.join(sorted(diff)))
    json.dump(lock, open(path, 'w', encoding='utf-8'), indent=2, sort_keys=True)
    print('V3-A.5 lock -> %s' % path)
    print('  commit  : %s' % lock['repo_commit'][:12])
    print('  seed/steps: %d / %d' % (lock['seed'], lock['steps']))
    print('  v43 anchors: summary %s  nesting %s'
          % (lock['v3a43_summary_sha256'][:12], lock['v3a43_nesting_sha256'][:12]))
    verify_v3a5_artifact_lock(a.root, a.src_root, v4_root=a.v4_root, v43_root=a.v43_root)
    print('  lock verified OK')
    print()
    print('══ §33 protocol summary ══')
    print('  commit            : %s  (clean tree required for a formal run)'
          % lock['repo_commit'][:12])
    print('  proposal SHA      : %s' % lock['proposal_sha256'][:12])
    print('  reference variant : %s' % lock['reference_variant'])
    print('  Y0 cache          : %s' % lock['cache_name'])
    print('  split SHA         : %s   mismatch train/dev: %s / %s'
          % (lock['split_sha256'][:12], lock['mismatch_train_sha256'][:12],
             lock['mismatch_dev_sha256'][:12]))
    print('  A0 target         : Block_H4 %s (native H/4 support)'
          % lock['dense_shape'])
    print('  A1 target         : exact nested G64 %s' % lock['g64_shape'])
    print('  feature pooling   : A0 native / A1 exact nested block mean'
          ' (no adaptive pooling, no interpolation)')
    print('  energy threshold  : %.6e (p%.0f, resolutions share it)'
          % (lock['energy_threshold'], lock['energy_pctl']))
    print('  seed / steps      : %d / %d   grad_accum %d   states %s (p=%s)'
          % (lock['seed'], lock['steps'], lock['grad_accum_default'],
             lock['states'], lock['state_probs']))
    print('  official Test     : disabled')


if __name__ == '__main__':
    main()
