#!/usr/bin/env python
"""V3-A.5 §31 acceptance: target algebra, G64 reproduction, frozen proposal, step0.

Run before any training. Hard-fails on the first violation:

  1. closed form holds for BOTH target geometries
     (D = H-Y0 -> q=1 ; D = 2(H-Y0) -> q=0.5 ; D = -(H-Y0) -> q=0)
  2. G64 reproduces the frozen V3-A.4.3 oracle: PSNR(AO64), cap64 and the pixel
     edges themselves (tol 2e-3 dB / 1e-3 capture)
  3. the proposal is frozen and its SHA matches the lock
  4. the two arms start from a bit-equal initialisation and produce identical
     step0 gates (q = 0.5) and identical step0 outputs
"""

import argparse
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import metrics                          # noqa: E402
from model.V3A5Verifier import V3A5Verifier, build_shared_init     # noqa: E402
from option import parser as option_parser                        # noqa: E402
from v3a42_runtime import _sha256                                 # noqa: E402
from v3a5_pipeline import (correction, load_proposal, load_rows,  # noqa: E402
                           make_dataset, sample_tensors)
from v3a5_runtime import (MODES, STATES, action_optimal_target,   # noqa: E402
                          bit_equal, expand_gate, prepare_geometry,
                          state_dict_sha, target_geometry,
                          validate_run_protocol, verify_v3a5_artifact_lock)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V43 = '/root/data/experiments/v3a43_fine_resolution'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
PSNR_TOL, CAP_TOL = 2e-3, 1e-3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='/root/data/experiments/v3a5_g64_verifier')
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--v43_root', default=V43)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0,
                    help='smoke: check the G64 reproduction on N dev images')
    a = ap.parse_args(_CLI)

    lock = verify_v3a5_artifact_lock(a.root, a.src_root, v4_root=a.v4_root,
                                     v43_root=a.v43_root)
    validate_run_protocol(lock, formal=(a.limit == 0), variant=a.variant,
                          cache_name=a.cache_name)
    checks = []
    print('V3-A.5 §31 acceptance checks')

    def ok(name, **extra):
        checks.append(dict(name=name, ok=True, **extra))
        print('  PASS %-42s %s' % (name, extra or ''))

    # ── 1. target algebra on both geometries ────────────────────────────────
    g = torch.Generator().manual_seed(0)
    y0 = torch.rand(1, 3, 400, 600, generator=g) * 2 - 1
    hr = torch.rand(1, 3, 400, 600, generator=g) * 2 - 1
    for mode in MODES:
        geom = prepare_geometry(target_geometry(400, 600, mode), a.device)
        for label, D, want in (('D=H-Y0', hr - y0, 1.0),
                               ('D=2(H-Y0)', 2 * (hr - y0), 0.5),
                               ('D=-(H-Y0)', -(hr - y0), 0.0)):
            tgt = action_optimal_target(y0.to(a.device), hr.to(a.device),
                                        D.to(a.device), geom)
            got = float(tgt['q_grid'].mean())
            if abs(got - want) > 1e-4:
                raise SystemExit('%s/%s: q=%.6f, expected %.3f'
                                 % (mode, label, got, want))
        ok('target algebra %s' % mode, shape=list(geom['shape']))

    # ── 2. G64 reproduces the frozen V3-A.4.3 oracle ────────────────────────
    nesting = json.load(open(lock['v3a43_nesting_path'], encoding='utf-8'))
    geom64 = target_geometry(400, 600, 'g64')
    ref_y = [int(v) for v in nesting['levels']['64']['pixel_edges_y']]
    ref_x = [int(v) for v in nesting['levels']['64']['pixel_edges_x']]
    if [int(v) for v in geom64['edges'][0]] != ref_y or \
            [int(v) for v in geom64['edges'][1]] != ref_x:
        raise SystemExit('G64 pixel edges differ from the frozen nesting.json')
    # the FEATURE-pooling partition must be the same base-cell grouping, i.e.
    # pixel_edges / cell_size on the native lattice (§8)
    base = target_geometry(400, 600, 'dense_block_h4')
    cell_y = int(base['edges'][0][1] - base['edges'][0][0])
    cell_x = int(base['edges'][1][1] - base['edges'][1][0])
    if [int(v) // cell_y for v in ref_y] != [int(v) for v in
                                             geom64['base_index_edges'][0]] or \
            [int(v) // cell_x for v in ref_x] != [int(v) for v in
                                                  geom64['base_index_edges'][1]]:
        raise SystemExit('feature-pooling base-index edges differ from the '
                         'V3-A.4.3 G64 partition')
    ok('g64 target+pooling edges == V3-A.4.3 nesting.json',
       nby=geom64['shape'][0])

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    rows = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                     os.path.join(a.v4_root, 'splits', 'split.json'))['dev']
    mmap = json.load(open(os.path.join(a.v4_root, 'mappings', 'mismatch_dev_64.json'),
                          encoding='utf-8'))
    ds = make_dataset(ns, rows, os.path.join(a.src_root, a.cache_name,
                                             'refiner_train'), mmap)
    proposal = load_proposal(os.path.join(a.src_root, R1_CK), a.device)
    if any(p.requires_grad for p in proposal.parameters()):
        raise SystemExit('the proposal is not frozen')
    if _sha256(os.path.join(a.src_root, R1_CK)) != lock['proposal_sha256']:
        raise SystemExit('proposal SHA does not match the lock')
    ok('proposal frozen + SHA matches the lock')

    dev64 = prepare_geometry(target_geometry(400, 600, 'g64'), a.device)
    devh4 = prepare_geometry(target_geometry(400, 600, 'dense_block_h4'), a.device)
    # the reproduction table is an INPUT, so it must come from the lock -- not
    # from --v43_root. Otherwise the nesting check could read tree A while the
    # reproduction reads tree B.
    frozen = {(r['split'], r['state'], r['sample_id']): r for r in
              csv.DictReader(open(lock['v3a43_per_image_path'], encoding='utf-8'))}
    n = len(rows) if not a.limit else min(a.limit, len(rows))
    worst_psnr, worst_cap, seen = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(n):
            for state in STATES:
                t = sample_tensors(ds, i, state, a.device)
                D, sr = correction(proposal.proposal, t['Y0'], t['R'])
                t64 = action_optimal_target(t['Y0'], t['H'], D, dev64)
                th4 = action_optimal_target(t['Y0'], t['H'], D, devh4)
                p_r1 = metrics(sr, t['H'])[0]
                p64 = metrics(t['Y0'] + t64['q_full'] * D, t['H'])[0]
                p_h4 = metrics(t['Y0'] + th4['q_full'] * D, t['H'])[0]
                ref = frozen[('dev', state, t['name'])]
                worst_psnr = max(worst_psnr, abs(p64 - float(ref['G64'])),
                                 abs(p_h4 - float(ref['Block_H4'])),
                                 abs(p_r1 - float(ref['R1'])))
                if p_h4 - p_r1 > 0:
                    worst_cap = max(worst_cap, abs(
                        (p64 - p_r1) / (p_h4 - p_r1) - float(ref['capture_G64'])))
                seen += 1
    if worst_psnr > PSNR_TOL:
        raise SystemExit('G64/Block_H4/R1 PSNR differs from V3-A.4.3 by %.4f dB '
                         '(tol %.0e)' % (worst_psnr, PSNR_TOL))
    if worst_cap > CAP_TOL:
        raise SystemExit('cap64 differs from V3-A.4.3 by %.4f (tol %.0e)'
                         % (worst_cap, CAP_TOL))
    ok('g64 reproduction vs V3-A.4.3', cells=seen,
       dPSNR=round(worst_psnr, 8), dcap64=round(worst_cap, 8))

    # ── 3. step-0 fairness (§15) ────────────────────────────────────────────
    init = build_shared_init('dense_block_h4', 'g64', seed=int(lock['seed']))
    if not bit_equal(init['dense_block_h4'], init['g64']):
        raise SystemExit('the two arms do not share a bit-equal initialisation')
    ok('shared init bit-equal', sha=state_dict_sha(init['g64'])[:12])
    t = sample_tensors(ds, 0, 'correct', a.device)
    step0 = {}
    for mode in MODES:
        m = V3A5Verifier(mode).to(a.device).eval()
        m.load_state_dict({k: v.clone() for k, v in init[mode].items()}, strict=True)
        gm = prepare_geometry(target_geometry(400, 600, mode), a.device)
        with torch.no_grad():
            D, _sr = correction(proposal.proposal, t['Y0'], t['R'])
            q = m(t['X'], t['Y0'], t['R'], geom=gm)
            step0[mode] = dict(q_mean=float(q.mean()), q_std=float(q.std()),
                               psnr=metrics(t['Y0'] + expand_gate(q, gm) * D,
                                            t['H'])[0])
        if abs(step0[mode]['q_mean'] - 0.5) > 1e-6 or step0[mode]['q_std'] > 1e-6:
            raise SystemExit('%s step0 gate is not the shared 0.5 constant: %s'
                             % (mode, step0[mode]))
    if abs(step0['g64']['psnr'] - step0['dense_block_h4']['psnr']) > 1e-9:
        raise SystemExit('step0 outputs differ across arms: %s' % step0)
    ok('step0 identical across arms', q_mean=round(step0['g64']['q_mean'], 6),
       psnr=round(step0['g64']['psnr'], 6))

    print('\n%d/%d checks passed' % (len(checks), len(checks)))
    json.dump(dict(checks=checks, limit=a.limit, commit=lock['repo_commit']),
              open(os.path.join(a.root, 'targets', 'acceptance.json'), 'w',
                   encoding='utf-8'), indent=2, sort_keys=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
