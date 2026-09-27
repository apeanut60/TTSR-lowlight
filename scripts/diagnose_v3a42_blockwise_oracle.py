#!/usr/bin/env python
"""V3-A.4.2 §9-§15: true blockwise gate-resolution oracle sweep.

For every image x state the frozen proposal is run ONCE,

    D = g_v2 * delta = proposal(Y0, R)

and then seven gates are compared:

    Base          no correction
    R1            q = 1 (the raw proposal output, i.e. V3-A.1's Y_hat)
    G1 .. G16     piecewise-constant *blockwise oracle* at that grid
    Spatial_H4    the V3-A.4.1 H/4 bilinear oracle (upper bound)

The G-grid gate is the optimum under a piecewise-constant parameterisation,

    q_B = clip( sum_B (H-Y0)*D / (sum_B D^2 + eps), 0, 1 )

and is expanded by nearest block assignment. It is never an AvgPool / resize of
the dense q map: that is a smoothed spatial gate, not a coarse oracle (§4.1).

Read-only: torch.no_grad(), every parameter frozen, no optimizer, no checkpoint,
official Test never touched.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# option.py calls parser.parse_args() at import time; stash this script's flags
# first so they are not consumed by the option parser.
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from dataset.lolv2real_v3a import (TrainSet, pairs_from_manifest,  # noqa: E402
                                   read_model_image)
from local_refine_runtime import metrics                          # noqa: E402
from model.V3A4Verifier import V3A4Refiner                        # noqa: E402
from option import parser as option_parser                        # noqa: E402
from v3a42_runtime import (blockwise_action_optimal_gate,         # noqa: E402
                           git_state,
                           spatial_oracle_q4,
                           verify_v3a42_artifact_lock)
from v3a4_runtime import load_r1_proposal_strict                  # noqa: E402
from v3a_runtime import exposure_gain                             # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
STATES = ('correct', 'true_dark_g0.5', 'mismatch')
# V3-A.4.1's Global/Spatial oracle means, used for the §14 reproduction check
BASELINE = '/root/data/experiments/v3a41_target_audit_v2/oracle/global_vs_spatial.json'
CAP_TOL, PSNR_TOL = 1e-3, 2e-3


class _Tee(object):
    """stdout + the run log, so logs/ matches §21 without shell plumbing."""

    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')

    def __call__(self, msg=''):
        print(msg)
        self.f.write(msg + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def parse_grids(text):
    grids = [int(g) for g in str(text).split(',') if g.strip()]
    if not grids:
        raise SystemExit('--grids is empty')
    if grids != sorted(set(grids)):
        raise SystemExit('--grids must be strictly increasing, got %r' % text)
    if 1 not in grids:
        raise SystemExit('--grids must contain 1: 1x1 IS the V3-A.4.1 Global-AO '
                         'and is the reproduction anchor')
    return grids


def gate_stats(q):
    """Descriptive stats of the gate at the resolution it is parameterised at
    (q_grid for a block oracle, q4 for Spatial-H/4), never of the upsampled map."""
    f = q.detach().float().reshape(-1)
    return dict(q_mean=float(f.mean()), q_std=float(f.std()),
                q_min=float(f.min()), q_max=float(f.max()),
                frac_q0=float((f == 0).float().mean()),
                frac_q1=float((f == 1).float().mean()))


def capture_stats(r1, arm, spatial):
    """Per-image H_G/H_S. Only defined where H_S > 0; a negative denominator
    would produce ratios with no meaning (§12: this is an AUXILIARY number)."""
    n = 0
    vals = []
    for a, b, c in zip(r1, arm, spatial):
        h_s = c - a
        if h_s > 0:
            vals.append((b - a) / h_s)
            n += 1
    if not vals:
        return dict(per_image_mean=None, per_image_median=None, n_valid=0)
    v = np.asarray(vals, dtype=np.float64)
    return dict(per_image_mean=float(v.mean()), per_image_median=float(np.median(v)),
                per_image_max=float(v.max()), n_valid=n)


def mean_gate_stats(rows):
    keys = sorted(rows[0])
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--root',
                    default='/root/data/experiments/v3a42_blockwise_oracle')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--splits', default='dev,train')
    ap.add_argument('--grids', default='1,2,4,8,16')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0,
                    help='debug/smoke: only the first N images per split')
    a = ap.parse_args(_CLI)

    dev = a.device
    grids = parse_grids(a.grids)
    arms = ['R1'] + ['G%d' % g for g in grids] + ['Spatial_H4']
    split_tags = [s.strip() for s in a.splits.split(',') if s.strip()]
    for tag in split_tags:
        if tag not in ('train', 'dev'):
            raise SystemExit('--splits only accepts train/dev, got %r' % tag)
    if not split_tags:
        raise SystemExit('--splits is empty')

    os.makedirs(os.path.join(a.root, 'oracle'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    log = _Tee(os.path.join(a.root, 'logs', 'diagnose_blockwise_oracle.log'))

    lock = verify_v3a42_artifact_lock(a.root, a.src_root, v4_root=a.v4_root)
    locked_grids = [int(g) for g in str(lock['grids']).split(',') if g.strip()]
    missing = [g for g in grids if g not in locked_grids]
    if missing:
        raise SystemExit('--grids %r asks for %r which the lock does not cover '
                         '(locked: %r) -- rerun scripts/setup_v3a42_oracle.py'
                         % (a.grids, missing, lock['grids']))

    log('V3-A.4.2 true blockwise gate-resolution oracle')
    log('  oracle def : %s' % lock['oracle_def_version'])
    log('  grids      : %s' % ','.join(str(g) for g in grids))
    log('  splits     : %s' % ','.join(split_tags))
    log('  states     : %s' % ','.join(STATES))
    log('  limit      : %s' % (a.limit or 'none'))
    head, dirty = git_state()
    log('  git        : %s%s' % ((head or 'unknown')[:8],
                                 '' if not dirty else '  DIRTY(%d: %s)'
                                 % (len(dirty), ', '.join(dirty[:3]))))
    log()

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    model = V3A4Refiner('none').to(dev).eval()
    load_r1_proposal_strict(model, os.path.join(a.src_root, R1_CK), dev)
    for p in model.parameters():
        p.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):
        raise SystemExit('a parameter is still trainable -- this is a read-only '
                         'diagnostic')

    split = json.load(open(os.path.join(a.v4_root, 'splits', 'split.json'),
                           encoding='utf-8'))
    all_pairs = {p[0]: p for p in pairs_from_manifest(
        os.path.join(a.src_root, 'manifests', 'refiner_train.csv'))}

    out, per_image, curve_rows = {}, [], []
    for tag in split_tags:
        rows = [all_pairs[i] for i in split['train' if tag == 'train' else 'dev']]
        ds = TrainSet(ns, crop_size=0, pairs=rows,
                      y0_cache=os.path.join(a.src_root, a.cache_name,
                                            'refiner_train'), split='Train')
        mmap = json.load(open(os.path.join(
            a.v4_root, 'mappings',
            'mismatch_%s.json' % ('train_575' if tag == 'train' else 'dev_64')),
            encoding='utf-8'))
        # one accumulator per (state, arm): a single shared 'R1' bucket used to
        # hide a KeyError at aggregation time in an earlier script
        acc = {s: {arm: [] for arm in arms} for s in STATES}
        qacc = {s: {arm: [] for arm in arms} for s in STATES}
        bases = []
        if a.limit:
            rows = rows[:a.limit]
        for i, (name, _low, _high) in enumerate(rows):
            _n, _lr, hr, ref, y0, _mis = ds._load(i)
            donor = ds.ref_map[mmap[name]]
            mis = read_model_image(donor, size=hr.shape[-2:])
            refs = {'correct': ref, 'true_dark_g0.5': exposure_gain(ref, 0.5),
                    'mismatch': mis}
            y0d, hrd = y0[None].to(dev), hr[None].to(dev)
            base = metrics(y0d, hrd)[0]
            bases.append(base)
            with torch.no_grad():
                for s in STATES:
                    rd = refs[s][None].to(dev)
                    sr, aux = model.proposal(y0d, rd)
                    D = aux['gate'] * aux['delta']
                    vals, qmap = {}, {}
                    # R1 is q = 1 by construction (sr == Y0 + D)
                    vals['R1'] = metrics(sr, hrd)[0]
                    qmap['R1'] = torch.ones_like(D)
                    for g in grids:
                        o = blockwise_action_optimal_gate(y0d, hrd, D, g)
                        arm = 'G%d' % g
                        vals[arm] = metrics(y0d + o['q_full'] * D, hrd)[0]
                        qmap[arm] = o['q_grid']
                    q4 = spatial_oracle_q4(y0d, hrd, D)
                    qf = F.interpolate(q4, size=y0d.shape[-2:], mode='bilinear',
                                       align_corners=False)
                    vals['Spatial_H4'] = metrics(y0d + qf * D, hrd)[0]
                    qmap['Spatial_H4'] = q4

                    h_s = vals['Spatial_H4'] - vals['R1']
                    rec = dict(sample_id=name, split=tag, state=s, Base=base)
                    for arm in arms:
                        rec[arm] = vals[arm]
                        rec['delta_%s_vs_R1' % arm] = vals[arm] - vals['R1']
                        gs = gate_stats(qmap[arm])
                        for k, v in gs.items():
                            rec['%s_%s' % (arm, k)] = v
                        acc[s][arm].append(vals[arm])
                        qacc[s][arm].append(gs)
                    for g in grids:
                        arm = 'G%d' % g
                        rec['capture_%s' % arm] = ((vals[arm] - vals['R1']) / h_s
                                                   if h_s > 0 else float('nan'))
                    per_image.append(rec)
            if (i + 1) % 50 == 0:
                log('  %s %d/%d' % (tag, i + 1, len(rows)))

        res = {}
        for s in STATES:
            mean_psnr = {arm: float(np.mean(acc[s][arm])) for arm in arms}
            r1, sp = mean_psnr['R1'], mean_psnr['Spatial_H4']
            h_s = sp - r1
            capture = {}
            for g in grids:
                arm = 'G%d' % g
                h_g = mean_psnr[arm] - r1
                capture[arm] = dict(
                    mean_psnr=mean_psnr[arm], H_G=h_g, H_S=h_s,
                    capture=(h_g / h_s) if h_s > 0 else None)
                capture[arm].update(
                    capture_stats(acc[s]['R1'], acc[s][arm], acc[s]['Spatial_H4']))
            seq = ['G%d' % g for g in grids] + ['Spatial_H4']
            marginal, prev = {}, 'R1'
            for arm in seq:
                marginal['%s->%s' % (prev, arm)] = mean_psnr[arm] - mean_psnr[prev]
                prev = arm
            res[s] = dict(mean_psnr=mean_psnr, mean_base=float(np.mean(bases)),
                          H_S=h_s, capture=capture, marginal_gain=marginal,
                          q_stats={arm: mean_gate_stats(qacc[s][arm])
                                   for arm in arms})
        out[tag] = res
        log('%s done (%d images)' % (tag, len(rows)))

    # ── repro check against V3-A.4.1 (§14) ──────────────────────────────────
    repro = dict(checked=False, tol_capture=CAP_TOL, tol_psnr=PSNR_TOL,
                 baseline=BASELINE, rows=[])
    if a.limit:
        log('[repro] --limit %d: skipped (debug/smoke run)' % a.limit)
    elif not os.path.isfile(BASELINE):
        log('[repro] V3-A.4.1 baseline missing (%s) -- skipped' % BASELINE)
    else:
        base = json.load(open(BASELINE, encoding='utf-8'))
        bad = []
        for tag in split_tags:
            for s in STATES:
                ref = base.get(tag, {}).get(s)
                if ref is None:
                    continue
                got = out[tag][s]
                d_psnr = abs(got['mean_psnr']['G1'] - ref['Global_AO'])
                d_spat = abs(got['mean_psnr']['Spatial_H4'] - ref['Spatial_AO'])
                d_cap = abs(got['capture']['G1']['capture'] - ref['capture_global'])
                row = dict(split=tag, state=s, psnr_G1=got['mean_psnr']['G1'],
                           psnr_G1_v3a41=ref['Global_AO'], d_psnr=d_psnr,
                           psnr_S=got['mean_psnr']['Spatial_H4'],
                           psnr_S_v3a41=ref['Spatial_AO'], d_psnr_spatial=d_spat,
                           capture_G1=got['capture']['G1']['capture'],
                           capture_v3a41=ref['capture_global'], d_capture=d_cap)
                repro['rows'].append(row)
                log('[repro] %-5s %-16s dPSNR(G1)=%.2e dPSNR(S)=%.2e dcap=%.2e'
                    % (tag, s, d_psnr, d_spat, d_cap))
                if d_psnr > PSNR_TOL or d_spat > PSNR_TOL or d_cap > CAP_TOL:
                    bad.append('%s/%s' % (tag, s))
        repro['checked'] = True
        repro['matched'] = not bad
        if bad:
            raise SystemExit(
                'V3-A.4.2 G1/Spatial did not reproduce V3-A.4.1 on %s '
                '(tol: psnr %.0e, capture %.0e) -- either the V3-A.4.1 baseline '
                'moved or the blockwise oracle is wrong; do not read the sweep'
                % (', '.join(bad), PSNR_TOL, CAP_TOL))
        log('[repro] G1 == V3-A.4.1 Global-AO and Spatial_H4 == Spatial-AO on '
            'all %d rows' % len(repro['rows']))

    # ── write artifacts ─────────────────────────────────────────────────────
    summary = dict(protocol=dict(
        experiment='v3a42_blockwise_oracle', oracle_def_version=lock['oracle_def_version'],
        grids=grids, arms=arms, states=list(STATES), splits=split_tags,
        limit=a.limit, src_root=a.src_root, v4_root=a.v4_root,
        data_dir=a.data_dir, variant=a.variant, cache_name=a.cache_name,
        proposal_ckpt=R1_CK, repro_baseline=BASELINE,
        lock_repo_commit=lock.get('repo_commit'), git_head=head,
        git_dirty=dirty[:20], git_dirty_count=len(dirty)),
        repro=repro, splits=out)
    json.dump(summary, open(os.path.join(a.root, 'oracle', 'summary.json'), 'w',
                            encoding='utf-8'), indent=2, sort_keys=True)

    with open(os.path.join(a.root, 'oracle', 'per_image.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)

    for tag in split_tags:
        for s in STATES:
            r1 = out[tag][s]['mean_psnr']['R1']
            prev = r1
            for arm in ['R1'] + ['G%d' % g for g in grids] + ['Spatial_H4']:
                mp = out[tag][s]['mean_psnr'][arm]
                cap = out[tag][s]['capture'].get(arm, {})
                curve_rows.append(dict(
                    split=tag, state=s, arm=arm, mean_psnr=mp,
                    H_vs_R1=mp - r1,
                    capture_vs_H4=cap.get('capture'),
                    per_image_capture_mean=cap.get('per_image_mean'),
                    per_image_capture_n=cap.get('n_valid'),
                    marginal_gain_vs_prev=mp - prev))
                prev = mp
    with open(os.path.join(a.root, 'oracle', 'capture_curve.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        w.writeheader()
        w.writerows(curve_rows)

    # ── report ──────────────────────────────────────────────────────────────
    log()
    log('══ Primary: mean PSNR by gate resolution (§14) ══')
    log('  %-5s %-16s' % ('split', 'state')
        + ''.join(' %8s' % k for k in ['Base', 'R1']
                  + ['G%d' % g for g in grids] + ['H/4']))
    for tag in split_tags:
        for s in STATES:
            e = out[tag][s]
            vals = ([e['mean_base'], e['mean_psnr']['R1']]
                    + [e['mean_psnr']['G%d' % g] for g in grids]
                    + [e['mean_psnr']['Spatial_H4']])
            log('  %-5s %-16s' % (tag, s) + ''.join(' %8.4f' % v for v in vals))
    log()
    log('══ Primary: headroom capture vs the H/4 upper bound (§11-§14) ══')
    log('  %-5s %-16s' % ('split', 'state')
        + ''.join(' %9s %6s' % ('H_G%d' % g, 'cap%d' % g) for g in grids))
    for tag in split_tags:
        for s in STATES:
            e = out[tag][s]
            cells = ''
            for g in grids:
                c = e['capture']['G%d' % g]
                cap = ('%.3f' % c['capture']) if c['capture'] is not None else 'n/a'
                cells += ' %9.4f %6s' % (c['H_G'], cap)
            log('  %-5s %-16s' % (tag, s) + cells)
        for s in STATES:
            log('    %-5s %-16s H_S = %+.4f  cap16 = %s  '
                'per-image cap16 median = %s'
                % (tag, s, out[tag][s]['H_S'],
                   ('%.3f' % out[tag][s]['capture']['G16']['capture'])
                   if out[tag][s]['capture']['G16']['capture'] is not None
                   else 'n/a',
                   ('%.3f' % out[tag][s]['capture']['G16']['per_image_median'])
                   if out[tag][s]['capture']['G16']['per_image_median'] is not None
                   else 'n/a'))
    log()
    log('══ Secondary: marginal gain (§13, §17) ══')
    for tag in split_tags:
        for s in STATES:
            m = out[tag][s]['marginal_gain']
            log('  %-5s %-16s %s' % (tag, s, '  '.join(
                '%s %+.4f' % (k, v) for k, v in m.items())))
    log()
    log('artifacts -> %s/oracle/{summary.json, per_image.csv, capture_curve.csv}'
        % a.root)
    log.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
