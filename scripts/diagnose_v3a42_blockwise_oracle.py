#!/usr/bin/env python
"""V3-A.4.2 §9-§15: true blockwise gate-resolution oracle sweep.

For every image x state the frozen proposal is run ONCE,

    D = g_v2 * delta = proposal(Y0, R)

and then these gates are compared:

    Base          no correction
    R1            q = 1 (the raw proposal output, i.e. V3-A.1's Y_hat)
    G1 .. G16     piecewise-constant *blockwise oracle* at that grid
    Block_H4      the SAME blockwise oracle at H/4 x W/4 cells (primary end)
    Spatial_H4_legacy  the V3-A.4.1 area-downsample + bilinear oracle

The G-grid gate is the optimum under a piecewise-constant parameterisation,

    q_B = clip( sum_B (H-Y0)*D / (sum_B D^2 + eps), 0, 1 )

and is expanded by nearest block assignment. It is never an AvgPool / resize of
the dense q map: that is a smoothed spatial gate, not a coarse oracle (§4.1).

WHY Block_H4 EXISTS: the legacy H/4 oracle changes **two** things at once --
spatial resolution AND the target formulation (it solves on area-downsampled
images and then bilinearly upsamples the gate). It is therefore NOT an upper
bound of the G curve: PSNR(G16) > PSNR(Spatial_H4_legacy) is possible for a
reason that has nothing to do with resolution. The primary denominator is
Block_H4, which uses the identical full-resolution blockwise formula as the G
curve and only varies the resolution. Legacy is kept as the V3-A.4.1
reproduction anchor and as a secondary (historical) capture.

The oracle minimises the *continuous* RGB MSE on [-1,1] tensors. The PSNR that
gets reported is computed on 8-bit rounded images (local_refine_runtime.metrics
rounds first), so the monotonicity guarantee holds for continuous MSE, not
bit-exactly for the reported PSNR.

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
                           check_v3a41_reproduction,
                           git_state,
                           h4_block_grid,
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


def _prefixed(tag, d):
    return {('%s_%s' % (tag, k)): v for k, v in d.items()}


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
    ap.add_argument('--repro_baseline', default=BASELINE,
                    help='V3-A.4.1 Global/Spatial oracle means; required on a '
                         'full (--limit 0) run')
    a = ap.parse_args(_CLI)

    dev = a.device
    grids = parse_grids(a.grids)
    arms = (['R1'] + ['G%d' % g for g in grids]
            + ['Block_H4', 'Spatial_H4_legacy'])
    split_tags = [s.strip() for s in a.splits.split(',') if s.strip()]
    for tag in split_tags:
        if tag not in ('train', 'dev'):
            raise SystemExit('--splits only accepts train/dev, got %r' % tag)
    if not split_tags:
        raise SystemExit('--splits is empty')
    # A formal sweep MUST be able to prove it reproduces V3-A.4.1. Checking
    # this before the sweep (not after) means a missing anchor fails in seconds
    # instead of after 20 minutes of GPU time.
    if a.limit == 0 and not os.path.isfile(a.repro_baseline):
        raise SystemExit('the V3-A.4.1 reproduction anchor %s does not exist -- '
                         'a --limit 0 run must be able to check G1 and '
                         'Spatial_H4_legacy against it (§14); pass '
                         '--repro_baseline, or use --limit N for a smoke run'
                         % a.repro_baseline)

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
    # consume the locked protocol instead of hard-coding it: a lock that claims
    # a different endpoint resolution must not silently run H/4 anyway
    factor = int(lock['block_h4_factor'])
    if factor != 4:
        raise SystemExit('this experiment is defined at H/4 (factor=4); the lock '
                         'says factor=%d -- the legacy V3-A.4.1 anchor is only '
                         'reproducible at H/4' % factor)
    if lock.get('states') != '+'.join(STATES):
        raise SystemExit('lock states %r != the states this diagnostic computes '
                         '(%r) -- rerun scripts/setup_v3a42_oracle.py'
                         % (lock.get('states'), '+'.join(STATES)))

    log('V3-A.4.2 true blockwise gate-resolution oracle')
    log('  oracle def : %s' % lock['oracle_def_version'])
    log('  grids      : %s' % ','.join(str(g) for g in grids))
    log('  grid scheme: G arms coarsen the Block_H4 base grid (nested families)')
    log('  h4 factor  : 1/%d (from the lock)' % factor)
    log('  splits     : %s' % ','.join(split_tags))
    log('  states     : %s' % ','.join(STATES))
    log('  limit      : %s' % (a.limit or 'none'))
    log('  repro anchor: %s' % (a.repro_baseline if a.limit == 0 else 'skipped'))
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
    block_h4_grid = [None]          # observed (Gy, Gx), recorded in the protocol
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
                    qmap['R1'] = torch.ones_like(D[:, :1])
                    # the single base partition every G arm coarsens
                    base_grid = h4_block_grid(y0d.shape[-2], y0d.shape[-1],
                                              factor)
                    block_h4_grid[0] = base_grid
                    for g in grids:
                        # the G arms COARSEN the Block_H4 base grid, so every
                        # G block is a union of whole base cells and the family
                        # chain is exactly nested (MSE(Block_H4) <= MSE(G))
                        o = blockwise_action_optimal_gate(y0d, hrd, D, g,
                                                          base_grid=base_grid)
                        arm = 'G%d' % g
                        vals[arm] = metrics(y0d + o['q_full'] * D, hrd)[0]
                        qmap[arm] = o['q_grid']
                    # same family as the G curve, only the resolution changes
                    ob = blockwise_action_optimal_gate(y0d, hrd, D, base_grid)
                    vals['Block_H4'] = metrics(y0d + ob['q_full'] * D, hrd)[0]
                    qmap['Block_H4'] = ob['q_grid']
                    # legacy V3-A.4.1 formulation: area-downsample the *data*,
                    # solve there, then bilinearly expand the gate
                    q4 = spatial_oracle_q4(y0d, hrd, D, factor=factor)
                    qf = F.interpolate(q4, size=y0d.shape[-2:], mode='bilinear',
                                       align_corners=False)
                    vals['Spatial_H4_legacy'] = metrics(y0d + qf * D, hrd)[0]
                    qmap['Spatial_H4_legacy'] = q4

                    h_block = vals['Block_H4'] - vals['R1']
                    h_legacy = vals['Spatial_H4_legacy'] - vals['R1']
                    rec = dict(sample_id=name, split=tag, state=s, Base=base)
                    for arm in arms:
                        rec[arm] = vals[arm]
                        rec['delta_%s_vs_R1' % arm] = vals[arm] - vals['R1']
                        gs = gate_stats(qmap[arm])
                        for k, v in gs.items():
                            rec['%s_%s' % (arm, k)] = v
                        acc[s][arm].append(vals[arm])
                        qacc[s][arm].append(gs)
                    rec['capture_block_vs_BlockH4'] = ((vals['Block_H4'] - vals['R1'])
                                                       / h_block if h_block > 0
                                                       else float('nan'))
                    for g in grids:
                        arm = 'G%d' % g
                        h_g = vals[arm] - vals['R1']
                        rec['capture_%s' % arm] = (h_g / h_block
                                                   if h_block > 0 else float('nan'))
                        rec['capture_%s_vs_LegacyH4' % arm] = (
                            h_g / h_legacy if h_legacy > 0 else float('nan'))
                    per_image.append(rec)
            if (i + 1) % 50 == 0:
                log('  %s %d/%d' % (tag, i + 1, len(rows)))

        res = {}
        for s in STATES:
            mean_psnr = {arm: float(np.mean(acc[s][arm])) for arm in arms}
            r1 = mean_psnr['R1']
            h_block = mean_psnr['Block_H4'] - r1
            h_legacy = mean_psnr['Spatial_H4_legacy'] - r1
            capture = {}
            for g in grids:
                arm = 'G%d' % g
                h_g = mean_psnr[arm] - r1
                capture[arm] = dict(
                    mean_psnr=mean_psnr[arm], H_G=h_g,
                    H_S_block_h4=h_block, H_S_legacy_h4=h_legacy,
                    capture=(h_g / h_block) if h_block > 0 else None,
                    capture_legacy=((h_g / h_legacy) if h_legacy > 0 else None))
                capture[arm].update(
                    _prefixed('block', capture_stats(acc[s]['R1'], acc[s][arm],
                                                     acc[s]['Block_H4'])))
                capture[arm].update(
                    _prefixed('legacy', capture_stats(
                        acc[s]['R1'], acc[s][arm], acc[s]['Spatial_H4_legacy'])))
            # the resolution curve stops at Block_H4: it is the SAME family as
            # the G arms. Legacy-H4 is a different formulation, so a
            # "Block_H4 -> Legacy" step is not a resolution marginal gain and
            # must never be read as one (§13/§17).
            seq = ['G%d' % g for g in grids] + ['Block_H4']
            marginal, prev = {}, 'R1'
            for arm in seq:
                marginal['%s->%s' % (prev, arm)] = mean_psnr[arm] - mean_psnr[prev]
                prev = arm
            legacy_marginal = dict(
                Spatial_H4_legacy_vs_R1=mean_psnr['Spatial_H4_legacy'] - r1,
                Spatial_H4_legacy_vs_Block_H4=(mean_psnr['Spatial_H4_legacy']
                                               - mean_psnr['Block_H4']))
            res[s] = dict(mean_psnr=mean_psnr, mean_base=float(np.mean(bases)),
                          H_S_block_h4=h_block, H_S_legacy_h4=h_legacy,
                          capture=capture, marginal_gain=marginal,
                          legacy_marginal=legacy_marginal,
                          q_stats={arm: mean_gate_stats(qacc[s][arm])
                                   for arm in arms})
        out[tag] = res
        log('%s done (%d images)' % (tag, len(rows)))

    # ── repro check against V3-A.4.1 (§14) ──────────────────────────────────
    repro = dict(checked=False, tol_capture=CAP_TOL, tol_psnr=PSNR_TOL,
                 baseline=a.repro_baseline, rows=[])
    if a.limit:
        log('[repro] --limit %d: skipped (debug/smoke run)' % a.limit)
    else:
        # a missing anchor already hard-failed at startup; reaching here means
        # the file exists and the comparison is mandatory
        base = json.load(open(a.repro_baseline, encoding='utf-8'))
        bad = []
        for tag in split_tags:
            for s in STATES:
                ref = base.get(tag, {}).get(s)
                if ref is None:
                    continue
                # capture_legacy (V3-A.4.1's denominator), never our primary
                # Block_H4 capture -- see check_v3a41_reproduction's docstring
                row = check_v3a41_reproduction(out[tag][s], ref,
                                               tol_capture=CAP_TOL,
                                               tol_psnr=PSNR_TOL)
                row.update(split=tag, state=s)
                repro['rows'].append(row)
                prim = row['capture_G1_primary']
                log('[repro] %-5s %-16s dPSNR(G1)=%.2e dPSNR(LegacyH4)=%.2e '
                    'd_capture[%s]=%.2e  (primary capture %.3f, not compared)'
                    % (tag, s, row['d_psnr'], row['d_psnr_spatial'],
                       row['denominator'], row['d_capture'],
                       prim if prim is not None else float('nan')))
                if not row['ok']:
                    bad.append('%s/%s' % (tag, s))
        repro['checked'] = True
        repro['matched'] = not bad
        if bad:
            raise SystemExit(
                'V3-A.4.2 G1 / Spatial_H4_legacy did not reproduce V3-A.4.1 on %s '
                '(tol: psnr %.0e, capture %.0e) -- either the V3-A.4.1 baseline '
                'moved or the blockwise oracle is wrong; do not read the sweep'
                % (', '.join(bad), PSNR_TOL, CAP_TOL))
        log('[repro] G1 == V3-A.4.1 Global-AO and Spatial_H4_legacy == '
            'Spatial-AO on all %d rows' % len(repro['rows']))

    # ── write artifacts ─────────────────────────────────────────────────────
    summary = dict(protocol=dict(
        experiment='v3a42_blockwise_oracle', oracle_def_version=lock['oracle_def_version'],
        grids=grids, arms=arms, states=list(STATES), splits=split_tags,
        limit=a.limit, src_root=a.src_root, v4_root=a.v4_root,
        data_dir=a.data_dir, variant=a.variant, cache_name=a.cache_name,
        proposal_ckpt=R1_CK, repro_baseline=a.repro_baseline,
        # the oracle minimises continuous RGB MSE on [-1,1] tensors; reported
        # PSNR is computed on 8-bit rounded images (local_refine_runtime.metrics)
        oracle_objective='continuous_rgb_mse_on_[-1,1]_tensors',
        primary_denominator='Block_H4',
        secondary_denominator='Spatial_H4_legacy',
        grid_scheme='nested_base_h4',
        block_h4_factor=factor,
        block_h4_grid=block_h4_grid[0],
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
            for arm in arms:
                mp = out[tag][s]['mean_psnr'][arm]
                cap = out[tag][s]['capture'].get(arm, {})
                # a legacy row is NOT a step of the resolution curve
                step = (None if arm == 'Spatial_H4_legacy' else mp - prev)
                curve_rows.append(dict(
                    split=tag, state=s, arm=arm, mean_psnr=mp,
                    H_vs_R1=mp - r1,
                    capture_vs_BlockH4=cap.get('capture'),
                    capture_vs_LegacyH4=cap.get('capture_legacy'),
                    per_image_capture_block_mean=cap.get('block_per_image_mean'),
                    per_image_capture_block_n=cap.get('block_n_valid'),
                    per_image_capture_legacy_mean=cap.get('legacy_per_image_mean'),
                    marginal_gain_vs_prev=step,
                    is_resolution_step=(arm != 'Spatial_H4_legacy')))
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
                  + ['G%d' % g for g in grids]
                  + ['BlkH4', 'LegH4']))
    for tag in split_tags:
        for s in STATES:
            e = out[tag][s]
            vals = ([e['mean_base'], e['mean_psnr']['R1']]
                    + [e['mean_psnr']['G%d' % g] for g in grids]
                    + [e['mean_psnr']['Block_H4'],
                       e['mean_psnr']['Spatial_H4_legacy']])
            log('  %-5s %-16s' % (tag, s) + ''.join(' %8.4f' % v for v in vals))
    log()
    log('══ Primary: capture vs Block_H4 -- SAME family, only resolution ══')
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
            e = out[tag][s]
            c16 = e['capture']['G16']
            cap_l = (('%.3f' % c16['capture_legacy'])
                     if c16['capture_legacy'] is not None else 'n/a')
            med_b = (('%.3f' % c16['block_per_image_median'])
                     if c16['block_per_image_median'] is not None else 'n/a')
            log('    %-5s %-16s H_BlockH4 = %+.4f  H_LegacyH4 = %+.4f  '
                'cap16 = %s (legacy %.3f)  per-image cap16 median = %s'
                % (tag, s, e['H_S_block_h4'], e['H_S_legacy_h4'],
                   ('%.3f' % c16['capture']) if c16['capture'] is not None else 'n/a',
                   c16['capture_legacy'] if c16['capture_legacy'] is not None else float('nan'),
                   med_b))
    log()
    log('══ Secondary: capture vs the LEGACY H/4 oracle (V3-A.4.1 link) ══')
    log('  %-5s %-16s' % ('split', 'state')
        + ''.join(' %6s' % ('capL%d' % g) for g in grids))
    for tag in split_tags:
        for s in STATES:
            e = out[tag][s]
            cells = ''.join(' %6s' % (
                ('%.3f' % e['capture']['G%d' % g]['capture_legacy'])
                if e['capture']['G%d' % g]['capture_legacy'] is not None else 'n/a')
                for g in grids)
            log('  %-5s %-16s' % (tag, s) + cells)
    log()
    log('══ Secondary: marginal gain (§13, §17) ══')
    for tag in split_tags:
        for s in STATES:
            m = out[tag][s]['marginal_gain']
            log('  %-5s %-16s %s' % (tag, s, '  '.join(
                '%s %+.4f' % (k, v) for k, v in m.items())))
            lm = out[tag][s]['legacy_marginal']
            log('  %-5s %-16s [legacy anchor, not a resolution step] %s'
                % (tag, s, '  '.join('%s %+.4f' % (k, v) for k, v in lm.items())))
    log()
    log('artifacts -> %s/oracle/{summary.json, per_image.csv, capture_curve.csv}'
        % a.root)
    log.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
