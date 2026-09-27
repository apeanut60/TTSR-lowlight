#!/usr/bin/env python
"""V3-A.4.3 §9-§12: fine-resolution saturation sweep (G16 / G32 / G64 / Block_H4).

Inherits V3-A.4.2 exactly: same base partition (Block_H4 = 100x150 cells on
400x600), same blockwise optimum, same primary denominator (Block_H4), same
legacy anchor outside the resolution curve.

New: the ladder is a **verified nested chain** (v3a43_runtime.build_level_chain).
``round(linspace)`` on the base lattice is tried first -- so G16 stays
bit-identical to the frozen V3-A.4.2 sweep -- and any level that would drop a
previous boundary is rebuilt by hierarchical refinement. The continuous-MSE
chain is additionally checked at run time on every image x state, so the
adjacent marginal gains can only reflect resolution.

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
                           h4_block_grid, spatial_oracle_q4)
from v3a43_runtime import (build_level_chain, check_v3a42_reproduction,  # noqa: E402
                           check_worktree, level_geometry_report,
                           mse_chain_violations,
                           saturation_metrics, verify_v3a43_artifact_lock)
from v3a4_runtime import load_r1_proposal_strict                  # noqa: E402
from v3a_runtime import exposure_gain                             # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
# NOTE: the V3-A.4.2 reproduction anchor is NOT configured here -- it is read
# from artifact_lock.json (see main), so the anchor cannot be swapped at run
# time while the lock validates a different file.
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
STATES = ('correct', 'true_dark_g0.5', 'mismatch')
BLOCK_ARM = 'Block_H4'
LEGACY_ARM = 'Spatial_H4_legacy'


class _Tee(object):
    """stdout + the run log, so logs/ matches §22 without shell plumbing."""

    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')

    def __call__(self, msg=''):
        print(msg)
        self.f.write(msg + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def parse_ints(text, what, required=()):
    vals = [int(v) for v in str(text).split(',') if v.strip()]
    if not vals:
        raise SystemExit('%s is empty' % what)
    if vals != sorted(set(vals)):
        raise SystemExit('%s must be strictly increasing, got %r' % (what, text))
    for r in required:
        if r not in vals:
            raise SystemExit('%s must contain %d' % (what, r))
    return vals


def gate_stats(q):
    """Gate statistics at the resolution the gate is parameterised at."""
    f = q.detach().float().reshape(-1)
    return dict(q_mean=float(f.mean()), q_std=float(f.std()),
                q_min=float(f.min()), q_max=float(f.max()),
                frac_q0=float((f == 0).float().mean()),
                frac_q1=float((f == 1).float().mean()))


def capture_stats(r1, arm, spatial):
    """Per-image H/denominator. Auxiliary only; primary is the mean-PSNR ratio."""
    vals = [(b - a) / (c - a) for a, b, c in zip(r1, arm, spatial) if c - a > 0]
    if not vals:
        return dict(per_image_mean=None, per_image_median=None, n_valid=0)
    v = np.asarray(vals, dtype=np.float64)
    return dict(per_image_mean=float(v.mean()), per_image_median=float(np.median(v)),
                per_image_max=float(v.max()), n_valid=len(vals))


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
                    default='/root/data/experiments/v3a43_fine_resolution')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--splits', default='dev,train')
    ap.add_argument('--grids', default='1,2,4,8,16,32,64')
    ap.add_argument('--levels', default='16,32,64',
                    help='primary saturation ladder (§5); must be in --grids')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0,
                    help='debug/smoke: only the first N images per split')
    a = ap.parse_args(_CLI)

    dev = a.device
    grids = parse_ints(a.grids, '--grids', required=(16,))
    levels = parse_ints(a.levels, '--levels', required=(16,))
    missing = [g for g in levels if g not in grids]
    if missing:
        raise SystemExit('--levels must be a subset of --grids (missing %r)' % missing)
    arms = (['R1'] + ['G%d' % g for g in grids]
            + [BLOCK_ARM, LEGACY_ARM])
    # ladder for the continuous-MSE nesting check (§13) and the saturation curve
    ladder = ['R1'] + ['G%d' % g for g in grids] + [BLOCK_ARM]
    split_tags = [s.strip() for s in a.splits.split(',') if s.strip()]
    for tag in split_tags:
        if tag not in ('train', 'dev'):
            raise SystemExit('--splits only accepts train/dev, got %r' % tag)
    if not split_tags:
        raise SystemExit('--splits is empty')

    # §20/P1: a formal sweep must be attributable to one revision. Checked
    # before anything else so a dirty tree costs seconds, not 20 minutes.
    git_head, git_dirty, wt_warning = check_worktree(a.limit)

    os.makedirs(os.path.join(a.root, 'oracle'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    log = _Tee(os.path.join(a.root, 'logs', 'diagnose_fine_resolution.log'))

    lock = verify_v3a43_artifact_lock(a.root, a.src_root, v4_root=a.v4_root)
    # P1: the LOCK is the source of truth for the reproduction anchor. Deriving
    # it from a CLI flag would let the lock validate summary A while the
    # reproduction actually read summary B.
    v42_summary = lock['v3a42_summary_path']
    locked_grids = [int(g) for g in str(lock['grids']).split(',') if g.strip()]
    extra = [g for g in grids if g not in locked_grids]
    if extra:
        raise SystemExit('--grids asks for %r which the lock does not cover '
                         '(locked %r) -- rerun setup with the same --grids'
                         % (extra, lock['grids']))
    if [int(v) for v in str(lock['resolution_levels']).split(',')] != levels:
        raise SystemExit('--levels %r != locked resolution_levels %r'
                         % (levels, lock['resolution_levels']))
    factor = int(lock['block_h4_factor'])
    if factor != 4:
        raise SystemExit('this experiment is defined at H/4; the lock says %d'
                         % factor)
    if lock.get('states') != '+'.join(STATES):
        raise SystemExit('lock states %r != the computed states %r'
                         % (lock.get('states'), '+'.join(STATES)))

    log('V3-A.4.3 fine-resolution saturation sweep')
    log('  oracle def : %s' % lock['oracle_def_version'])
    log('  grids      : %s' % ','.join(str(g) for g in grids))
    log('  levels     : %s (primary saturation ladder)' % ','.join(str(l) for l in levels))
    log('  nesting    : %s (from the lock)' % lock['nesting_scheme'])
    log('  h4 factor  : 1/%d' % factor)
    log('  splits     : %s' % ','.join(split_tags))
    log('  states     : %s' % ','.join(STATES))
    log('  limit      : %s' % (a.limit or 'none'))
    log('  repro anchor: %s' % (v42_summary if a.limit == 0 else 'skipped'))
    log('  git        : %s  dirty=%d' % ((git_head or 'unknown')[:8],
                                         len(git_dirty)))
    if wt_warning:
        log('  git WARNING: %s' % wt_warning)
    log()

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    model = V3A4Refiner('none').to(dev).eval()
    load_r1_proposal_strict(model, os.path.join(a.src_root, R1_CK), dev)
    for p in model.parameters():
        p.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):
        raise SystemExit('a parameter is still trainable -- read-only diagnostic')

    split = json.load(open(os.path.join(a.v4_root, 'splits', 'split.json'),
                           encoding='utf-8'))
    all_pairs = {p[0]: p for p in pairs_from_manifest(
        os.path.join(a.src_root, 'manifests', 'refiner_train.csv'))}

    out, per_image, curve_rows = {}, [], []
    seen_base_grids, mse_violations = set(), []
    nesting_geo = [None]
    n_checked = [0]
    for tag in split_tags:
        rows = [all_pairs[i] for i in split['train' if tag == 'train' else 'dev']]
        ds = TrainSet(ns, crop_size=0, pairs=rows,
                      y0_cache=os.path.join(a.src_root, a.cache_name,
                                            'refiner_train'), split='Train')
        mmap = json.load(open(os.path.join(
            a.v4_root, 'mappings',
            'mismatch_%s.json' % ('train_575' if tag == 'train' else 'dev_64')),
            encoding='utf-8'))
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
            base_psnr = metrics(y0d, hrd)[0]
            bases.append(base_psnr)
            h, w = y0d.shape[-2], y0d.shape[-1]
            base_grid = h4_block_grid(h, w, factor)
            seen_base_grids.add(tuple(int(v) for v in base_grid))
            chain = build_level_chain(h, w, base_grid, grids)
            if nesting_geo[0] is None:
                nesting_geo[0] = level_geometry_report(h, w, base_grid, chain)
                if not chain['all_nested']:
                    raise SystemExit('the ladder is not nested on the real '
                                     'geometry: %s' % chain['containment'])
            with torch.no_grad():
                for s in STATES:
                    rd = refs[s][None].to(dev)
                    sr, aux = model.proposal(y0d, rd)
                    D = aux['gate'] * aux['delta']
                    vals, qmap, mses = {}, {}, {}
                    vals['R1'] = metrics(sr, hrd)[0]
                    qmap['R1'] = torch.ones_like(D[:, :1])
                    mses['R1'] = float((y0d + D - hrd).pow(2).mean())
                    for g in grids:
                        # verified nested chain: every block is a union of
                        # Block_H4 base cells and of every finer level
                        o = blockwise_action_optimal_gate(
                            y0d, hrd, D, g,
                            edges=(chain['y'][g], chain['x'][g]))
                        arm = 'G%d' % g
                        vals[arm] = metrics(y0d + o['q_full'] * D, hrd)[0]
                        qmap[arm] = o['q_grid']
                        mses[arm] = float((o['Y_oracle'] - hrd).pow(2).mean())
                    ob = blockwise_action_optimal_gate(y0d, hrd, D, base_grid)
                    vals[BLOCK_ARM] = metrics(y0d + ob['q_full'] * D, hrd)[0]
                    qmap[BLOCK_ARM] = ob['q_grid']
                    mses[BLOCK_ARM] = float((ob['Y_oracle'] - hrd).pow(2).mean())
                    bad = mse_chain_violations(mses, ladder)
                    n_checked[0] += 1
                    if bad:
                        mse_violations.extend([dict(split=tag, state=s,
                                                    sample_id=name, **b)
                                               for b in bad])
                    # legacy anchor: different formulation, never in the ladder
                    q4 = spatial_oracle_q4(y0d, hrd, D, factor=factor)
                    qf = F.interpolate(q4, size=y0d.shape[-2:], mode='bilinear',
                                       align_corners=False)
                    vals[LEGACY_ARM] = metrics(y0d + qf * D, hrd)[0]
                    qmap[LEGACY_ARM] = q4

                    h_block = vals[BLOCK_ARM] - vals['R1']
                    h_legacy = vals[LEGACY_ARM] - vals['R1']
                    rec = dict(sample_id=name, split=tag, state=s, Base=base_psnr,
                               block_h4_nby=ob['nby'], block_h4_nbx=ob['nbx'])
                    for arm in arms:
                        rec[arm] = vals[arm]
                        rec['delta_%s_vs_R1' % arm] = vals[arm] - vals['R1']
                        rec['mse_%s' % arm] = mses.get(arm)
                        gs = gate_stats(qmap[arm])
                        for k, v in gs.items():
                            rec['%s_%s' % (arm, k)] = v
                        acc[s][arm].append(vals[arm])
                        qacc[s][arm].append(gs)
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
            h_block = mean_psnr[BLOCK_ARM] - r1
            h_legacy = mean_psnr[LEGACY_ARM] - r1
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
                                                     acc[s][BLOCK_ARM])))
                capture[arm].update(
                    _prefixed('legacy', capture_stats(acc[s]['R1'], acc[s][arm],
                                                      acc[s][LEGACY_ARM])))
            res[s] = dict(
                mean_psnr=mean_psnr, mean_base=float(np.mean(bases)),
                H_S_block_h4=h_block, H_S_legacy_h4=h_legacy,
                capture=capture,
                saturation=saturation_metrics(mean_psnr, levels,
                                              block_arm=BLOCK_ARM),
                marginal_gain=saturation_metrics(mean_psnr, grids,
                                                 block_arm=BLOCK_ARM,
                                                 start_arm='R1'),
                legacy_marginal=dict(
                    Spatial_H4_legacy_vs_R1=h_legacy,
                    Spatial_H4_legacy_vs_Block_H4=(mean_psnr[LEGACY_ARM]
                                                   - mean_psnr[BLOCK_ARM])),
                q_stats={arm: mean_gate_stats(qacc[s][arm]) for arm in arms})
        out[tag] = res
        log('%s done (%d images)' % (tag, len(rows)))

    # ── §21 reproduction against the frozen V3-A.4.2 sweep ──────────────────
    repro = dict(checked=False, anchor=v42_summary, rows=[])
    if a.limit:
        log('[repro] --limit %d: skipped (debug/smoke run)' % a.limit)
    else:
        base = json.load(open(v42_summary, encoding='utf-8'))['splits']
        bad_rows = []
        for tag in split_tags:
            for s in STATES:
                ref = base.get(tag, {}).get(s)
                if ref is None:
                    raise SystemExit('V3-A.4.2 summary has no %s/%s cell -- the '
                                     'anchor is not usable' % (tag, s))
                row = check_v3a42_reproduction(out[tag][s], ref,
                                               tol_psnr=2e-3, tol_capture=1e-3)
                row.update(split=tag, state=s)
                repro['rows'].append(row)
                log('[repro] %-5s %-16s dPSNR(G16)=%.2e dPSNR(BlockH4)=%.2e '
                    'dPSNR(LegacyH4)=%.2e d_cap16=%.2e'
                    % (tag, s, row['psnr']['G16']['d'],
                       row['psnr']['Block_H4']['d'],
                       row['psnr']['Spatial_H4_legacy']['d'],
                       row['capture_G16']['d']))
                if not row['ok']:
                    bad_rows.append('%s/%s' % (tag, s))
        repro['checked'] = True
        repro['matched'] = not bad_rows
        if bad_rows:
            raise SystemExit(
                'V3-A.4.3 did not reproduce V3-A.4.2 on %s (tol psnr 2e-3, '
                'capture 1e-3) -- do not read this sweep'
                % ', '.join(bad_rows))
        log('[repro] G16 / Block_H4 / Spatial_H4_legacy / cap16 all reproduced '
            'on %d rows' % len(repro['rows']))

    if mse_violations:
        raise SystemExit('the continuous-MSE nesting chain was violated (%d '
                         'cases, first: %s) -- the ladder is not a refinement '
                         'chain, so the marginal gains would not be pure '
                         'resolution' % (len(mse_violations), mse_violations[0]))
    log('[nesting] continuous-MSE chain verified on %d image x state cells '
        '(MSE(Block_H4) <= ... <= MSE(G1) <= MSE(R1))' % n_checked[0])

    # ── write artifacts (§22) ───────────────────────────────────────────────
    # NOTE: the geometry report already owns the key 'levels' (the per-level
    # block counts / pixel edges), so the plan's resolution ladder goes in as
    # 'analysis_levels' -- overwriting it would silently drop the geometry.
    nesting = dict(nesting_geo[0],
                   grids=grids, analysis_levels=levels,
                   observed_base_grids=[list(g) for g in sorted(seen_base_grids)],
                   ladder=ladder,
                   mse_chain=dict(checked=n_checked[0],
                                  violations=mse_violations[:20],
                                  n_violations=len(mse_violations),
                                  tol=1e-9),
                   lock_nesting_scheme=lock['nesting_scheme'])
    json.dump(nesting, open(os.path.join(a.root, 'oracle', 'nesting.json'), 'w',
                            encoding='utf-8'), indent=2, sort_keys=True)

    summary = dict(protocol=dict(
        experiment='v3a43_fine_resolution',
        oracle_def_version=lock['oracle_def_version'],
        grids=grids, levels=levels, arms=arms, ladder=ladder,
        states=list(STATES), splits=split_tags, limit=a.limit,
        src_root=a.src_root, v4_root=a.v4_root,
        data_dir=a.data_dir, variant=a.variant, cache_name=a.cache_name,
        proposal_ckpt=R1_CK, repro_anchor=v42_summary,
        repro_anchor_source='artifact_lock.json',
        oracle_objective='continuous_rgb_mse_on_[-1,1]_tensors',
        primary_denominator=BLOCK_ARM, secondary_denominator=LEGACY_ARM,
        grid_scheme='verified_nested_from_block_h4',
        nesting_scheme=lock['nesting_scheme'],
        block_h4_factor=factor, block_h4_grid=nesting_geo[0]['base_grid'],
        lock_repo_commit=lock.get('repo_commit'),
        # the REAL worktree state, never a hard-coded zero
        git_head=git_head, git_dirty_count=len(git_dirty),
        git_dirty=git_dirty[:20]),
        repro=repro, nesting_summary=dict(
            all_nested=nesting['all_nested'], scheme=nesting['scheme'],
            mse_chain_checked=n_checked[0], mse_chain_violations=0),
        splits=out)
    json.dump(summary, open(os.path.join(a.root, 'oracle', 'summary.json'), 'w',
                            encoding='utf-8'), indent=2, sort_keys=True)

    with open(os.path.join(a.root, 'oracle', 'per_image.csv'), 'w', newline='',
              encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        wr.writeheader()
        wr.writerows(per_image)

    for tag in split_tags:
        for s in STATES:
            r1 = out[tag][s]['mean_psnr']['R1']
            prev = r1
            for arm in arms:
                mp = out[tag][s]['mean_psnr'][arm]
                cap = out[tag][s]['capture'].get(arm, {})
                step = (None if arm == LEGACY_ARM else mp - prev)
                curve_rows.append(dict(
                    split=tag, state=s, arm=arm, mean_psnr=mp, H_vs_R1=mp - r1,
                    capture_vs_BlockH4=cap.get('capture'),
                    capture_vs_LegacyH4=cap.get('capture_legacy'),
                    per_image_capture_block_mean=cap.get('block_per_image_mean'),
                    per_image_capture_block_n=cap.get('block_n_valid'),
                    marginal_gain_vs_prev=step,
                    is_resolution_step=(arm != LEGACY_ARM)))
                prev = mp
    with open(os.path.join(a.root, 'oracle', 'fine_capture_curve.csv'), 'w',
              newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        wr.writeheader()
        wr.writerows(curve_rows)

    # ── report (§24) ────────────────────────────────────────────────────────
    log()
    log('══ Primary: mean PSNR by resolution (§24) ══')
    log('  %-5s %-16s' % ('split', 'state')
        + ''.join(' %9s' % k for k in ['Base', 'R1']
                  + ['G%d' % g for g in grids] + ['Block_H4', 'Legacy']))
    for tag in split_tags:
        for s in STATES:
            e = out[tag][s]
            vals = ([e['mean_base'], e['mean_psnr']['R1']]
                    + [e['mean_psnr']['G%d' % g] for g in grids]
                    + [e['mean_psnr'][BLOCK_ARM], e['mean_psnr'][LEGACY_ARM]])
            log('  %-5s %-16s' % (tag, s) + ''.join(' %9.4f' % v for v in vals))
    log()
    log('══ Primary: capture vs Block_H4 (same family, resolution only) ══')
    log('  %-5s %-16s' % ('split', 'state')
        + ''.join(' %7s' % ('cap%d' % g) for g in grids)
        + ' %9s' % 'H_BlockH4')
    for tag in split_tags:
        for s in STATES:
            e = out[tag][s]
            cells = ''.join(' %7s' % (
                ('%.3f' % e['capture']['G%d' % g]['capture'])
                if e['capture']['G%d' % g]['capture'] is not None else 'n/a')
                for g in grids)
            log('  %-5s %-16s' % (tag, s) + cells
                + ' %9.4f' % e['H_S_block_h4'])
    log()
    log('══ Primary: saturation ladder Δ (§10-§12) ══')
    log('  %-5s %-16s' % ('split', 'state')
        + ''.join(' %16s' % k for k in
                  ['G%d->G%d' % (a_, b_) for a_, b_ in zip(levels[:-1], levels[1:])]
                  + ['G%d->Block_H4' % levels[-1]]))
    for tag in split_tags:
        for s in STATES:
            sat = out[tag][s]['saturation']
            log('  %-5s %-16s' % (tag, s)
                + ''.join(' %+9.4f(%s)' % (v['delta'], v['band'][:3])
                          for v in sat.values()))
    log()
    log('══ Secondary: capture vs the legacy H/4 oracle ══')
    log('  %-5s %-16s' % ('split', 'state')
        + ''.join(' %7s' % ('capL%d' % g) for g in grids))
    for tag in split_tags:
        for s in STATES:
            e = out[tag][s]
            log('  %-5s %-16s' % (tag, s) + ''.join(' %7s' % (
                ('%.3f' % e['capture']['G%d' % g]['capture_legacy'])
                if e['capture']['G%d' % g]['capture_legacy'] is not None else 'n/a')
                for g in grids))
    log()
    log('══ Full ladder marginal gain (dB) ══')
    for tag in split_tags:
        for s in STATES:
            m = out[tag][s]['marginal_gain']
            log('  %-5s %-16s %s' % (tag, s, '  '.join(
                '%s %+.4f' % (k, v['delta']) for k, v in m.items())))
            lm = out[tag][s]['legacy_marginal']
            log('  %-5s %-16s [legacy anchor, not a resolution step] %s'
                % (tag, s, '  '.join('%s %+.4f' % (k, v) for k, v in lm.items())))
    log()
    log('artifacts -> %s/oracle/{summary.json, per_image.csv, '
        'fine_capture_curve.csv, nesting.json}' % a.root)
    log.close()
    return 0
if __name__ == '__main__':
    sys.exit(main())
