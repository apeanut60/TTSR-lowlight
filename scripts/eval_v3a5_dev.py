#!/usr/bin/env python
"""V3-A.5 §16-§20: dev64 (and optionally train575) comparison of A0 vs A1.

Per split x state reports:

    Base, R1, A0 dense-BlockH4, A1 G64, AO64, Block_H4 oracle
    Recovery64 = (PSNR(V) - PSNR(R1)) / (PSNR(AO64) - PSNR(R1))
    gate metrics vs the arm's own target (MAE / corr / masked / fracs / mask)

AO64 and the Block_H4 oracle are NOT recomputed: they are read from the frozen,
locked V3-A.4.3 per-image table (the oracle is the learning ceiling, not a
training input). Readings are cross-checked against the locked V3-A.4.3 summary.

Read-only: no training, no optimizer, official Test never touched.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import metrics                          # noqa: E402
from model.V3A5Verifier import V3A5Verifier                       # noqa: E402
from option import parser as option_parser                        # noqa: E402
from v3a5_pipeline import (correction, load_proposal, load_rows,  # noqa: E402
                           make_dataset, sample_tensors)
from v3a5_runtime import (ARM_MODE, ARMS, STATES, action_optimal_target,  # noqa: E402
                          block_energy, check_worktree, energy_mask,
                          expand_gate, gate_metrics, per_image_qmean_corr,
                          prepare_geometry, recovery, target_geometry,
                          validate_run_protocol, verdict_5a,
                          verify_v3a5_artifact_lock)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V43 = '/root/data/experiments/v3a43_fine_resolution'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
ORACLE_COLS = ('R1', 'G64', 'Block_H4')


def load_oracle_table(path):
    """-> {(split, state, sample_id): row} from the frozen V3-A.4.3 table."""
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    table, means = {}, {}
    for r in rows:
        table[(r['split'], r['state'], r['sample_id'])] = r
        for c in ORACLE_COLS:
            if c not in r:
                raise SystemExit('V3-A.4.3 per_image.csv lacks the %s column' % c)
            means.setdefault((r['split'], r['state']), {}).setdefault(
                c, []).append(float(r[c]))
    means = {k: {c: float(np.mean(v[c])) for c in v} for k, v in means.items()}
    return table, means


def cross_check_oracle(means, summary_path):
    """The frozen summary must agree with the per-image means we read."""
    s = json.load(open(summary_path, encoding='utf-8'))['splits']
    bad = []
    for (tag, state), got in means.items():
        ref = s[tag][state]['mean_psnr']
        for c, arm in (('G64', 'G64'), ('Block_H4', 'Block_H4'), ('R1', 'R1')):
            if abs(got[c] - ref[arm]) > 1e-6:
                bad.append('%s/%s/%s' % (tag, state, c))
    if bad:
        raise SystemExit('V3-A.4.3 per_image.csv disagrees with its summary on %s '
                         '-- the oracle anchor is inconsistent' % ', '.join(bad))


def nanmean_dict(rows):
    """Average per-image metric dicts, skipping undefined values (§19).

    A handful of images have an undefined correlation because the G64 oracle's
    optimal gate is constant (q* == 1 everywhere -> AO64 == R1). A plain mean
    would turn that into NaN for the whole state; the count of finite values is
    reported alongside so the reader can see how many images support the number.
    """
    out = {}
    for k in sorted(rows[0]):
        v = np.asarray([r[k] for r in rows], dtype=np.float64)
        finite = np.isfinite(v)
        out[k] = float(v[finite].mean()) if finite.any() else float('nan')
        out[k + '_n_valid'] = int(finite.sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='/root/data/experiments/v3a5_g64_verifier')
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--v43_root', default=V43)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--splits', default='dev,train')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0,
                    help='smoke: only the first N images per split')
    a = ap.parse_args(_CLI)

    split_tags = [s.strip() for s in a.splits.split(',') if s.strip()]
    for tag in split_tags:
        if tag not in ('train', 'dev'):
            raise SystemExit('--splits only accepts train/dev, got %r' % tag)
    git_head, git_dirty, wt_warning = check_worktree(a.limit)
    lock = verify_v3a5_artifact_lock(a.root, a.src_root, v4_root=a.v4_root,
                                     v43_root=a.v43_root)
    proto = validate_run_protocol(lock, formal=(a.limit == 0), variant=a.variant,
                                  cache_name=a.cache_name)
    a.variant = proto['variant']
    a.cache_name = proto['cache_name']
    diag_dir = os.path.join(a.root, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    logf = open(os.path.join(a.root, 'logs', 'eval_dev.log'), 'w', encoding='utf-8')

    def log(msg=''):
        print(msg)
        logf.write(msg + '\n')
        logf.flush()

    log('V3-A.5 dev evaluation (A0 dense-BlockH4 vs A1 G64)')
    log('  commit %s dirty=%d  splits %s  limit %s'
        % ((git_head or '?')[:8], len(git_dirty), ','.join(split_tags),
           a.limit or 'none'))
    if wt_warning:
        log('  git WARNING: %s' % wt_warning)

    # §25/§26: the eval anchor is the LOCKED V3-A.4.3 table, not whatever
    # --v43_root points at
    anchor = lock['v3a43_per_image_path']
    oracle, oracle_means = load_oracle_table(anchor)
    cross_check_oracle(oracle_means, lock['v3a43_summary_path'])
    log('  oracle anchor: V3-A.4.3 per_image.csv cross-checked OK')

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    rows = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                     os.path.join(a.v4_root, 'splits', 'split.json'))
    mmaps = {tag: json.load(open(os.path.join(
        a.v4_root, 'mappings',
        'mismatch_%s.json' % ('train_575' if tag == 'train' else 'dev_64')),
        encoding='utf-8')) for tag in ('train', 'dev')}
    proposal = load_proposal(os.path.join(a.src_root, R1_CK), a.device)
    geoms = {m: prepare_geometry(target_geometry(400, 600, m), a.device)
             for m in ('g64', 'dense_block_h4')}
    thr = float(lock['energy_threshold'])

    arms = {}
    used = {}
    for arm in ARMS:
        # a formal run must be the locked step; a smoke run may use last.pt
        ck = os.path.join(a.root, arm, 'checkpoints', 'ckpt_%06d.pt' % lock['steps'])
        if a.limit and not os.path.isfile(ck):
            ck = os.path.join(a.root, arm, 'checkpoints', 'last.pt')
        if not os.path.isfile(ck):
            raise SystemExit('checkpoint missing for %s: %s (train it first)'
                             % (arm, ck))
        used[arm] = os.path.relpath(ck, a.root)
        blob = torch.load(ck, map_location=a.device)
        model = V3A5Verifier(blob['mode']).to(a.device).eval()
        model.load_state_dict(blob['model'], strict=True)
        for p in model.parameters():
            p.requires_grad_(False)
        arms[arm] = model
        log('  loaded %s (mode %s, step %s, %s)'
            % (arm, blob['mode'], blob.get('step'), used[arm]))

    out, per_image, gate_rows, rec_rows = {}, [], [], []
    for tag in split_tags:
        ds = make_dataset(ns, rows[tag], os.path.join(a.src_root, a.cache_name,
                                                      'refiner_train'), mmaps[tag])
        n = len(rows[tag]) if not a.limit else min(a.limit, len(rows[tag]))
        # the arm keys stay FULL names everywhere in the artifacts, so a
        # reader can never confuse A0_dense_blockh4 with a column named 'A0'
        acc = {s: {k: [] for k in ('Base', 'R1', 'AO64', 'BlockH4', *ARMS)}
               for s in STATES}
        qv = {arm: {s: [] for s in STATES} for arm in ARMS}
        qo = {arm: {s: [] for s in STATES} for arm in ARMS}
        gm = {arm: {s: [] for s in STATES} for arm in ARMS}
        for i in range(n):
            with torch.no_grad():
                for state in STATES:
                    t = sample_tensors(ds, i, state, a.device)
                    D, sr = correction(proposal.proposal, t['Y0'], t['R'])
                    o_row = oracle.get((tag, state, t['name']))
                    if o_row is None:
                        raise SystemExit('no frozen oracle row for %s/%s/%s'
                                         % (tag, state, t['name']))
                    rec = dict(sample_id=t['name'], split=tag, state=state,
                               Base=metrics(t['Y0'], t['H'])[0],
                               R1=metrics(sr, t['H'])[0],
                               AO64=float(o_row['G64']),
                               BlockH4=float(o_row['Block_H4']),
                               oracle_R1=float(o_row['R1']))
                    for arm in ARMS:
                        mode = ARM_MODE[arm]
                        q_v = arms[arm](t['X'], t['Y0'], t['R'],
                                        geom=geoms[mode])
                        q_full = expand_gate(q_v, geoms[mode])
                        rec[arm] = metrics(t['Y0'] + q_full * D, t['H'])[0]
                        tgt = action_optimal_target(t['Y0'], t['H'], D, geoms[mode])
                        mask = energy_mask(block_energy(D, geoms[mode]), thr)
                        m = gate_metrics(q_v, tgt['q_grid'], mask)
                        gm[arm][state].append(m)
                        qv[arm][state].append(q_v[0, 0].cpu())
                        qo[arm][state].append(tgt['q_grid'][0, 0].cpu())
                        rec['%s_qmae' % arm] = m['MAE']
                        rec['%s_corr' % arm] = m['corr']
                        rec['recovery64_%s' % arm] = recovery(
                            rec[arm], rec['R1'], rec['AO64'])
                    per_image.append(rec)
                    for k in ('Base', 'R1', 'AO64', 'BlockH4', *ARMS):
                        acc[state][k].append(rec[k])

        res = {}
        for state in STATES:
            e = {k: float(np.mean(v)) for k, v in acc[state].items()}
            # the frozen oracle R1 and our R1 must agree (same proposal, same data)
            e['oracle_R1'] = float(np.mean([r['oracle_R1'] for r in per_image
                                            if r['split'] == tag
                                            and r['state'] == state]))
            res[state] = dict(
                psnr=e,
                recovery64={arm: recovery(e[arm], e['R1'], e['AO64']) for arm in ARMS},
                recovery64_by_state_denominator=bool(e['AO64'] - e['R1'] > 0),
                recovery64_cells_with_zero_denominator=sum(
                    1 for r in per_image if r['split'] == tag and r['state'] == state
                    and r['recovery64_%s' % ARMS[0]] is None),
                gate={arm: nanmean_dict(gm[arm][state]) for arm in ARMS},
                per_image_qmean_corr={arm: per_image_qmean_corr(
                    qv[arm][state], qo[arm][state]) for arm in ARMS})
            for arm in ARMS:
                g = res[state]['gate'][arm]
                rec_rows.append(dict(split=tag, state=state, arm=arm,
                                     psnr=e[arm], R1=e['R1'], AO64=e['AO64'],
                                     Block_H4=e['BlockH4'],
                                     headroom_vs_R1=e[arm] - e['R1'],
                                     recovery64=res[state]['recovery64'][arm],
                                     **{('gate_' + k): v for k, v in g.items()},
                                     per_image_qmean_corr=res[state]
                                     ['per_image_qmean_corr'][arm]))
                gate_rows.append(dict(split=tag, state=state, arm=arm, **g))
        out[tag] = res
        log('%s done (%d images)' % (tag, n))

    report = dict(arm=[], criteria=None)
    for tag in split_tags:
        for state in STATES:
            e = out[tag][state]['psnr']
            row = dict(split=tag, state=state, Base=e['Base'], R1=e['R1'],
                       A0=e['A0_dense_blockh4'], A1=e['A1_g64'], AO64=e['AO64'],
                       Block_H4=e['BlockH4'], oracle_R1=e['oracle_R1'],
                       recovery64_A0=out[tag][state]['recovery64']['A0_dense_blockh4'],
                       recovery64_A1=out[tag][state]['recovery64']['A1_g64'])
            report['arm'].append(row)
    if 'dev' in out:
        dev = {s: dict(R1=out['dev'][s]['psnr']['R1'],
                       A0=out['dev'][s]['psnr']['A0_dense_blockh4'],
                       A1=out['dev'][s]['psnr']['A1_g64']) for s in STATES}
        report['criteria'] = verdict_5a(dev, out['dev']['correct']['psnr']['Base'])

    json.dump(dict(protocol=dict(commit=lock['repo_commit'], splits=split_tags,
                                 limit=a.limit, states=list(STATES), arms=list(ARMS),
                                 energy_threshold=thr,
                                 checkpoints=used,
                                 oracle_anchor=anchor,
                                 oracle_anchor_source='artifact_lock.json',
                                 git_head=git_head, git_dirty_count=len(git_dirty)),
                  splits=out, report=report),
              open(os.path.join(diag_dir, 'summary.json'), 'w', encoding='utf-8'),
              indent=2, sort_keys=True)
    for arm in ARMS:
        json.dump(dict(arm=arm, splits={t: {s: dict(
            psnr=out[t][s]['psnr'],
            recovery64=out[t][s]['recovery64'][arm],
            gate=out[t][s]['gate'][arm],
            per_image_qmean_corr=out[t][s]['per_image_qmean_corr'][arm])
            for s in STATES} for t in split_tags}),
            open(os.path.join(a.root, arm, 'dev_metrics.json' if 'dev' in out
                              else 'train_metrics_eval.json'), 'w', encoding='utf-8'),
            indent=2, sort_keys=True)
    for name, rows_ in (('per_image.csv', per_image),
                        ('gate_metrics.csv', gate_rows),
                        ('recovery64.csv', rec_rows)):
        with open(os.path.join(diag_dir, name), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0].keys()))
            w.writeheader()
            w.writerows(rows_)

    log()
    log('══ §17 PSNR ══')
    log('  %-5s %-16s %9s %9s %9s %9s %9s %9s %9s'
        % ('split', 'state', 'Base', 'R1', 'A0', 'A1', 'AO64', 'Block_H4', 'orR1'))
    for r in report['arm']:
        log('  %-5s %-16s %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f'
            % (r['split'], r['state'], r['Base'], r['R1'], r['A0'], r['A1'],
               r['AO64'], r['Block_H4'], r['oracle_R1']))
    log()
    log('══ §18 Recovery64 ══')
    for r in report['arm']:
        f = lambda v: ('%.3f' % v) if v is not None else 'n/a'
        log('  %-5s %-16s A0 %s   A1 %s' % (r['split'], r['state'],
                                            f(r['recovery64_A0']),
                                            f(r['recovery64_A1'])))
    if report['criteria']:
        c = report['criteria']
        log()
        log('══ §20 criteria (dev64) ══')
        log('  correct: A1 >= R1 - 0.02 : %s   (A1 >= R1: %s)'
            % (c['correct_not_worse'], c['correct_beats_r1']))
        log('  harmful: A1 >= Base      : %s   (>= Base + 0.05: %s)'
            % (c['harmful_ge_base'], c['harmful_ge_base_005']))
        log('  A1 - A0 by state        : %s'
            % ', '.join('%s %+.4f' % (k, v) for k, v in c['gain_by_state'].items()))
        log('  gain rule (mean >= +0.05 or 2/3 >= +0.05 & none < -0.02): %s'
            % c['gain_rule'])
    log()
    log('artifacts -> %s/{summary.json,per_image.csv,gate_metrics.csv,recovery64.csv}'
        % diag_dir)
    logf.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
