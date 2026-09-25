#!/usr/bin/env python
"""Aggregate the external evaluation (plan §F/§G).

Generic dataset/group aggregation (no Huawei/Nikon weights), paired bootstrap
on the two primary differences, and the tail-failure lists. Recomputes the
out-of-range fraction from the saved float predictions.
"""

import argparse
import collections
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The CSV records the display label; the prediction dirs use the internal name.
LABEL = {'N0': 'N0', 'Self-fixed': 'V2-Self',
         'Nano-fixed-correct': 'V2-Nano-correct',
         'Nano-fixed-bypass': 'V2-Nano-bypass',
         'Nano-fixed-mismatch': 'V2-Nano-mismatch'}
ALL_CONDS = list(LABEL)


def load(tdir):
    p = os.path.join(tdir, 'per_image_external.csv')
    by = collections.defaultdict(dict)
    for r in csv.DictReader(open(p, encoding='utf-8')):
        by[r['sample_id']][r['condition']] = r
    return by


def oob_fraction(tdir, condition, sample_id):
    f = os.path.join(tdir, 'predictions', condition, sample_id + '.npy')
    if not os.path.isfile(f):
        return None
    a = np.load(f)
    return float((np.abs(a) > 1.0).mean())


def boot(delta, groups, n=5000, seed=20260924, lo=0.025, hi=0.975):
    """Paired bootstrap over groups (scene == image here)."""
    rng = np.random.default_rng(seed)
    keys = sorted(set(groups))
    idx = {k: np.array([i for i, g in enumerate(groups) if g == k]) for k in keys}
    d = np.asarray(delta, dtype=np.float64)
    stats = []
    for _ in range(n):
        draw = rng.choice(len(keys), size=len(keys), replace=True)
        sel = np.concatenate([idx[keys[j]] for j in draw])
        stats.append(d[sel].mean())
    stats = np.sort(np.array(stats))
    return (float(stats[int(lo * n)]), float(stats[int(hi * n)]),
            float(d.mean()), float(np.median(d)),
            int((d > 0).sum()), len(d),
            float(np.percentile(d, 5)), float(d.min()), float(d.max()),
            int((d < -1.0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding='utf-8'))
    tdir = os.path.join(cfg['output_root'], 'target_lolv2real')
    by = load(tdir)
    sids = sorted(by)

    def series(cond, key='psnr_rgb'):
        return np.array([float(by[s][LABEL[cond]][key]) for s in sids])

    # Resampling unit = image, not the dataset ``group`` column (which is a
    # single constant here -- grouping by it would collapse the bootstrap to a
    # zero-width interval). For LOL-v2-real Test each image is its own scene,
    # so image-level resampling IS scene-level resampling.
    groups = list(sids)
    if len(set(groups)) < 2:
        raise SystemExit('bootstrap needs >=2 resampling units')
    summary = dict(
        dataset=cfg['target']['dataset'], split=cfg['target']['split'],
        sample_count=len(sids), prior_use_status=cfg['target']['prior_use_status'],
        means={c: float(series(c).mean()) for c in ALL_CONDS},
        means_ssim_y={c: float(series(c, 'ssim_y').mean()) for c in ALL_CONDS},
        bootstrap=dict(resamples=5000, seed=20260924, unit='image (scene == image here)',
                       marginal_confidence=0.95,
                       joint_option='97.5% marginal intervals', stats={}),
        oob={}, notes=[])
    for name, (x, y) in {'scheme(Nano-N0)': ('Nano-fixed-correct', 'N0'),
                         'external(Nano-Self)': ('Nano-fixed-correct', 'Self-fixed'),
                         'content(correct-mismatch)': ('Nano-fixed-correct',
                                                       'Nano-fixed-mismatch'),
                         'presence(mismatch-N0)': ('Nano-fixed-mismatch', 'N0')
                         }.items():
        d = series(x) - series(y)
        lo, hi, mean, med, better, n, p05, worst, best, gt1 = boot(d, groups)
        summary['bootstrap']['stats'][name] = dict(
            mean=mean, median=med, ci95=[lo, hi], benefit_count=better,
            n=n, p05=p05, worst=worst, best=best, count_below_minus_1db=gt1,
            ci_excludes_zero=(lo > 0 or hi < 0))
    for c in ALL_CONDS:
        v = [oob_fraction(tdir, c, s) for s in sids]
        v = [x for x in v if x is not None]
        summary['oob'][c] = float(np.mean(v)) if v else None

    print('══ external summary: %s %s, n=%d ══'
          % (summary['dataset'], summary['split'], len(sids)))
    print('  %-20s %10s %10s' % ('condition', 'PSNR(rgb)', 'SSIM(Y)'))
    for c, v in summary['means'].items():
        print('  %-20s %10.4f %10.4f' % (c, v, summary['means_ssim_y'][c]))
    print()
    print('  %-26s %9s %20s %8s %8s' % ('difference', 'mean', '95% CI', 'median', 'better'))
    for k, v in summary['bootstrap']['stats'].items():
        print('  %-26s %+9.4f  [%+.4f, %+.4f] %+8.4f %5d/%d'
              % (k, v['mean'], v['ci95'][0], v['ci95'][1], v['median'],
                 v['benefit_count'], v['n']))
    print()
    for k, v in summary['bootstrap']['stats'].items():
        print('  %-26s p05 %+.3f worst %+.3f best %+.3f  <-1dB: %d'
              % (k, v['p05'], v['worst'], v['best'], v['count_below_minus_1db']))
    print()
    print('  oob fraction:', {k: (round(v, 6) if v is not None else None)
                             for k, v in summary['oob'].items()})

    d = series('Nano-fixed-correct') - series('N0')
    order = np.argsort(d)
    summary['worst5'] = [dict(sample_id=sids[i], delta=float(d[i])) for i in order[:5]]
    summary['best5'] = [dict(sample_id=sids[i], delta=float(d[i])) for i in order[-5:]]
    print('\n  worst5:', ', '.join('%s %+.3f' % (x['sample_id'], x['delta'])
                                   for x in summary['worst5']))
    print('  best5 :', ', '.join('%s %+.3f' % (x['sample_id'], x['delta'])
                                  for x in summary['best5']))

    preds = glob.glob(os.path.join(tdir, 'predictions', '*', '*.npy'))
    summary['prediction_files'] = len(preds)
    out = os.path.join(tdir, 'summary_external.json')
    json.dump(summary, open(out, 'w', encoding='utf-8'), indent=2, sort_keys=True)
    print('\nsummary -> %s  (%d prediction files)' % (out, len(preds)))


if __name__ == '__main__':
    main()
