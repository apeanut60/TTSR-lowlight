#!/usr/bin/env python
"""Aggregate the V2 per-image CSVs across seeds.

Answers the only question this sweep exists for: is the Nano-vs-N0 and
Nano-vs-Self gap stable in sign and roughly in size across seeds, or was the
seed-42 result a single lucky draw?

    python scripts/summarize_local_refine_seeds.py \
        --out_root /root/data/experiments/retinex_v2_localref
"""

import argparse
import csv
import glob
import os
import statistics as st
import sys


def load(path):
    by = {}
    for r in csv.DictReader(open(path, encoding='utf-8')):
        by.setdefault(r['sample_id'], {})[r['condition']] = r
    return by


def proj(by, cond):
    acc = 0.0
    for cam, w in (('Huawei', 0.6), ('Nikon', 0.4)):
        v = [float(d[cond]['psnr_rgb']) for d in by.values()
             if d['N0']['camera'] == cam]
        if not v:
            raise SystemExit('no %s rows in %s' % (cam, cond))
        acc += w * st.mean(v)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root',
                    default='/root/data/experiments/retinex_v2_localref')
    ap.add_argument('--pattern', default='per_image_v2*.csv')
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.out_root, a.pattern)))
    if not paths:
        raise SystemExit('no per-image CSVs under %s' % a.out_root)

    rows = []
    per_image = {}
    for p in paths:
        seed = '42'
        base = os.path.basename(p)
        if '_s' in base:
            seed = base.split('_s')[-1].split('.')[0]
        by = load(p)
        n0 = proj(by, 'N0')
        slf = proj(by, 'V2-Self')
        nan = proj(by, 'V2-Nano-correct')
        byp = proj(by, 'V2-Nano-bypass')
        mis = proj(by, 'V2-Nano-mismatch')
        rows.append(dict(seed=seed, n0=n0, self_=slf, nano=nan, bypass=byp,
                         mismatch=mis, ds=nan - n0, de=nan - slf))
        d = {k: float(v['V2-Nano-correct']['psnr_rgb'])
                - float(v['N0']['psnr_rgb']) for k, v in by.items()}
        per_image[seed] = d

    print('%-6s %9s %9s %9s %9s | %9s %9s' %
          ('seed', 'N0', 'Self', 'Nano', 'bypass', 'Nano-N0', 'Nano-Self'))
    for r in rows:
        print('%-6s %9.4f %9.4f %9.4f %9.4f | %+9.4f %+9.4f' %
              (r['seed'], r['n0'], r['self_'], r['nano'], r['bypass'],
               r['ds'], r['de']))

    ds = [r['ds'] for r in rows]
    de = [r['de'] for r in rows]
    print()
    print('Δscheme   (Nano−N0)   over %d seeds: mean %+.4f  min %+.4f  max %+.4f'
          % (len(ds), st.mean(ds), min(ds), max(ds)))
    print('Δexternal (Nano−Self) over %d seeds: mean %+.4f  min %+.4f  max %+.4f'
          % (len(de), st.mean(de), min(de), max(de)))
    print('sign held (Nano > N0) in %d/%d seeds;  (Nano > Self) in %d/%d seeds'
          % (sum(1 for x in ds if x > 0), len(ds),
             sum(1 for x in de if x > 0), len(de)))

    print()
    print('per-image Nano−N0, per seed:')
    for s, d in per_image.items():
        v = sorted(d.values())
        print('  seed %-4s mean %+.4f median %+.4f better %2d/%d worst %+.3f'
              % (s, st.mean(v), st.median(v), sum(1 for t in v if t > 0),
                 len(v), min(v)))

    # Where do seeds disagree? Samples that flip sign across seeds are the ones
    # a "reference helps" story must eventually explain.
    common = set.intersection(*[set(d) for d in per_image.values()])
    if len(per_image) > 1 and common:
        flips = [k for k in common
                 if max(per_image[s][k] for s in per_image) > 0
                 and min(per_image[s][k] for s in per_image) < 0]
        print()
        print('samples with a sign flip across seeds: %d/%d' % (len(flips), len(common)))
        for k in sorted(flips)[:12]:
            print('   %-20s %s' % (k, '  '.join('%+.3f' % per_image[s][k]
                                                 for s in per_image)))


if __name__ == '__main__':
    main()
