#!/usr/bin/env python
"""Precompute random 192 crop schedule for Nikon train (v2.6 style)."""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from local_refine_runtime import read_manifest_csv  # noqa: E402

EXP = '/root/data/experiments/TTSR-lowlight-v27-nikon'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--crop', type=int, default=192)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--out_root', default=EXP)
    a = ap.parse_args()

    tr = read_manifest_csv(os.path.join(a.out_root, 'manifests/manifests_train.csv'))
    n = len(tr)
    assert n > 0
    epochs = int(np.ceil(a.steps / float(n)))
    rng = np.random.RandomState(a.seed)
    out = os.path.join(a.out_root, 'manifests', 'random_crop_s%d.csv' % a.seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rows = []
    step = 0
    for ep in range(epochs):
        order = rng.permutation(n)
        for sample_idx in order:
            if step >= a.steps:
                break
            step += 1
            row = tr[int(sample_idx)]
            W, H = map(int, row['low_hw'].split('x'))  # W×H in manifest
            ch = a.crop
            if H < ch or W < ch:
                raise SystemExit('crop too large for %s (%dx%d)' % (
                    row['sample_id'], H, W))
            top = int(rng.randint(0, H - ch + 1))
            left = int(rng.randint(0, W - ch + 1))
            rows.append(dict(
                step=step, epoch=ep, sample_idx=int(sample_idx),
                sample_id=row['sample_id'], crop_y=top, crop_x=left,
                flip_h=0, flip_v=0, rotation=0, H=H, W=W, crop=ch,
            ))
        if step >= a.steps:
            break

    cols = ['step', 'epoch', 'sample_idx', 'sample_id', 'crop_y', 'crop_x',
            'flip_h', 'flip_v', 'rotation', 'H', 'W', 'crop']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # uniqueness check: 4 visits → preferably unique crops
    from collections import defaultdict
    by_id = defaultdict(set)
    for r in rows:
        by_id[r['sample_id']].add((r['crop_y'], r['crop_x']))
    n_unique = float(np.mean([len(v) for v in by_id.values()]))
    print('wrote %s steps=%d n_train=%d mean_unique_crops=%.2f' % (
        out, len(rows), n, n_unique))


if __name__ == '__main__':
    main()
