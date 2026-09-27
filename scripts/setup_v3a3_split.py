#!/usr/bin/env python
"""V3-A.3 §6: fixed 575/64 split from the 639 Nano-v2 train images.

Written once, then frozen: all corruption audits, headroom measurements and
structural decisions use dev64; the official Test set is touched only after the
structure is frozen.
"""

import argparse
import csv
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/data/experiments/v3a3_lolv2real')
    ap.add_argument('--src_manifest',
                    default='/root/data/experiments/v3a1_lolv2real/manifests/refiner_train.csv')
    ap.add_argument('--seed', type=int, default=20260927)
    ap.add_argument('--n_dev', type=int, default=64)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.src_manifest, encoding='utf-8')))
    ids = sorted(r['sample_id'] for r in rows)
    if len(ids) != 639:
        raise SystemExit('expected 639 source samples, got %d' % len(ids))
    rng = random.Random(a.seed)
    dev = sorted(rng.sample(ids, a.n_dev))
    devset = set(dev)
    train = [i for i in ids if i not in devset]
    if len(train) != 575 or set(train) & devset or len(set(train) | devset) != 639:
        raise SystemExit('split arithmetic failed')

    os.makedirs(os.path.join(a.root, 'splits'), exist_ok=True)
    p = os.path.join(a.root, 'splits', 'split.json')
    if os.path.isfile(p):
        old = json.load(open(p, encoding='utf-8'))
        if old['dev'] != dev:
            raise SystemExit('split already exists and differs; refusing to rewrite')
    json.dump(dict(seed=a.seed, n_train=len(train), n_dev=len(dev),
                   source_manifest=a.src_manifest, train=train, dev=dev),
              open(p, 'w', encoding='utf-8'), indent=2, sort_keys=True)
    print('verifier_train %d  verifier_dev %d  (seed %d)' % (len(train), len(dev), a.seed))
    print('-> %s' % p)


if __name__ == '__main__':
    main()
