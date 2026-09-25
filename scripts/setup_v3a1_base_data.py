#!/usr/bin/env python
"""Stage a base-training dataset with a held-out validation split (V3-A.1,方案 A).

The base must not be selected on the 100-image test set, and the 50 train
images that have no Nano-v2 reference are the natural held-out split:

    base_data/Train/  <- the 639 samples that DO have a reference
    base_data/Test/   <- the 50 samples that do NOT  (validation only)

Only symlinks are created; nothing is copied and no original is modified.
"""

import argparse
import csv
import os


def link(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    d = os.path.join(dst_dir, os.path.basename(src))
    if not os.path.lexists(d):
        os.symlink(src, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/data/experiments/v3a1_lolv2real')
    ap.add_argument('--out', default='/root/data/experiments/v3a1_lolv2real/base_data')
    a = ap.parse_args()

    with_ref = list(csv.DictReader(
        open(os.path.join(a.root, 'manifests', 'refiner_train.csv'), encoding='utf-8')))
    all_rows = list(csv.DictReader(
        open(os.path.join(a.root, 'manifests', 'base_train.csv'), encoding='utf-8')))
    have = {os.path.basename(r['low_path']) for r in with_ref}
    held = [r for r in all_rows if os.path.basename(r['low_path']) not in have]
    if len(with_ref) != 639 or len(held) != 50:
        raise SystemExit('expected 639/50, got %d/%d' % (len(with_ref), len(held)))

    for split, rows in (('Train', with_ref), ('Test', held)):
        for r in rows:
            link(r['low_path'], os.path.join(a.out, split, 'Low'))
            link(r['high_path'], os.path.join(a.out, split, 'Normal'))
        # the loader requires the reference directory to exist even when the
        # model is reference-free; it may be empty
        os.makedirs(os.path.join(a.out, split, 'nanobanana_ref_v2'), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'Train', 'nanobanana_ref_v2'), exist_ok=True)
    print('base_data staged: Train(low/high)=%d  Test(low/high)=%d'
          % (len(os.listdir(os.path.join(a.out, 'Train', 'Low'))),
             len(os.listdir(os.path.join(a.out, 'Test', 'Low')))))
    print('  validation split = the 50 samples with no Nano-v2 reference')
    print('  -> %s' % a.out)


if __name__ == '__main__':
    main()
