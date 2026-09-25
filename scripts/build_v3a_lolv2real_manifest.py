#!/usr/bin/env python
"""Build the strict V3-A manifests for LOL-v2-real (plan §1 / §8).

Writes three CSVs plus a protocol.json. Every row carries SHA256 for low, GT
and reference; a sample without a reference is reported, never silently paired
and never back-filled with the GT.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# option.py calls parser.parse_args() at import time, so argv must be cleared
# before anything that imports it (local_refine_runtime does).
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from dataset.lolv2real_v3a import collect_pairs, gt_name_for, list_files  # noqa: E402
from local_refine_runtime import sha256                                   # noqa: E402

FIELDS = ['sample_id', 'camera', 'local_path', 'low_path', 'high_path',
          'nano_path', 'low_sha256', 'high_sha256', 'nano_sha256',
          'low_h', 'low_w', 'nano_h', 'nano_w', 'reference_variant']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--ref_variant', default='nanobanana_ref_v2')
    ap.add_argument('--out_root', default='/root/data/experiments/v3a_lolv2real')
    a = ap.parse_args(_CLI)
    os.makedirs(os.path.join(a.out_root, 'manifests'), exist_ok=True)

    from PIL import Image
    ref_dir = os.path.join(a.dataset_dir, 'Test', a.ref_variant)
    refs = set(list_files(ref_dir))

    rows = []
    for split in ('Train', 'Test'):
        rd = os.path.join(a.dataset_dir, split, a.ref_variant)
        rmap = {f: os.path.join(rd, f) for f in list_files(rd)} if os.path.isdir(rd) else {}
        for name, low, high in collect_pairs(a.dataset_dir, split):
            hp = rmap.get(name)
            lw, lh = Image.open(low).size
            if hp:
                nw, nh = Image.open(hp).size
            else:
                nw = nh = None
            rows.append(dict(
                sample_id=name, camera='LOLv2real', local_path=split,
                low_path=low, high_path=high, nano_path=hp or '',
                low_sha256=sha256(low), high_sha256=sha256(high),
                nano_sha256=sha256(hp) if hp else '',
                low_h=lh, low_w=lw, nano_h=nh, nano_w=nw,
                reference_variant=a.ref_variant))

    tr = [r for r in rows if r['local_path'] == 'Train']
    te = [r for r in rows if r['local_path'] == 'Test']
    tr_ref = [r for r in tr if r['nano_path']]
    te_ref = [r for r in te if r['nano_path']]

    # --- hard checks -------------------------------------------------------
    for r in rows:
        if r['nano_path'] and os.path.samefile(r['nano_path'], r['high_path']):
            raise SystemExit('%s: reference path IS the GT path' % r['sample_id'])
    if len({r['sample_id'] for r in tr}) != len(tr):
        raise SystemExit('duplicate train sample_id')
    if len({r['low_sha256'] for r in tr}) != len(tr):
        raise SystemExit('duplicate train low content')
    resize_tr = sum(1 for r in tr_ref if (r['nano_h'], r['nano_w']) != (r['low_h'], r['low_w']))
    resize_te = sum(1 for r in te_ref if (r['nano_h'], r['nano_w']) != (r['low_h'], r['low_w']))

    def write(path, rs):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in rs:
                w.writerow(r)

    mdir = os.path.join(a.out_root, 'manifests')
    write(os.path.join(mdir, 'base_train.csv'), tr)
    write(os.path.join(mdir, 'refiner_train.csv'), tr_ref)
    write(os.path.join(mdir, 'test.csv'), te)

    missing = sorted(os.path.splitext(r['sample_id'])[0] for r in tr if not r['nano_path'])
    proto = dict(
        dataset_dir=a.dataset_dir, reference_variant=a.ref_variant,
        base_train=len(tr), refiner_train=len(tr_ref), test=len(te),
        test_with_ref=len(te_ref),
        train_missing_ref=len(missing),
        train_missing_ref_ids=missing,
        reference_resize_needed=dict(train=resize_tr, test=resize_te),
        pairing_rule="lowNNNNN.png -> normalNNNNN.png (prefix replacement); "
                     "reference by identical basename",
        hr_fallback_used=False,
        manifests={k: dict(n=len(v), sha256=sha256(os.path.join(mdir, k + '.csv')))
                   for k, v in (('base_train', tr), ('refiner_train', tr_ref),
                                ('test', te))})
    json.dump(proto, open(os.path.join(a.out_root, 'protocol.json'), 'w',
                          encoding='utf-8'), indent=2, sort_keys=True)

    print('base_train     %d  (all of Train)' % len(tr))
    print('refiner_train  %d  (%d without %s, e.g. %s)'
          % (len(tr_ref), len(missing), a.ref_variant, missing[:4]))
    print('test           %d  (with ref %d)' % (len(te), len(te_ref)))
    print('reference resize needed: train %d, test %d' % (resize_tr, resize_te))
    print('-> %s' % mdir)


if __name__ == '__main__':
    main()
