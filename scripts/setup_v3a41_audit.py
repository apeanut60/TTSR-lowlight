#!/usr/bin/env python
"""V3-A.4.1 §4/§18: fixed crop manifest + audit artifact lock. Read-only."""

import argparse
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from dataset.lolv2real_v3a import pairs_from_manifest    # noqa: E402
from local_refine_runtime import sha256                  # noqa: E402
from v3a41_runtime import build_fixed_crop_manifest      # noqa: E402
from v3a4_runtime import verify_v3a4_artifact_lock       # noqa: E402

V4 = '/root/data/experiments/v3a4_lolv2real'
V1 = '/root/data/experiments/v3a1_lolv2real'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--src_root', default=V1)
    ap.add_argument('--root', default='/root/data/experiments/v3a41_target_audit')
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--seed', type=int, default=20260927)
    a = ap.parse_args(_CLI)
    os.makedirs(os.path.join(a.root, 'crops'), exist_ok=True)

    # the audit inherits V3-A.4's protocol; fail loudly if any of it moved
    v4lock = verify_v3a4_artifact_lock(a.v4_root, a.src_root)
    split = json.load(open(os.path.join(a.v4_root, 'splits', 'split.json'),
                           encoding='utf-8'))
    all_pairs = pairs_from_manifest(os.path.join(a.src_root, 'manifests',
                                                 'refiner_train.csv'))
    by_id = {p[0]: p for p in all_pairs}
    manifest_path = os.path.join(a.root, 'crops', 'crop_manifest.csv')
    rows = (build_fixed_crop_manifest([by_id[i] for i in split['train']], 'train',
                                      a.crop, a.k, a.seed)
            + build_fixed_crop_manifest([by_id[i] for i in split['dev']], 'dev',
                                        a.crop, a.k, a.seed))
    # Compare the FULL protocol key, geometry included: rot_k/flip_h/flip_w are
    # scientific variables now, so a geometry change must refuse, not silently
    # overwrite.
    KEY = ('sample_id', 'split', 'crop_id', 'top', 'left', 'height', 'width',
           'rot_k', 'flip_h', 'flip_w', 'seed')
    if os.path.isfile(manifest_path):
        old = list(csv.DictReader(open(manifest_path, encoding='utf-8')))
        old_keys = {tuple(str(r.get(k, '<missing>')) for k in KEY) for r in old}
        new_keys = {tuple(str(r[k]) for k in KEY) for r in rows}
        if old_keys != new_keys:
            raise SystemExit(
                'crop manifest already exists and differs (%d rows vs %d, %d '
                'keys differ) -- refusing to resample; use a new --root for a '
                'new protocol' % (len(old), len(rows),
                                  len(old_keys ^ new_keys)))
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('crop manifest: %d crops (%d train + %d dev, K=%d, seed=%d)'
          % (len(rows), len(split['train']) * a.k, len(split['dev']) * a.k, a.k, a.seed))

    lock = dict(
        repo_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).decode().strip(),
        proposal_sha256=v4lock['proposal_sha256'],
        cache_metadata_sha256=v4lock['cache_metadata_sha256'],
        manifest_sha256=v4lock['manifest_sha256'],
        split_sha256=v4lock['split_sha256'],
        mismatch_train_sha256=v4lock['mismatch_train_sha256'],
        mismatch_dev_sha256=v4lock['mismatch_dev_sha256'],
        energy_stats_sha256=v4lock['energy_stats_sha256'],
        crop_manifest_sha256=sha256(manifest_path),
        seed=a.seed, crop_size=a.crop, k=a.k,
        states='correct+true_dark_g0.5+mismatch')
    json.dump(lock, open(os.path.join(a.root, 'artifact_lock.json'), 'w',
                         encoding='utf-8'), indent=2, sort_keys=True)
    print('-> %s' % a.root)


if __name__ == '__main__':
    main()
