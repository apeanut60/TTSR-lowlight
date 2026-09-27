#!/usr/bin/env python
"""V3-A.4.2 §20: artifact lock for the blockwise gate-resolution oracle sweep.

Binds every input the sweep depends on to a SHA256, plus the oracle definition
version and the grid list. No crop manifest: V3-A.4.2 evaluates full images
only (§20), so there is no crop protocol to freeze.

The V3-A.4 protocol is inherited wholesale -- ``verify_v3a4_artifact_lock`` runs
first, so this lock cannot be generated against a moved split / mismatch map /
energy statistic.
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3a42_runtime import ORACLE_DEF_VERSION, _sha256          # noqa: E402
from v3a4_runtime import verify_v3a4_artifact_lock             # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--root',
                    default='/root/data/experiments/v3a42_blockwise_oracle')
    ap.add_argument('--grids', default='1,2,4,8,16')
    ap.add_argument('--states', default='correct+true_dark_g0.5+mismatch')
    a = ap.parse_args()

    grids = [int(g) for g in a.grids.split(',') if g.strip()]
    if grids != sorted(set(grids)):
        raise SystemExit('--grids must be strictly increasing, got %r' % a.grids)
    if 1 not in grids:
        raise SystemExit('--grids must contain 1: the 1x1 block oracle IS the '
                         'V3-A.4.1 Global-AO and is the reproduction anchor')

    v4lock = verify_v3a4_artifact_lock(a.v4_root, a.src_root)
    lock = dict(
        repo_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ).decode().strip(),
        proposal_sha256=v4lock['proposal_sha256'],
        cache_metadata_sha256=v4lock['cache_metadata_sha256'],
        manifest_sha256=v4lock['manifest_sha256'],
        split_sha256=v4lock['split_sha256'],
        mismatch_train_sha256=v4lock['mismatch_train_sha256'],
        mismatch_dev_sha256=v4lock['mismatch_dev_sha256'],
        energy_stats_sha256=v4lock['energy_stats_sha256'],
        grids=','.join(str(g) for g in grids),
        states=a.states,
        oracle_def_version=ORACLE_DEF_VERSION)

    os.makedirs(os.path.join(a.root, 'oracle'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    path = os.path.join(a.root, 'artifact_lock.json')
    if os.path.isfile(path):
        old = json.load(open(path, encoding='utf-8'))
        # repo_commit is MEANT to move (the workflow re-runs this on the final
        # revision); every scientific field must not.
        diff = [k for k in lock if k != 'repo_commit' and old.get(k) != lock[k]]
        if diff:
            raise SystemExit(
                'existing V3-A.4.2 lock differs on %s -- refusing to overwrite a '
                'frozen protocol; use a new --root for a new protocol'
                % ', '.join(sorted(diff)))
    json.dump(lock, open(path, 'w', encoding='utf-8'), indent=2, sort_keys=True)
    print('V3-A.4.2 lock -> %s' % path)
    print('  commit   : %s' % lock['repo_commit'][:12])
    print('  grids    : %s' % lock['grids'])
    print('  states   : %s' % lock['states'])
    print('  oracle   : %s' % lock['oracle_def_version'])
    print('  proposal : %s' % lock['proposal_sha256'][:12])
    print('  lock sha : %s' % _sha256(path)[:12])


if __name__ == '__main__':
    main()
