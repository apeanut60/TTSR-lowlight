#!/usr/bin/env python
"""V3-A.4.3 §19/§20: artifact lock for the fine-resolution saturation sweep.

New root, new oracle definition version -- the V3-A.4.2 root is never touched.
This lock additionally pins the frozen V3-A.4.2 summary (the reproduction
anchor, §21) and the nesting scheme that the sweep will actually use.
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3a42_runtime import _sha256                          # noqa: E402
from v3a43_runtime import (ORACLE_DEF_VERSION, build_level_chain,  # noqa: E402
                           level_geometry_report)
from v3a4_runtime import verify_v3a4_artifact_lock         # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V42 = '/root/data/experiments/v3a42_blockwise_oracle'
REF_H, REF_W = 400, 600          # every LOLv2-real image


def parse_ints(text, what):
    vals = [int(v) for v in str(text).split(',') if v.strip()]
    if not vals:
        raise SystemExit('%s is empty' % what)
    if vals != sorted(set(vals)):
        raise SystemExit('%s must be strictly increasing, got %r' % (what, text))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--v3a42_root', default=V42)
    ap.add_argument('--root',
                    default='/root/data/experiments/v3a43_fine_resolution')
    ap.add_argument('--grids', default='1,2,4,8,16,32,64')
    ap.add_argument('--levels', default='16,32,64',
                    help='the fine levels the primary analysis reads (§5); must '
                         'be a subset of --grids and contain 16')
    ap.add_argument('--states', default='correct+true_dark_g0.5+mismatch')
    a = ap.parse_args()

    grids = parse_ints(a.grids, '--grids')
    levels = parse_ints(a.levels, '--levels')
    if 16 not in levels:
        raise SystemExit('--levels must contain 16: G16 is the reproduction '
                         'anchor against V3-A.4.2')
    missing = [g for g in levels if g not in grids]
    if missing:
        raise SystemExit('--levels %r must be a subset of --grids %r (missing %r)'
                         % (levels, grids, missing))

    v4lock = verify_v3a4_artifact_lock(a.v4_root, a.src_root)
    v42_summary = os.path.join(a.v3a42_root, 'oracle', 'summary.json')
    if not os.path.isfile(v42_summary):
        raise SystemExit('the V3-A.4.2 summary is missing (%s) -- this round must '
                         'be able to reproduce G16/Block_H4/Legacy against it '
                         '(§21)' % v42_summary)
    v42 = json.load(open(v42_summary, encoding='utf-8'))

    # freeze the nesting scheme for the real geometry, so a lock cannot claim a
    # hierarchy the sweep will not build
    base = (REF_H // 4, REF_W // 4)
    chain = build_level_chain(REF_H, REF_W, base, grids)
    geo = level_geometry_report(REF_H, REF_W, base, chain)
    if not chain['all_nested']:
        raise SystemExit('the requested ladder is not nested even after the '
                         'hierarchical fallback -- refusing to lock: %s'
                         % chain['containment'])

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
        resolution_levels=','.join(str(g) for g in levels),
        block_h4_factor=4,
        states=a.states,
        oracle_def_version=ORACLE_DEF_VERSION,
        nesting_scheme='%s/%s' % (chain['scheme_y'], chain['scheme_x']),
        nesting_all_nested=bool(chain['all_nested']),
        nesting_levels={str(g): dict(nby=geo['levels'][str(g)]['nby'],
                                     nbx=geo['levels'][str(g)]['nbx'],
                                     note_y=geo['levels'][str(g)]['note_y'],
                                     note_x=geo['levels'][str(g)]['note_x'])
                        for g in grids},
        reference_geometry=[REF_H, REF_W],
        v3a42_summary_path=v42_summary,
        v3a42_summary_sha256=_sha256(v42_summary),
        v3a42_oracle_def_version=v42.get('protocol', {}).get('oracle_def_version'),
        v3a42_repo_commit=v42.get('protocol', {}).get('lock_repo_commit'))

    os.makedirs(os.path.join(a.root, 'oracle'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    path = os.path.join(a.root, 'artifact_lock.json')
    if os.path.isfile(path):
        old = json.load(open(path, encoding='utf-8'))
        diff = [k for k in lock if k != 'repo_commit' and old.get(k) != lock[k]]
        if diff:
            results = os.path.join(a.root, 'oracle', 'summary.json')
            if os.path.isfile(results):
                raise SystemExit(
                    'existing V3-A.4.3 lock differs on %s AND %s already holds a '
                    'sweep -- refusing to mix two protocols under one root; use a '
                    'new --root' % (', '.join(sorted(diff)), results))
            print('note: no sweep under this root yet -> regenerating a changed '
                  'protocol (%s)' % ', '.join(sorted(diff)))
    json.dump(lock, open(path, 'w', encoding='utf-8'), indent=2, sort_keys=True)

    print('V3-A.4.3 lock -> %s' % path)
    print('  commit   : %s' % lock['repo_commit'][:12])
    print('  grids    : %s' % lock['grids'])
    print('  levels   : %s (primary analysis)' % lock['resolution_levels'])
    print('  nesting  : %s  all_nested=%s' % (lock['nesting_scheme'],
                                              lock['nesting_all_nested']))
    for g in grids:
        e = lock['nesting_levels'][str(g)]
        print('    G%-3d -> %3d x %3d blocks  [%s]' % (g, e['nby'], e['nbx'],
                                                       e['note_y']))
    print('  oracle   : %s' % lock['oracle_def_version'])
    print('  v3a4.2   : %s  sha %s' % (lock['v3a42_summary_path'],
                                       lock['v3a42_summary_sha256'][:12]))


if __name__ == '__main__':
    main()
