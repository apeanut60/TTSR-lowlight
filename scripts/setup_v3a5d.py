#!/usr/bin/env python
"""V3-A.5D0 setup: artifact lock + dirs. Does NOT train or audit."""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from v3a42_runtime import _sha256                                      # noqa: E402
from v3a5d_runtime import (evidence_schema_payload,                     # noqa: E402
                           evidence_schema_sha, pool_geometry_coverage)
from v3a5_runtime import prepare_geometry, target_geometry              # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V5A = '/root/data/experiments/v3a5_g64_verifier'
ROOT = '/root/data/experiments/v3a5d_evidence'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
REF_H, REF_W = 400, 600


def git_meta():
    cwd = '/root/projects/TTSR-lowlight'
    try:
        head = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd, text=True).strip()
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=cwd, text=True).strip()
    except Exception:
        head, dirty = None, ''
    return head, bool(dirty)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--v5a_root', default=V5A)
    a = ap.parse_args(_CLI)

    for d in ('evidence', 'diagnostics', 'logs'):
        os.makedirs(os.path.join(a.root, d), exist_ok=True)

    v5a_lock = json.load(open(os.path.join(a.v5a_root, 'artifact_lock.json'),
                              encoding='utf-8'))
    thr = float(v5a_lock['energy_threshold'])
    geom = prepare_geometry(target_geometry(REF_H, REF_W, 'g64'), 'cpu')
    cov1 = pool_geometry_coverage(1, geom)
    cov2 = pool_geometry_coverage(2, geom)
    if not (cov1['all_pixels_covered_once'] and cov1['no_empty_block']
            and cov2['all_pixels_covered_once'] and cov2['no_empty_block']):
        raise SystemExit('G64 pool coverage failed: %s / %s' % (cov1, cov2))

    schema = evidence_schema_payload()
    schema_path = os.path.join(a.root, 'evidence', 'schema.json')
    json.dump(schema, open(schema_path, 'w', encoding='utf-8'),
              indent=2, sort_keys=True)

    head, dirty = git_meta()
    lock = dict(
        stage='V3-A.5D0',
        root=a.root,
        repo_commit=head,
        repo_dirty=dirty,
        src_root=a.src_root,
        v4_root=a.v4_root,
        v5a_root=a.v5a_root,
        proposal_ckpt=os.path.join(a.src_root, R1_CK),
        proposal_sha256=v5a_lock['proposal_sha256'],
        cache_name=v5a_lock.get('cache_name', 'cache_y0_lolbase'),
        cache_metadata_sha256=v5a_lock['cache_metadata_sha256'],
        manifest_sha256=v5a_lock['manifest_sha256'],
        split_sha256=v5a_lock['split_sha256'],
        mismatch_train_sha256=v5a_lock['mismatch_train_sha256'],
        mismatch_dev_sha256=v5a_lock['mismatch_dev_sha256'],
        energy_threshold=thr,
        energy_pctl=v5a_lock.get('energy_pctl', 10.0),
        energy_stats_sha256=v5a_lock['energy_stats_sha256'],
        g64_shape=list(geom['shape']),
        g64_edges_y=[int(v) for v in geom['edges'][0]],
        g64_edges_x=[int(v) for v in geom['edges'][1]],
        reference_geometry=[REF_H, REF_W],
        evidence_schema_sha256=evidence_schema_sha(),
        evidence_schema_path=schema_path,
        pool_coverage_full=cov1,
        pool_coverage_h2=cov2,
        splits=['train', 'dev'],
        states=['correct', 'true_dark_g0.5', 'mismatch'],
        official_test_forbidden=True,
        auto_train_forbidden=True,
        note='D0 zero-training evidence audit; inherits V3-A.5A energy/G64/split',
    )
    lock_path = os.path.join(a.root, 'artifact_lock.json')
    json.dump(lock, open(lock_path, 'w', encoding='utf-8'),
              indent=2, sort_keys=True)
    # verify sha of written schema file matches runtime
    if _sha256(schema_path) and evidence_schema_sha() != lock['evidence_schema_sha256']:
        raise SystemExit('schema sha drift')
    print('V3-A.5D0 lock -> %s' % lock_path)
    print('  energy_threshold=%.6e' % thr)
    print('  evidence_schema_sha256=%s' % lock['evidence_schema_sha256'][:16])
    print('  g64_shape=%s' % lock['g64_shape'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
