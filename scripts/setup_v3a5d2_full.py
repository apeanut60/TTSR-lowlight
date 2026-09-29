#!/usr/bin/env python
"""V3-A.5D2-full setup: lock inheriting V5A energy/G64 + shared init. No train."""

import argparse
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.V3A5D2Verifier import ARMS, build_d2_shared_init              # noqa: E402
from v3a42_runtime import _sha256                                       # noqa: E402
from v3a5_runtime import state_dict_sha                                 # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V5A = '/root/data/experiments/v3a5_g64_verifier'
V43 = '/root/data/experiments/v3a43_fine_resolution'
TINY = '/root/data/experiments/v3a5d2_rf_tiny'
ROOT = '/root/data/experiments/v3a5d2_rf_full'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'


def dump(path, obj):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    json.dump(obj, open(path, 'w', encoding='utf-8'), indent=2, sort_keys=True)


def git_head():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd='/root/projects/TTSR-lowlight',
            text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--v5a_root', default=V5A)
    ap.add_argument('--tiny_root', default=TINY)
    a = ap.parse_args(_CLI)

    for d in ('init', 'diagnostics', 'logs'):
        os.makedirs(os.path.join(a.root, d), exist_ok=True)

    v5a = json.load(open(os.path.join(a.v5a_root, 'artifact_lock.json')))
    thr = float(v5a['energy_threshold'])
    tiny_verdict = None
    tv = os.path.join(a.tiny_root, 'diagnostics', 'd2_verdict.json')
    if os.path.isfile(tv):
        tiny_verdict = json.load(open(tv))

    init = build_d2_shared_init(seed=42)
    torch.save(init, os.path.join(a.root, 'init', 'shared_init_s42.pt'))
    init_sha = state_dict_sha(init['A0_control'])

    lock = dict(
        stage='V3-A.5D2-full',
        root=a.root,
        repo_commit=git_head(),
        src_root=SRC,
        v4_root=V4,
        v5a_root=a.v5a_root,
        v43_root=V43,
        tiny_root=a.tiny_root,
        proposal_ckpt=os.path.join(SRC, R1_CK),
        proposal_sha256=v5a.get('proposal_sha256'),
        cache_name=v5a.get('cache_name', 'cache_y0_lolbase'),
        cache_metadata_sha256=v5a.get('cache_metadata_sha256'),
        split_sha256=v5a.get('split_sha256'),
        mismatch_train_sha256=v5a.get('mismatch_train_sha256'),
        mismatch_dev_sha256=v5a.get('mismatch_dev_sha256'),
        energy_threshold=thr,
        energy_pctl=v5a.get('energy_pctl', 10.0),
        energy_stats_sha256=v5a.get('energy_stats_sha256'),
        g64_shape=v5a.get('g64_shape', [64, 64]),
        init_sha=init_sha,
        arms=list(ARMS),
        out_weight=0.0,
        steps=3000,
        grad_accum=4,
        seed=42,
        lr='1e-4 (0-2000) -> 5e-5 (2000-3000)',
        state_probs=[0.5, 0.25, 0.25],
        official_test_forbidden=True,
        tiny_verdict_label=(tiny_verdict or {}).get('label'),
        note='D2-full MultiScale RF; gate-only; inherits V5A energy/G64/split',
    )
    dump(os.path.join(a.root, 'artifact_lock.json'), lock)
    print('V3-A.5D2-full lock -> %s' % os.path.join(a.root, 'artifact_lock.json'))
    print('  energy_threshold=%.6e  init_sha=%s' % (thr, init_sha[:16]))
    print('  tiny_verdict=%s' % lock['tiny_verdict_label'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
