#!/usr/bin/env python
"""V3-A.5D2 setup: reuse V3A5C tiny16 + shared A0/A1 init. No training."""

import argparse
import json
import os
import shutil
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.V3A5D2Verifier import ARMS, build_d2_shared_init              # noqa: E402
from v3a42_runtime import _sha256                                       # noqa: E402
from v3a5_runtime import state_dict_sha                                 # noqa: E402
from v3a5d2_runtime import TINY_ROOT_SRC                                # noqa: E402

ROOT = '/root/data/experiments/v3a5d2_rf_tiny'


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
    ap.add_argument('--tiny_src', default=TINY_ROOT_SRC)
    a = ap.parse_args(_CLI)

    tiny_dst = os.path.join(a.root, 'tiny')
    os.makedirs(tiny_dst, exist_ok=True)
    os.makedirs(os.path.join(a.root, 'diagnostics'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)

    src_tiny = os.path.join(a.tiny_src, 'tiny')
    for name in ('tiny16_ids.json', 'tiny16_mismatch_map.json',
                 'oracle_stats.json', 'target_stats.json', 'pairs.json',
                 'cache.pt'):
        src = os.path.join(src_tiny, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(tiny_dst, name))

    ids = json.load(open(os.path.join(tiny_dst, 'tiny16_ids.json')))['ids']
    v5c_lock = json.load(open(os.path.join(a.tiny_src, 'artifact_lock.json')))
    thr = float(v5c_lock['energy_threshold'])

    init = build_d2_shared_init(seed=42)
    torch.save(init, os.path.join(tiny_dst, 'shared_init_s42.pt'))
    init_sha = state_dict_sha(init['A0_control'])
    print('shared init sha=%s' % init_sha[:16])
    print('arms=%s' % list(ARMS))

    lock = dict(
        stage='V3-A.5D2',
        root=a.root,
        repo_commit=git_head(),
        tiny_src=a.tiny_src,
        tiny16_ids=ids,
        tiny16_ids_sha256=_sha256(os.path.join(tiny_dst, 'tiny16_ids.json')),
        v3a5c_lock_sha256=_sha256(os.path.join(a.tiny_src, 'artifact_lock.json')),
        energy_threshold=thr,
        proposal_sha256=v5c_lock.get('proposal_sha256'),
        init_sha=init_sha,
        out_weight=0.0,
        arms=list(ARMS),
        updates=20000,
        seed=42,
        bottleneck=64,
        note='RF/MultiScale tiny overfit; no evidence; no Transformer/NonLocal',
        d1_verdict='D1_null_close_evidence',
    )
    dump(os.path.join(a.root, 'artifact_lock.json'), lock)
    print('V3-A.5D2 lock -> %s' % os.path.join(a.root, 'artifact_lock.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
