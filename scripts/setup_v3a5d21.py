#!/usr/bin/env python
"""V3-A.5D2.1 setup: deterministic train64 + mismatch slice + shared D2 init."""

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
from v3a5_pipeline import load_rows                                     # noqa: E402
from v3a5_runtime import state_dict_sha                                 # noqa: E402
from v3a5c_runtime import dump_json, slice_mismatch_map                 # noqa: E402
from v3a5d21_runtime import (EXPOSURE_STEPS, N_SCALE, SCALE_SEED,       # noqa: E402
                             choose_tiny_ids)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V5A = '/root/data/experiments/v3a5_g64_verifier'
TINY16 = '/root/data/experiments/v3a5c_tiny_overfit'
ROOT = '/root/data/experiments/v3a5d21_scale64'


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
    ap.add_argument('--n', type=int, default=N_SCALE)
    ap.add_argument('--seed', type=int, default=SCALE_SEED)
    a = ap.parse_args(_CLI)

    for d in ('subset', 'init', 'diagnostics', 'logs'):
        os.makedirs(os.path.join(a.root, d), exist_ok=True)

    v5a = json.load(open(os.path.join(V5A, 'artifact_lock.json')))
    thr = float(v5a['energy_threshold'])
    train_rows = load_rows(
        os.path.join(SRC, 'manifests', 'refiner_train.csv'),
        os.path.join(V4, 'splits', 'split.json'))['train']
    train_ids = [r[0] for r in train_rows]
    ids = choose_tiny_ids(train_ids, n=a.n, seed=a.seed)
    # record overlap with tiny16 for transparency
    tiny16 = json.load(open(os.path.join(TINY16, 'tiny', 'tiny16_ids.json')))['ids']
    overlap = sorted(set(ids) & set(tiny16))

    mmap_full = json.load(open(os.path.join(
        V4, 'mappings', 'mismatch_train_575.json'), encoding='utf-8'))
    mmap = slice_mismatch_map(mmap_full, ids)

    dump_json(os.path.join(a.root, 'subset', 'train64_ids.json'), dict(
        n=a.n, seed=a.seed, ids=ids, n_pairs=a.n * 3,
        tiny16_overlap=overlap, n_overlap=len(overlap),
        note='deterministic subset of train575; donors may lie outside the 64',
    ))
    dump_json(os.path.join(a.root, 'subset', 'train64_mismatch_map.json'), mmap)

    # Reuse V3A5C-style cache construction is heavy; training will compute D on
    # the fly with CorrectionCache (192 pairs fit easily). Still copy nothing
    # bulky — only ids + init.
    init = build_d2_shared_init(seed=42)
    torch.save(init, os.path.join(a.root, 'init', 'shared_init_s42.pt'))
    init_sha = state_dict_sha(init['A0_control'])

    lock = dict(
        stage='V3-A.5D2.1',
        root=a.root,
        repo_commit=git_head(),
        n_images=a.n,
        n_pairs=a.n * 3,
        subset_seed=a.seed,
        train64_ids_sha256=_sha256(os.path.join(a.root, 'subset', 'train64_ids.json')),
        energy_threshold=thr,
        proposal_sha256=v5a.get('proposal_sha256'),
        init_sha=init_sha,
        arms=list(ARMS),
        primary_arm='A1_multiscale',
        out_weight=0.0,
        updates=20000,
        exposure_ckpts=list(EXPOSURE_STEPS),
        grad_accum=4,
        seed=42,
        lr='1e-4 constant (match D2 tiny)',
        state_cycle='equal-freq over 192 pairs (like tiny)',
        tiny16_overlap=overlap,
        note='exposure/scale audit; no evidence; no target change',
        prior='D2-full @3k FAIL but train575 also unfit → undertrained vs ungeneralizable',
    )
    dump_json(os.path.join(a.root, 'artifact_lock.json'), lock)
    print('D2.1 lock -> %s' % os.path.join(a.root, 'artifact_lock.json'))
    print('  n=%d pairs=%d overlap_tiny16=%d init=%s'
          % (a.n, a.n * 3, len(overlap), init_sha[:16]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
