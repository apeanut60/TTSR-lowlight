#!/usr/bin/env python
"""V6.0 setup: zero-init RefineBlocks; HARD step0 V6 == frozen LLFormer Base."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.V3BResidualFusion import count_params  # noqa: E402
from model.V6LLFormerRefBridge import (CANONICAL_BASE, INJECTION_POINT,  # noqa: E402
                                       base_forward_m11, load_frozen_llformer_base,
                                       tiled_v6_forward)
from model.V6Model import V6Model, v6_step0_deltas_zero  # noqa: E402
from option import parser as option_parser  # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import snapshot_, state_dict_sha  # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head  # noqa: E402
from v3a72_runtime import json_ready  # noqa: E402
from v3b_runtime import GRAD_ACCUM, LR, SEED  # noqa: E402
from v6_runtime import (ARM_A1, CACHE_NAME, CKPT_STEPS, DEFAULT_UPDATES,  # noqa: E402
                        LOSS, MISMATCH_DEV, MISMATCH_TRAIN, OPTIMIZER,
                        REFERENCE_VARIANT, ROOT, SPLIT_JSON, SRC,
                        lock_architecture_fields)

CHECK_N = 8
ABS_TOL = 1e-6


def _check(base, branch, ds, indices, device, tag):
    worst = 0.0
    with torch.no_grad():
        for i in indices:
            t = sample_tensors(ds, int(i), 'correct', device)
            y_base = base_forward_m11(base, t['X'])
            y_v6 = tiled_v6_forward(base, branch, t['X'], t['Y0'], t['R'])
            db = float((y_base - y_v6).abs().max())
            dy0 = float((y_v6 - t['Y0']).abs().max())
            worst = max(worst, db)
            if db > ABS_TOL:
                raise SystemExit(
                    'HARD STOP step0 %s[%d] %s: d(Base,V6)=%.3e'
                    % (tag, i, t['name'], db))
            if dy0 > 1e-3:
                # cache should match live Base; warn-level for pad/feather
                print('  warn Y0 vs V6 step0 %.3e %s' % (dy0, t['name']),
                      flush=True)
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--base_ckpt', default=CANONICAL_BASE)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(_CLI)

    for d in ('init', 'diagnostics', 'logs', ARM_A1 + '/checkpoints', CACHE_NAME):
        os.makedirs(os.path.join(a.root, d), exist_ok=True)
    lock_path = os.path.join(a.root, 'artifact_lock.json')
    if os.path.isfile(lock_path) and not a.force:
        raise SystemExit('lock exists; pass --force: %s' % lock_path)

    cache_meta = os.path.join(a.root, CACHE_NAME, 'refiner_train', 'metadata.json')
    if not os.path.isfile(cache_meta):
        raise SystemExit('missing Y0 cache; run prepare_v6_y0_cache.py first: %s'
                         % cache_meta)

    torch.manual_seed(SEED)
    model = V6Model()
    if not v6_step0_deltas_zero(model):
        raise SystemExit('RefineBlock out not zero')
    n_p = count_params(model)
    sd = snapshot_(model)
    sha = state_dict_sha(sd)
    init_path = os.path.join(a.root, 'init', 'A1_v6_s42.pt')
    torch.save(dict(model=sd, arm=ARM_A1, injection_point=INJECTION_POINT),
               init_path)

    print('=== loading frozen LLFormer Base ===', flush=True)
    base = load_frozen_llformer_base(a.base_ckpt, a.device)
    model.to(a.device).eval()

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = REFERENCE_VARIANT
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       SPLIT_JSON)
    mmap_tr = json.load(open(MISMATCH_TRAIN))
    mmap_dv = json.load(open(MISMATCH_DEV))
    y0_dir = os.path.join(a.root, CACHE_NAME, 'refiner_train')
    ds_tr = make_dataset(ns, splits['train'], y0_dir, mmap_tr)
    ds_dv = make_dataset(ns, splits['dev'], y0_dir, mmap_dv)

    print('=== step0 V6 vs Base train8 ===', flush=True)
    wvt = _check(base, model, ds_tr, range(CHECK_N), a.device, 'train')
    print('  worst d(Base,V6)=%.3e' % wvt, flush=True)
    print('=== step0 V6 vs Base dev8 ===', flush=True)
    wvd = _check(base, model, ds_dv, range(CHECK_N), a.device, 'dev')
    print('  worst d(Base,V6)=%.3e' % wvd, flush=True)

    fields = lock_architecture_fields()
    if fields.get('official_test_allowed') is not False:
        raise SystemExit('official_test_allowed must be False')
    lock = dict(
        **fields,
        repo_commit=git_head(),
        root=a.root,
        src_root=a.src_root,
        base_ckpt=a.base_ckpt,
        base_ckpt_sha256=file_sha256(a.base_ckpt),
        base_cache_metadata_sha256=file_sha256(cache_meta),
        cache_metadata_path=cache_meta,
        cache_metadata_sha256=file_sha256(cache_meta),
        cache_name=CACHE_NAME,
        split_json=SPLIT_JSON,
        split_sha256=file_sha256(SPLIT_JSON),
        mismatch_train=MISMATCH_TRAIN,
        mismatch_train_sha256=file_sha256(MISMATCH_TRAIN),
        mismatch_dev=MISMATCH_DEV,
        mismatch_dev_sha256=file_sha256(MISMATCH_DEV),
        reference_variant=REFERENCE_VARIANT,
        optimizer=OPTIMIZER,
        lr=LR,
        weight_decay=0.0,
        updates=DEFAULT_UPDATES,
        grad_accum=GRAD_ACCUM,
        seed=SEED,
        pair_schedule_seed=SEED,
        checkpoint_steps=list(CKPT_STEPS),
        n_params=n_p,
        v6_init_path=init_path,
        v6_init_sha=sha,
        step0_check=dict(train_worst=wvt, dev_worst=wvd, n=CHECK_N),
        note='A0=frozen LLFormer E110 Y0; A1=V5-structure Ref H/4+H/2; MSE; no B0',
    )
    dump_json(lock_path, json_ready(lock))
    print('V6.0 setup OK  n=%d sha=%s' % (n_p, sha[:12]), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
