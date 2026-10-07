#!/usr/bin/env python
"""V5.0 setup: zero-init RefineBlocks; HARD V5 step0 == Base. No training."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import metrics as _metrics                        # noqa: E402
from model.V3BResidualFusion import count_params                            # noqa: E402
from model.V5Model import V5Model, v5_step0_deltas_zero                     # noqa: E402
from model.V5RetinexBridge import (INJECTION_POINT, load_frozen_retinex_mainnet,  # noqa: E402
                                   tiled_bridge_decode, tiled_v5_forward)
from option import parser as option_parser                                  # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors           # noqa: E402
from v3a5_runtime import snapshot_, state_dict_sha                          # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head                   # noqa: E402
from v3b_runtime import GRAD_ACCUM, LR, SEED, json_ready                    # noqa: E402
from v5_runtime import (ARM_A1, CKPT_STEPS, DEFAULT_UPDATES, LOSS,           # noqa: E402
                        OPTIMIZER, lock_architecture_fields)

SRC = '/root/data/experiments/v3a1_lolv2real'
B0 = '/root/data/experiments/v3b0_implicit_residual'
ROOT = '/root/data/experiments/v5_aligned_ref'
CACHE_META = os.path.join(SRC, 'cache_y0_lolbase', 'refiner_train', 'metadata.json')
CHECK_N = 8
ABS_TOL = 1e-6


def _check(main, branch, ds, indices, device, tag):
    worst_v5 = 0.0
    with torch.no_grad():
        for i in indices:
            t = sample_tensors(ds, int(i), 'correct', device)
            y_base = tiled_bridge_decode(main, t['X'], delta_fn=None)
            y_v5 = tiled_v5_forward(main, branch, t['X'], t['Y0'], t['R'])
            db = float((y_base - y_v5).abs().max())
            worst_v5 = max(worst_v5, db)
            p0 = float(_metrics(y_base, t['H'])[0])
            p1 = float(_metrics(y_v5, t['H'])[0])
            if db > ABS_TOL or abs(p0 - p1) > ABS_TOL:
                raise SystemExit(
                    'HARD STOP step0 %s[%d] %s: dY=%.3e dPSNR=%.3e'
                    % (tag, i, t['name'], db, abs(p0 - p1)))
    return worst_v5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--b0_root', default=B0)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(_CLI)

    for d in ('init', 'diagnostics', 'logs', ARM_A1 + '/checkpoints'):
        os.makedirs(os.path.join(a.root, d), exist_ok=True)
    lock_path = os.path.join(a.root, 'artifact_lock.json')
    if os.path.isfile(lock_path) and not a.force:
        raise SystemExit('lock exists; pass --force: %s' % lock_path)

    b0 = json.load(open(os.path.join(a.b0_root, 'artifact_lock.json')))
    b0_psnr = json.load(open(os.path.join(
        a.b0_root, 'diagnostics', 'eval_020000', 'summary_psnr.json')))
    meta = json.load(open(CACHE_META))
    base_ckpt = meta['base_checkpoint']
    base_run = os.path.dirname(os.path.dirname(base_ckpt))

    torch.manual_seed(SEED)
    model = V5Model()
    if not v5_step0_deltas_zero(model):
        raise SystemExit('RefineBlock out not zero')
    n_p = count_params(model)
    sd = snapshot_(model)
    sha = state_dict_sha(sd)
    init_path = os.path.join(a.root, 'init', 'A1_v5_s42.pt')
    torch.save(dict(model=sd, arm=ARM_A1, injection_point=INJECTION_POINT),
               init_path)

    print('=== loading frozen Retinexformer MainNet (no VGG/LTE) ===', flush=True)
    mainnet = load_frozen_retinex_mainnet(base_ckpt, base_run, a.device)
    model.to(a.device).eval()

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = b0['reference_variant']
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       b0['split_json'])
    mmap_tr = json.load(open(b0['mismatch_train']))
    mmap_dv = json.load(open(b0['mismatch_dev']))
    ds_tr = make_dataset(
        ns, splits['train'],
        os.path.join(a.src_root, b0.get('cache_name', 'cache_y0_lolbase'),
                     'refiner_train'), mmap_tr)
    ds_dv = make_dataset(
        ns, splits['dev'],
        os.path.join(a.src_root, b0.get('cache_name', 'cache_y0_lolbase'),
                     'refiner_train'), mmap_dv)

    print('=== step0 V5 vs Base train8 ===', flush=True)
    wvt = _check(mainnet, model, ds_tr, range(CHECK_N), a.device, 'train')
    print('  worst d(Base,V5)=%.3e' % wvt, flush=True)
    print('=== step0 V5 vs Base dev8 ===', flush=True)
    wvd = _check(mainnet, model, ds_dv, range(CHECK_N), a.device, 'dev')
    print('  worst d(Base,V5)=%.3e' % wvd, flush=True)

    # official_test_allowed=False is required in lock_architecture_fields()
    fields = lock_architecture_fields()
    if fields.get('official_test_allowed') is not False:
        raise SystemExit('official_test_allowed must be False')
    lock = dict(
        **fields,
        repo_commit=git_head(),
        root=a.root,
        src_root=a.src_root,
        b0_root=a.b0_root,
        base_ckpt=base_ckpt,
        base_ckpt_sha256=file_sha256(base_ckpt),
        base_run_dir=base_run,
        base_cache_metadata_sha256=file_sha256(CACHE_META),
        cache_metadata_path=CACHE_META,
        cache_metadata_sha256=file_sha256(CACHE_META),
        cache_name=b0.get('cache_name', 'cache_y0_lolbase'),
        proposal_ckpt=b0['proposal_ckpt'],
        proposal_sha256=b0['proposal_sha256'],
        split_json=b0['split_json'],
        split_sha256=b0['split_sha256'],
        mismatch_train=b0['mismatch_train'],
        mismatch_train_sha256=b0['mismatch_train_sha256'],
        mismatch_dev=b0['mismatch_dev'],
        mismatch_dev_sha256=b0['mismatch_dev_sha256'],
        reference_variant=b0['reference_variant'],
        b0_head_ckpt=os.path.join(a.b0_root, 'checkpoints', 'ckpt_020000.pt'),
        frozen_b0_psnr={k: float(v) for k, v in b0_psnr['B0_normal'].items()},
        optimizer=OPTIMIZER,
        lr=LR,
        weight_decay=0.0,
        updates=DEFAULT_UPDATES,
        grad_accum=GRAD_ACCUM,
        seed=SEED,
        pair_schedule_seed=SEED,
        checkpoint_steps=list(CKPT_STEPS),
        n_params=n_p,
        v5_init_path=init_path,
        v5_init_sha=sha,
        step0_check=dict(train_worst=wvt, dev_worst=wvd, n=CHECK_N),
        note='A0=frozen B0@20k eval; A1=V5.0 explicit align + H/4+H/2 refine; MSE; Base frozen',
    )
    dump_json(lock_path, json_ready(lock))
    print('V5.0 setup OK  n=%d sha=%s' % (n_p, sha[:12]), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
