#!/usr/bin/env python
"""V7-RIVER setup: zero-init fusion + step0 ≡ tiled Base HARD."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.V3BResidualFusion import count_params  # noqa: E402
from model.V6LLFormerRefBridge import (base_forward_m11,  # noqa: E402
                                       load_frozen_llformer_base, tiled_v6_forward)
from model.V7Model import V7Model, v7_step0_deltas_zero  # noqa: E402
from option import parser as option_parser  # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import snapshot_, state_dict_sha  # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head  # noqa: E402
from v3a72_runtime import json_ready  # noqa: E402
from v3b_runtime import GRAD_ACCUM, LR, SEED  # noqa: E402
from v7_river_runtime import (BASE_CKPT, CACHE_NAME, CKPT_STEPS,  # noqa: E402
                              DEFAULT_UPDATES, LOSS, MISMATCH_TRAIN, OPTIMIZER,
                              REFERENCE_VARIANT, ROOT, SPLIT_JSON, SRC,
                              ensure_y0_cache_link, lock_architecture_fields,
                              setting_root)

CHECK_N = 4
ABS_TOL = 1e-6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--setting', default='A', choices=['A', 'B'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(_CLI)

    arm_root = setting_root(a.setting, a.root)
    for d in ('init', 'diagnostics', 'logs', 'checkpoints', 'smoke_checkpoints'):
        os.makedirs(os.path.join(arm_root, d), exist_ok=True)
    # shared cache at experiments root
    ensure_y0_cache_link(a.root)
    lock_path = os.path.join(arm_root, 'artifact_lock.json')
    if os.path.isfile(lock_path) and not a.force:
        raise SystemExit('lock exists; pass --force: %s' % lock_path)

    cache_meta = os.path.join(a.root, CACHE_NAME, 'refiner_train', 'metadata.json')
    if not os.path.isfile(cache_meta):
        raise SystemExit('missing Y0 cache metadata: %s' % cache_meta)

    torch.manual_seed(SEED)
    model = V7Model()
    if not v7_step0_deltas_zero(model):
        raise SystemExit('fusion out not zero-init')
    train_names = model.apply_setting_freeze(a.setting)
    n_p = count_params(model)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    sd = snapshot_(model)
    sha = state_dict_sha(sd)
    init_path = os.path.join(arm_root, 'init', 'v7_river_s42.pt')
    torch.save(dict(model=sd, setting=a.setting, arm=lock_architecture_fields(
        a.setting)['arm']), init_path)

    print('=== loading frozen LLFormer Base ===', flush=True)
    base = load_frozen_llformer_base(BASE_CKPT, a.device)
    model.to(a.device).eval()

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = REFERENCE_VARIANT
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       SPLIT_JSON)
    mmap = json.load(open(MISMATCH_TRAIN))
    ds = make_dataset(ns, splits['train'],
                      os.path.join(a.root, CACHE_NAME, 'refiner_train'), mmap)
    worst = 0.0
    with torch.no_grad():
        for i in range(min(CHECK_N, len(splits['train']))):
            t = sample_tensors(ds, i, 'correct', a.device)
            y_base = base_forward_m11(base, t['X'])
            y_v7 = tiled_v6_forward(base, model, t['X'], t['Y0'], t['R'])
            db = float((y_base - y_v7).abs().max())
            worst = max(worst, db)
            if db > ABS_TOL:
                raise SystemExit(
                    'HARD STOP step0[%d] %s d(Base,V7)=%.3e' % (i, t['name'], db))
    print('  step0 worst=%.3e OK' % worst, flush=True)

    # Setting A: align params must not require grad
    if a.setting.upper() == 'A':
        for name, p in model.named_parameters():
            if name.startswith(('match_encoder', 'texture_encoder', 'align_')):
                if p.requires_grad:
                    raise SystemExit('Setting A: align still trainable: %s' % name)

    lock = lock_architecture_fields(a.setting)
    lock.update(dict(
        repo_commit=git_head(),
        root=a.root,
        arm_root=arm_root,
        src_root=a.src_root,
        data_dir=a.data_dir,
        split_json=SPLIT_JSON,
        mismatch_train=MISMATCH_TRAIN,
        cache_name=CACHE_NAME,
        base_ckpt_sha256=file_sha256(BASE_CKPT),
        base_cache_metadata_sha256=file_sha256(cache_meta),
        split_sha256=file_sha256(SPLIT_JSON),
        mismatch_train_sha256=file_sha256(MISMATCH_TRAIN),
        optimizer=OPTIMIZER,
        lr=LR,
        weight_decay=0.0,
        updates=DEFAULT_UPDATES,
        grad_accum=GRAD_ACCUM,
        seed=SEED,
        pair_schedule_seed=SEED,
        checkpoint_steps=list(CKPT_STEPS),
        v7_init_path=init_path,
        v7_init_sha=sha,
        n_params=n_p,
        n_trainable=n_train,
        trainable_param_names_sample=train_names[:20],
        step0_worst=worst,
        gate_step0_ok=True,
        selection_split='official_Test',
        note='Selection/ablation on Official Test per user request; not dev64',
    ))
    dump_json(lock_path, json_ready(lock))
    # also copy pointer at experiments root for current setting
    dump_json(os.path.join(a.root, 'current_lock_%s.json' % a.setting),
              json_ready(lock))
    print('=== V7-RIVER setup OK setting=%s ===' % a.setting, flush=True)
    print('  trainable %d / %d' % (n_train, n_p), flush=True)
    print('  lock %s' % lock_path, flush=True)


if __name__ == '__main__':
    main()
