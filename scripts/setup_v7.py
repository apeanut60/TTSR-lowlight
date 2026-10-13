#!/usr/bin/env python
"""V7 setup: C2 load + zero-init ATT + Gate0/2 step0 Y==Y0 HARD checks."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.V3BResidualFusion import count_params  # noqa: E402
from model.td_v6_adapter import V7TaskDecoupledModel, _rgb01  # noqa: E402
from model.td_transfer import td_reconstruction_losses  # noqa: E402
from option import parser as option_parser  # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import snapshot_, state_dict_sha  # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head  # noqa: E402
from v3a72_runtime import json_ready  # noqa: E402
from v3b_runtime import GRAD_ACCUM, LR, SEED  # noqa: E402
from v7_runtime import (ARM_A1, CACHE_NAME, CKPT_STEPS, DEFAULT_UPDATES,  # noqa: E402
                        LOSS, MISMATCH_DEV, MISMATCH_TRAIN, OPTIMIZER,
                        PAIR_STATES_A1, REFERENCE_VARIANT, ROOT, SRC,
                        SPLIT_JSON, BASE_CKPT, C2_CKPT,
                        ensure_y0_cache_link, lock_architecture_fields)

CHECK_N = 8
ABS_TOL = 1e-6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(_CLI)

    for d in ('init', 'diagnostics', 'logs', ARM_A1 + '/checkpoints'):
        os.makedirs(os.path.join(a.root, d), exist_ok=True)
    lock_path = os.path.join(a.root, 'artifact_lock.json')
    if os.path.isfile(lock_path) and not a.force:
        raise SystemExit('lock exists; pass --force: %s' % lock_path)

    ensure_y0_cache_link(a.root)
    cache_meta = os.path.join(a.root, CACHE_NAME, 'refiner_train', 'metadata.json')
    if not os.path.isfile(cache_meta):
        raise SystemExit('missing Y0 cache metadata: %s' % cache_meta)

    torch.manual_seed(SEED)
    print('=== Gate1: load C2 feature_extraction ===', flush=True)
    model = V7TaskDecoupledModel(c2_ckpt=C2_CKPT).to(a.device)
    model.assert_formal_ready()
    model.eval()

    # Gate2: zero-init → Y==Y0
    print('=== Gate2: step0 identity ===', flush=True)
    with torch.no_grad():
        y0 = torch.rand(1, 3, 64, 64, device=a.device)
        ref = torch.rand(1, 3, 64, 64, device=a.device)
        out = model(y0, ref)
        d = float((out['y'] - y0).abs().max())
        if d > ABS_TOL:
            raise SystemExit('HARD STOP step0 Y!=Y0 max_abs=%.3e' % d)
        # C=0 → fused == base
        base = model.transfer.content(y0)
        out0 = model.transfer(y0, out['aligned_h4'], torch.zeros_like(out['raw_confidence_h4']))
        if float((out0['fused'] - base).abs().max()) > ABS_TOL:
            raise SystemExit('HARD STOP Eq.10 C=0 fused!=base')
        # I_tex zero at init
        if float(out['i_tex'].abs().max()) > ABS_TOL:
            raise SystemExit('HARD STOP I_tex not zero at init')

    # Grad: L_tex reaches I_tex path
    print('=== Gate2: L_tex / L_rec grads ===', flush=True)
    model.train()
    y0 = torch.rand(1, 3, 64, 64, device=a.device, requires_grad=False)
    ref = torch.rand(1, 3, 64, 64, device=a.device)
    gt = torch.rand(1, 3, 64, 64, device=a.device)
    out = model(y0, ref)
    losses = td_reconstruction_losses(out, gt)
    losses['loss'].backward()
    g_tex = float(model.transfer.texture_decoder.out.weight.grad.abs().sum())
    g_att = float(model.transfer.att.rgb_out.weight.grad.abs().sum())
    if g_tex <= 0 or g_att <= 0:
        raise SystemExit('HARD STOP missing grads tex=%.3e att=%.3e' % (g_tex, g_att))
    # C2 frozen
    for p in model.match_features.parameters():
        if p.requires_grad:
            raise SystemExit('HARD STOP C2 not frozen')
        if p.grad is not None:
            raise SystemExit('HARD STOP C2 got grad')

    n_p = count_params(model)
    n_train = sum(p.numel() for _, p in model.trainable_named_parameters())
    sd = snapshot_(model)
    # strip C2 buffers from init sha? keep full state for resume
    sha = state_dict_sha(sd)
    init_path = os.path.join(a.root, 'init', 'A1_v7_s42.pt')
    torch.save(dict(model=sd, arm=ARM_A1, c2_ckpt=C2_CKPT), init_path)

    # Real-data step0 on a few tiles
    print('=== Gate0/2: real-data step0 (correct) ===', flush=True)
    model.eval()
    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = REFERENCE_VARIANT
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       SPLIT_JSON)
    mmap_tr = json.load(open(MISMATCH_TRAIN))
    y0_dir = os.path.join(a.root, CACHE_NAME, 'refiner_train')
    ds = make_dataset(ns, splits['train'], y0_dir, mmap_tr)
    worst = 0.0
    with torch.no_grad():
        for i in range(min(CHECK_N, len(splits['train']))):
            t = sample_tensors(ds, i, 'correct', a.device)
            out = model.forward_m11(t['Y0'], t['R'])
            # compare in [0,1]
            y0_01 = _rgb01(t['Y0'])
            db = float((out['y'] - y0_01).abs().max())
            worst = max(worst, db)
            if db > ABS_TOL:
                raise SystemExit(
                    'HARD STOP real step0[%d] %s d=%.3e' % (i, t['name'], db))
    print('  real step0 worst=%.3e OK' % worst, flush=True)

    lock = lock_architecture_fields()
    lock.update(dict(
        repo_commit=git_head(),
        root=a.root,
        src_root=a.src_root,
        data_dir=a.data_dir,
        split_json=SPLIT_JSON,
        mismatch_train=MISMATCH_TRAIN,
        mismatch_dev=MISMATCH_DEV,
        cache_name=CACHE_NAME,
        base_ckpt_sha256=file_sha256(BASE_CKPT),
        base_cache_metadata_sha256=file_sha256(cache_meta),
        c2_ckpt_sha256=file_sha256(C2_CKPT),
        split_sha256=file_sha256(SPLIT_JSON),
        mismatch_train_sha256=file_sha256(MISMATCH_TRAIN),
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
        v7_init_path=init_path,
        v7_init_sha=sha,
        n_params=n_p,
        n_trainable=n_train,
        gate0_ok=True,
        gate1_c2_strict=True,
        gate2_step0_ok=True,
        gate2_grad_ok=True,
        step0_real_worst=worst,
        pair_states=list(PAIR_STATES_A1),
        loss=LOSS,
    ))
    dump_json(lock_path, json_ready(lock))
    print('=== V7 setup OK ===', flush=True)
    print('  init %s sha=%s' % (init_path, sha[:16]), flush=True)
    print('  trainable %d / total %d' % (n_train, n_p), flush=True)
    print('  lock %s' % lock_path, flush=True)


if __name__ == '__main__':
    main()
