#!/usr/bin/env python
"""V4.0 setup: shared RGB-head init + artifact_lock. No training."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.V3BResidualFusion import V3B0ResidualFusion, count_params    # noqa: E402
from model.V4RefCanvas import CANVAS_A0, CANVAS_A1, DIR_A0, DIR_A1       # noqa: E402
from v3a5_runtime import bit_equal, snapshot_, state_dict_sha           # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head               # noqa: E402
from v3b_runtime import CKPT_STEPS, DEFAULT_UPDATES, GRAD_ACCUM, LR, SEED, json_ready  # noqa: E402
from v4_runtime import ARM_A0, ARM_A1, ARMS                             # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
B0 = '/root/data/experiments/v3b0_implicit_residual'
ROOT = '/root/data/experiments/v4_ref_canvas'
CACHE_META = os.path.join(SRC, 'cache_y0_lolbase', 'refiner_train', 'metadata.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--b0_root', default=B0)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(_CLI)

    for d in ('init', 'diagnostics', 'logs', ARM_A0 + '/checkpoints',
              ARM_A1 + '/checkpoints'):
        os.makedirs(os.path.join(a.root, d), exist_ok=True)
    lock_path = os.path.join(a.root, 'artifact_lock.json')
    if os.path.isfile(lock_path) and not a.force:
        raise SystemExit('lock exists; pass --force: %s' % lock_path)

    b0 = json.load(open(os.path.join(a.b0_root, 'artifact_lock.json')))
    b0_psnr = json.load(open(os.path.join(
        a.b0_root, 'diagnostics', 'eval_020000', 'summary_psnr.json')))
    audit_path = os.path.join(a.root, 'diagnostics', 'ref_quality_audit.json')
    if not os.path.isfile(audit_path):
        raise SystemExit('run scripts/audit_v4_ref_quality.py first: %s' % audit_path)
    audit = json.load(open(audit_path))
    meta = json.load(open(CACHE_META))

    torch.manual_seed(SEED)
    shared = V3B0ResidualFusion(in_ch=96)
    f0 = torch.randn(1, 32, 8, 8)
    tt = torch.randn(1, 32, 8, 8)
    if float(shared(f0, tt, (16, 16)).abs().max()) > 1e-7:
        raise SystemExit('shared head step0 Δ not zero')
    sd = snapshot_(shared)
    n_p = count_params(shared)
    sha = state_dict_sha(sd)

    a0 = V3B0ResidualFusion(in_ch=96)
    a1 = V3B0ResidualFusion(in_ch=96)
    a0.load_state_dict(sd, strict=True)
    a1.load_state_dict(sd, strict=True)
    if not bit_equal(snapshot_(a0), snapshot_(a1)):
        raise SystemExit('A0/A1 init not bit-equal')
    if state_dict_sha(snapshot_(a0)) != sha or state_dict_sha(snapshot_(a1)) != sha:
        raise SystemExit('shared init sha mismatch')

    a0_path = os.path.join(a.root, 'init', 'A0_b0_replay_s42.pt')
    a1_path = os.path.join(a.root, 'init', 'A1_ref_canvas_s42.pt')
    shared_path = os.path.join(a.root, 'init', 'shared_head_s42.pt')
    torch.save(dict(model=sd, role='shared'), shared_path)
    torch.save(dict(model=snapshot_(a0), arm=ARM_A0, direction=DIR_A0,
                    canvas=CANVAS_A0, shared_head_init_sha=sha), a0_path)
    torch.save(dict(model=snapshot_(a1), arm=ARM_A1, direction=DIR_A1,
                    canvas=CANVAS_A1, shared_head_init_sha=sha), a1_path)

    lock = dict(
        stage='V4.0',
        root=a.root,
        prior='V3B4_CASE_D_HARM',
        b0_root=a.b0_root,
        repo_commit=git_head(),
        proposal_ckpt=b0['proposal_ckpt'],
        proposal_sha256=b0['proposal_sha256'],
        cache_name=b0.get('cache_name', 'cache_y0_lolbase'),
        base_cache_metadata_sha256=b0['base_cache_metadata_sha256'],
        cache_metadata_path=CACHE_META,
        cache_metadata_sha256=file_sha256(CACHE_META),
        split_json=b0['split_json'],
        split_sha256=b0['split_sha256'],
        mismatch_train=b0['mismatch_train'],
        mismatch_train_sha256=b0['mismatch_train_sha256'],
        mismatch_dev=b0['mismatch_dev'],
        mismatch_dev_sha256=b0['mismatch_dev_sha256'],
        reference_variant=b0['reference_variant'],
        direction_A0=DIR_A0,
        direction_A1=DIR_A1,
        canvas_A0=CANVAS_A0,
        canvas_A1=CANVAS_A1,
        architecture='V3B0ResidualFusion',
        arms=list(ARMS),
        n_params=n_p,
        shared_head_init_path=shared_path,
        shared_head_init_sha=sha,
        init_paths={ARM_A0: a0_path, ARM_A1: a1_path},
        a0_init_sha=sha,
        a1_init_sha=sha,
        init_shas={ARM_A0: sha, ARM_A1: sha},
        optimizer='Adam',
        lr=LR,
        weight_decay=0.0,
        updates=DEFAULT_UPDATES,
        grad_accum=GRAD_ACCUM,
        seed=SEED,
        pair_schedule_seed=SEED,
        checkpoint_steps=list(CKPT_STEPS),
        official_test_allowed=False,
        frozen_b0_psnr=b0_psnr['B0_normal'],
        ref_quality_audit=audit_path,
        ref_quality_audit_sha256=file_sha256(audit_path),
        base_ckpt=meta.get('base_checkpoint'),
        note='A0=Y0 canvas F0-query; A1=R canvas FR-query; same head init',
    )
    dump_json(lock_path, json_ready(lock))
    print('V4.0 setup OK  n=%d shared_sha=%s' % (n_p, sha[:12]), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
