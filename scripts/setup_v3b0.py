#!/usr/bin/env python
"""V3-B.0 setup: residual-head init + artifact_lock. No training."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.V3BResidualFusion import V3B0ResidualFusion, count_params     # noqa: E402
from v3a5_runtime import snapshot_, state_dict_sha                      # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head               # noqa: E402
from v3b_runtime import (ARM, DEFAULT_UPDATES, GRAD_ACCUM, LR, SEED,    # noqa: E402
                         json_ready)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V5A = '/root/data/experiments/v3a5_g64_verifier'
V6 = '/root/data/experiments/v3a6_decision_gate'
V7 = '/root/data/experiments/v3a7_utility_gate'
ROOT = '/root/data/experiments/v3b0_implicit_residual'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--v5a_root', default=V5A)
    ap.add_argument('--v6_root', default=V6)
    ap.add_argument('--v7_root', default=V7)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(_CLI)

    for d in ('init', 'checkpoints', 'diagnostics', 'logs'):
        os.makedirs(os.path.join(a.root, d), exist_ok=True)
    lock_path = os.path.join(a.root, 'artifact_lock.json')
    if os.path.isfile(lock_path) and not a.force:
        raise SystemExit('lock exists; pass --force: %s' % lock_path)

    v5a = json.load(open(os.path.join(a.v5a_root, 'artifact_lock.json')))
    v6s = json.load(open(os.path.join(
        a.v6_root, 'diagnostics', 'eval_020000', 'summary_psnr.json')))
    v7s = json.load(open(os.path.join(
        a.v7_root, 'diagnostics', 'eval_020000', 'summary_psnr.json')))

    proposal_path = os.path.join(a.src_root, R1_CK)
    split_path = os.path.join(a.v4_root, 'splits', 'split.json')
    mmap_tr = os.path.join(a.v4_root, 'mappings', 'mismatch_train_575.json')
    mmap_dv = os.path.join(a.v4_root, 'mappings', 'mismatch_dev_64.json')

    torch.manual_seed(SEED)
    head = V3B0ResidualFusion()
    init_path = os.path.join(a.root, 'init', 'b0_zero_out_s42.pt')
    sd = snapshot_(head)
    torch.save(dict(model=sd, arm=ARM), init_path)
    n_params = count_params(head)
    delta = head(torch.randn(1, 32, 8, 8), torch.randn(1, 32, 8, 8), (16, 16))
    if float(delta.abs().max()) > 1e-7:
        raise SystemExit('init ΔY not zero: %.3e' % float(delta.abs().max()))

    def _mean(d):
        return float(sum(d.values()) / len(d))

    lock = dict(
        stage='V3-B.0',
        root=a.root,
        prior='V3A72_CASE_C_EXPLORATORY_WEAK',
        repo_commit=git_head(),
        proposal_ckpt=proposal_path,
        proposal_sha256=file_sha256(proposal_path),
        cache_name=v5a.get('cache_name', 'cache_y0_lolbase'),
        base_cache_metadata_sha256=v5a.get('cache_metadata_sha256'),
        split_json=split_path,
        split_sha256=file_sha256(split_path),
        mismatch_train=mmap_tr,
        mismatch_train_sha256=file_sha256(mmap_tr),
        mismatch_dev=mmap_dv,
        mismatch_dev_sha256=file_sha256(mmap_dv),
        reference_variant=v5a.get('reference_variant', 'nanobanana_ref_v2'),
        architecture='V3B0ResidualFusion',
        arm=ARM,
        n_params=n_params,
        init_path=init_path,
        init_sha=state_dict_sha(sd),
        optimizer='Adam',
        lr=LR,
        weight_decay=0.0,
        updates=DEFAULT_UPDATES,
        grad_accum=GRAD_ACCUM,
        seed=SEED,
        pair_schedule_seed=SEED,
        official_test_allowed=False,
        fusion_inputs=('F0', 'T', 'F0-T'),
        forbidden_inputs=('D', 'g_v2', 'q_star', 'U', 'X', 'Y0_as_feature',
                          'evidence'),
        frozen_baseline_psnr=dict(
            v3a6_A1=_mean(v6s['A1']),
            v3a7_A1=_mean(v7s['A1']),
            v3a6_A1_by_state=v6s['A1'],
            v3a7_A1_by_state=v7s['A1'],
        ),
        note='implicit RGB residual on frozen matcher F0/T; no explicit gate',
    )
    dump_json(lock_path, json_ready(lock))
    print('V3-B.0 setup OK  n_params=%d  init_sha=%s' % (n_params, lock['init_sha'][:16]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
