#!/usr/bin/env python
"""V3-A.7.2 setup: write artifact_lock only. No training. Audit must not overwrite."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from v3a5e1b_runtime import file_sha256                                  # noqa: E402
from v3a6_runtime import dump_json, git_head                             # noqa: E402
from v3a72_runtime import (BOOTSTRAP_B, BOOTSTRAP_SEED, COVERAGES,       # noqa: E402
                           inspect_f2_ckpt, json_ready)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V5A = '/root/data/experiments/v3a5_g64_verifier'
V6 = '/root/data/experiments/v3a6_decision_gate'
V7 = '/root/data/experiments/v3a7_utility_gate'
V71 = '/root/data/experiments/v3a71_utility_predictability'
D21 = '/root/data/experiments/v3a5d21_scale64'
ROOT = '/root/data/experiments/v3a72_selective_closure'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
F2_CKPT = os.path.join(D21, 'A1_multiscale/checkpoints/ckpt_020000.pt')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--v5a_root', default=V5A)
    ap.add_argument('--v6_root', default=V6)
    ap.add_argument('--v7_root', default=V7)
    ap.add_argument('--v71_root', default=V71)
    ap.add_argument('--f2_ckpt', default=F2_CKPT)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(_CLI)

    os.makedirs(os.path.join(a.root, 'diagnostics'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    lock_path = os.path.join(a.root, 'artifact_lock.json')
    if os.path.isfile(lock_path) and not a.force:
        raise SystemExit('lock exists; pass --force to rebuild: %s' % lock_path)

    v5a = json.load(open(os.path.join(a.v5a_root, 'artifact_lock.json')))
    v6 = json.load(open(os.path.join(a.v6_root, 'artifact_lock.json')))
    v71 = json.load(open(os.path.join(a.v71_root, 'artifact_lock.json')))
    proposal_path = os.path.join(a.src_root, R1_CK)
    split_path = os.path.join(a.v4_root, 'splits', 'split.json')
    mmap_tr = os.path.join(a.v4_root, 'mappings', 'mismatch_train_575.json')
    mmap_dv = os.path.join(a.v4_root, 'mappings', 'mismatch_dev_64.json')

    blob = torch.load(a.f2_ckpt, map_location='cpu')
    meta = inspect_f2_ckpt(blob, a.f2_ckpt)
    if not meta['ok_step'] or not meta['ok_arm']:
        raise SystemExit('F2 ckpt semantic fail: %s' % meta)

    v6_sum = json.load(open(os.path.join(
        a.v6_root, 'diagnostics', 'eval_020000', 'summary_psnr.json')))
    v7_sum = json.load(open(os.path.join(
        a.v7_root, 'diagnostics', 'eval_020000', 'summary_psnr.json')))
    def _mean(d):
        return float(sum(d.values()) / len(d))

    lock = dict(
        stage='V3-A.7.2',
        root=a.root,
        prior='V3A71_CASE_C_RANKING_COVERAGE',
        repo_commit=git_head(),
        proposal_ckpt=proposal_path,
        proposal_sha256=file_sha256(proposal_path),
        cache_name=v5a.get('cache_name', 'cache_y0_lolbase'),
        cache_metadata_sha256=v5a.get('cache_metadata_sha256'),
        split_json=split_path,
        split_sha256=file_sha256(split_path),
        mismatch_train=mmap_tr,
        mismatch_train_sha256=file_sha256(mmap_tr),
        mismatch_dev=mmap_dv,
        mismatch_dev_sha256=file_sha256(mmap_dv),
        energy_threshold=float(v5a['energy_threshold']),
        energy_stats_sha256=v5a.get('energy_stats_sha256'),
        reference_variant=v5a.get('reference_variant', 'nanobanana_ref_v2'),
        geometry='g64',
        f0_source='V3A5E_FEATURE_NAMES',
        f2_ckpt=a.f2_ckpt,
        f2_ckpt_sha256=file_sha256(a.f2_ckpt),
        f2_ckpt_step=int(meta['step']),
        f2_ckpt_arm=meta['arm'],
        f2_source_stage='V3-A.5D2.1',
        f2_out_weight=0.0,
        f2_training='qstar_G64_gate_only_memorization',
        v71_block_rows=os.path.join(a.v71_root, 'diagnostics', 'block_rows.npz'),
        init_sha=v6.get('init_sha'),
        per_image=int(v71.get('per_image', 96)),
        bootstrap_B=BOOTSTRAP_B,
        bootstrap_seed=BOOTSTRAP_SEED,
        coverage_grid=list(COVERAGES),
        primary_coverage=0.10,
        seed=42,
        official_test_allowed=False,
        frozen_baseline_psnr=dict(
            v3a6_A1=_mean(v6_sum['A1']),
            v3a7_A1=_mean(v7_sum['A1']),
            v3a6_A1_by_state=v6_sum['A1'],
            v3a7_A1_by_state=v7_sum['A1'],
            global_constant_q=0.5,
        ),
        note='zero-train selective coverage closure; train-calibrated tau only',
    )
    dump_json(lock_path, json_ready(lock))
    print('V3-A.7.2 setup OK')
    print('  lock=%s' % lock_path)
    print('  F2 step=%s arm=%s sha=%s' % (
        meta['step'], meta['arm'], lock['f2_ckpt_sha256'][:16]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
