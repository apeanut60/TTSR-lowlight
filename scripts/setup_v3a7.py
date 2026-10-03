#!/usr/bin/env python
"""V3-A.7 setup: reuse V3-A.6 shared MultiScale init; lock A0=V3A6 decision-MSE."""

import argparse
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from v3a5_runtime import state_dict_sha                                  # noqa: E402
from v3a6_runtime import CKPT_STEPS, DEFAULT_UPDATES, GRAD_ACCUM, LR, SEED  # noqa: E402
from v3a7_runtime import (BOTTLENECK, OBJECTIVES, dump_json,             # noqa: E402
                          file_sha256, git_head)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V5A = '/root/data/experiments/v3a5_g64_verifier'
V6 = '/root/data/experiments/v3a6_decision_gate'
ROOT = '/root/data/experiments/v3a7_utility_gate'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--v6_root', default=V6)
    a = ap.parse_args(_CLI)

    for d in ('init', 'diagnostics', 'logs', 'A1_utility_bce/checkpoints'):
        os.makedirs(os.path.join(a.root, d), exist_ok=True)

    v6lock = json.load(open(os.path.join(a.v6_root, 'artifact_lock.json')))
    src_init = v6lock['init_path']
    dst_init = os.path.join(a.root, 'init', 'shared_init_s42.pt')
    shutil.copy2(src_init, dst_init)
    blob = torch.load(dst_init, map_location='cpu')
    # V3A6 init keys A0_qstar / A1_decision_mse are bit-equal MultiScale
    sd = blob['A1_decision_mse']
    init_sha = state_dict_sha(sd)

    a0_ckpts = {}
    for step in (0, 1000, 3000, 5000, 10000, 20000):
        p = os.path.join(a.v6_root, 'A1_decision_mse', 'checkpoints',
                         'ckpt_%06d.pt' % step)
        if not os.path.isfile(p):
            raise SystemExit('V3-A.6 A1 ckpt missing (A0 control): %s' % p)
        a0_ckpts[str(step)] = p

    lock = dict(
        stage='V3-A.7',
        root=a.root,
        prior='V3A6_CASE_D_FAIL',
        repo_commit=git_head(),
        v6_root=a.v6_root,
        a0_control=A0_NAME,
        a0_source_arm='A1_decision_mse',
        a0_checkpoints=a0_ckpts,
        proposal_ckpt=os.path.join(SRC, R1_CK),
        proposal_sha256=v6lock['proposal_sha256'],
        cache_name=v6lock['cache_name'],
        cache_metadata_sha256=v6lock['cache_metadata_sha256'],
        split_json=v6lock['split_json'],
        split_sha256=v6lock['split_sha256'],
        mismatch_train=v6lock['mismatch_train'],
        mismatch_train_sha256=v6lock['mismatch_train_sha256'],
        mismatch_dev=v6lock['mismatch_dev'],
        mismatch_dev_sha256=v6lock['mismatch_dev_sha256'],
        energy_threshold=v6lock['energy_threshold'],
        energy_stats_sha256=v6lock['energy_stats_sha256'],
        reference_variant=v6lock['reference_variant'],
        geometry='g64',
        architecture='V3A5D2Verifier.A1_multiscale',
        bottleneck=BOTTLENECK,
        init_path=dst_init,
        init_sha=init_sha,
        v6_init_sha=v6lock['init_sha'],
        arms=['A0_decision_mse', 'A1_utility_bce'],
        objectives=dict(A0_decision_mse='decision_mse', **OBJECTIVES),
        updates=DEFAULT_UPDATES,
        grad_accum=GRAD_ACCUM,
        lr=LR,
        weight_decay=0.0,
        seed=SEED,
        pair_schedule_seed=SEED,
        states=list(v6lock['states']),
        mask_mode=v6lock['mask_mode'],
        checkpoint_steps=list(CKPT_STEPS),
        official_test_allowed=False,
        note=('A0=V3-A.6 A1 decision_mse ckpts (not retrained); '
              'A1=masked BCE on 1[U_B>0]; sole variable=loss'),
    )
    dump_json(os.path.join(a.root, 'artifact_lock.json'), lock)
    if init_sha != v6lock['init_sha']:
        raise SystemExit('copied init sha mismatch vs V3-A.6')
    print('V3-A.7 setup OK')
    print('  root=%s' % a.root)
    print('  init_sha=%s  thr=%.6e' % (init_sha[:16], lock['energy_threshold']))
    return 0


A0_NAME = 'A0_decision_mse'


if __name__ == '__main__':
    raise SystemExit(main())
