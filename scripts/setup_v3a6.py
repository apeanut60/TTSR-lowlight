#!/usr/bin/env python
"""V3-A.6 setup: shared MultiScale init + formal artifact_lock. No train."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.V3A6DecisionVerifier import (build_v3a6_model,                   # noqa: E402
                                        build_v3a6_shared_init)
from v3a5_runtime import (bit_equal, prepare_geometry, state_dict_sha,  # noqa: E402
                          target_geometry)
from v3a6_runtime import (ARMS, BOTTLENECK, CKPT_STEPS, DEFAULT_UPDATES,  # noqa: E402
                          GRAD_ACCUM, LR, OBJECTIVES, SEED, dump_json,
                          file_sha256, git_head)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V5A = '/root/data/experiments/v3a5_g64_verifier'
ROOT = '/root/data/experiments/v3a6_decision_gate'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--v5a_root', default=V5A)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    a = ap.parse_args(_CLI)

    for d in ('init', 'diagnostics', 'logs', 'A0_qstar/checkpoints',
              'A1_decision_mse/checkpoints'):
        os.makedirs(os.path.join(a.root, d), exist_ok=True)

    v5a = json.load(open(os.path.join(a.v5a_root, 'artifact_lock.json')))
    thr = float(v5a['energy_threshold'])
    proposal_path = os.path.join(a.src_root, R1_CK)
    split_path = os.path.join(a.v4_root, 'splits', 'split.json')
    mmap_tr = os.path.join(a.v4_root, 'mappings', 'mismatch_train_575.json')
    mmap_dv = os.path.join(a.v4_root, 'mappings', 'mismatch_dev_64.json')

    init = build_v3a6_shared_init(seed=SEED, bottleneck=BOTTLENECK)
    init_path = os.path.join(a.root, 'init', 'shared_init_s42.pt')
    torch.save(init, init_path)
    # verify bit-equal across arms
    if not bit_equal(init['A0_qstar'], init['A1_decision_mse']):
        raise SystemExit('shared init arms not bit-equal')
    init_sha = state_dict_sha(init['A0_qstar'])

    # step0 q equality smoke
    geom = prepare_geometry(target_geometry(40, 60, 'g64'), 'cpu')
    m0 = build_v3a6_model('A0_qstar'); m0.load_state_dict(init['A0_qstar'])
    m1 = build_v3a6_model('A1_decision_mse'); m1.load_state_dict(init['A1_decision_mse'])
    m0.eval(); m1.eval()
    x = torch.zeros(1, 3, 40, 60)
    with torch.no_grad():
        dq = float((m0(x, x, x, geom=geom) - m1(x, x, x, geom=geom)).abs().max())
    if dq > 1e-6:
        raise SystemExit('step0 q mismatch %.3e' % dq)

    lock = dict(
        stage='V3-A.6',
        root=a.root,
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
        energy_threshold=thr,
        energy_stats_sha256=v5a.get('energy_stats_sha256'),
        reference_variant=v5a.get('reference_variant', 'nanobanana_ref_v2'),
        geometry='g64',
        architecture='V3A5D2Verifier.A1_multiscale',
        bottleneck=BOTTLENECK,
        init_path=init_path,
        init_sha=init_sha,
        init_sha_A0=state_dict_sha(init['A0_qstar']),
        init_sha_A1=state_dict_sha(init['A1_decision_mse']),
        arms=list(ARMS),
        objectives=dict(OBJECTIVES),
        updates=DEFAULT_UPDATES,
        grad_accum=GRAD_ACCUM,
        lr=LR,
        weight_decay=0.0,
        seed=SEED,
        pair_schedule_seed=SEED,
        states=list(v5a.get('states', 'correct+true_dark_g0.5+mismatch').split('+'))
        if isinstance(v5a.get('states'), str) else v5a.get('states'),
        mask_mode='g64_proposal_energy_expand',
        checkpoint_steps=list(CKPT_STEPS),
        official_test_allowed=False,
        note='A0 q*-regression vs A1 output-MSE-only; sole variable = loss',
    )
    dump_json(os.path.join(a.root, 'artifact_lock.json'), lock)
    print('V3-A.6 setup OK')
    print('  root=%s' % a.root)
    print('  init_sha=%s  step0_dq=%.3e  thr=%.6e'
          % (init_sha[:16], dq, thr))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
