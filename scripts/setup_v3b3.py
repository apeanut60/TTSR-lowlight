#!/usr/bin/env python
"""V3-B.3 setup: A0/A1 shared init + train-only RGB global stats. No training."""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.V3BGlobalStats import (A0_IN, GLOBAL_STAT_NAMES, V3B3A1,  # noqa: E402
                                  assert_shared_init, copy_a0_to_a1_head,
                                  rgb01_mean_std)
from model.V3BResidualFusion import V3B0ResidualFusion, count_params    # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors       # noqa: E402
from v3a5_runtime import STATES, snapshot_, state_dict_sha              # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head               # noqa: E402
from v3b_runtime import DEFAULT_UPDATES, GRAD_ACCUM, LR, SEED, json_ready  # noqa: E402
from v3b3_runtime import ARM_A0, ARM_A1, ARMS, CKPT_STEPS               # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
B0 = '/root/data/experiments/v3b0_implicit_residual'
ROOT = '/root/data/experiments/v3b3_global_stats'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--b0_root', default=B0)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--norm_limit', type=int, default=0)
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
    b0_safe = json.load(open(os.path.join(
        a.b0_root, 'diagnostics', 'eval_020000', 'safety.json')))

    torch.manual_seed(SEED)
    a0 = V3B0ResidualFusion(in_ch=A0_IN)
    a1 = V3B3A1()
    copy_a0_to_a1_head(a0, a1.head)
    f0 = torch.randn(1, 32, 8, 8)
    t = torch.randn(1, 32, 8, 8)
    s = torch.randn(1, 6)
    assert_shared_init(a0, a1, f0, t, s, (16, 16))
    if float(a0(f0, t, (16, 16)).abs().max()) > 1e-7:
        raise SystemExit('A0 step0 ΔY not zero')

    a0_path = os.path.join(a.root, 'init', 'A0_b0_replay_s42.pt')
    a1_path = os.path.join(a.root, 'init', 'A1_global_stats_s42.pt')
    sd0, sd1 = snapshot_(a0), snapshot_(a1)
    torch.save(dict(model=sd0, arm=ARM_A0, in_ch=A0_IN), a0_path)
    torch.save(dict(model=sd1, arm=ARM_A1, in_ch=160), a1_path)
    a0_sha = state_dict_sha(sd0)
    a1_sha = state_dict_sha(sd1)
    cw_sha = a0_sha

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = b0['reference_variant']
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       b0['split_json'])
    mmap = json.load(open(b0['mismatch_train']))
    train_rows = splits['train'][:a.norm_limit] if a.norm_limit else splits['train']
    ds = make_dataset(
        ns, train_rows,
        os.path.join(a.src_root, b0.get('cache_name', 'cache_y0_lolbase'),
                     'refiner_train'), mmap)

    vecs = []
    print('=== train global RGB stats n_img=%d ===' % len(train_rows), flush=True)
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(train_rows)):
            for state in STATES:
                sample = sample_tensors(ds, i, state, a.device)
                s = rgb01_mean_std(sample['R']).reshape(-1).cpu().numpy()
                vecs.append(s)
            if (i + 1) % 50 == 0 or i + 1 == len(train_rows):
                print('  %d/%d (%.0fs)' % (i + 1, len(train_rows), time.time() - t0),
                      flush=True)
    arr = np.stack(vecs, axis=0)  # [1725, 6]
    stats = {}
    for c, name in enumerate(GLOBAL_STAT_NAMES):
        col = arr[:, c]
        stats[name] = dict(
            mean=float(col.mean()), std=float(col.std(ddof=0)),
            min=float(col.min()), max=float(col.max()),
            p01=float(np.percentile(col, 1)),
            p99=float(np.percentile(col, 99)),
        )
    payload = dict(
        names=list(GLOBAL_STAT_NAMES),
        per_channel=stats,
        n_vectors=int(arr.shape[0]),
        n_images=len(train_rows),
        n_states=len(STATES),
        stat_source_space='RGB01',
        std_unbiased=False,
        normalization='zscore_clip5',
        clip=5.0,
        eps=1e-6,
        seed=SEED,
        note='R in [-1,1] → R01=(R+1)/2; mean/std over HW; unbiased=False',
    )
    stats_path = os.path.join(a.root, 'global_stats.json')
    dump_json(stats_path, json_ready(payload))
    stats_sha = file_sha256(stats_path)

    lock = dict(
        stage='V3-B.3',
        root=a.root,
        prior='V3B2_CASE_C_NULL',
        b0_root=a.b0_root,
        repo_commit=git_head(),
        proposal_ckpt=b0['proposal_ckpt'],
        proposal_sha256=b0['proposal_sha256'],
        cache_name=b0.get('cache_name', 'cache_y0_lolbase'),
        base_cache_metadata_sha256=b0['base_cache_metadata_sha256'],
        split_json=b0['split_json'],
        split_sha256=b0['split_sha256'],
        mismatch_train=b0['mismatch_train'],
        mismatch_train_sha256=b0['mismatch_train_sha256'],
        mismatch_dev=b0['mismatch_dev'],
        mismatch_dev_sha256=b0['mismatch_dev_sha256'],
        reference_variant=b0['reference_variant'],
        architecture='V3B0ResidualFusion+GlobalStatMLP',
        arms=list(ARMS),
        n_params={ARM_A0: count_params(a0), ARM_A1: count_params(a1)},
        init_paths={ARM_A0: a0_path, ARM_A1: a1_path},
        a0_init_sha=a0_sha,
        a1_init_sha=a1_sha,
        init_shas={ARM_A0: a0_sha, ARM_A1: a1_sha},
        common_weight_sha=cw_sha,
        global_stat_names=list(GLOBAL_STAT_NAMES),
        stat_source_space='RGB01',
        std_unbiased=False,
        global_stats_path=stats_path,
        global_stats_sha256=stats_sha,
        normalization_method='zscore_clip5',
        mlp_architecture='Linear6-32-GELU-Linear32-64',
        broadcast_channels=64,
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
        frozen_b0_safety=b0_safe,
        note='A0=B0 replay; A1=+RGB mean/std global prior; sole variable=G',
    )
    dump_json(lock_path, json_ready(lock))
    print('V3-B.3 setup OK')
    print('  A0 sha=%s n=%d' % (a0_sha[:16], count_params(a0)))
    print('  A1 sha=%s n=%d' % (a1_sha[:16], count_params(a1)))
    print('  global_stats_sha=%s n_vec=%d' % (stats_sha[:16], arr.shape[0]))
    for name in GLOBAL_STAT_NAMES:
        st = stats[name]
        print('  %s mean=%.4g std=%.4g' % (name, st['mean'], st['std']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
