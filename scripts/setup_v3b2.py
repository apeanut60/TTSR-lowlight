#!/usr/bin/env python
"""V3-B.2 setup: A0/A1 shared init + train-only evidence stats. No training."""

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

from local_refine_runtime import EVAL_QUERY_CHUNK                       # noqa: E402
from model.V3BEvidence import (EVIDENCE_NAMES, assert_shared_init,     # noqa: E402
                               attach_match_evidence, common_weight_sha,
                               copy_a0_to_a1, extract_match_maps)
from model.V3BResidualFusion import V3B0ResidualFusion, count_params    # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_proposal, load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import STATES, snapshot_, state_dict_sha              # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head               # noqa: E402
from v3b_runtime import DEFAULT_UPDATES, GRAD_ACCUM, LR, SEED, json_ready  # noqa: E402
from v3b2_runtime import ARM_A0, ARM_A1, ARMS, CKPT_STEPS               # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
B0 = '/root/data/experiments/v3b0_implicit_residual'
ROOT = '/root/data/experiments/v3b2_evidence_fusion'
RESERVOIR = 200000


def _reservoir_push(buf, rng, n_seen, values):
    """Push 1d numpy values into fixed-size reservoir (modified Alg R)."""
    for v in values:
        n_seen += 1
        if len(buf) < RESERVOIR:
            buf.append(float(v))
        else:
            j = int(rng.integers(0, n_seen))
            if j < RESERVOIR:
                buf[j] = float(v)
    return n_seen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--b0_root', default=B0)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--norm_limit', type=int, default=0,
                    help='limit train images for evidence stats (0=all)')
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

    # ── shared init ─────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    a0 = V3B0ResidualFusion(in_ch=96)
    a1 = V3B0ResidualFusion(in_ch=100)
    copy_a0_to_a1(a0, a1)
    f0 = torch.randn(1, 32, 8, 8)
    t = torch.randn(1, 32, 8, 8)
    e = torch.randn(1, 4, 8, 8)
    assert_shared_init(a0, a1, f0, t, e, (16, 16))
    if float(a0(f0, t, (16, 16)).abs().max()) > 1e-7:
        raise SystemExit('A0 step0 ΔY not zero')
    if float(a1(f0, t, (16, 16), E=e).abs().max()) > 1e-7:
        raise SystemExit('A1 step0 ΔY not zero')

    a0_path = os.path.join(a.root, 'init', 'A0_b0_replay_s42.pt')
    a1_path = os.path.join(a.root, 'init', 'A1_evidence_s42.pt')
    sd0, sd1 = snapshot_(a0), snapshot_(a1)
    torch.save(dict(model=sd0, arm=ARM_A0, in_ch=96), a0_path)
    torch.save(dict(model=sd1, arm=ARM_A1, in_ch=100), a1_path)
    a0_sha = state_dict_sha(sd0)
    a1_sha = state_dict_sha(sd1)
    cw_sha = common_weight_sha(a0)

    # ── train-only evidence stats ───────────────────────────────────────
    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = b0['reference_variant']
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       b0['split_json'])
    mmap = json.load(open(b0['mismatch_train']))
    train_rows = splits['train']
    if a.norm_limit:
        train_rows = train_rows[:a.norm_limit]
    ds = make_dataset(
        ns, train_rows,
        os.path.join(a.src_root, b0.get('cache_name', 'cache_y0_lolbase'),
                     'refiner_train'), mmap)

    wrapper = load_proposal(b0['proposal_ckpt'], a.device)
    core = wrapper.proposal if hasattr(wrapper, 'proposal') else wrapper
    core.match.chunk = EVAL_QUERY_CHUNK
    match_ev = attach_match_evidence(core).to(a.device)
    match_ev.chunk = EVAL_QUERY_CHUNK
    encoder = core.encoder

    n_ch = len(EVIDENCE_NAMES)
    acc_sum = np.zeros(n_ch, dtype=np.float64)
    acc_sq = np.zeros(n_ch, dtype=np.float64)
    acc_n = 0
    acc_min = np.full(n_ch, np.inf)
    acc_max = np.full(n_ch, -np.inf)
    reservoirs = [[] for _ in range(n_ch)]
    n_seen = [0] * n_ch
    rng = np.random.default_rng(SEED)

    print('=== train evidence stats n_img=%d ===' % len(train_rows), flush=True)
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(train_rows)):
            for state in STATES:
                sample = sample_tensors(ds, i, state, a.device)
                _, _, e_raw = extract_match_maps(
                    encoder, match_ev, sample['Y0'], sample['R'])
                flat = e_raw.reshape(n_ch, -1).float().cpu().numpy()
                acc_sum += flat.sum(axis=1)
                acc_sq += (flat ** 2).sum(axis=1)
                acc_n += flat.shape[1]
                acc_min = np.minimum(acc_min, flat.min(axis=1))
                acc_max = np.maximum(acc_max, flat.max(axis=1))
                # subsample ~1/32 pixels into reservoir for speed
                idx = rng.integers(0, flat.shape[1], size=max(1, flat.shape[1] // 32))
                for c in range(n_ch):
                    n_seen[c] = _reservoir_push(
                        reservoirs[c], rng, n_seen[c], flat[c, idx])
            if (i + 1) % 25 == 0 or i + 1 == len(train_rows):
                print('  %d/%d (%.0fs)' % (i + 1, len(train_rows), time.time() - t0),
                      flush=True)

    mean = acc_sum / max(acc_n, 1)
    var = np.maximum(acc_sq / max(acc_n, 1) - mean ** 2, 1e-12)
    std = np.sqrt(var)
    stats = {}
    for c, name in enumerate(EVIDENCE_NAMES):
        arr = np.asarray(reservoirs[c], dtype=np.float64)
        p01 = float(np.percentile(arr, 1)) if arr.size else float('nan')
        p99 = float(np.percentile(arr, 99)) if arr.size else float('nan')
        stats[name] = dict(
            mean=float(mean[c]), std=float(std[c]),
            min=float(acc_min[c]), max=float(acc_max[c]),
            p01=p01, p99=p99,
        )
    evidence_stats = dict(
        names=list(EVIDENCE_NAMES),
        per_channel=stats,
        normalization='zscore_clip5',
        clip=5.0,
        eps=1e-6,
        n_pixels=int(acc_n),
        n_images=len(train_rows),
        n_states=len(STATES),
        resolution='H/2',
        note='confidence_entropy := matcher entropy/log(K); no polarity flip',
        seed=SEED,
    )
    stats_path = os.path.join(a.root, 'evidence_stats.json')
    dump_json(stats_path, json_ready(evidence_stats))
    stats_sha = file_sha256(stats_path)

    lock = dict(
        stage='V3-B.2',
        root=a.root,
        prior='V3B0_CASE_A_STRONG_GO_B1_C_UNSAFE',
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
        architecture='V3B0ResidualFusion',
        arms=list(ARMS),
        arm=None,
        n_params={ARM_A0: count_params(a0), ARM_A1: count_params(a1)},
        init_paths={ARM_A0: a0_path, ARM_A1: a1_path},
        a0_init_sha=a0_sha,
        a1_init_sha=a1_sha,
        init_shas={ARM_A0: a0_sha, ARM_A1: a1_sha},
        common_weight_sha=cw_sha,
        evidence_names=list(EVIDENCE_NAMES),
        evidence_stats_path=stats_path,
        evidence_stats_sha256=stats_sha,
        normalization_method='zscore_clip5',
        evidence_resolution='H/2',
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
        note='A0=B0 replay; A1=+4 evidence channels; sole variable=E',
    )
    dump_json(lock_path, json_ready(lock))
    print('V3-B.2 setup OK')
    print('  A0 sha=%s n=%d' % (a0_sha[:16], count_params(a0)))
    print('  A1 sha=%s n=%d' % (a1_sha[:16], count_params(a1)))
    print('  evidence_stats_sha=%s' % stats_sha[:16])
    for name in EVIDENCE_NAMES:
        st = stats[name]
        print('  %s mean=%.4g std=%.4g' % (name, st['mean'], st['std']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
