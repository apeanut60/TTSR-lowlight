#!/usr/bin/env python
"""V3-A.4 §2.3/§5.4/§10/§22: build the frozen protocol artifacts.

Produces, once and then read-only:
  splits/split.json               575/64 (copied from V3-A.3, not resampled)
  mappings/mismatch_train_575.json   donors strictly inside train575
  mappings/mismatch_dev_64.json      donors strictly inside dev64
  action_stats/energy.json           eps_energy from train575 correct refs
  action_stats/action_norm.json      per-channel RMS of D4 from train575 ONLY
  artifact_lock.json                 every SHA the results bind to
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from dataset.lolv2real_v3a import TrainSet, pairs_from_manifest   # noqa: E402
from local_refine_runtime import sha256                           # noqa: E402
from model.V3A4Verifier import V3A4Refiner                        # noqa: E402
from option import parser as option_parser                        # noqa: E402
from v3a2_runtime import action_optimal_gate, energy_threshold    # noqa: E402
from v3a4_runtime import (build_mismatch_map, compute_action_rms,  # noqa: E402
                          load_r1_proposal_strict)

R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
SEED = 20260927


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_root', default='/root/data/experiments/v3a1_lolv2real')
    ap.add_argument('--v3a3_root', default='/root/data/experiments/v3a3_lolv2real')
    ap.add_argument('--root', default='/root/data/experiments/v3a4_lolv2real')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(_CLI)
    dev = a.device
    for sub in ('splits', 'mappings', 'action_stats', 'init'):
        os.makedirs(os.path.join(a.root, sub), exist_ok=True)

    # ── split: reuse V3-A.3's, do not resample ────────────────────────────
    src_split = os.path.join(a.v3a3_root, 'splits', 'split.json')
    split = json.load(open(src_split, encoding='utf-8'))
    if split['seed'] != SEED or len(split['train']) != 575 or len(split['dev']) != 64:
        raise SystemExit('unexpected source split')
    outp = os.path.join(a.root, 'splits', 'split.json')
    json.dump(split, open(outp, 'w', encoding='utf-8'), indent=2, sort_keys=True)
    print('split: train %d dev %d (copied from V3-A.3, seed %d)'
          % (len(split['train']), len(split['dev']), SEED))

    # ── split-isolated mismatch maps (§2.3) ───────────────────────────────
    maps = {}
    for tag, ids in (('train_575', split['train']), ('dev_64', split['dev'])):
        m = build_mismatch_map(ids, seed=SEED)
        p = os.path.join(a.root, 'mappings', 'mismatch_%s.json' % tag)
        json.dump(m, open(p, 'w', encoding='utf-8'), indent=2, sort_keys=True)
        other = set(split['dev'] if tag == 'train_575' else split['train'])
        if (set(m) | set(m.values())) & other:
            raise SystemExit('%s leaks across the split' % tag)
        maps[tag] = dict(path=p, sha256=sha256(p), n=len(m))
        print('%s: %d pairs, no cross-split donor' % (tag, len(m)))

    # ── train-only action statistics (§5.4, §10) ──────────────────────────
    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    all_pairs = pairs_from_manifest(os.path.join(a.src_root, 'manifests',
                                                 'refiner_train.csv'))
    tr_ids = set(split['train'])
    tr_pairs = sorted([p for p in all_pairs if p[0] in tr_ids], key=lambda p: p[0])
    cache = os.path.join(a.src_root, a.cache_name, 'refiner_train')
    ds = TrainSet(ns, crop_size=0, pairs=tr_pairs, y0_cache=cache, split='Train')

    m = V3A4Refiner('none').to(dev).eval()
    load_r1_proposal_strict(m, os.path.join(a.src_root, R1_CK), dev)
    rms = compute_action_rms(m, tr_pairs, ds, dev)
    p = os.path.join(a.root, 'action_stats', 'action_norm.json')
    json.dump(dict(method='per-channel RMS of D4 on verifier_train575 correct refs',
                   rms_D=rms, eps=1e-6, n=len(tr_pairs),
                   split_sha256=sha256(outp),
                   proposal_sha256=sha256(os.path.join(a.src_root, R1_CK)),
                   manifest_sha256=sha256(os.path.join(a.src_root, 'manifests',
                                                       'refiner_train.csv'))),
              open(p, 'w', encoding='utf-8'), indent=2, sort_keys=True)
    print('action RMS (per RGB channel): %s' % ['%.5f' % x for x in rms])

    energies = []
    with torch.no_grad():
        for i in range(len(tr_pairs)):
            _n, _lr, hr, ref, y0, _mm = ds._load(i)
            _sr, aux = m.proposal(y0[None].to(dev), ref[None].to(dev))
            _q, e = action_optimal_gate(y0[None].to(dev), hr[None].to(dev),
                                        aux['gate'] * aux['delta'])
            energies.append(e.cpu())
    eps = energy_threshold(energies)
    ep = os.path.join(a.root, 'action_stats', 'energy.json')
    json.dump(dict(eps_energy=eps, method='p10 of per-pixel proposal energy',
                   split='verifier_train575', n=len(tr_pairs),
                   split_sha256=sha256(outp)),
              open(ep, 'w', encoding='utf-8'), indent=2, sort_keys=True)
    print('eps_energy = %.6e' % eps)
    if eps <= 0:
        raise SystemExit('degenerate energy threshold')

    # ── artifact lock (§22) ───────────────────────────────────────────────
    lock = dict(
        repo_commit=__import__('subprocess').check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).decode().strip(),
        proposal_sha256=sha256(os.path.join(a.src_root, R1_CK)),
        base_checkpoint_sha256=json.load(open(os.path.join(
            cache, 'metadata.json'), encoding='utf-8'))['base_checkpoint_sha256'],
        cache_metadata_sha256=sha256(os.path.join(cache, 'metadata.json')),
        manifest_sha256=sha256(os.path.join(a.src_root, 'manifests',
                                            'refiner_train.csv')),
        split_sha256=sha256(outp),
        mismatch_train_sha256=maps['train_575']['sha256'],
        mismatch_dev_sha256=maps['dev_64']['sha256'],
        energy_stats_sha256=sha256(ep),
        action_norm_sha256=sha256(p),
        seed=42, states='correct+true_dark_g0.5+mismatch',
        loss='1.0*masked_smoothl1 + 0.1*L1', budget=3000,
        primary_checkpoint=3000)
    json.dump(lock, open(os.path.join(a.root, 'artifact_lock.json'), 'w',
                         encoding='utf-8'), indent=2, sort_keys=True)
    print('-> %s' % a.root)


if __name__ == '__main__':
    main()
