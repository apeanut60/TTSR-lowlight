#!/usr/bin/env python
"""V3-A.5D1 setup: reuse V3A5C tiny16 + cache evidence_g64 + train575 norm.

Does NOT train.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.V3A5DEvidenceProbe import V3A5DEvidenceProbe                 # noqa: E402
from model.V3A5DVerifier import EVIDENCE_NAMES, build_d1_shared_init     # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a42_runtime import _sha256                                       # noqa: E402
from v3a5_pipeline import (load_proposal, load_rows, make_dataset,      # noqa: E402
                           sample_tensors)
from v3a5_runtime import prepare_geometry, state_dict_sha, target_geometry  # noqa: E402
from v3a5d1_runtime import (TINY_ROOT_SRC, stack_narrow_evidence_g64)   # noqa: E402
from v3a5d_runtime import evidence_schema_sha                           # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
ROOT = '/root/data/experiments/v3a5d1_tiny_evidence'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
REF_H, REF_W = 400, 600


def dump(path, obj):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    json.dump(obj, open(path, 'w', encoding='utf-8'), indent=2, sort_keys=True)


def git_head():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd='/root/projects/TTSR-lowlight',
            text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--tiny_src', default=TINY_ROOT_SRC)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--norm_limit', type=int, default=0,
                    help='debug: first N train575 images for norm stats')
    a = ap.parse_args(_CLI)

    tiny_dst = os.path.join(a.root, 'tiny')
    os.makedirs(tiny_dst, exist_ok=True)
    os.makedirs(os.path.join(a.root, 'diagnostics'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)

    # ── copy locked tiny16 artifacts from V3A5C ──────────────────────────
    src_tiny = os.path.join(a.tiny_src, 'tiny')
    for name in ('tiny16_ids.json', 'tiny16_mismatch_map.json',
                 'oracle_stats.json', 'target_stats.json', 'pairs.json'):
        src = os.path.join(src_tiny, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(tiny_dst, name))
    ids = json.load(open(os.path.join(tiny_dst, 'tiny16_ids.json')))['ids']
    mmap = json.load(open(os.path.join(tiny_dst, 'tiny16_mismatch_map.json')))
    base_cache = torch.load(os.path.join(src_tiny, 'cache.pt'), map_location='cpu')
    v5c_lock = json.load(open(os.path.join(a.tiny_src, 'artifact_lock.json')))
    thr = float(v5c_lock['energy_threshold'])

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       os.path.join(a.v4_root, 'splits', 'split.json'))
    mmap_train = json.load(open(os.path.join(
        a.v4_root, 'mappings', 'mismatch_train_575.json'), encoding='utf-8'))

    model = load_proposal(os.path.join(a.src_root, R1_CK), a.device)
    probe = V3A5DEvidenceProbe(model.proposal).to(a.device).eval()
    geom = prepare_geometry(target_geometry(REF_H, REF_W, 'g64'), a.device)

    # ── train575 evidence norm (sim_max, f0_minus_t z-scored; gate raw) ──
    print('=== pooling train575 evidence for norm stats ===')
    ds_full = make_dataset(
        ns, splits['train'],
        os.path.join(a.src_root, a.cache_name, 'refiner_train'), mmap_train)
    n_train = len(splits['train']) if not a.norm_limit else min(a.norm_limit, len(splits['train']))
    acc_sum = np.zeros(3, dtype=np.float64)
    acc_sq = np.zeros(3, dtype=np.float64)
    acc_n = 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_train):
            for state in ('correct', 'true_dark_g0.5', 'mismatch'):
                t = sample_tensors(ds_full, i, state, a.device)
                _, _, evidence = probe(t['Y0'], t['R'], check_equiv=(i == 0 and state == 'correct'))
                if i == 0 and state == 'correct':
                    print('  equiv T_max=0 check via probe on first sample')
                ev = stack_narrow_evidence_g64(evidence, geom)  # [1,3,64,64]
                flat = ev.reshape(3, -1).float().cpu().numpy()
                acc_sum += flat.sum(axis=1)
                acc_sq += (flat ** 2).sum(axis=1)
                acc_n += flat.shape[1]
            if (i + 1) % 50 == 0 or (i + 1) == n_train:
                print('  norm pooled %d/%d (%.0fs)' % (i + 1, n_train, time.time() - t0))
    mean = acc_sum / max(acc_n, 1)
    var = acc_sq / max(acc_n, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    # gate_v2 kept raw: store identity mean/std but zscore_mask disables it
    norm = dict(
        names=list(EVIDENCE_NAMES),
        mean=mean.tolist(),
        std=std.tolist(),
        zscore_mask=[1.0, 1.0, 0.0],
        n_blocks=int(acc_n),
        n_images=n_train,
        note='z-score sim_max + f0_minus_t; gate_v2 left in [0,1]',
    )
    dump(os.path.join(a.root, 'tiny', 'evidence_norm.json'), norm)
    print('  mean=%s std=%s' % (mean.round(4), std.round(4)))

    # ── attach evidence_g64 to each tiny cache entry ─────────────────────
    print('=== caching evidence for tiny16 x 3 states ===')
    rows_by_name = {r[0]: r for r in splits['train']}
    rows = [rows_by_name[i] for i in ids]
    ds = make_dataset(ns, rows,
                      os.path.join(a.src_root, a.cache_name, 'refiner_train'), mmap)
    name_to_i = {name: i for i, name in enumerate(ids)}
    entries = dict(base_cache['entries'])
    with torch.no_grad():
        for p in base_cache['pairs']:
            t = sample_tensors(ds, name_to_i[p['name']], p['state'], a.device)
            _, _, evidence = probe(t['Y0'], t['R'], check_equiv=False)
            ev = stack_narrow_evidence_g64(evidence, geom).detach().cpu().half()
            key = p['key']
            ent = dict(entries[key])
            ent['evidence_g64'] = ev  # [1,3,64,64]
            ent['evidence_names'] = list(EVIDENCE_NAMES)
            entries[key] = ent
    cache = dict(base_cache)
    cache['entries'] = entries
    cache['evidence_names'] = list(EVIDENCE_NAMES)
    cache['evidence_norm'] = norm
    torch.save(cache, os.path.join(tiny_dst, 'cache.pt'))
    print('  wrote tiny/cache.pt with evidence_g64')

    # ── shared init ──────────────────────────────────────────────────────
    init = build_d1_shared_init(seed=42, ev_mean=mean, ev_std=std)
    torch.save(init, os.path.join(tiny_dst, 'shared_init_s42.pt'))
    init_sha = state_dict_sha(init['A0_control'])
    print('  shared init sha=%s' % init_sha[:16])

    lock = dict(
        stage='V3-A.5D1',
        root=a.root,
        repo_commit=git_head(),
        tiny_src=a.tiny_src,
        tiny16_ids=ids,
        tiny16_ids_sha256=_sha256(os.path.join(tiny_dst, 'tiny16_ids.json')),
        v3a5c_lock_sha256=_sha256(os.path.join(a.tiny_src, 'artifact_lock.json')),
        energy_threshold=thr,
        energy_threshold_source='v3a5c_tiny_overfit/artifact_lock.json',
        proposal_sha256=v5c_lock.get('proposal_sha256'),
        evidence_names=list(EVIDENCE_NAMES),
        evidence_norm=norm,
        evidence_schema_sha256=evidence_schema_sha(),
        init_sha=init_sha,
        out_weight=0.0,
        arms=['A0_control', 'A1_evidence'],
        updates=20000,
        seed=42,
        note='narrow evidence tiny overfit; no MultiScale; no full train',
    )
    dump(os.path.join(a.root, 'artifact_lock.json'), lock)
    print('V3-A.5D1 lock -> %s' % os.path.join(a.root, 'artifact_lock.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
