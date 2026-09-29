#!/usr/bin/env python
"""V3-A.5D0 evidence audit: train575+dev64 x 3 states. Zero training.

Writes under --root:
  evidence/{correlations,polarized,deciles,per_image,redundancy}.json
  diagnostics/d0_verdict.json
Does NOT train D1/D2/D3.
"""

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

from model.V3A5DEvidenceProbe import V3A5DEvidenceProbe                 # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import (load_proposal, load_rows, make_dataset,      # noqa: E402
                           sample_tensors)
from v3a5d_runtime import (ANALYSIS_CHANNELS, STATES,                   # noqa: E402
                           action_optimal_target, block_energy,
                           decile_analysis, decile_monotonic_score,
                           energy_mask, evidence_schema_sha,
                           pearson_corr, pool_all_evidence_to_g64,
                           polarized_labels, polarized_score,
                           pr_auc_binary, prepare_geometry,
                           roc_auc_binary, spearman_corr,
                           target_geometry, verdict_d0)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
ROOT = '/root/data/experiments/v3a5d_evidence'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
REF_H, REF_W = 400, 600
CHANNEL_NAMES = [c[0] for c in ANALYSIS_CHANNELS]
CHANNEL_DIR = {c[0]: c[3] for c in ANALYSIS_CHANNELS}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k) if isinstance(k, tuple) else k: _jsonable(v)
                for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(_jsonable(obj), open(path, 'w', encoding='utf-8'),
              indent=2, sort_keys=True)


def collect_split(probe, ds, rows, id_to_i, geom, thr, device, split,
                  check_equiv_first=True, limit=0):
    """Accumulate per-(state,channel) flat arrays + per-image means."""
    # store lists of 1d numpy arrays then concatenate
    blocks = {st: {ch: [] for ch in CHANNEL_NAMES} for st in STATES}
    q_blocks = {st: [] for st in STATES}
    mask_blocks = {st: [] for st in STATES}
    per_image = []

    n = len(rows) if not limit else min(limit, len(rows))
    t0 = time.time()
    equiv_done = False
    with torch.no_grad():
        for i in range(n):
            name = rows[i][0]
            local = id_to_i[name]
            for state in STATES:
                t = sample_tensors(ds, local, state, device)
                check = check_equiv_first and (not equiv_done)
                sr, aux, evidence = probe(t['Y0'], t['R'], check_equiv=check)
                if check:
                    equiv_done = True
                    print('[%s] equiv ok T_max=%.3e gate_max=%.3e'
                          % (split, aux['equiv']['T_max_abs'],
                             aux['equiv']['gate_max_abs']))
                D = aux['D']
                tgt = action_optimal_target(t['Y0'], t['H'], D, geom)
                mask = energy_mask(block_energy(D, geom), thr)
                pooled = pool_all_evidence_to_g64(evidence, geom)

                q = tgt['q_grid'].detach().float().cpu().numpy().reshape(-1)
                m = mask.detach().cpu().numpy().reshape(-1).astype(bool)
                q_blocks[state].append(q)
                mask_blocks[state].append(m)

                img_row = dict(split=split, name=name, state=state,
                               mean_q=float(q.mean()),
                               mean_q_masked=float(q[m].mean()) if m.any() else float('nan'))
                for ch in CHANNEL_NAMES:
                    arr = pooled[ch].detach().float().cpu().numpy().reshape(-1)
                    blocks[state][ch].append(arr)
                    img_row['mean_%s' % ch] = float(arr.mean())
                per_image.append(img_row)

            if (i + 1) % 25 == 0 or (i + 1) == n:
                print('  [%s] %d/%d images (%.0fs)'
                      % (split, i + 1, n, time.time() - t0))

    out_blocks = {}
    for st in STATES:
        out_blocks[st] = dict(
            q=np.concatenate(q_blocks[st]),
            mask=np.concatenate(mask_blocks[st]),
            ev={ch: np.concatenate(blocks[st][ch]) for ch in CHANNEL_NAMES},
        )
    return out_blocks, per_image


def stats_for_split_state(q, mask, ev_dict):
    """Per-channel pearson/spearman (masked), polarized AUC, deciles."""
    rows = {}
    for ch in CHANNEL_NAMES:
        ev = ev_dict[ch]
        direction = CHANNEL_DIR[ch]
        pear = pearson_corr(ev, q, mask)
        spear = spearman_corr(ev, q, mask)
        keep, labels = polarized_labels(q)
        # polarized on ALL blocks (not energy mask) — q* polarity is the label
        keep = keep & np.isfinite(ev)
        score = polarized_score(ev, direction if direction != 0 else +1)
        # for direction==0 still report AUC on raw values (no flip cherry-pick)
        if direction == 0:
            score = ev
        roc = roc_auc_binary(score[keep], labels[keep]) if keep.any() else float('nan')
        pr = pr_auc_binary(score[keep], labels[keep]) if keep.any() else float('nan')
        dec = decile_analysis(ev[mask], q[mask])
        rows[ch] = dict(
            pearson=pear, spearman=spear,
            roc_auc=roc, pr_auc=pr,
            n_blocks=int(mask.sum()),
            n_polarized=int(keep.sum()),
            decile=dec,
            decile_dir=decile_monotonic_score(dec),
            direction=direction,
        )
    return rows


def image_level_stats(per_image, split, state):
    rows = [r for r in per_image if r['split'] == split and r['state'] == state]
    if len(rows) < 3:
        return {}
    q = np.array([r['mean_q'] for r in rows], dtype=np.float64)
    out = {}
    for ch in CHANNEL_NAMES:
        ev = np.array([r['mean_%s' % ch] for r in rows], dtype=np.float64)
        out[ch] = dict(
            pearson=pearson_corr(ev, q),
            spearman=spearman_corr(ev, q),
            n=len(rows),
        )
    return out


def redundancy_matrix(ev_dict, mask):
    """Spearman evidence x evidence on masked blocks."""
    mat = {a: {} for a in CHANNEL_NAMES}
    for i, a in enumerate(CHANNEL_NAMES):
        for b in CHANNEL_NAMES[i:]:
            rho = spearman_corr(ev_dict[a], ev_dict[b], mask)
            mat[a][b] = rho
            mat[b][a] = rho
    return mat


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0,
                    help='smoke: first N images per split')
    a = ap.parse_args(_CLI)

    lock_path = os.path.join(a.root, 'artifact_lock.json')
    if not os.path.isfile(lock_path):
        raise SystemExit('run setup_v3a5d.py first (%s missing)' % lock_path)
    lock = json.load(open(lock_path, encoding='utf-8'))
    if lock.get('evidence_schema_sha256') != evidence_schema_sha():
        raise SystemExit('evidence schema SHA drifted vs lock')
    thr = float(lock['energy_threshold'])

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       os.path.join(a.v4_root, 'splits', 'split.json'))
    mmap_train = json.load(open(os.path.join(
        a.v4_root, 'mappings', 'mismatch_train_575.json'), encoding='utf-8'))
    mmap_dev = json.load(open(os.path.join(
        a.v4_root, 'mappings', 'mismatch_dev_64.json'), encoding='utf-8'))
    cache = os.path.join(a.src_root, a.cache_name, 'refiner_train')

    model = load_proposal(os.path.join(a.src_root, R1_CK), a.device)
    probe = V3A5DEvidenceProbe(model.proposal).to(a.device).eval()
    geom = prepare_geometry(target_geometry(REF_H, REF_W, 'g64'), a.device)

    all_blocks = {}
    all_per_image = []
    for split, rows, mmap in (
            ('train', splits['train'], mmap_train),
            ('dev', splits['dev'], mmap_dev)):
        ds = make_dataset(ns, rows, cache, mmap)
        # TrainSet indexes by position in pairs; id_to_i from row order
        id_to_i = {rows[i][0]: i for i in range(len(rows))}
        print('=== collecting %s (%d images) ===' % (split, len(rows)))
        blocks, per_img = collect_split(
            probe, ds, rows, id_to_i, geom, thr, a.device, split,
            check_equiv_first=True, limit=a.limit)
        all_blocks[split] = blocks
        all_per_image.extend(per_img)

    # ── correlations + polarized + deciles ────────────────────────────────
    correlations = {}
    polarized = {}
    deciles = {}
    per_channel_for_verdict = {ch: dict(spearman={}, roc_auc={},
                                        decile_dir={}) for ch in CHANNEL_NAMES}

    for split in ('train', 'dev'):
        for state in STATES:
            st = stats_for_split_state(
                all_blocks[split][state]['q'],
                all_blocks[split][state]['mask'],
                all_blocks[split][state]['ev'])
            key = '%s|%s' % (split, state)
            correlations[key] = {
                ch: dict(pearson=st[ch]['pearson'], spearman=st[ch]['spearman'],
                         n_blocks=st[ch]['n_blocks'])
                for ch in CHANNEL_NAMES
            }
            polarized[key] = {
                ch: dict(roc_auc=st[ch]['roc_auc'], pr_auc=st[ch]['pr_auc'],
                         n_polarized=st[ch]['n_polarized'],
                         direction=st[ch]['direction'])
                for ch in CHANNEL_NAMES
            }
            deciles[key] = {
                ch: dict(bins=st[ch]['decile'], dir=st[ch]['decile_dir'])
                for ch in CHANNEL_NAMES
            }
            for ch in CHANNEL_NAMES:
                per_channel_for_verdict[ch]['spearman'][(split, state)] = \
                    st[ch]['spearman']
                per_channel_for_verdict[ch]['roc_auc'][(split, state)] = \
                    st[ch]['roc_auc']

        # pooled-across-states decile dir for criterion C
        q_all = np.concatenate([all_blocks[split][s]['q'] for s in STATES])
        m_all = np.concatenate([all_blocks[split][s]['mask'] for s in STATES])
        deciles_all = {}
        for ch in CHANNEL_NAMES:
            ev_all = np.concatenate([all_blocks[split][s]['ev'][ch] for s in STATES])
            dec = decile_analysis(ev_all[m_all], q_all[m_all])
            ddir = decile_monotonic_score(dec)
            per_channel_for_verdict[ch]['decile_dir'][split] = ddir
            deciles_all[ch] = dict(bins=dec, dir=ddir)
        deciles['%s|all' % split] = deciles_all

    # ── image-level ───────────────────────────────────────────────────────
    image_stats = {}
    for split in ('train', 'dev'):
        for state in STATES:
            image_stats['%s|%s' % (split, state)] = image_level_stats(
                all_per_image, split, state)

    # ── redundancy (train correct, masked) ────────────────────────────────
    redundancy = redundancy_matrix(
        all_blocks['train']['correct']['ev'],
        all_blocks['train']['correct']['mask'])

    verdict = verdict_d0(per_channel_for_verdict)
    verdict['energy_threshold'] = thr
    verdict['limit'] = a.limit
    verdict['n_train'] = len(splits['train']) if not a.limit else min(a.limit, len(splits['train']))
    verdict['n_dev'] = len(splits['dev']) if not a.limit else min(a.limit, len(splits['dev']))

    ev_dir = os.path.join(a.root, 'evidence')
    dump(os.path.join(ev_dir, 'correlations.json'), correlations)
    dump(os.path.join(ev_dir, 'polarized.json'), polarized)
    dump(os.path.join(ev_dir, 'deciles.json'), deciles)
    dump(os.path.join(ev_dir, 'per_image_stats.json'), image_stats)
    dump(os.path.join(ev_dir, 'per_image_rows.json'), all_per_image)
    dump(os.path.join(ev_dir, 'redundancy.json'), redundancy)
    dump(os.path.join(a.root, 'diagnostics', 'd0_verdict.json'), verdict)

    print('=== D0 verdict: %s (next=%s) ===' % (verdict['verdict'], verdict['next_step']))
    for r in verdict['reasons'][:12]:
        print('  ', r)
    print('wrote evidence under %s' % ev_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
