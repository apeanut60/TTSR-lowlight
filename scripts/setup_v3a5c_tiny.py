#!/usr/bin/env python
"""V3-A.5C setup: tiny16 subset + precomputed (D, q*, mask) cache + lock.

Does NOT train. Inherits V3-A.5A energy threshold / proposal / split SHAs.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import metrics                          # noqa: E402
from option import parser as option_parser                        # noqa: E402
from v3a41_runtime import global_action_optimal_gate              # noqa: E402
from v3a5_pipeline import (correction, load_proposal, load_rows,  # noqa: E402
                           make_dataset, sample_tensors)
from v3a5_runtime import (STATES, action_optimal_target,          # noqa: E402
                          block_energy, energy_mask, expand_gate,
                          prepare_geometry, target_geometry)
from v3a5c_runtime import (N_MICRO, N_TINY, TINY_SEED,            # noqa: E402
                           choose_tiny_ids, dump_json, file_sha256,
                           slice_mismatch_map)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
V5A = '/root/data/experiments/v3a5_g64_verifier'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'


def git_meta():
    try:
        head = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd='/root/projects/TTSR-lowlight',
            text=True).strip()
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd='/root/projects/TTSR-lowlight',
            text=True).strip().splitlines()
    except Exception:
        head, dirty = None, []
    return head, dirty


def accept_subset(oracle_by_state):
    """Plan §24: mean G64 headroom > 0; >=2 states with G64-G1 >= 0.05."""
    heads = []
    regional = 0
    for st, o in oracle_by_state.items():
        h = o['psnr_g64'] - o['psnr_r1']
        heads.append(h)
        if o['psnr_g64'] - o['psnr_g1'] >= 0.05:
            regional += 1
    return dict(ok=bool(float(np.mean(heads)) > 0 and regional >= 2),
                mean_g64_headroom=float(np.mean(heads)),
                n_states_regional=regional, per_state_headroom=dict(
                    zip(oracle_by_state.keys(), [float(x) for x in heads])))


def build_for_ids(ids, rows_by_name, mmap_full, ns, cache_y0, proposal,
                  geom_g64, geom_h4, thr, device):
    mmap = slice_mismatch_map(mmap_full, ids)
    # dataset over only these rows, but mismatch donors may be outside the 16;
    # TrainSet looks up donors by path via mismatch_map values as names in
    # ref_map built from pairs — so we must include donor rows too OR rely on
    # mismatch_map paths. Looking at TrainSet: it uses mismatch_map[name] as
    # a key into ref_map built from `pairs`. So donors must be in `pairs`.
    donor_ids = sorted(set(mmap.values()))
    # ref_map is built from the whole ref_dir listing, so donors need not be in
    # `pairs`; keep the tiny set only.
    all_ids = list(ids)
    rows = [rows_by_name[i] for i in all_ids]
    id_to_local = {name: i for i, name in enumerate(all_ids)}
    ds = make_dataset(ns, rows, cache_y0, mmap)

    pairs = []
    cache_entries = {}
    target_stats = {st: [] for st in STATES}
    oracle = {st: dict(psnr_base=[], psnr_r1=[], psnr_g1=[],
                       psnr_g64=[], psnr_h4=[]) for st in STATES}

    for name in ids:
        i = id_to_local[name]
        for state in STATES:
            t = sample_tensors(ds, i, state, device)
            with torch.no_grad():
                D, _ = correction(proposal.proposal, t['Y0'], t['R'])
                tgt = action_optimal_target(t['Y0'], t['H'], D, geom_g64)
                mask = energy_mask(block_energy(D, geom_g64), thr)
                q_g1, _, _ = global_action_optimal_gate(t['Y0'], t['H'], D)
                tgt_h4 = action_optimal_target(t['Y0'], t['H'], D, geom_h4)
                y0, hrd = t['Y0'], t['H']
                p_base = metrics(y0, hrd)[0]
                p_r1 = metrics(y0 + D, hrd)[0]
                p_g1 = metrics(y0 + q_g1 * D, hrd)[0]
                p_g64 = metrics(y0 + tgt['q_full'] * D, hrd)[0]
                p_h4 = metrics(y0 + tgt_h4['q_full'] * D, hrd)[0]

            key = '%s|%s' % (name, state)
            cache_entries[key] = dict(
                D=D.detach().cpu().half(),  # store fp16; reload as float
                q_grid=tgt['q_grid'].detach().cpu(),
                q_full=tgt['q_full'].detach().cpu(),
                mask=mask.detach().cpu().bool(),
                name=name, state=state,
            )
            pairs.append(dict(name=name, state=state, key=key,
                              local_index=i))
            q = tgt['q_grid'].float().reshape(-1)
            target_stats[state].append(dict(
                mean=float(q.mean()), std=float(q.std()),
                frac0=float((q <= 0.05).float().mean()),
                frac1=float((q >= 0.95).float().mean()),
                frac_mid=float(((q > 0.05) & (q < 0.95)).float().mean()),
            ))
            oracle[state]['psnr_base'].append(p_base)
            oracle[state]['psnr_r1'].append(p_r1)
            oracle[state]['psnr_g1'].append(p_g1)
            oracle[state]['psnr_g64'].append(p_g64)
            oracle[state]['psnr_h4'].append(p_h4)

    # aggregate oracle / target stats
    oracle_agg = {}
    for st in STATES:
        o = oracle[st]
        oracle_agg[st] = {k: float(np.mean(v)) for k, v in o.items()}
        oracle_agg[st]['g64_minus_r1'] = (oracle_agg[st]['psnr_g64']
                                          - oracle_agg[st]['psnr_r1'])
        oracle_agg[st]['g64_minus_g1'] = (oracle_agg[st]['psnr_g64']
                                          - oracle_agg[st]['psnr_g1'])
    tgt_agg = {}
    for st in STATES:
        rows_st = target_stats[st]
        tgt_agg[st] = dict(
            mean=float(np.mean([r['mean'] for r in rows_st])),
            std=float(np.mean([r['std'] for r in rows_st])),
            frac0=float(np.mean([r['frac0'] for r in rows_st])),
            frac1=float(np.mean([r['frac1'] for r in rows_st])),
            frac_mid=float(np.mean([r['frac_mid'] for r in rows_st])),
            n=len(rows_st),
        )
    return pairs, cache_entries, mmap, oracle_agg, tgt_agg, id_to_local, all_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='/root/data/experiments/v3a5c_tiny_overfit')
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--v5a_root', default=V5A)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n_tiny', type=int, default=N_TINY)
    ap.add_argument('--n_micro', type=int, default=N_MICRO)
    ap.add_argument('--tiny_seed', type=int, default=TINY_SEED)
    ap.add_argument('--max_redraw', type=int, default=3)
    a = ap.parse_args(_CLI)

    t0 = time.time()
    os.makedirs(os.path.join(a.root, 'tiny'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    log_path = os.path.join(a.root, 'logs', 'setup_v3a5c.log')
    logf = open(log_path, 'w', encoding='utf-8')

    def log(msg=''):
        print(msg)
        logf.write(msg + '\n')
        logf.flush()

    v5a_lock = json.load(open(os.path.join(a.v5a_root, 'artifact_lock.json'),
                              encoding='utf-8'))
    thr = float(v5a_lock['energy_threshold'])
    log('V3-A.5C setup')
    log('  energy_threshold (from V5A) = %.12e' % thr)

    split = json.load(open(os.path.join(a.v4_root, 'splits', 'split.json'),
                           encoding='utf-8'))
    mmap_full = json.load(open(os.path.join(a.v4_root, 'mappings',
                                            'mismatch_train_575.json'),
                               encoding='utf-8'))
    train_ids = list(split['train'])
    all_rows = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                         os.path.join(a.v4_root, 'splits', 'split.json'))
    rows_by_name = {r[0]: r for r in all_rows['train']}

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    proposal = load_proposal(os.path.join(a.src_root, R1_CK), a.device)
    geom_g64 = prepare_geometry(target_geometry(400, 600, 'g64'), a.device)
    geom_h4 = prepare_geometry(target_geometry(400, 600, 'dense_block_h4'),
                               a.device)
    cache_y0 = os.path.join(a.src_root, a.cache_name, 'refiner_train')

    # --- micro4 (always from seed, first 4 of a separate draw) ---
    micro_ids = choose_tiny_ids(train_ids, n=a.n_micro, seed=a.tiny_seed + 1000)
    dump_json(os.path.join(a.root, 'tiny', 'micro4_ids.json'),
              dict(ids=micro_ids, seed=a.tiny_seed + 1000, n=a.n_micro))

    # --- tiny16 with redraw ---
    seed = int(a.tiny_seed)
    accepted = None
    for attempt in range(a.max_redraw):
        ids = choose_tiny_ids(train_ids, n=a.n_tiny, seed=seed + attempt)
        log('attempt %d seed=%d ids[0:3]=%s' % (attempt, seed + attempt, ids[:3]))
        pairs, entries, mmap, oracle_agg, tgt_agg, id_to_local, all_ids = \
            build_for_ids(ids, rows_by_name, mmap_full, ns, cache_y0, proposal,
                          geom_g64, geom_h4, thr, a.device)
        acc = accept_subset(oracle_agg)
        log('  accept: %s  mean_g64_headroom=%.4f  n_regional=%d'
            % (acc['ok'], acc['mean_g64_headroom'], acc['n_states_regional']))
        if acc['ok']:
            accepted = dict(ids=ids, seed=seed + attempt, attempt=attempt,
                            pairs=pairs, entries=entries, mmap=mmap,
                            oracle=oracle_agg, targets=tgt_agg,
                            id_to_local=id_to_local, all_ids=all_ids,
                            accept=acc)
            break
        log('  redraw')
    if accepted is None:
        raise SystemExit('tiny16 failed acceptance after %d redraws' % a.max_redraw)

    # persist ids / mismatch / stats
    dump_json(os.path.join(a.root, 'tiny', 'tiny16_ids.json'),
              dict(ids=accepted['ids'], seed=accepted['seed'],
                   attempt=accepted['attempt'], n=len(accepted['ids']),
                   accept=accepted['accept']))
    dump_json(os.path.join(a.root, 'tiny', 'tiny16_mismatch_map.json'),
              accepted['mmap'])
    dump_json(os.path.join(a.root, 'tiny', 'target_stats.json'),
              accepted['targets'])
    dump_json(os.path.join(a.root, 'tiny', 'oracle_stats.json'),
              accepted['oracle'])
    dump_json(os.path.join(a.root, 'tiny', 'pairs.json'),
              accepted['pairs'])

    cache_path = os.path.join(a.root, 'tiny', 'cache.pt')
    torch.save(dict(
        entries=accepted['entries'],
        pairs=accepted['pairs'],
        energy_threshold=thr,
        ids=accepted['ids'],
        all_ids=accepted['all_ids'],
        id_to_local=accepted['id_to_local'],
        geom_mode='g64',
        shape=list(geom_g64['shape']),
    ), cache_path)
    log('wrote cache (%d entries) -> %s' % (len(accepted['entries']), cache_path))

    # also build micro cache for sanity
    mpairs, mentries, mmmap, moracle, mtgt, midloc, mall = build_for_ids(
        micro_ids, rows_by_name, mmap_full, ns, cache_y0, proposal,
        geom_g64, geom_h4, thr, a.device)
    dump_json(os.path.join(a.root, 'tiny', 'micro4_mismatch_map.json'), mmmap)
    torch.save(dict(entries=mentries, pairs=mpairs, energy_threshold=thr,
                    ids=micro_ids, all_ids=mall, id_to_local=midloc,
                    geom_mode='g64', shape=list(geom_g64['shape'])),
               os.path.join(a.root, 'tiny', 'micro4_cache.pt'))

    head, dirty = git_meta()
    lock = dict(
        plan='V3A5C_TINY_OVERFIT_AUDIT',
        repo_commit=head,
        git_dirty_count=len(dirty),
        git_dirty_files=dirty,
        energy_threshold=thr,
        energy_threshold_source='v3a5_g64_verifier/artifact_lock.json',
        v5a_lock_sha256=file_sha256(os.path.join(a.v5a_root, 'artifact_lock.json')),
        proposal_sha256=file_sha256(os.path.join(a.src_root, R1_CK)),
        split_sha256=file_sha256(os.path.join(a.v4_root, 'splits', 'split.json')),
        mismatch_train_sha256=file_sha256(
            os.path.join(a.v4_root, 'mappings', 'mismatch_train_575.json')),
        cache_metadata_sha256=file_sha256(
            os.path.join(a.src_root, a.cache_name, 'refiner_train',
                         'metadata.json')),
        tiny_ids_sha256=file_sha256(os.path.join(a.root, 'tiny', 'tiny16_ids.json')),
        tiny_mismatch_sha256=file_sha256(
            os.path.join(a.root, 'tiny', 'tiny16_mismatch_map.json')),
        tiny_cache_sha256=file_sha256(cache_path),
        micro4_ids_sha256=file_sha256(
            os.path.join(a.root, 'tiny', 'micro4_ids.json')),
        tiny_seed=accepted['seed'],
        tiny_attempt=accepted['attempt'],
        n_tiny=len(accepted['ids']),
        n_pairs=len(accepted['pairs']),
        states=list(STATES),
        variant=a.variant,
        cache_name=a.cache_name,
        train_seed=42,
        grad_accum=4,
        updates_default=20000,
        accept=accepted['accept'],
        wall_seconds=time.time() - t0,
    )
    dump_json(os.path.join(a.root, 'artifact_lock.json'), lock)
    log('lock written. wall=%.1fs' % lock['wall_seconds'])
    logf.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
