#!/usr/bin/env python
"""V3-A.5C train: C0 (current loss) or C1 (gate-only) on the locked tiny set."""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.V3A5Verifier import V3A5Verifier, build_shared_init     # noqa: E402
from option import parser as option_parser                        # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import (bit_equal, expand_gate, prepare_geometry,  # noqa: E402
                          state_dict_sha, target_geometry, verifier_loss)
from v3a5c_eval_lib import eval_all_pairs                         # noqa: E402
from v3a5c_runtime import (CKPT_STEPS, DEFAULT_UPDATES, GRAD_ACCUM,  # noqa: E402
                           OUT_WEIGHT_C0, OUT_WEIGHT_C1, TRAIN_SEED,
                           dump_json, early_overfit_success,
                           make_pair_schedule)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
ARMS = {
    'C0_current_loss': OUT_WEIGHT_C0,
    'C1_gate_only': OUT_WEIGHT_C1,
}


def load_tiny_bundle(root, micro=False):
    tiny = os.path.join(root, 'tiny')
    if micro:
        cache = torch.load(os.path.join(tiny, 'micro4_cache.pt'),
                           map_location='cpu')
        ids = json.load(open(os.path.join(tiny, 'micro4_ids.json')))['ids']
        mmap = json.load(open(os.path.join(tiny, 'micro4_mismatch_map.json')))
    else:
        cache = torch.load(os.path.join(tiny, 'cache.pt'), map_location='cpu')
        ids = json.load(open(os.path.join(tiny, 'tiny16_ids.json')))['ids']
        mmap = json.load(open(os.path.join(tiny, 'tiny16_mismatch_map.json')))
    return cache, ids, mmap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='/root/data/experiments/v3a5c_tiny_overfit')
    ap.add_argument('--arm', required=True, choices=list(ARMS))
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--updates', type=int, default=DEFAULT_UPDATES)
    ap.add_argument('--grad_accum', type=int, default=GRAD_ACCUM)
    ap.add_argument('--seed', type=int, default=TRAIN_SEED)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--micro', action='store_true')
    a = ap.parse_args(_CLI)

    out_w = ARMS[a.arm]
    arm_name = ('_micro_' + a.arm) if a.micro else a.arm
    arm_dir = os.path.join(a.root, arm_name)
    os.makedirs(os.path.join(arm_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    logf = open(os.path.join(a.root, 'logs', 'train_%s%s.log'
                             % (a.arm, '_micro' if a.micro else '')), 'w',
                encoding='utf-8')

    def log(msg=''):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    thr = float(lock['energy_threshold'])
    cache, ids, mmap = load_tiny_bundle(a.root, micro=a.micro)
    pairs = cache['pairs']
    n_pairs = len(pairs)
    entries = cache['entries']

    log('V3-A.5C train %s  out_weight=%.3f  pairs=%d  updates=%d  accum=%d'
        % (a.arm, out_w, n_pairs, a.updates, a.grad_accum))

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    all_rows = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                         os.path.join(a.v4_root, 'splits', 'split.json'))
    rows_by_name = {r[0]: r for r in all_rows['train']}
    rows = [rows_by_name[i] for i in ids]
    ds = make_dataset(ns, rows, os.path.join(a.src_root, a.cache_name,
                                             'refiner_train'), mmap)
    name_to_i = {name: i for i, name in enumerate(ids)}

    geom = prepare_geometry(target_geometry(400, 600, 'g64'), a.device)
    init = build_shared_init('dense_block_h4', 'g64', seed=int(a.seed))
    model = V3A5Verifier('g64').to(a.device)
    model.load_state_dict({k: v.clone() for k, v in init['g64'].items()},
                          strict=True)
    if not bit_equal(init['dense_block_h4'], init['g64']):
        raise SystemExit('shared init not bit-equal')
    init_sha = state_dict_sha(init['g64'])
    log('  init_sha=%s' % init_sha[:16])

    model.eval()
    with torch.no_grad():
        p0 = pairs[0]
        t = sample_tensors(ds, name_to_i[p0['name']], p0['state'], a.device)
        q0 = model(t['X'], t['Y0'], t['R'], geom=geom)
        q0m = float(q0.mean())
    if abs(q0m - 0.5) > 1e-5:
        raise SystemExit('step0 q mean = %.6f (want 0.5)' % q0m)
    log('  step0 q_mean=%.6f OK' % q0m)

    def save_ckpt(step):
        blob = dict(model=model.state_dict(), step=step, arm=a.arm,
                    out_weight=out_w, energy_threshold=thr, seed=a.seed,
                    init_sha=init_sha, n_pairs=n_pairs, micro=bool(a.micro))
        path = os.path.join(arm_dir, 'checkpoints', 'ckpt_%06d.pt' % step)
        torch.save(blob, path)
        torch.save(blob, os.path.join(arm_dir, 'checkpoints', 'last.pt'))
        return path

    save_ckpt(0)

    schedule = make_pair_schedule(a.updates, a.grad_accum, n_pairs, a.seed + 7)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    model.train()
    hist = []
    overfit_rows = []
    t0 = time.time()
    unique_seen = set()
    micro_seen = 0
    early = False
    prev_early = False
    ckpt_set = set(CKPT_STEPS)
    last_step = 0

    def maybe_eval(step):
        nonlocal early, prev_early
        summary = eval_all_pairs(model, ds, name_to_i, pairs, entries, geom,
                                 thr, a.device, out_weight=out_w)
        summary['step'] = step
        overfit_rows.append(summary)
        o = summary['overall']
        log('  [overfit @%d] masked_MAE=%.4f masked_corr=%.4f dec_acc=%.3f '
            'std_ratio=%.3f PSNR=%.3f'
            % (step, o['masked_MAE'], o['masked_corr'], o['decision_accuracy'],
               o.get('std_ratio', float('nan')), o.get('PSNR', float('nan'))))
        model.train()
        now_early = early_overfit_success(o)
        if step > 0 and now_early and prev_early:
            early = True
            log('  early_overfit_success (2 consecutive ckpts) at step %d' % step)
        prev_early = now_early
        return summary

    maybe_eval(0)

    for step in range(a.updates):
        last_step = step + 1
        opt.zero_grad(set_to_none=True)
        agg = np.zeros(3, dtype=np.float64)
        for k in range(a.grad_accum):
            j = step * a.grad_accum + k
            pi = int(schedule[j])
            unique_seen.add(pi)
            micro_seen += 1
            p = pairs[pi]
            ent = entries[p['key']]
            t = sample_tensors(ds, name_to_i[p['name']], p['state'], a.device)
            D = ent['D'].to(a.device).float()
            q_star = ent['q_grid'].to(a.device).float()
            mask = ent['mask'].to(a.device)
            q_v = model(t['X'], t['Y0'], t['R'], geom=geom)
            q_full = expand_gate(q_v, geom)
            loss, gate_t, out_t = verifier_loss(
                q_v, q_star, mask, q_full, t['Y0'], t['H'], D,
                out_weight=out_w)
            (loss / a.grad_accum).backward()
            agg += np.array([float(loss), float(gate_t), float(out_t)])
        opt.step()
        agg /= a.grad_accum

        if step % a.log_every == 0 or last_step == a.updates:
            row = dict(optimizer_step=step, micro_batches_seen=micro_seen,
                       unique_pairs_seen_so_far=len(unique_seen),
                       loss=float(agg[0]), gate=float(agg[1]),
                       out=float(agg[2]), lr=1e-4,
                       q_v_mean=float(q_v.mean()),
                       q_opt_mean=float(q_star.mean()))
            hist.append(row)
            log('  step %5d/%d  loss %.5f (gate %.5f out %.5f)  '
                'uniq_pairs %d/%d  %.0fs'
                % (step, a.updates, row['loss'], row['gate'], row['out'],
                   len(unique_seen), n_pairs, time.time() - t0))

        if last_step in ckpt_set:
            save_ckpt(last_step)
            maybe_eval(last_step)
            if early:
                break

    if last_step not in ckpt_set:
        save_ckpt(last_step)
        maybe_eval(last_step)

    if overfit_rows:
        keys = sorted(overfit_rows[0]['overall'].keys())
        with open(os.path.join(arm_dir, 'overfit_curve.csv'), 'w',
                  newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['step'] + keys)
            w.writeheader()
            for r in overfit_rows:
                row = {'step': r['step']}
                row.update({k: r['overall'].get(k) for k in keys})
                w.writerow(row)

    dump_json(os.path.join(arm_dir, 'train_metrics.json'), dict(
        arm=a.arm, out_weight=out_w, updates_requested=a.updates,
        updates_done=last_step, grad_accum=a.grad_accum, seed=a.seed,
        init_sha=init_sha, energy_threshold=thr, n_pairs=n_pairs,
        micro=bool(a.micro), history=hist, early_overfit_success=early,
        wall_seconds=time.time() - t0,
        unique_pairs_final=len(unique_seen),
        overfit_checkpoints=[r['step'] for r in overfit_rows],
    ))
    dump_json(os.path.join(arm_dir, 'args.json'), dict(
        arm=a.arm, args=vars(a), out_weight=out_w, init_sha=init_sha,
        lock_commit=lock.get('repo_commit'),
    ))
    # drop bulky per_pair lists from summaries on disk during train
    slim = []
    for r in overfit_rows:
        slim.append(dict(step=r['step'], overall=r['overall'],
                         by_state=r['by_state'], baselines=r['baselines']))
    dump_json(os.path.join(arm_dir, 'overfit_summaries.json'), slim)
    log('done %s in %.1fs (early=%s)' % (a.arm, time.time() - t0, early))
    logf.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
