#!/usr/bin/env python
"""V3-A.5D2.1 train: A0 or A1 MultiScale on fixed train64 (192 pairs), gate-only."""

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

from local_refine_runtime import metrics as _metrics                    # noqa: E402
from model.V3A5D2Verifier import ARMS, V3A5D2Verifier                    # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import (correction, load_proposal, load_rows,        # noqa: E402
                           make_dataset, sample_tensors)
from v3a5_runtime import (STATES, action_optimal_target, bit_equal,     # noqa: E402
                          block_energy, energy_mask, expand_gate,
                          prepare_geometry, state_dict_sha,
                          target_geometry, verifier_loss)
from v3a5c_runtime import (GRAD_ACCUM, aggregate_pair_metrics,          # noqa: E402
                           dump_json, early_overfit_success,
                           make_pair_schedule, pair_gate_bundle)
from v3a5d21_runtime import (DEFAULT_UPDATES, EXPOSURE_STEPS,           # noqa: E402
                             OUT_WEIGHT)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
ROOT_EXP = '/root/data/experiments/v3a5d21_scale64'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT_EXP)
    ap.add_argument('--arm', required=True, choices=list(ARMS))
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--updates', type=int, default=DEFAULT_UPDATES)
    ap.add_argument('--grad_accum', type=int, default=GRAD_ACCUM)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--log_every', type=int, default=100)
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    thr = float(lock['energy_threshold'])
    ids = json.load(open(os.path.join(a.root, 'subset', 'train64_ids.json')))['ids']
    mmap = json.load(open(os.path.join(a.root, 'subset', 'train64_mismatch_map.json')))

    arm_dir = os.path.join(a.root, a.arm)
    os.makedirs(os.path.join(arm_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    logf = open(os.path.join(a.root, 'logs', 'train_%s.log' % a.arm), 'w',
                encoding='utf-8')

    def log(msg=''):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    all_rows = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                         os.path.join(a.v4_root, 'splits', 'split.json'))
    rows_by_name = {r[0]: r for r in all_rows['train']}
    rows = [rows_by_name[i] for i in ids]
    ds = make_dataset(ns, rows,
                      os.path.join(a.src_root, a.cache_name, 'refiner_train'), mmap)
    name_to_i = {name: i for i, name in enumerate(ids)}
    geom = prepare_geometry(target_geometry(400, 600, 'g64'), a.device)
    proposal = load_proposal(os.path.join(a.src_root, R1_CK), a.device)

    # Build 192 pairs + preload tensors (D,q*,mask computed once).
    pairs = []
    for name in ids:
        for state in STATES:
            pairs.append(dict(name=name, state=state,
                              key='%s|%s' % (name, state)))
    n_pairs = len(pairs)
    log('V3-A.5D2.1 train %s  pairs=%d  updates=%d  thr=%.3e'
        % (a.arm, n_pairs, a.updates, thr))
    log('  preloading...')
    pair_bank = []
    with torch.no_grad():
        for p in pairs:
            t = sample_tensors(ds, name_to_i[p['name']], p['state'], a.device)
            D, _ = correction(proposal.proposal, t['Y0'], t['R'])
            tgt = action_optimal_target(t['Y0'], t['H'], D, geom)
            mask = energy_mask(block_energy(D, geom), thr)
            pair_bank.append(dict(
                X=t['X'], Y0=t['Y0'], H=t['H'], R=t['R'],
                D=D.detach(), q_star=tgt['q_grid'].detach(),
                mask=mask.detach(),
                name=p['name'], state=p['state'], key=p['key'],
            ))
    log('  preload done (%d)' % len(pair_bank))
    expos_20k = a.updates * a.grad_accum / float(n_pairs)
    log('  planned exposures/pair @%dk ≈ %.1f'
        % (a.updates // 1000, expos_20k))

    init = torch.load(os.path.join(a.root, 'init', 'shared_init_s42.pt'),
                      map_location='cpu')
    model = V3A5D2Verifier(a.arm).to(a.device)
    model.load_state_dict(init[a.arm], strict=True)
    init_sha = state_dict_sha(init[a.arm])
    if a.arm == 'A1_multiscale':
        a0 = V3A5D2Verifier('A0_control')
        a0.load_state_dict(init['A0_control'], strict=True)
        if not bit_equal(
                a0.net.state_dict(),
                {k: v.detach().cpu() for k, v in model.net.state_dict().items()}):
            raise SystemExit('A0/A1 net not bit-equal')
        if model.context_residual_max_abs() != 0.0:
            raise SystemExit('context residual not zero')
    log('  init_sha=%s' % init_sha[:16])

    model.eval()
    with torch.no_grad():
        b0 = pair_bank[0]
        q0 = model(b0['X'], b0['Y0'], b0['R'], geom=geom)
        if abs(float(q0.mean()) - 0.5) > 1e-5:
            raise SystemExit('step0 q mean bad')
        if a.arm == 'A1_multiscale':
            m0 = V3A5D2Verifier('A0_control').to(a.device)
            m0.load_state_dict(init['A0_control'], strict=True)
            qa = m0(b0['X'], b0['Y0'], b0['R'], geom=geom)
            if float((q0 - qa).abs().max()) > 1e-5:
                raise SystemExit('step0 A0!=A1')
    log('  step0 OK')

    def save_ckpt(step):
        blob = dict(model=model.state_dict(), step=step, arm=a.arm,
                    out_weight=OUT_WEIGHT, energy_threshold=thr,
                    seed=a.seed, init_sha=init_sha, n_pairs=n_pairs)
        path = os.path.join(arm_dir, 'checkpoints', 'ckpt_%06d.pt' % step)
        torch.save(blob, path)
        torch.save(blob, os.path.join(arm_dir, 'checkpoints', 'last.pt'))
        return path

    def eval_bank(step):
        model.eval()
        rows = []
        by = {s: [] for s in STATES}
        with torch.no_grad():
            for b in pair_bank:
                q_v = model(b['X'], b['Y0'], b['R'], geom=geom)
                bundle = pair_gate_bundle(q_v, b['q_star'], b['mask'])
                y_hat = b['Y0'] + expand_gate(q_v, geom) * b['D']
                bundle['PSNR'] = float(_metrics(y_hat, b['H'])[0])
                bundle['name'] = b['name']
                bundle['state'] = b['state']
                bundle['Recovery64'] = float('nan')
                rows.append(bundle)
                by[b['state']].append(bundle)
        summary = dict(step=step, overall=aggregate_pair_metrics(rows),
                       by_state={s: aggregate_pair_metrics(by[s]) for s in STATES})
        o = summary['overall']
        log('  [train64 @%d] MAE=%.4f corr=%.4f acc=%.3f std_r=%.3f spat=%.3f'
            % (step, o['masked_MAE'], o['masked_corr'], o['decision_accuracy'],
               o.get('std_ratio', float('nan')),
               o.get('spatial_corr_mean', float('nan'))))
        model.train()
        return summary

    save_ckpt(0)
    schedule = make_pair_schedule(a.updates, a.grad_accum, n_pairs, a.seed + 7)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    model.train()
    hist, overfit_rows = [], []
    t0 = time.time()
    unique_seen = set()
    ckpt_set = set(EXPOSURE_STEPS) | {0}
    last_step = 0
    overfit_rows.append(eval_bank(0))

    for step in range(a.updates):
        last_step = step + 1
        opt.zero_grad(set_to_none=True)
        agg = 0.0
        for k in range(a.grad_accum):
            j = step * a.grad_accum + k
            pi = int(schedule[j])
            unique_seen.add(pi)
            b = pair_bank[pi]
            q_v = model(b['X'], b['Y0'], b['R'], geom=geom)
            q_full = expand_gate(q_v, geom)
            loss, gate_t, out_t = verifier_loss(
                q_v, b['q_star'], b['mask'], q_full, b['Y0'], b['H'], b['D'],
                out_weight=OUT_WEIGHT)
            (loss / a.grad_accum).backward()
            agg += float(loss)
        opt.step()
        agg /= a.grad_accum
        if step % a.log_every == 0 or last_step == a.updates:
            hist.append(dict(step=step, loss=agg, uniq=len(unique_seen)))
            log('  step %5d/%d  loss %.5f  uniq %d/%d  %.0fs'
                % (step, a.updates, agg, len(unique_seen), n_pairs,
                   time.time() - t0))
        if last_step in ckpt_set:
            save_ckpt(last_step)
            overfit_rows.append(eval_bank(last_step))

    if last_step not in ckpt_set:
        save_ckpt(last_step)
        overfit_rows.append(eval_bank(last_step))

    keys = sorted(overfit_rows[0]['overall'].keys())
    with open(os.path.join(arm_dir, 'overfit_curve.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['step'] + keys)
        w.writeheader()
        for r in overfit_rows:
            row = {'step': r['step']}
            row.update({k: r['overall'].get(k) for k in keys})
            w.writerow(row)
    dump_json(os.path.join(arm_dir, 'train_metrics.json'), dict(
        arm=a.arm, updates=a.updates, n_pairs=n_pairs, init_sha=init_sha,
        exposures_per_pair=expos_20k, wall_seconds=time.time() - t0,
        history=hist,
    ))
    dump_json(os.path.join(arm_dir, 'overfit_summaries.json'),
              [dict(step=r['step'], overall=r['overall'], by_state=r['by_state'])
               for r in overfit_rows])
    log('done %s in %.1fs' % (a.arm, time.time() - t0))
    logf.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
