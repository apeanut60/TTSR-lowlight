#!/usr/bin/env python
"""V3-A.5C C0/C1 tiny-set overfit trainer.

Cache (from setup) holds D / q* / mask; X/Y0/H/R are reloaded from the dataset
so the cache stays small. Equal-frequency 48-pair cycle; no Test / no dev.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import metrics                          # noqa: E402
from model.V3A5Verifier import V3A5Verifier, build_shared_init     # noqa: E402
from option import parser as option_parser                        # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import (STATES, bit_equal, expand_gate,         # noqa: E402
                          prepare_geometry, snapshot_, state_dict_sha,
                          target_geometry, verifier_loss)
from v3a5c_runtime import (CKPT_STEPS, DEFAULT_UPDATES, GRAD_ACCUM,  # noqa: E402
                           OUT_WEIGHT_C0, OUT_WEIGHT_C1, TRAIN_SEED,
                           aggregate_pair_metrics, dump_json,
                           early_overfit_success, file_sha256,
                           make_pair_schedule, pair_gate_bundle,
                           verdict_c0, verdict_c1)

DEFAULT_ROOT = '/root/data/experiments/v3a5c_tiny_overfit'
SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'


def load_locked(root, device):
    lock = json.load(open(os.path.join(root, 'artifact_lock.json'), encoding='utf-8'))
    cache_path = os.path.join(root, 'tiny', 'cache.pt')
    if file_sha256(cache_path) != lock['tiny_cache_sha256']:
        raise SystemExit('tiny cache sha mismatch vs lock')
    blob = torch.load(cache_path, map_location='cpu')
    ids = json.load(open(os.path.join(root, 'tiny', 'tiny16_ids.json'),
                         encoding='utf-8'))['ids']
    mmap = json.load(open(os.path.join(root, 'tiny', 'tiny16_mismatch_map.json'),
                          encoding='utf-8'))
    oracle = json.load(open(os.path.join(root, 'tiny', 'oracle_stats.json'),
                            encoding='utf-8'))
    return lock, blob, ids, mmap, oracle


def materialize_pairs(root, blob, ids, mmap, lock, device):
    """Attach live X/Y0/H/R tensors + float D/q/mask for each of 48 pairs."""
    ns = option_parser.parse_args([])
    ns.dataset_dir = '/root/data/datasets/lol-v2-real'
    ns.v3a_ref_variant = lock.get('variant', 'nanobanana_ref_v2')
    all_rows = load_rows(os.path.join(SRC, 'manifests', 'refiner_train.csv'),
                         os.path.join(V4, 'splits', 'split.json'))
    rows_by_name = {r[0]: r for r in all_rows['train']}
    # include donors so TrainSet can resolve mismatch refs if needed
    donor_ids = sorted(set(mmap.values()))
    row_list = [rows_by_name[i] for i in ids]
    # donors not already in tiny set still need to be loadable; TrainSet builds
    # ref_map from filesystem, so donors need not be in pairs — keep tiny only.
    ds = make_dataset(ns, row_list,
                      os.path.join(SRC, lock.get('cache_name', 'cache_y0_lolbase'),
                                   'refiner_train'),
                      mmap)
    id_to_local = {name: i for i, name in enumerate(ids)}

    pairs = []
    for meta in blob['pairs']:
        name, state = meta['name'], meta['state']
        key = meta.get('key', '%s|%s' % (name, state))
        ent = blob['entries'][key]
        t = sample_tensors(ds, id_to_local[name], state, device)
        D = ent['D'].float().to(device)
        if D.dim() == 3:
            D = D.unsqueeze(0)
        q_grid = ent['q_grid'].float().to(device)
        if q_grid.dim() == 3:
            q_grid = q_grid.unsqueeze(0)
        mask = ent['mask'].to(device).float()
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        # oracles for Recovery64 (per-pair recomputed once)
        with torch.no_grad():
            r1 = metrics(t['Y0'] + D, t['H'])[0]
            ao = metrics(t['Y0'] + expand_gate(
                q_grid,
                prepare_geometry(target_geometry(400, 600, 'g64'), device)) * D,
                t['H'])[0]
        pairs.append(dict(
            name=name, state=state, key=key,
            X=t['X'], Y0=t['Y0'], H=t['H'], R=t['R'],
            D=D, q_grid=q_grid, mask=mask, R1=r1, AO64=ao,
        ))
    if len(pairs) != 48:
        raise SystemExit('expected 48 pairs, got %d' % len(pairs))
    return pairs


def evaluate_pairs(model, pairs, geom):
    model.eval()
    rows = []
    with torch.no_grad():
        for p in pairs:
            q_v = model(p['X'], p['Y0'], p['R'], geom=geom)
            bund = pair_gate_bundle(q_v, p['q_grid'], p['mask'])
            q_full = expand_gate(q_v, geom)
            psnr = metrics(p['Y0'] + q_full * p['D'], p['H'])[0]
            den = p['AO64'] - p['R1']
            rec = ((psnr - p['R1']) / den) if den > 1e-8 else float('nan')
            h = model.prepare_features(
                model.common_features(p['X'], p['Y0'], p['R']), geom)
            h1 = model.net.act(model.net.head1(model.net.act(model.net.head0(h))))
            logit = model.net.head2(h1)
            rows.append(dict(
                name=p['name'], state=p['state'], **bund,
                PSNR=psnr, Recovery64=rec, R1=p['R1'], AO64=p['AO64'],
                logit_mean=float(logit.mean()), logit_std=float(logit.std()),
                logit_min=float(logit.min()), logit_max=float(logit.max()),
            ))
    model.train()
    return rows


def summarize(rows):
    overall = aggregate_pair_metrics(rows)
    by_state = {s: aggregate_pair_metrics([r for r in rows if r['state'] == s])
                for s in STATES}
    return dict(overall=overall, by_state=by_state)


def flatten_curve_row(step, loss_row, summary, split='overall'):
    o = summary['overall'] if split == 'overall' else summary['by_state'][split]
    return dict(
        step=step, split=split,
        loss_total=loss_row.get('loss'),
        loss_gate=loss_row.get('gate'),
        loss_output=loss_row.get('out'),
        MAE=o.get('MAE'), masked_MAE=o.get('masked_MAE'),
        corr=o.get('corr'), masked_corr=o.get('masked_corr'),
        decision_accuracy=o.get('decision_accuracy'),
        q_v_mean=o.get('q_v_mean'), q_v_std=o.get('q_v_std'),
        q_opt_mean=o.get('q_opt_mean'), q_opt_std=o.get('q_opt_std'),
        std_ratio=o.get('std_ratio'),
        spatial_corr_mean=o.get('spatial_corr_mean'),
        spatial_corr_median=o.get('spatial_corr_median'),
        PSNR=o.get('PSNR'), Recovery64=o.get('Recovery64'))


def write_curve_csv(path, curve):
    if not curve:
        return
    keys = sorted({k for row in curve for k in row})
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in curve:
            w.writerow(row)


def ensure_shared_init(root, seed=TRAIN_SEED):
    path = os.path.join(root, 'tiny', 'shared_init_s42.pt')
    if os.path.isfile(path):
        return path
    init = build_shared_init('dense_block_h4', 'g64', seed=seed)
    if not bit_equal(init['dense_block_h4'], init['g64']):
        raise SystemExit('shared init not bit-equal')
    torch.save(init, path)
    return path


def train_arm(root, arm, out_weight, steps, grad_accum, seed, device, log):
    lock, blob, ids, mmap, _oracle = load_locked(root, device)
    pairs = materialize_pairs(root, blob, ids, mmap, lock, device)
    arm_dir = os.path.join(root, arm)
    os.makedirs(os.path.join(arm_dir, 'checkpoints'), exist_ok=True)

    geom = prepare_geometry(target_geometry(400, 600, 'g64'), device)
    init_path = ensure_shared_init(root, seed=seed)
    init = torch.load(init_path, map_location='cpu')
    model = V3A5Verifier('g64').to(device)
    model.load_state_dict({k: v.clone() for k, v in init['g64'].items()},
                          strict=True)

    rows0 = evaluate_pairs(model, pairs, geom)
    sum0 = summarize(rows0)
    if abs(sum0['overall']['q_v_mean'] - 0.5) > 1e-5:
        raise SystemExit('step0 q_v_mean=%s' % sum0['overall']['q_v_mean'])
    log('  step0: q=%.6f mMAE=%.4f mCorr=%.4f acc=%.3f'
        % (sum0['overall']['q_v_mean'], sum0['overall']['masked_MAE'],
           sum0['overall']['masked_corr'], sum0['overall']['decision_accuracy']))

    schedule = make_pair_schedule(steps, grad_accum, 48, seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    before = snapshot_(model)

    curve, hist = [], []
    last_loss = dict(loss=0.0, gate=0.0, out=0.0)
    dump_json(os.path.join(arm_dir, 'checkpoints', 'metrics_%06d.json' % 0),
              dict(step=0, summary=sum0,
                   verdict=verdict_c0(sum0['overall'], sum0['by_state'])))
    curve.append(flatten_curve_row(0, last_loss, sum0))
    torch.save(dict(model=snapshot_(model), step=0, out_weight=out_weight,
                    arm=arm),
               os.path.join(arm_dir, 'checkpoints', 'ckpt_%06d.pt' % 0))

    model.train()
    t0 = time.time()
    early = False
    strong_streak = 0
    ckpt_set = set(CKPT_STEPS)
    unique_seen = set()
    micro_seen = 0
    upd = 0

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        agg = np.zeros(3, dtype=np.float64)
        for k in range(grad_accum):
            j = int(schedule[step * grad_accum + k])
            micro_seen += 1
            unique_seen.add(j)
            p = pairs[j]
            q_v = model(p['X'], p['Y0'], p['R'], geom=geom)
            q_full = expand_gate(q_v, geom)
            loss, gate_t, out_t = verifier_loss(
                q_v, p['q_grid'], p['mask'], q_full, p['Y0'], p['H'], p['D'],
                out_weight=out_weight)
            (loss / grad_accum).backward()
            agg += np.array([float(loss), float(gate_t), float(out_t)])
        opt.step()
        agg /= grad_accum
        last_loss = dict(loss=float(agg[0]), gate=float(agg[1]), out=float(agg[2]))
        upd = step + 1

        if upd % 100 == 0 or upd in ckpt_set:
            hist.append(dict(step=upd, **last_loss,
                             micro_batches_seen=micro_seen,
                             unique_pairs_seen=len(unique_seen), lr=1e-4))
            log('  step %5d/%d  loss %.5f (g %.5f + %.2f*o %.5f)  uniq %d/48  %.0fs'
                % (upd, steps, last_loss['loss'], last_loss['gate'],
                   out_weight, last_loss['out'], len(unique_seen),
                   time.time() - t0))

        if upd in ckpt_set:
            rows = evaluate_pairs(model, pairs, geom)
            summary = summarize(rows)
            v = verdict_c0(summary['overall'], summary['by_state'])
            torch.save(dict(model=snapshot_(model), step=upd,
                            out_weight=out_weight, arm=arm),
                       os.path.join(arm_dir, 'checkpoints',
                                    'ckpt_%06d.pt' % upd))
            dump_json(os.path.join(arm_dir, 'checkpoints',
                                   'metrics_%06d.json' % upd),
                      dict(step=upd, summary=summary, verdict=v, per_pair=rows))
            curve.append(flatten_curve_row(upd, last_loss, summary))
            for s in STATES:
                curve.append(flatten_curve_row(upd, last_loss, summary, split=s))
            log('  eval@%d: mMAE=%.4f mCorr=%.4f acc=%.3f std_r=%.3f -> %s'
                % (upd, v['thresholds']['masked_MAE'],
                   v['thresholds']['masked_corr'],
                   v['thresholds']['decision_accuracy'],
                   v['thresholds']['std_ratio'], v['label']))
            if early_overfit_success(summary['overall']):
                strong_streak += 1
            else:
                strong_streak = 0
            if strong_streak >= 2:
                early = True
                log('  early_overfit_success at step %d' % upd)
                break

    final_step = upd if early else steps
    if final_step not in ckpt_set and final_step != 0:
        rows = evaluate_pairs(model, pairs, geom)
        summary = summarize(rows)
        v = verdict_c0(summary['overall'], summary['by_state'])
        torch.save(dict(model=snapshot_(model), step=final_step,
                        out_weight=out_weight, arm=arm),
                   os.path.join(arm_dir, 'checkpoints',
                                'ckpt_%06d.pt' % final_step))
        dump_json(os.path.join(arm_dir, 'checkpoints',
                               'metrics_%06d.json' % final_step),
                  dict(step=final_step, summary=summary, verdict=v,
                       per_pair=rows))
        curve.append(flatten_curve_row(final_step, last_loss, summary))
    else:
        meta = json.load(open(os.path.join(arm_dir, 'checkpoints',
                                           'metrics_%06d.json' % final_step),
                              encoding='utf-8'))
        summary, v = meta['summary'], meta['verdict']

    write_curve_csv(os.path.join(arm_dir, 'overfit_curve.csv'), curve)
    after = snapshot_(model)
    train_json = dict(
        arm=arm, out_weight=out_weight, steps_requested=steps,
        steps_done=final_step, grad_accum=grad_accum, seed=seed,
        early_overfit_success=early, history=hist,
        final_summary=summary, final_verdict=v,
        param_l1_change=float(sum(
            float((after[k] - before[k]).abs().sum()) for k in before)),
        wall_seconds=time.time() - t0,
        init_sha=state_dict_sha(init['g64']),
        micro_batches_seen=micro_seen,
        unique_pairs_seen=len(unique_seen),
        lock_tiny_cache_sha=lock['tiny_cache_sha256'],
        shared_init=init_path,
    )
    dump_json(os.path.join(arm_dir, 'train_metrics.json'), train_json)
    dump_json(os.path.join(arm_dir, 'args.json'),
              dict(arm=arm, out_weight=out_weight, steps=steps,
                   grad_accum=grad_accum, seed=seed, device=device))
    log('done %s: %s steps=%d wall=%.0fs' %
        (arm, v['label'], final_step, time.time() - t0))
    return v, summary, final_step


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--arm', required=True,
                    choices=('C0_current_loss', 'C1_gate_only'))
    ap.add_argument('--steps', type=int, default=DEFAULT_UPDATES)
    ap.add_argument('--grad_accum', type=int, default=GRAD_ACCUM)
    ap.add_argument('--seed', type=int, default=TRAIN_SEED)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(_CLI)

    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'diagnostics'), exist_ok=True)
    logf = open(os.path.join(a.root, 'logs', 'train_%s.log' % a.arm), 'w',
                encoding='utf-8')

    def log(msg=''):
        print(msg)
        logf.write(msg + '\n')
        logf.flush()

    out_weight = OUT_WEIGHT_C0 if a.arm == 'C0_current_loss' else OUT_WEIGHT_C1
    if a.arm == 'C1_gate_only' and not a.force:
        c0p = os.path.join(a.root, 'C0_current_loss', 'train_metrics.json')
        c0 = json.load(open(c0p, encoding='utf-8'))
        if c0['final_verdict']['action'].startswith('stop'):
            raise SystemExit('C0 already succeeded (%s); refuse C1'
                             % c0['final_verdict']['label'])

    log('V3-A.5C train %s out_w=%.2f steps=%d accum=%d' %
        (a.arm, out_weight, a.steps, a.grad_accum))
    v, summary, final_step = train_arm(
        a.root, a.arm, out_weight, a.steps, a.grad_accum, a.seed,
        a.device, log)

    if a.arm == 'C0_current_loss':
        dump_json(os.path.join(a.root, 'diagnostics', 'c0_verdict.json'),
                  dict(verdict=v, final_step=final_step,
                       overall=summary['overall']))
    else:
        c0 = json.load(open(os.path.join(a.root, 'C0_current_loss',
                                         'train_metrics.json'), encoding='utf-8'))
        c1v = verdict_c1(c0['final_summary']['overall'], summary['overall'])
        dump_json(os.path.join(a.root, 'diagnostics', 'c1_verdict.json'),
                  dict(verdict=c1v, c0_label=c0['final_verdict']['label'],
                       c1_fit_label=v['label'], final_step=final_step,
                       overall=summary['overall']))
        log('C1 verdict: %s (maeΔ=%.3f corrΔ=%.3f acc=%.3f)' %
            (c1v['label'], c1v['delta_masked_MAE'], c1v['delta_masked_corr'],
             c1v['decision_accuracy']))
    logf.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
