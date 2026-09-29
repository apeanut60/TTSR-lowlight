#!/usr/bin/env python
"""V3-A.5D1 train: A0_control or A1_evidence on locked tiny16 (gate-only)."""

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

from model.V3A5DVerifier import ARMS, V3A5DVerifier                      # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors       # noqa: E402
from v3a5_runtime import (bit_equal, expand_gate, prepare_geometry,     # noqa: E402
                          state_dict_sha, target_geometry, verifier_loss)
from v3a5c_runtime import (CKPT_STEPS, DEFAULT_UPDATES, GRAD_ACCUM,     # noqa: E402
                           TRAIN_SEED, dump_json, early_overfit_success,
                           make_pair_schedule)
from v3a5d1_runtime import OUT_WEIGHT                                   # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
ROOT_EXP = '/root/data/experiments/v3a5d1_tiny_evidence'


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
    ap.add_argument('--seed', type=int, default=TRAIN_SEED)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--log_every', type=int, default=100)
    a = ap.parse_args(_CLI)

    arm_dir = os.path.join(a.root, a.arm)
    os.makedirs(os.path.join(arm_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    logf = open(os.path.join(a.root, 'logs', 'train_%s.log' % a.arm), 'w',
                encoding='utf-8')

    def log(msg=''):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    thr = float(lock['energy_threshold'])
    norm = lock['evidence_norm']
    tiny = os.path.join(a.root, 'tiny')
    cache = torch.load(os.path.join(tiny, 'cache.pt'), map_location='cpu')
    ids = json.load(open(os.path.join(tiny, 'tiny16_ids.json')))['ids']
    mmap = json.load(open(os.path.join(tiny, 'tiny16_mismatch_map.json')))
    pairs = cache['pairs']
    entries = cache['entries']
    n_pairs = len(pairs)
    out_w = OUT_WEIGHT

    log('V3-A.5D1 train %s  out_weight=%.3f  pairs=%d  updates=%d'
        % (a.arm, out_w, n_pairs, a.updates))

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

    # Preload all 48 (X,Y0,H,R) + cached D/q*/mask/evidence onto device.
    log('  preloading %d pair tensors...' % n_pairs)
    pair_bank = []
    for p in pairs:
        t = sample_tensors(ds, name_to_i[p['name']], p['state'], a.device)
        ent = entries[p['key']]
        pair_bank.append(dict(
            X=t['X'], Y0=t['Y0'], H=t['H'], R=t['R'],
            D=ent['D'].to(a.device).float(),
            q_star=ent['q_grid'].to(a.device).float(),
            mask=ent['mask'].to(a.device),
            evidence_g64=ent['evidence_g64'].to(a.device).float(),
            q_full=ent['q_full'].to(a.device).float(),
            name=p['name'], state=p['state'], key=p['key'],
        ))
    log('  preload done')

    init_blob = torch.load(os.path.join(tiny, 'shared_init_s42.pt'),
                           map_location='cpu')
    model = V3A5DVerifier(a.arm).to(a.device)
    model.load_state_dict(init_blob[a.arm], strict=True)
    model.set_evidence_norm(norm['mean'], norm['std'], norm['zscore_mask'])
    init_sha = state_dict_sha(init_blob[a.arm])
    # Fairness: common trunk of A0/A1 must match at init (ev_head excluded).
    if a.arm == 'A1_evidence':
        a0 = V3A5DVerifier('A0_control')
        a0.load_state_dict(init_blob['A0_control'], strict=True)
        # Compare on CPU — model.net may already be on CUDA.
        if not bit_equal(a0.net.state_dict(),
                         {k: v.detach().cpu() for k, v in model.net.state_dict().items()}):
            raise SystemExit('A0/A1 common net not bit-equal at init')
        if model.evidence_weight_max_abs() != 0.0:
            raise SystemExit('A1 evidence weights not zero at step0')
    log('  init_sha=%s  ev_w_max=%g' % (init_sha[:16], model.evidence_weight_max_abs()))

    model.eval()
    with torch.no_grad():
        b0 = pair_bank[0]
        q0 = model(b0['X'], b0['Y0'], b0['R'], geom=geom,
                   evidence_g64=b0['evidence_g64'])
        q0m = float(q0.mean())
        if a.arm == 'A1_evidence':
            m0 = V3A5DVerifier('A0_control').to(a.device)
            m0.load_state_dict(init_blob['A0_control'], strict=True)
            m0.set_evidence_norm(norm['mean'], norm['std'], norm['zscore_mask'])
            q_a0 = m0(b0['X'], b0['Y0'], b0['R'], geom=geom)
            if float((q0 - q_a0).abs().max()) > 1e-5:
                raise SystemExit('step0 A0!=A1 max_abs=%.3e'
                                 % float((q0 - q_a0).abs().max()))
            log('  step0 A0≡A1 OK (max_abs<=1e-5)')
    if abs(q0m - 0.5) > 1e-5:
        raise SystemExit('step0 q mean = %.6f (want 0.5)' % q0m)
    log('  step0 q_mean=%.6f OK' % q0m)

    def save_ckpt(step):
        blob = dict(model=model.state_dict(), step=step, arm=a.arm,
                    out_weight=out_w, energy_threshold=thr, seed=a.seed,
                    init_sha=init_sha, n_pairs=n_pairs,
                    evidence_norm=norm)
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

    def eval_from_bank(step):
        nonlocal early, prev_early
        model.eval()
        from v3a5c_runtime import pair_gate_bundle, aggregate_pair_metrics
        from local_refine_runtime import metrics as _metrics
        from v3a5_runtime import STATES
        rows = []
        by = {s: [] for s in STATES}
        with torch.no_grad():
            for b in pair_bank:
                q_v = model(b['X'], b['Y0'], b['R'], geom=geom,
                            evidence_g64=b['evidence_g64'])
                q_full = expand_gate(q_v, geom)
                bundle = pair_gate_bundle(q_v, b['q_star'], b['mask'])
                y_hat = b['Y0'] + q_full * b['D']
                psnr = _metrics(y_hat, b['H'])[0]
                bundle.update(dict(PSNR=float(psnr), name=b['name'],
                                   state=b['state'], Recovery64=float('nan')))
                rows.append(bundle)
                by[b['state']].append(bundle)
        summary = dict(
            step=step,
            overall=aggregate_pair_metrics(rows),
            by_state={s: aggregate_pair_metrics(by[s]) for s in STATES},
        )
        o = summary['overall']
        overfit_rows.append(summary)
        log('  [overfit @%d] masked_MAE=%.4f masked_corr=%.4f dec_acc=%.3f '
            'std_ratio=%.3f spat_corr=%.3f'
            % (step, o['masked_MAE'], o['masked_corr'], o['decision_accuracy'],
               o.get('std_ratio', float('nan')),
               o.get('spatial_corr_mean', float('nan'))))
        model.train()
        now_early = early_overfit_success(o)
        if step > 0 and now_early and prev_early:
            early = True
            log('  early_overfit_success at step %d' % step)
        prev_early = now_early
        return summary

    eval_from_bank(0)

    for step in range(a.updates):
        last_step = step + 1
        opt.zero_grad(set_to_none=True)
        agg = np.zeros(3, dtype=np.float64)
        for k in range(a.grad_accum):
            j = step * a.grad_accum + k
            pi = int(schedule[j])
            unique_seen.add(pi)
            micro_seen += 1
            b = pair_bank[pi]
            q_v = model(b['X'], b['Y0'], b['R'], geom=geom,
                        evidence_g64=b['evidence_g64'])
            q_full = expand_gate(q_v, geom)
            loss, gate_t, out_t = verifier_loss(
                q_v, b['q_star'], b['mask'], q_full, b['Y0'], b['H'], b['D'],
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
                       q_opt_mean=float(b['q_star'].mean()))
            hist.append(row)
            log('  step %5d/%d  loss %.5f  uniq %d/%d  %.0fs'
                % (step, a.updates, row['loss'], len(unique_seen), n_pairs,
                   time.time() - t0))

        if last_step in ckpt_set:
            save_ckpt(last_step)
            eval_from_bank(last_step)
            if early:
                break

    if last_step not in ckpt_set:
        save_ckpt(last_step)
        eval_from_bank(last_step)

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
        history=hist, early_overfit_success=early,
        wall_seconds=time.time() - t0,
        unique_pairs_final=len(unique_seen),
        overfit_checkpoints=[r['step'] for r in overfit_rows],
    ))
    slim = [dict(step=r['step'], overall=r['overall'], by_state=r['by_state'])
            for r in overfit_rows]
    dump_json(os.path.join(arm_dir, 'overfit_summaries.json'), slim)
    log('done %s in %.1fs (early=%s)' % (a.arm, time.time() - t0, early))
    logf.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
