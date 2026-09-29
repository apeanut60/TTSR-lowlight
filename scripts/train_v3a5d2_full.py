#!/usr/bin/env python
"""V3-A.5D2-full train: A0_control or A1_multiscale on train575 (gate-only)."""

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

from local_refine_runtime import metrics                                # noqa: E402
from model.V3A5D2Verifier import ARMS, V3A5D2Verifier                    # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import (correction, load_proposal, load_rows,        # noqa: E402
                           make_dataset, sample_tensors)
from v3a5_runtime import (STATES, STATE_PROBS, action_optimal_target,   # noqa: E402
                          bit_equal, block_energy, energy_mask,
                          expand_gate, parameter_l1_drift, prepare_geometry,
                          sample_states, snapshot_, state_dict_sha,
                          target_geometry, verifier_loss)
from v3a5c_runtime import dump_json                                     # noqa: E402
from v3a5d2_runtime import OUT_WEIGHT                                   # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
ROOT = '/root/data/experiments/v3a5d2_rf_full'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'


def lr_at(step, lr0=1e-4, lr1=5e-5, switch=2000):
    return lr0 if step < switch else lr1


class CorrectionCache(object):
    def __init__(self, max_gb=8.0):
        self.store = {}
        self.hits = 0
        self.misses = 0
        self.bytes = 0
        self.max_bytes = int(float(max_gb) * (1 << 30))

    def get(self, key, produce, device):
        v = self.store.get(key)
        if v is None:
            self.misses += 1
            v = produce().detach().to('cpu')
            if self.bytes + v.numel() * v.element_size() <= self.max_bytes:
                self.store[key] = v.clone()
                self.bytes += v.numel() * v.element_size()
        else:
            self.hits += 1
        return v.to(device)

    def stats(self):
        return dict(entries=len(self.store), hits=self.hits, misses=self.misses,
                    gigabytes=round(self.bytes / float(1 << 30), 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--arm', required=True, choices=list(ARMS))
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--cache_d_max_gb', type=float, default=8.0)
    a = ap.parse_args(_CLI)

    if a.limit and a.steps == 3000:
        a.steps = max(4, a.limit * 2)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    thr = float(lock['energy_threshold'])
    out_w = OUT_WEIGHT
    arm_dir = os.path.join(a.root, a.arm)
    os.makedirs(os.path.join(arm_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    logf = open(os.path.join(a.root, 'logs', 'train_%s.log' % a.arm), 'w',
                encoding='utf-8')

    def log(msg=''):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    log('V3-A.5D2-full train %s  out_weight=%.3f  steps=%d  accum=%d'
        % (a.arm, out_w, a.steps, a.grad_accum))
    log('  energy thr=%.6e  seed=%d  states p=%s'
        % (thr, a.seed, STATE_PROBS))

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    all_rows = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                         os.path.join(a.v4_root, 'splits', 'split.json'))
    train_rows = all_rows['train']
    mmap_train = json.load(open(os.path.join(
        a.v4_root, 'mappings', 'mismatch_train_575.json'), encoding='utf-8'))
    ds = make_dataset(ns, train_rows,
                      os.path.join(a.src_root, a.cache_name, 'refiner_train'),
                      mmap_train)
    n_train = len(train_rows) if not a.limit else min(a.limit, len(train_rows))

    proposal = load_proposal(os.path.join(a.src_root, R1_CK), a.device)
    prop_before = snapshot_(proposal.proposal)

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
            raise SystemExit('A0/A1 common net not bit-equal')
        if model.context_residual_max_abs() != 0.0:
            raise SystemExit('context residual not zero at init')
    log('  init_sha=%s  ctx_res=%g' % (init_sha[:16], model.context_residual_max_abs()))

    geom = prepare_geometry(target_geometry(400, 600, 'g64'), a.device)

    # step0 check
    model.eval()
    with torch.no_grad():
        t = sample_tensors(ds, 0, 'correct', a.device)
        q0 = model(t['X'], t['Y0'], t['R'], geom=geom)
        q0m = float(q0.mean())
    if abs(q0m - 0.5) > 1e-5:
        raise SystemExit('step0 q mean=%.6f want 0.5' % q0m)
    log('  step0 q_mean=%.6f OK' % q0m)

    d_cache = CorrectionCache(a.cache_d_max_gb)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    model.train()
    rng = np.random.default_rng(int(a.seed))
    idx_seq = rng.integers(0, n_train, size=a.steps * a.grad_accum)
    state_seq = sample_states(a.steps * a.grad_accum, int(a.seed) + 1)
    hist = []
    t0 = time.time()

    for step in range(a.steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g['lr'] = lr
        opt.zero_grad(set_to_none=True)
        agg = np.zeros(3, dtype=np.float64)
        for k in range(a.grad_accum):
            j = step * a.grad_accum + k
            i, state = int(idx_seq[j]), state_seq[j]
            t = sample_tensors(ds, i, state, a.device)
            with torch.no_grad():
                D = d_cache.get(
                    (i, state),
                    lambda: correction(proposal.proposal, t['Y0'], t['R'])[0],
                    a.device)
                tgt = action_optimal_target(t['Y0'], t['H'], D, geom)
                mask = energy_mask(block_energy(D, geom), thr)
            q_v = model(t['X'], t['Y0'], t['R'], geom=geom)
            q_full = expand_gate(q_v, geom)
            loss, gate_t, out_t = verifier_loss(
                q_v, tgt['q_grid'], mask, q_full, t['Y0'], t['H'], D,
                out_weight=out_w)
            (loss / a.grad_accum).backward()
            agg += np.array([float(loss), float(gate_t), float(out_t)])
        opt.step()
        agg /= a.grad_accum

        if step % a.log_every == 0 or step == a.steps - 1:
            row = dict(step=step, loss=float(agg[0]), gate=float(agg[1]),
                       out=float(agg[2]), lr=lr,
                       q_v_mean=float(q_v.mean()),
                       q_opt_mean=float(tgt['q_grid'].mean()),
                       mask_valid=float(mask.mean()))
            hist.append(row)
            log('  step %5d/%d  loss %.5f (gate %.5f)  lr %.1e  '
                'q_v %.3f  %.0fs'
                % (step, a.steps, row['loss'], row['gate'], lr,
                   row['q_v_mean'], time.time() - t0))

        if step + 1 in (1000, 2000, a.steps):
            blob = dict(model=model.state_dict(), step=step + 1, arm=a.arm,
                        out_weight=out_w, energy_threshold=thr, seed=a.seed,
                        init_sha=init_sha, limit=a.limit)
            path = os.path.join(arm_dir, 'checkpoints',
                                'ckpt_%06d.pt' % (step + 1))
            torch.save(blob, path)
            torch.save(blob, os.path.join(arm_dir, 'checkpoints', 'last.pt'))
            log('  checkpoint -> %s' % path)

    drift = parameter_l1_drift(prop_before, snapshot_(proposal.proposal))
    if drift != 0.0:
        raise SystemExit('proposal drifted %g' % drift)
    dump_json(os.path.join(arm_dir, 'train_metrics.json'), dict(
        arm=a.arm, steps=a.steps, grad_accum=a.grad_accum, seed=a.seed,
        out_weight=out_w, energy_threshold=thr, init_sha=init_sha,
        history=hist, wall_seconds=time.time() - t0,
        correction_cache=d_cache.stats(), proposal_drift=drift,
        n_train=n_train, limit=a.limit,
    ))
    log('done %s in %.1fs (cache=%s)' % (a.arm, time.time() - t0, d_cache.stats()))
    logf.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
