#!/usr/bin/env python
"""V3-A.7 train: A1_utility_bce on train575 (equal-freq pairs)."""

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

from model.V3A6DecisionVerifier import build_v3a6_model                  # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import (correction, load_proposal, load_rows,        # noqa: E402
                           make_dataset, sample_tensors)
from v3a5_runtime import (STATES, bit_equal, block_energy, energy_mask, # noqa: E402
                          prepare_geometry, snapshot_, state_dict_sha,
                          target_geometry)
from v3a5c_runtime import make_pair_schedule                            # noqa: E402
from v3a6_runtime import CKPT_STEPS, DEFAULT_UPDATES, GRAD_ACCUM, LR, SEED  # noqa: E402
from v3a7_runtime import (ARMS, FORMAL_LOCK_KEYS, OBJECTIVES, arm_loss,  # noqa: E402
                          dump_json, file_sha256, git_head,
                          hard_verify_lock, masked_fraction)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
ROOT = '/root/data/experiments/v3a7_utility_gate'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
REF_H, REF_W = 400, 600


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
            nbytes = v.numel() * v.element_size()
            if self.bytes + nbytes <= self.max_bytes:
                self.store[key] = v.clone()
                self.bytes += nbytes
        else:
            self.hits += 1
        return v.to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--arm', default='A1_utility_bce', choices=list(ARMS))
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--updates', type=int, default=DEFAULT_UPDATES)
    ap.add_argument('--grad_accum', type=int, default=GRAD_ACCUM)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--formal', action='store_true', default=True)
    ap.add_argument('--cache_d_max_gb', type=float, default=8.0)
    ap.add_argument('--start_step', type=int, default=0)
    a = ap.parse_args(_CLI)
    if a.smoke:
        a.formal = False
        if a.updates == DEFAULT_UPDATES:
            a.updates = 8
        if not a.limit:
            a.limit = 4

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    thr = float(lock['energy_threshold'])
    missing = [k for k in FORMAL_LOCK_KEYS if k not in lock]
    if missing and a.formal and not a.smoke:
        raise SystemExit('artifact_lock missing keys: %s' % missing)
    hard_verify_lock(lock, dict(
        energy_threshold=thr, seed=a.seed, pair_schedule_seed=a.seed,
        official_test_allowed=False,
        architecture='V3A5D2Verifier.A1_multiscale',
        geometry='g64', bottleneck=64,
        proposal_sha256=file_sha256(lock['proposal_ckpt']),
        split_sha256=file_sha256(lock['split_json']),
        mismatch_train_sha256=file_sha256(lock['mismatch_train']),
        mismatch_dev_sha256=file_sha256(lock['mismatch_dev']),
        reference_variant=a.variant,
        init_sha=lock['init_sha'],
        updates=a.updates, grad_accum=a.grad_accum, lr=a.lr,
        mask_mode='g64_proposal_energy_expand',
    ), formal=a.formal and not a.smoke)

    arm_dir = os.path.join(a.root, a.arm)
    os.makedirs(os.path.join(arm_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    log_name = 'train_%s%s.log' % (a.arm, '_smoke' if a.smoke else '')
    logf = open(os.path.join(a.root, 'logs', log_name),
                'a' if a.start_step > 0 else 'w', encoding='utf-8')

    def log(msg=''):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       os.path.join(a.v4_root, 'splits', 'split.json'))
    train_rows = splits['train'][:a.limit] if a.limit else splits['train']
    mmap = json.load(open(os.path.join(a.v4_root, 'mappings',
                                       'mismatch_train_575.json')))
    ds = make_dataset(
        ns, train_rows,
        os.path.join(a.src_root, a.cache_name, 'refiner_train'), mmap)
    pairs = []
    for i, row in enumerate(train_rows):
        for state in STATES:
            pairs.append(dict(i=i, name=row[0], state=state,
                              key='%s|%s' % (row[0], state)))
    n_pairs = len(pairs)
    log('V3-A.7 train %s  objective=%s  pairs=%d  updates=%d'
        % (a.arm, OBJECTIVES[a.arm], n_pairs, a.updates))

    proposal_path = os.path.join(a.src_root, R1_CK)
    proposal = load_proposal(proposal_path, a.device)
    prop_before = snapshot_(proposal.proposal)
    geom = prepare_geometry(target_geometry(REF_H, REF_W, 'g64'), a.device)

    init = torch.load(os.path.join(a.root, 'init', 'shared_init_s42.pt'),
                      map_location='cpu')
    sd_init = init['A1_decision_mse'] if 'A1_decision_mse' in init else init
    init_sha = state_dict_sha(sd_init)
    model = build_v3a6_model('A1_decision_mse').to(a.device)
    if a.start_step > 0:
        resume_path = os.path.join(arm_dir, 'checkpoints',
                                   'ckpt_%06d.pt' % a.start_step)
        blob = torch.load(resume_path, map_location='cpu')
        if int(blob.get('step', -1)) != int(a.start_step):
            raise SystemExit('resume step mismatch')
        model.load_state_dict(blob['model'], strict=True)
        log('  RESUME from %s' % resume_path)
    else:
        model.load_state_dict(sd_init, strict=True)
    if a.formal and init_sha != lock.get('init_sha'):
        raise SystemExit('init_sha mismatch formal')

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=0.0)
    schedule = make_pair_schedule(a.updates, a.grad_accum, n_pairs, a.seed)
    cursor0 = int(a.start_step) * int(a.grad_accum)
    mask_fracs, pos_fracs = [], []
    dcache = CorrectionCache(max_gb=a.cache_d_max_gb)

    def get_D(t, key):
        return dcache.get(
            key,
            lambda: correction(proposal.proposal, t['Y0'], t['R'])[0],
            a.device)

    def save_ckpt(step):
        blob = dict(
            model=model.state_dict(), step=int(step), arm=a.arm,
            objective=OBJECTIVES[a.arm], seed=a.seed, init_sha=init_sha,
            proposal_sha=lock.get('proposal_sha256') or file_sha256(proposal_path),
            energy_threshold=thr, repo_commit=git_head(),
            updates=a.updates, grad_accum=a.grad_accum, lr=a.lr,
        )
        path = os.path.join(arm_dir, 'checkpoints', 'ckpt_%06d.pt' % step)
        torch.save(blob, path)
        torch.save(blob, os.path.join(arm_dir, 'checkpoints', 'last.pt'))
        return path

    if a.start_step <= 0:
        save_ckpt(0)
        model.train()
        p0 = pairs[int(schedule[0])]
        t0 = sample_tensors(ds, p0['i'], p0['state'], a.device)
        D0 = get_D(t0, p0['key'])
        m0 = energy_mask(block_energy(D0, geom), thr)
        q0 = model(t0['X'], t0['Y0'].detach(), t0['R'].detach(), geom=geom)
        loss0, _ = arm_loss(a.arm, q0, t0['Y0'].detach(), t0['H'], D0, geom, m0)
        loss0.backward()
        g_head = float(model.net.head2.weight.grad.abs().sum()) if \
            model.net.head2.weight.grad is not None else 0.0
        opt.zero_grad(set_to_none=True)
        if not bit_equal(prop_before, snapshot_(proposal.proposal)):
            raise SystemExit('proposal mutated during grad sanity')
        log('  grad_sanity head=%.3e' % g_head)
        if g_head <= 0:
            raise SystemExit('head grad is zero')

    t_wall = time.time()
    running = 0.0
    n_run = 0
    cursor = cursor0
    for step in range(a.start_step + 1, a.updates + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        for _ in range(a.grad_accum):
            p = pairs[int(schedule[cursor])]; cursor += 1
            t = sample_tensors(ds, p['i'], p['state'], a.device)
            D = get_D(t, p['key'])
            y0 = t['Y0'].detach()
            mask = energy_mask(block_energy(D, geom), thr)
            mask_fracs.append(masked_fraction(mask))
            q = model(t['X'], y0, t['R'].detach(), geom=geom)
            loss, aux = arm_loss(a.arm, q, y0, t['H'], D, geom, mask)
            pos_fracs.append(aux['pos_frac'])
            (loss / a.grad_accum).backward()
            running += float(loss.detach())
            n_run += 1
        opt.step()
        if step % a.log_every == 0 or step == 1 or step == a.updates:
            log('  step %d  loss=%.5f  mask=%.3f  posU=%.3f  cache=%s  (%.0fs)'
                % (step, running / max(n_run, 1),
                   float(np.mean(mask_fracs[-a.grad_accum * a.log_every:])),
                   float(np.mean(pos_fracs[-a.grad_accum * a.log_every:])),
                   dict(hits=dcache.hits, misses=dcache.misses,
                        gb=round(dcache.bytes / float(1 << 30), 2)),
                   time.time() - t_wall))
            running = 0.0
            n_run = 0
        if step in CKPT_STEPS or step == a.updates:
            if step != 0:
                save_ckpt(step)

    if not bit_equal(prop_before, snapshot_(proposal.proposal)):
        raise SystemExit('proposal mutated during training')
    dump_json(os.path.join(arm_dir, 'train_mask_stats.json'), dict(
        mean_masked_fraction=float(np.mean(mask_fracs)) if mask_fracs else None,
        mean_pos_utility=float(np.mean(pos_fracs)) if pos_fracs else None,
        n=len(mask_fracs),
    ))
    log('DONE %s' % a.arm)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
