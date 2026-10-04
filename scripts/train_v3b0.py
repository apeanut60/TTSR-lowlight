#!/usr/bin/env python
"""V3-B.0 train: MSE(Y0+ΔY, H); only residual head gets grad."""

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

from local_refine_runtime import EVAL_QUERY_CHUNK                       # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion                  # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_proposal, load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import (STATES, bit_equal, snapshot_, state_dict_sha,  # noqa: E402
                          )
from v3a5c_runtime import make_pair_schedule                            # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head, hard_verify_lock  # noqa: E402
from v3b_runtime import (ARM, CKPT_STEPS, DEFAULT_UPDATES, FORMAL_LOCK_KEYS,  # noqa: E402
                         GRAD_ACCUM, LR, SEED, b0_forward, b0_loss,
                         match_features, proposal_core)

SRC = '/root/data/experiments/v3a1_lolv2real'
ROOT = '/root/data/experiments/v3b0_implicit_residual'


class MapCache(object):
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
            f0, t = produce()
            f0 = f0.detach().to('cpu')
            t = t.detach().to('cpu')
            nbytes = (f0.numel() + t.numel()) * f0.element_size()
            if self.bytes + nbytes <= self.max_bytes:
                self.store[key] = (f0.clone(), t.clone())
                self.bytes += nbytes
            return f0.to(device), t.to(device)
        self.hits += 1
        return v[0].to(device), v[1].to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default='/root/data/experiments/v3a4_lolv2real')
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--updates', type=int, default=DEFAULT_UPDATES)
    ap.add_argument('--grad_accum', type=int, default=GRAD_ACCUM)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--formal', action='store_true', default=True)
    ap.add_argument('--cache_max_gb', type=float, default=8.0)
    ap.add_argument('--start_step', type=int, default=0)
    a = ap.parse_args(_CLI)
    if a.smoke:
        a.formal = False
        if a.updates == DEFAULT_UPDATES:
            a.updates = 8
        if not a.limit:
            a.limit = 4

    lock_path = os.path.join(a.root, 'artifact_lock.json')
    lock = json.load(open(lock_path))
    lock_mtime = os.stat(lock_path).st_mtime
    missing = [k for k in FORMAL_LOCK_KEYS if k not in lock]
    if missing and a.formal:
        raise SystemExit('lock missing %s' % missing)
    hard_verify_lock(lock, dict(
        official_test_allowed=False,
        architecture='V3B0ResidualFusion',
        arm=ARM,
        seed=a.seed,
        lr=a.lr if not a.smoke else lock['lr'],
        grad_accum=a.grad_accum if not a.smoke else lock['grad_accum'],
        proposal_sha256=file_sha256(lock['proposal_ckpt']),
        split_sha256=file_sha256(lock['split_json']),
        mismatch_train_sha256=file_sha256(lock['mismatch_train']),
        mismatch_dev_sha256=file_sha256(lock['mismatch_dev']),
        reference_variant=lock['reference_variant'],
    ), formal=a.formal and not a.smoke)

    os.makedirs(os.path.join(a.root, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    logf = open(os.path.join(a.root, 'logs',
                             'train_b0%s.log' % ('_smoke' if a.smoke else '')),
                'a' if a.start_step > 0 else 'w', encoding='utf-8')

    def log(msg=''):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = lock['reference_variant']
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       lock['split_json'])
    train_rows = splits['train'][:a.limit] if a.limit else splits['train']
    mmap = json.load(open(lock['mismatch_train']))
    ds = make_dataset(
        ns, train_rows,
        os.path.join(a.src_root, lock.get('cache_name', 'cache_y0_lolbase'),
                     'refiner_train'), mmap)
    pairs = []
    for i, row in enumerate(train_rows):
        for state in STATES:
            pairs.append(dict(i=i, name=row[0], state=state,
                              key='%s|%s' % (row[0], state)))
    log('V3-B.0 train  pairs=%d  updates=%d' % (len(pairs), a.updates))

    wrapper = load_proposal(lock['proposal_ckpt'], a.device)
    core = proposal_core(wrapper)
    core.match.chunk = EVAL_QUERY_CHUNK
    for p in wrapper.parameters():
        p.requires_grad_(False)
    wrapper.eval()
    snap_enc = snapshot_(core.encoder)
    snap_match = snapshot_(core.match)

    head = V3B0ResidualFusion().to(a.device)
    init = torch.load(lock['init_path'], map_location='cpu')
    if a.start_step > 0:
        ck = os.path.join(a.root, 'checkpoints', 'ckpt_%06d.pt' % a.start_step)
        blob = torch.load(ck, map_location='cpu')
        if int(blob.get('step', -1)) != int(a.start_step):
            raise SystemExit('resume step mismatch')
        head.load_state_dict(blob['model'], strict=True)
        log('  RESUME %s' % ck)
    else:
        head.load_state_dict(init['model'], strict=True)
    init_sha = state_dict_sha(init['model'])
    if a.formal and init_sha != lock.get('init_sha'):
        raise SystemExit('init_sha mismatch')

    opt = torch.optim.Adam(head.parameters(), lr=a.lr, weight_decay=0.0)
    schedule = make_pair_schedule(a.updates, a.grad_accum, len(pairs), a.seed)
    cache = MapCache(max_gb=a.cache_max_gb)

    def get_ft(t, key):
        return cache.get(
            key,
            lambda: match_features(wrapper, t['Y0'], t['R']),
            a.device)

    def save_ckpt(step):
        blob = dict(
            model=head.state_dict(), step=int(step), arm=ARM,
            objective='mse_reconstruction', seed=a.seed, init_sha=init_sha,
            proposal_sha=lock['proposal_sha256'], repo_commit=git_head(),
            updates=a.updates, grad_accum=a.grad_accum, lr=a.lr,
        )
        path = os.path.join(a.root, 'checkpoints', 'ckpt_%06d.pt' % step)
        torch.save(blob, path)
        torch.save(blob, os.path.join(a.root, 'checkpoints', 'last.pt'))
        return path

    if a.start_step <= 0:
        save_ckpt(0)
        t0 = sample_tensors(ds, pairs[0]['i'], pairs[0]['state'], a.device)
        f0, tt = get_ft(t0, pairs[0]['key'])
        y, d0 = b0_forward(head, f0.detach(), tt.detach(), t0['Y0'].detach())
        if float(d0.abs().max()) > 1e-6:
            raise SystemExit('step0 ΔY not ~0: %.3e' % float(d0.abs().max()))
        loss0 = b0_loss(y, t0['H'])
        loss0.backward()
        g = float(head.out.weight.grad.abs().sum()) if head.out.weight.grad is not None else 0.0
        opt.zero_grad(set_to_none=True)
        if g <= 0:
            raise SystemExit('head grad is zero')
        log('  grad_sanity out=%.3e' % g)

    t_wall = time.time()
    running = 0.0
    n_run = 0
    cursor = int(a.start_step) * int(a.grad_accum)
    for step in range(a.start_step + 1, a.updates + 1):
        head.train()
        opt.zero_grad(set_to_none=True)
        for _ in range(a.grad_accum):
            p = pairs[int(schedule[cursor])]; cursor += 1
            t = sample_tensors(ds, p['i'], p['state'], a.device)
            f0, tt = get_ft(t, p['key'])
            y, _ = b0_forward(head, f0.detach(), tt.detach(), t['Y0'].detach())
            loss = b0_loss(y, t['H'])
            (loss / a.grad_accum).backward()
            running += float(loss.detach())
            n_run += 1
        opt.step()
        if step % a.log_every == 0 or step == 1 or step == a.updates:
            log('  step %d  loss=%.5f  cache=%s  (%.0fs)'
                % (step, running / max(n_run, 1),
                   dict(hits=cache.hits, misses=cache.misses,
                        gb=round(cache.bytes / float(1 << 30), 2)),
                   time.time() - t_wall))
            running = 0.0
            n_run = 0
        if step in CKPT_STEPS or step == a.updates:
            save_ckpt(step)

    if not bit_equal(snap_enc, snapshot_(core.encoder)):
        raise SystemExit('encoder mutated')
    if not bit_equal(snap_match, snapshot_(core.match)):
        raise SystemExit('match mutated')
    if os.stat(lock_path).st_mtime != lock_mtime:
        raise SystemExit('lock mutated during train')
    log('DONE B0')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
