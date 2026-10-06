#!/usr/bin/env python
"""V5.0 train: Match/Texture/Align/Refine only. Frozen Base. MSE. 30k formal."""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import load_frozen_n0                             # noqa: E402
from model.V5Model import V5Model                                           # noqa: E402
from model.V5RetinexBridge import INJECTION_POINT, tiled_v5_forward         # noqa: E402
from option import parser as option_parser                                  # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors           # noqa: E402
from v3a5_runtime import STATES, bit_equal, snapshot_, state_dict_sha       # noqa: E402
from v3a5c_runtime import make_pair_schedule                                # noqa: E402
from v3a6_runtime import file_sha256, git_head, hard_verify_lock            # noqa: E402
from v3b_runtime import GRAD_ACCUM, LR, SEED, b0_loss                       # noqa: E402
from v5_runtime import (ARM_A1, CKPT_STEPS, DEFAULT_UPDATES,                 # noqa: E402
                        FORMAL_LOCK_KEYS_V5, LOSS, OPTIMIZER)

SRC = '/root/data/experiments/v3a1_lolv2real'
ROOT = '/root/data/experiments/v5_aligned_ref'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
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
    ap.add_argument('--start_step', type=int, default=0)
    a = ap.parse_args(_CLI)
    if a.smoke:
        a.formal = False
        if a.updates == DEFAULT_UPDATES:
            a.updates = 8
        if not a.limit:
            a.limit = 4
    if a.formal and not a.smoke and a.start_step > 0:
        raise SystemExit('formal V5.0 forbids start_step>0 (no Adam state restore)')

    lock_path = os.path.join(a.root, 'artifact_lock.json')
    lock = json.load(open(lock_path))
    lock_mtime = os.stat(lock_path).st_mtime
    missing = [k for k in FORMAL_LOCK_KEYS_V5 if k not in lock]
    if missing and a.formal:
        raise SystemExit('lock missing %s' % missing)
    live_head = git_head()
    if a.formal and not a.smoke and live_head != lock['repo_commit']:
        raise SystemExit('repo HEAD drift: live=%s lock=%s'
                         % (live_head, lock['repo_commit']))
    hard_verify_lock(lock, dict(
        official_test_allowed=False,
        frozen_base=True,
        zero_init=True,
        injection_point=INJECTION_POINT,
        loss=LOSS,
        optimizer=OPTIMIZER,
        seed=a.seed,
        pair_schedule_seed=a.seed,
        lr=a.lr if not a.smoke else lock['lr'],
        updates=a.updates if not a.smoke else lock['updates'],
        grad_accum=a.grad_accum if not a.smoke else lock['grad_accum'],
        split_sha256=file_sha256(lock['split_json']),
        mismatch_train_sha256=file_sha256(lock['mismatch_train']),
        mismatch_dev_sha256=file_sha256(lock['mismatch_dev']),
        reference_variant=lock['reference_variant'],
        v5_init_sha=lock['v5_init_sha'],
        base_ckpt_sha256=file_sha256(lock['base_ckpt']),
    ), formal=a.formal and not a.smoke)

    ckpt_dir = os.path.join(a.root, ARM_A1, 'smoke_checkpoints' if a.smoke else 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    formal_30k = os.path.join(a.root, ARM_A1, 'checkpoints', 'ckpt_030000.pt')
    if a.formal and not a.smoke and a.start_step == 0 and os.path.isfile(formal_30k):
        raise SystemExit('formal 30k exists; refuse overwrite: %s' % formal_30k)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    logf = open(os.path.join(a.root, 'logs',
                             'train_%s%s.log' % (ARM_A1, '_smoke' if a.smoke else '')),
                'w', encoding='utf-8')

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
    log('V5.0 train %s  pairs=%d  updates=%d' % (ARM_A1, len(pairs), a.updates))

    n0, _trainer, _cfg = load_frozen_n0(
        lock['base_ckpt'], lock['base_run_dir'], a.device)
    mainnet = n0.MainNet
    for p in mainnet.parameters():
        p.requires_grad_(False)
    mainnet.eval()
    snap_base = snapshot_(mainnet)

    model = V5Model().to(a.device)
    init = torch.load(lock['v5_init_path'], map_location='cpu')
    model.load_state_dict(init['model'], strict=True)
    init_sha = state_dict_sha(init['model'])
    if a.formal and init_sha != lock['v5_init_sha']:
        raise SystemExit('v5 init_sha mismatch')

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=0.0)
    schedule = make_pair_schedule(a.updates, a.grad_accum, len(pairs), a.seed)

    def save_ckpt(step):
        blob = dict(
            model=model.state_dict(), step=int(step), arm=ARM_A1,
            objective='mse_reconstruction', seed=a.seed, init_sha=init_sha,
            repo_commit=git_head(), injection_point=INJECTION_POINT,
            frozen_base=True, loss=LOSS, updates=a.updates,
            grad_accum=a.grad_accum, lr=a.lr,
        )
        path = os.path.join(ckpt_dir, 'ckpt_%06d.pt' % step)
        torch.save(blob, path)
        torch.save(blob, os.path.join(ckpt_dir, 'last.pt'))
        return path

    if a.start_step <= 0:
        save_ckpt(0)
        t0 = sample_tensors(ds, pairs[0]['i'], pairs[0]['state'], a.device)
        y = tiled_v5_forward(mainnet, model, t0['X'], t0['Y0'], t0['R'])
        if float((y - t0['Y0']).abs().max()) > 1e-4:
            raise SystemExit('A1 step0 Y != Y0 max_abs=%.3e'
                             % float((y - t0['Y0']).abs().max()))
        loss0 = b0_loss(y, t0['H'])
        loss0.backward()
        gsum = 0.0
        for p in model.parameters():
            if p.grad is not None:
                gsum += float(p.grad.abs().sum())
        for p in mainnet.parameters():
            if p.grad is not None:
                raise SystemExit('frozen Base received grad')
        opt.zero_grad(set_to_none=True)
        if gsum <= 0:
            raise SystemExit('Ref branch grad is zero')
        log('  grad_sanity %.3e' % gsum)

    t_wall = time.time()
    running = 0.0
    n_run = 0
    cursor = 0
    for step in range(1, a.updates + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        for _ in range(a.grad_accum):
            p = pairs[int(schedule[cursor])]; cursor += 1
            t = sample_tensors(ds, p['i'], p['state'], a.device)
            y = tiled_v5_forward(mainnet, model, t['X'], t['Y0'], t['R'])
            loss = b0_loss(y, t['H'])
            (loss / a.grad_accum).backward()
            running += float(loss.detach())
            n_run += 1
        opt.step()
        if step % a.log_every == 0 or step == 1 or step == a.updates:
            log('  step %d  loss=%.5f  (%.0fs)'
                % (step, running / max(n_run, 1), time.time() - t_wall))
            running = 0.0
            n_run = 0
        if step in CKPT_STEPS or step == a.updates:
            save_ckpt(step)

    if not bit_equal(snap_base, snapshot_(mainnet)):
        raise SystemExit('Base mutated')
    if os.stat(lock_path).st_mtime != lock_mtime:
        raise SystemExit('lock mutated during train')
    log('DONE %s' % ARM_A1)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
