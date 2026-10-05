#!/usr/bin/env python
"""V4.0 train: A0 Y0-canvas or A1 Ref-canvas. Same head. MSE only."""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import EVAL_QUERY_CHUNK                       # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion                  # noqa: E402
from model.V4RefCanvas import (CANVAS_A0, CANVAS_A1, DIR_A0, DIR_A1,    # noqa: E402
                               features_a0, features_a1)
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_proposal, load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import STATES, bit_equal, snapshot_, state_dict_sha   # noqa: E402
from v3a5c_runtime import make_pair_schedule                            # noqa: E402
from v3a6_runtime import file_sha256, git_head, hard_verify_lock        # noqa: E402
from v3b_runtime import (CKPT_STEPS, DEFAULT_UPDATES, GRAD_ACCUM, LR,   # noqa: E402
                         SEED, MapCache, b0_loss, proposal_core)
from v4_runtime import (ARM_A0, ARM_A1, ARMS, FORMAL_LOCK_KEYS_V4,      # noqa: E402
                        arm_meta)

SRC = '/root/data/experiments/v3a1_lolv2real'
ROOT = '/root/data/experiments/v4_ref_canvas'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--arm', required=True, choices=list(ARMS))
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
    ap.add_argument('--cache_max_gb', type=float, default=16.0)
    ap.add_argument('--start_step', type=int, default=0)
    a = ap.parse_args(_CLI)
    if a.smoke:
        a.formal = False
        if a.updates == DEFAULT_UPDATES:
            a.updates = 8
        if not a.limit:
            a.limit = 4
    if a.formal and not a.smoke and a.start_step > 0:
        raise SystemExit('formal V4 forbids start_step>0 (no Adam state restore)')

    lock_path = os.path.join(a.root, 'artifact_lock.json')
    lock = json.load(open(lock_path))
    lock_mtime = os.stat(lock_path).st_mtime
    missing = [k for k in FORMAL_LOCK_KEYS_V4 if k not in lock]
    if missing and a.formal:
        raise SystemExit('lock missing %s' % missing)
    live_head = git_head()
    if a.formal and not a.smoke and live_head != lock['repo_commit']:
        raise SystemExit('repo HEAD drift: live=%s lock=%s'
                         % (live_head, lock['repo_commit']))
    direction, canvas = arm_meta(a.arm)
    hard_verify_lock(lock, dict(
        official_test_allowed=False,
        seed=a.seed,
        pair_schedule_seed=a.seed,
        optimizer='Adam',
        lr=a.lr if not a.smoke else lock['lr'],
        updates=a.updates if not a.smoke else lock['updates'],
        grad_accum=a.grad_accum if not a.smoke else lock['grad_accum'],
        proposal_sha256=file_sha256(lock['proposal_ckpt']),
        split_sha256=file_sha256(lock['split_json']),
        mismatch_train_sha256=file_sha256(lock['mismatch_train']),
        mismatch_dev_sha256=file_sha256(lock['mismatch_dev']),
        reference_variant=lock['reference_variant'],
        direction_A0=DIR_A0,
        direction_A1=DIR_A1,
        canvas_A0=CANVAS_A0,
        canvas_A1=CANVAS_A1,
        shared_head_init_sha=lock['shared_head_init_sha'],
        a0_init_sha=lock['a0_init_sha'],
        a1_init_sha=lock['a1_init_sha'],
    ), formal=a.formal and not a.smoke)

    arm_dir = os.path.join(a.root, a.arm)
    ckpt_dir = os.path.join(arm_dir, 'smoke_checkpoints' if a.smoke else 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    formal_20k = os.path.join(arm_dir, 'checkpoints', 'ckpt_020000.pt')
    if a.formal and not a.smoke and a.start_step == 0 and os.path.isfile(formal_20k):
        raise SystemExit('formal 20k exists; refuse overwrite: %s' % formal_20k)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    logf = open(os.path.join(a.root, 'logs',
                             'train_%s%s.log' % (a.arm, '_smoke' if a.smoke else '')),
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
                              key='%s|%s|%s' % (a.arm, row[0], state)))
    log('V4.0 train %s  dir=%s canvas=%s  pairs=%d  updates=%d'
        % (a.arm, direction, canvas, len(pairs), a.updates))

    wrapper = load_proposal(lock['proposal_ckpt'], a.device)
    core = proposal_core(wrapper)
    core.match.chunk = EVAL_QUERY_CHUNK
    for p in wrapper.parameters():
        p.requires_grad_(False)
    wrapper.eval()
    snap_enc = snapshot_(core.encoder)
    snap_match = snapshot_(core.match)

    model = V3B0ResidualFusion(in_ch=96).to(a.device)
    init = torch.load(lock['init_paths'][a.arm], map_location='cpu')
    model.load_state_dict(init['model'], strict=True)
    init_sha = state_dict_sha(init['model'])
    if a.formal and init_sha != lock['init_shas'][a.arm]:
        raise SystemExit('init_sha mismatch')
    if a.formal and init_sha != lock['shared_head_init_sha']:
        raise SystemExit('shared_head_init_sha mismatch')

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=0.0)
    schedule = make_pair_schedule(a.updates, a.grad_accum, len(pairs), a.seed)
    cache = MapCache(max_gb=a.cache_max_gb)

    def get_ft(t, key):
        if a.arm == ARM_A0:
            return cache.get(key, lambda: features_a0(wrapper, t['Y0'], t['R']),
                             a.device)
        return cache.get(key, lambda: features_a1(wrapper, t['Y0'], t['R']),
                         a.device)

    def predict_y(t, q, src_t):
        delta = model(q, src_t, t['Y0'].shape[-2:], E=None)
        if a.arm == ARM_A0:
            return t['Y0'].detach() + delta, delta
        return t['R'].detach() + delta, delta

    def save_ckpt(step):
        blob = dict(
            model=model.state_dict(), step=int(step), arm=a.arm,
            objective='mse_reconstruction', seed=a.seed, init_sha=init_sha,
            shared_head_init_sha=lock['shared_head_init_sha'],
            proposal_sha=lock['proposal_sha256'], repo_commit=git_head(),
            direction=direction, canvas=canvas,
            updates=a.updates, grad_accum=a.grad_accum, lr=a.lr,
        )
        path = os.path.join(ckpt_dir, 'ckpt_%06d.pt' % step)
        torch.save(blob, path)
        torch.save(blob, os.path.join(ckpt_dir, 'last.pt'))
        return path

    if a.start_step <= 0:
        save_ckpt(0)
        t0 = sample_tensors(ds, pairs[0]['i'], pairs[0]['state'], a.device)
        q, src_t = get_ft(t0, pairs[0]['key'])
        y, delta = predict_y(t0, q.detach(), src_t.detach())
        canvas_t = t0['Y0'] if a.arm == ARM_A0 else t0['R']
        if float((y - canvas_t).abs().max()) > 1e-6:
            raise SystemExit('%s step0 Y != canvas max_abs=%.3e'
                             % (a.arm, float((y - canvas_t).abs().max())))
        if float(delta.abs().max()) > 1e-6:
            raise SystemExit('%s step0 Δ not zero' % a.arm)
        loss0 = b0_loss(y, t0['H'])
        loss0.backward()
        g = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g += float(p.grad.abs().sum())
        opt.zero_grad(set_to_none=True)
        if g <= 0:
            raise SystemExit('grad is zero')
        log('  grad_sanity %.3e' % g)

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
            q, src_t = get_ft(t, p['key'])
            y, _d = predict_y(t, q.detach(), src_t.detach())
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
    log('DONE %s' % a.arm)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
