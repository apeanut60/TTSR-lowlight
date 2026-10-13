#!/usr/bin/env python
"""V8-B0 Gate1 train — B0-Self or B0-Nano, correct-only exclusive-32, MSE.

Same seed / init / pair schedule across arms; only Ref source differs.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

REPO = '/root/projects/TTSR-lowlight'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import EVAL_QUERY_CHUNK  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion  # noqa: E402
from option import parser as option_parser  # noqa: E402
from v3a5_pipeline import load_proposal, load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import bit_equal, snapshot_, state_dict_sha  # noqa: E402
from v3a5c_runtime import make_pair_schedule  # noqa: E402
from v3a6_runtime import file_sha256, git_head  # noqa: E402
from v3b_runtime import MapCache, b0_forward, b0_loss, match_features, proposal_core  # noqa: E402
from v8_b0_runtime import (  # noqa: E402
    ARM_NANO, ARM_SELF, GRAD_ACCUM, LR, ROOT, SEED, SRC, UPDATES_GATE1,
    Y0_CACHE, arm_root)


def _sample(ds, i, arm, device):
    """correct-state tensors; Self arm forces R=Y0."""
    t = sample_tensors(ds, i, 'correct', device)
    if arm == ARM_SELF:
        t = dict(t)
        t['R'] = t['Y0'].clone()
    return t


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--arm', required=True, choices=[ARM_SELF, ARM_NANO])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--updates', type=int, default=UPDATES_GATE1)
    ap.add_argument('--grad_accum', type=int, default=GRAD_ACCUM)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--log_every', type=int, default=50)
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    if lock.get('official_test_allowed') is not False:
        raise SystemExit('official test must be forbidden')
    if a.seed != lock['seed'] or a.lr != lock['lr']:
        raise SystemExit('seed/lr drift vs lock')
    if a.updates != lock['updates']:
        raise SystemExit('updates drift vs lock')

    arm_dir = arm_root(a.arm, a.root)
    os.makedirs(os.path.join(arm_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(arm_dir, 'logs'), exist_ok=True)
    logf = open(os.path.join(arm_dir, 'logs', 'train.log'), 'w', encoding='utf-8')

    def log(msg=''):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    train_ids = list(lock['train_ids'])
    splits = load_rows(os.path.join(SRC, 'manifests', 'refiner_train.csv'),
                       lock['split_json'])
    name_to = {r[0]: r for r in splits['train']}
    train_rows = [name_to[n] for n in train_ids]
    ns = option_parser.parse_args([])
    ns.dataset_dir = '/root/data/datasets/lol-v2-real'
    ns.v3a_ref_variant = lock['reference_variant']
    mmap = json.load(open(lock['mismatch_train']))
    ds = make_dataset(
        ns, train_rows,
        os.path.join(Y0_CACHE, 'refiner_train'), mmap)
    pairs = [dict(i=i, name=train_ids[i], state='correct',
                  key='%s|correct|%s' % (train_ids[i], a.arm))
             for i in range(len(train_ids))]
    log('V8-B0 train arm=%s n=%d updates=%d seed=%d'
        % (a.arm, len(pairs), a.updates, a.seed))

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
    head.load_state_dict(init['model'], strict=True)
    init_sha = state_dict_sha(init['model'])
    if init_sha != lock['init_sha']:
        raise SystemExit('init_sha mismatch')

    opt = torch.optim.Adam(head.parameters(), lr=a.lr, weight_decay=0.0)
    schedule = make_pair_schedule(a.updates, a.grad_accum, len(pairs), a.seed)
    cache = MapCache(max_gb=4.0)

    def get_ft(t, key):
        return cache.get(
            key,
            lambda: match_features(wrapper, t['Y0'], t['R']),
            a.device)

    def save_ckpt(step):
        blob = dict(
            model=head.state_dict(), step=int(step), arm=a.arm,
            objective='mse_reconstruction', seed=a.seed, init_sha=init_sha,
            proposal_sha=lock['proposal_sha256'], repo_commit=git_head(),
            updates=a.updates, grad_accum=a.grad_accum, lr=a.lr,
            train_ids=train_ids,
        )
        path = os.path.join(arm_dir, 'checkpoints', 'ckpt_%06d.pt' % step)
        torch.save(blob, path)
        torch.save(blob, os.path.join(arm_dir, 'checkpoints', 'last.pt'))
        return path

    save_ckpt(0)
    t0 = _sample(ds, pairs[0]['i'], a.arm, a.device)
    f0, tt = get_ft(t0, pairs[0]['key'])
    y, d0 = b0_forward(head, f0.detach(), tt.detach(), t0['Y0'].detach())
    if float(d0.abs().max()) > 1e-6:
        raise SystemExit('step0 ΔY not ~0')
    loss0 = b0_loss(y, t0['H'])
    loss0.backward()
    g = float(head.out.weight.grad.abs().sum())
    opt.zero_grad(set_to_none=True)
    if g <= 0:
        raise SystemExit('grad smoke fail')
    log('  step0 ok  grad_out=%.3e' % g)

    t_wall = time.time()
    running = 0.0
    n_run = 0
    cursor = 0
    for step in range(1, a.updates + 1):
        head.train()
        opt.zero_grad(set_to_none=True)
        for _ in range(a.grad_accum):
            p = pairs[int(schedule[cursor])]
            cursor += 1
            t = _sample(ds, p['i'], a.arm, a.device)
            f0, tt = get_ft(t, p['key'])
            y, _ = b0_forward(head, f0.detach(), tt.detach(), t['Y0'].detach())
            loss = b0_loss(y, t['H'])
            (loss / a.grad_accum).backward()
            running += float(loss.detach())
            n_run += 1
            for p_e in core.encoder.parameters():
                if p_e.grad is not None:
                    raise SystemExit('encoder grad')
            for p_m in core.match.parameters():
                if p_m.grad is not None:
                    raise SystemExit('match grad')
        opt.step()
        if step % a.log_every == 0 or step == 1 or step == a.updates:
            log('  step %d  loss=%.5f  (%.0fs)'
                % (step, running / max(n_run, 1), time.time() - t_wall))
            running = 0.0
            n_run = 0
        if step in (100, 200, 400) or step == a.updates:
            log('  saved %s' % save_ckpt(step))

    if not bit_equal(snap_enc, snapshot_(core.encoder)):
        raise SystemExit('encoder mutated')
    if not bit_equal(snap_match, snapshot_(core.match)):
        raise SystemExit('match mutated')
    # fixed-batch MSE at end
    head.eval()
    mses = []
    with torch.no_grad():
        for p in pairs:
            t = _sample(ds, p['i'], a.arm, a.device)
            f0, tt = get_ft(t, p['key'])
            y, _ = b0_forward(head, f0, tt, t['Y0'])
            mses.append(float(b0_loss(y, t['H'])))
    batch_mse = float(sum(mses) / len(mses))
    meta = dict(
        arm=a.arm, batch_mse_train32=batch_mse,
        per_image_mse=mses, updates=a.updates, seed=a.seed,
        init_sha=init_sha, schedule_seed=a.seed,
        proposal_sha=file_sha256(lock['proposal_ckpt']),
        repo_commit=git_head(),
    )
    with open(os.path.join(arm_dir, 'train_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    log('DONE arm=%s batch_mse=%.6f' % (a.arm, batch_mse))
    logf.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
