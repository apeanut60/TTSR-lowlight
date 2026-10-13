#!/usr/bin/env python
"""V7 train: Texture+ATT on frozen LLFormer Y0 + frozen C2. L_rec+L_tex L1."""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.td_v6_adapter import V7TaskDecoupledModel, _rgb01  # noqa: E402
from model.td_transfer import td_reconstruction_losses  # noqa: E402
from option import parser as option_parser  # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import snapshot_, state_dict_sha  # noqa: E402
from v3a5c_runtime import make_pair_schedule  # noqa: E402
from v3a6_runtime import file_sha256, git_head, hard_verify_lock  # noqa: E402
from v3b_runtime import GRAD_ACCUM, LR, SEED  # noqa: E402
from v7_runtime import (ARM_A1, CKPT_STEPS, DEFAULT_UPDATES,  # noqa: E402
                        FORMAL_LOCK_KEYS_V7, LOSS, OPTIMIZER, ROOT, SRC)


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
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--formal', action='store_true', default=True)
    ap.add_argument('--start_step', type=int, default=0)
    a = ap.parse_args(_CLI)
    if a.smoke:
        a.formal = False
        if a.updates == DEFAULT_UPDATES:
            a.updates = 100
        if not a.limit:
            a.limit = 8

    lock_path = os.path.join(a.root, 'artifact_lock.json')
    lock = json.load(open(lock_path))
    missing = [k for k in FORMAL_LOCK_KEYS_V7 if k not in lock]
    if missing and a.formal:
        raise SystemExit('lock missing %s' % missing)
    live_head = git_head()
    if a.formal and not a.smoke and live_head != lock['repo_commit']:
        raise SystemExit('repo HEAD drift: live=%s lock=%s'
                         % (live_head, lock['repo_commit']))
    hard_verify_lock(lock, dict(
        official_test_allowed=False,
        frozen_base=True,
        frozen_c2=True,
        zero_init=True,
        injection_point='post_y0_att',
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
        v7_init_sha=lock['v7_init_sha'],
        base_ckpt_sha256=file_sha256(lock['base_ckpt']),
        c2_ckpt_sha256=file_sha256(lock['c2_ckpt']),
    ), formal=a.formal and not a.smoke)

    ckpt_dir = os.path.join(
        a.root, ARM_A1, 'smoke_checkpoints' if a.smoke else 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    formal_end = os.path.join(a.root, ARM_A1, 'checkpoints',
                              'ckpt_%06d.pt' % DEFAULT_UPDATES)
    if a.formal and not a.smoke and a.start_step == 0 and os.path.isfile(formal_end):
        raise SystemExit('formal end ckpt exists; refuse overwrite: %s' % formal_end)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    logf = open(os.path.join(
        a.root, 'logs', 'train_%s%s.log' % (ARM_A1, '_smoke' if a.smoke else '')),
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
        os.path.join(a.root, lock['cache_name'], 'refiner_train'), mmap)
    # A1: correct-only
    pairs = [dict(i=i, name=row[0], state='correct',
                  key='%s|correct' % row[0])
             for i, row in enumerate(train_rows)]
    log('V7 train %s  pairs=%d  updates=%d  smoke=%s'
        % (ARM_A1, len(pairs), a.updates, a.smoke))

    model = V7TaskDecoupledModel(c2_ckpt=lock['c2_ckpt']).to(a.device)
    init = torch.load(lock['v7_init_path'], map_location='cpu')
    model.load_state_dict(init['model'], strict=True)
    init_sha = state_dict_sha(init['model'])
    if a.formal and init_sha != lock['v7_init_sha']:
        raise SystemExit('v7 init_sha mismatch')
    model.assert_formal_ready()

    train_params = [p for _, p in model.trainable_named_parameters()]
    opt = torch.optim.Adam(train_params, lr=a.lr, weight_decay=0.0)
    schedule = make_pair_schedule(a.updates, a.grad_accum, len(pairs), a.seed)

    def save_ckpt(step):
        blob = dict(
            model=model.state_dict(), step=int(step), arm=ARM_A1,
            objective=LOSS, seed=a.seed, init_sha=init_sha,
            repo_commit=git_head(), injection_point='post_y0_att',
            frozen_base=True, frozen_c2=True, loss=LOSS, updates=a.updates,
            grad_accum=a.grad_accum, lr=a.lr,
        )
        path = os.path.join(ckpt_dir, 'ckpt_%06d.pt' % step)
        torch.save(blob, path)
        torch.save(blob, os.path.join(ckpt_dir, 'last.pt'))
        return path

    if a.start_step <= 0:
        save_ckpt(0)
        t0 = sample_tensors(ds, pairs[0]['i'], pairs[0]['state'], a.device)
        with torch.no_grad():
            y = model.forward_m11(t0['Y0'], t0['R'])['y']
            y0_01 = _rgb01(t0['Y0'])
        if float((y - y0_01).abs().max()) > 1e-5:
            raise SystemExit('A1 step0 Y != Y0 max_abs=%.3e'
                             % float((y - y0_01).abs().max()))

    model.train()
    # keep C2 in eval
    model.match_features.eval()
    opt.zero_grad(set_to_none=True)
    t_start = time.time()
    run_loss = 0.0
    run_n = 0
    for step in range(max(a.start_step, 0) + 1, a.updates + 1):
        pi = schedule[step - 1]
        p = pairs[pi]
        t = sample_tensors(ds, p['i'], p['state'], a.device)
        out = model.forward_m11(t['Y0'], t['R'])
        gt_01 = _rgb01(t['H'])
        losses = td_reconstruction_losses(out, gt_01)
        (losses['loss'] / a.grad_accum).backward()
        run_loss += float(losses['loss'].detach())
        run_n += 1
        if step % a.grad_accum == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
            model.match_features.eval()
        if step % a.log_every == 0 or step == a.updates:
            avg = run_loss / max(run_n, 1)
            log('step %6d  loss=%.6f  l_rec=%.6f  l_tex=%.6f  %.1fs'
                % (step, avg, float(losses['l_rec']), float(losses['l_tex']),
                   time.time() - t_start))
            run_loss = 0.0
            run_n = 0
        if step in CKPT_STEPS or step == a.updates:
            path = save_ckpt(step)
            log('  saved %s' % path)

    log('done. elapsed=%.1fs' % (time.time() - t_start))
    logf.close()


if __name__ == '__main__':
    main()
