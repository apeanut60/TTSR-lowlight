#!/usr/bin/env python
"""V7-RIVER train Setting A (fusion only) or B (fusion+align). MSE."""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from model.V6LLFormerRefBridge import load_frozen_llformer_base, tiled_v6_forward  # noqa: E402
from model.V7Model import V7Model  # noqa: E402
from option import parser as option_parser  # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import STATES, snapshot_, state_dict_sha  # noqa: E402
from v3a5c_runtime import make_pair_schedule  # noqa: E402
from v3a6_runtime import file_sha256, git_head, hard_verify_lock  # noqa: E402
from v3b_runtime import GRAD_ACCUM, LR, SEED, b0_loss  # noqa: E402
from v7_river_runtime import (CKPT_STEPS, DEFAULT_UPDATES, FORMAL_LOCK_KEYS,  # noqa: E402
                              LOSS, OPTIMIZER, ROOT, SRC, SETTING_ARMS,
                              setting_root)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--setting', default='A', choices=['A', 'B'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--updates', type=int, default=DEFAULT_UPDATES)
    ap.add_argument('--grad_accum', type=int, default=GRAD_ACCUM)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--formal', action='store_true', default=True)
    a = ap.parse_args(_CLI)
    if a.smoke:
        a.formal = False
        if a.updates == DEFAULT_UPDATES:
            a.updates = 100
        if not a.limit:
            a.limit = 8

    arm_root = setting_root(a.setting, a.root)
    lock_path = os.path.join(arm_root, 'artifact_lock.json')
    lock = json.load(open(lock_path))
    missing = [k for k in FORMAL_LOCK_KEYS if k not in lock]
    if missing and a.formal:
        raise SystemExit('lock missing %s' % missing)
    if a.formal and not a.smoke and git_head() != lock['repo_commit']:
        raise SystemExit('repo HEAD drift')
    hard_verify_lock(lock, dict(
        official_test_allowed=True,
        frozen_base=True,
        frozen_align=(a.setting.upper() == 'A'),
        zero_init=True,
        injection_point=lock['injection_point'],
        loss=LOSS,
        optimizer=OPTIMIZER,
        seed=a.seed,
        pair_schedule_seed=a.seed,
        lr=a.lr if not a.smoke else lock['lr'],
        updates=a.updates if not a.smoke else lock['updates'],
        grad_accum=a.grad_accum if not a.smoke else lock['grad_accum'],
        split_sha256=file_sha256(lock['split_json']),
        mismatch_train_sha256=file_sha256(lock['mismatch_train']),
        reference_variant=lock['reference_variant'],
        v7_init_sha=lock['v7_init_sha'],
        base_ckpt_sha256=file_sha256(lock['base_ckpt']),
        setting=a.setting.upper(),
    ), formal=a.formal and not a.smoke)

    ckpt_dir = os.path.join(
        arm_root, 'smoke_checkpoints' if a.smoke else 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(arm_root, 'logs'), exist_ok=True)
    logf = open(os.path.join(
        arm_root, 'logs',
        'train_%s%s.log' % (SETTING_ARMS[a.setting.upper()],
                            '_smoke' if a.smoke else '')),
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
    pairs = []
    for i, row in enumerate(train_rows):
        for state in STATES:
            pairs.append(dict(i=i, name=row[0], state=state))
    log('V7-RIVER train setting=%s pairs=%d updates=%d smoke=%s'
        % (a.setting, len(pairs), a.updates, a.smoke))

    base = load_frozen_llformer_base(lock['base_ckpt'], a.device)
    snap_base = snapshot_(base)
    model = V7Model().to(a.device)
    init = torch.load(lock['v7_init_path'], map_location='cpu')
    model.load_state_dict(init['model'], strict=True)
    if state_dict_sha(init['model']) != lock['v7_init_sha'] and a.formal:
        raise SystemExit('init_sha mismatch')
    model.apply_setting_freeze(a.setting)

    opt = torch.optim.Adam(list(model.trainable_parameters()),
                           lr=a.lr, weight_decay=0.0)
    schedule = make_pair_schedule(a.updates, a.grad_accum, len(pairs), a.seed)

    def save_ckpt(step):
        blob = dict(
            model=model.state_dict(), step=int(step),
            arm=lock['arm'], setting=a.setting.upper(),
            objective='mse_reconstruction', seed=a.seed,
            init_sha=lock['v7_init_sha'], repo_commit=git_head(),
            injection_point=lock['injection_point'],
            frozen_base=True, frozen_align=lock['frozen_align'],
            loss=LOSS, updates=a.updates, grad_accum=a.grad_accum, lr=a.lr,
        )
        path = os.path.join(ckpt_dir, 'ckpt_%06d.pt' % step)
        torch.save(blob, path)
        torch.save(blob, os.path.join(ckpt_dir, 'last.pt'))
        return path

    save_ckpt(0)
    t0 = sample_tensors(ds, pairs[0]['i'], pairs[0]['state'], a.device)
    with torch.no_grad():
        y = tiled_v6_forward(base, model, t0['X'], t0['Y0'], t0['R'])
    if float((y - t0['Y0']).abs().max()) > 1e-3:
        raise SystemExit('step0 Y!=Y0')

    t_wall = time.time()
    running = 0.0
    n_run = 0
    cursor = 0
    for step in range(1, a.updates + 1):
        model.train()
        # keep frozen align in eval under Setting A
        if a.setting.upper() == 'A':
            for name in ('match_encoder', 'texture_encoder', 'matcher',
                         'align_h4', 'align_h2'):
                getattr(model, name).eval()
        opt.zero_grad(set_to_none=True)
        for _ in range(a.grad_accum):
            p = pairs[int(schedule[cursor])]
            cursor += 1
            t = sample_tensors(ds, p['i'], p['state'], a.device)
            y = tiled_v6_forward(base, model, t['X'], t['Y0'], t['R'])
            loss = b0_loss(y, t['H'])
            (loss / a.grad_accum).backward()
            running += float(loss.detach())
            n_run += 1
            # Base must not get grad
            for p_b in base.parameters():
                if p_b.grad is not None:
                    raise SystemExit('Base received grad')
            if a.setting.upper() == 'A':
                for n, p_a in model.named_parameters():
                    if n.startswith(('match_encoder', 'texture_encoder',
                                     'align_')) and p_a.grad is not None:
                        raise SystemExit('Setting A align got grad: %s' % n)
        opt.step()
        if step % a.log_every == 0 or step == 1 or step == a.updates:
            log('  step %d  loss=%.5f  (%.0fs)'
                % (step, running / max(n_run, 1), time.time() - t_wall))
            running = 0.0
            n_run = 0
        if step in CKPT_STEPS or step == a.updates:
            log('  saved %s' % save_ckpt(step))

    from v3a5_runtime import bit_equal
    if not bit_equal(snap_base, snapshot_(base)):
        raise SystemExit('Base mutated')
    log('DONE setting=%s' % a.setting)
    logf.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
