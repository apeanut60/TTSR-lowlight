#!/usr/bin/env python
"""v2 H3: V3B0ResidualFusion head on frozen H1 Encoder+LocalSoftMatch.

Uses the same P2 Y0 cache and Huawei train500 as H1/H2. Reconstruction = MSE.
Budget matches H1 (3000 updates, batch 8, lr 1e-4 drop at 2000) for comparability;
historical V3-B.0 on LOL used a different stack — noted in the report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from dataset.dataloader import _worker_init_fn as _worker_init  # noqa: E402
from dataset.data1_localrefine import LocalRefineTrainSet  # noqa: E402
from local_refine_runtime import (  # noqa: E402
    evaluate_v3b0, read_manifest_csv, sha256, verify_cache,
)
from model.LocalRefine import LocalSoftMatch, SharedEncoder, count_params  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion  # noqa: E402

EXP = '/root/data/experiments/TTSR-lowlight-v2'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', default=EXP)
    ap.add_argument('--h1_ckpt', default='',
                    help='H1 checkpoint providing encoder+match weights')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--drop_step', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr_after_drop', type=float, default=5e-5)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--crop_size', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--eval_every', type=int, default=1000)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--limit_steps', type=int, default=0)
    ap.add_argument('--allow_overwrite', action='store_true')
    a = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_dir = os.path.join(a.out_root, 'checkpoints', 'H3_v3b0_s%d' % a.seed)
    if os.path.isdir(run_dir) and not a.allow_overwrite:
        if any(f.startswith('checkpoint_') or f == 'train.jsonl'
               for f in os.listdir(run_dir)):
            raise SystemExit('refusing overwrite %s' % run_dir)
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()

    if not a.h1_ckpt:
        # prefer final H1 checkpoint
        h1_dir = os.path.join(a.out_root, 'checkpoints', 'H1_nano_s%d' % a.seed)
        cands = sorted(
            f for f in os.listdir(h1_dir)
            if f.startswith('checkpoint_') and f.endswith('.pt')
        ) if os.path.isdir(h1_dir) else []
        if not cands:
            raise SystemExit('no H1 checkpoint; train H1 first or pass --h1_ckpt')
        a.h1_ckpt = os.path.join(h1_dir, cands[-1])
    print('H3 using frozen features from', a.h1_ckpt)

    torch.manual_seed(a.seed)
    tr_rows = read_manifest_csv(os.path.join(a.out_root, 'manifests',
                                             'manifests_train.csv'))
    ev_rows = read_manifest_csv(os.path.join(a.out_root, 'manifests',
                                             'manifests_eval.csv'))
    train_cache = os.path.join(a.out_root, 'cache_p2_train')
    eval_cache = os.path.join(a.out_root, 'cache_p2_eval')
    train_meta = verify_cache(train_cache)
    verify_cache(eval_cache)

    encoder = SharedEncoder(32).to(device)
    match = LocalSoftMatch(32, radius=4, tau=0.1).to(device)
    head = V3B0ResidualFusion(in_ch=96).to(device)
    h1 = torch.load(a.h1_ckpt, map_location=device)
    sd = h1['refiner'] if 'refiner' in h1 else h1
    enc_sd = {k.replace('encoder.', '', 1): v for k, v in sd.items()
              if k.startswith('encoder.')}
    mat_sd = {k.replace('match.', '', 1): v for k, v in sd.items()
              if k.startswith('match.')}
    encoder.load_state_dict(enc_sd)
    match.load_state_dict(mat_sd)
    for p in list(encoder.parameters()) + list(match.parameters()):
        p.requires_grad_(False)
    encoder.eval()
    match.eval()
    if float(head.out.weight.abs().max()) != 0:
        raise SystemExit('V3B0 out not zero-init')
    print('head params', count_params(head))

    ds = LocalRefineTrainSet(tr_rows, train_cache, crop_size=a.crop_size,
                             seed=a.seed)
    loader_gen = torch.Generator()
    loader_gen.manual_seed(a.seed)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                    num_workers=a.workers, drop_last=True,
                    worker_init_fn=_worker_init, generator=loader_gen)
    opt = torch.optim.Adam(head.parameters(), lr=a.lr, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=0)
    steps = a.limit_steps or a.steps
    log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w')
    hist = []

    def run_eval(step):
        print('--- H3 eval @', step)
        res = evaluate_v3b0(head, encoder, match, ev_rows, eval_cache, device)
        summary = {l: float(np.mean([d['psnr_rgb'] for d in res[l].values()]))
                   for l in res}
        hist.append(dict(step=step, mean_psnr=summary))
        torch.save(dict(head=head.state_dict(), encoder=encoder.state_dict(),
                        match=match.state_dict(), global_step=step,
                        h1_ckpt=a.h1_ckpt, mean_psnr=summary, config=vars(a)),
                   os.path.join(run_dir, 'checkpoint_%05d.pt' % step))
        print('  means', summary)
        head.train()

    # step0 identity
    with torch.no_grad():
        y = torch.randn(1, 3, 128, 128, device=device)
        n = torch.randn(1, 3, 128, 128, device=device)
        f0 = encoder((y + 1) * 0.5)
        fr = encoder((n + 1) * 0.5)
        t = match(f0, fr)
        dy = head(f0, t, y.shape[-2:])
        if float(dy.abs().max()) > 1e-6:
            raise SystemExit('H3 step0 residual not ~0')
        print('H3 step0 identity OK')

    run_eval(0)
    step = 0
    data_pass = 0
    head.train()
    while step < steps:
        ds.set_data_pass(data_pass)
        for batch in dl:
            if step >= steps:
                break
            step += 1
            lr = a.lr if step <= a.drop_step else a.lr_after_drop
            for g in opt.param_groups:
                g['lr'] = lr
            y0 = batch['y0'].to(device).detach()
            hr = batch['high'].to(device)
            nano = batch['nano'].to(device).detach()
            with torch.no_grad():
                f0 = encoder((y0 + 1.) * 0.5)
                fr = encoder((nano + 1.) * 0.5)
                t = match(f0, fr)
            dy = head(f0, t, y0.shape[-2:])
            sr = y0 + dy
            loss = torch.nn.functional.mse_loss(sr, hr)
            if not torch.isfinite(loss):
                raise SystemExit('non-finite at %d' % step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % a.log_every == 0 or step == 1:
                row = dict(step=step, data_pass=data_pass, lr=lr,
                           loss=float(loss.detach()),
                           dy_absmean=float(dy.detach().abs().mean()))
                log_f.write(json.dumps(row) + '\n')
                log_f.flush()
                print('  step %d/%d loss=%.6f' % (step, steps, row['loss']))
            if step % a.eval_every == 0 or step == steps:
                run_eval(step)
        if step < steps:
            data_pass += 1

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(dict(vars(a), head_params=count_params(head),
                       h1_ckpt_sha256=sha256(a.h1_ckpt),
                       base_checkpoint_sha256=train_meta['_base_sha'],
                       data_passes=data_pass + 1,
                       note='MSE recon; frozen H1 encoder+match; comparable budget to H1'),
                  f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, 'eval_history.json'), 'w') as f:
        json.dump(hist, f, indent=2)
    log_f.close()
    print('=== done H3 in %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
