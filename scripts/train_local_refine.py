#!/usr/bin/env python
"""V2 single-arm training: V2-Self or V2-Nano (plan section E).

The frozen N0 is *not* in the loop -- both arms read the same float32 Y0 cache,
so the only difference between them is which tensor becomes the refiner's
second input. Everything else (init, data draw, budget, LR schedule, losses)
is identical by construction.
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from dataset.dataloader import _worker_init                    # noqa: E402
from dataset.data1_localrefine import LocalRefineTrainSet      # noqa: E402
from loss.loss_enhance import (ColorConstancyLoss, ExposureControlLoss,  # noqa: E402
                               IlluminationSmoothnessLoss, PerceptualLoss,
                               ReconstructionLoss)
from model.LocalRefine import Refiner, count_params            # noqa: E402
from model.Vgg19 import Vgg19                                  # noqa: E402
from local_refine_runtime import (evaluate_conditions, read_manifest_csv,  # noqa: E402
                                  sha256)


def build_rows(out_root, tag):
    return read_manifest_csv(os.path.join(out_root, 'manifests_%s.csv' % tag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['self', 'nano'])
    ap.add_argument('--out_root', default='/root/data/experiments/retinex_v2_localref')
    ap.add_argument('--dataset_dir', default='/root/data/datasets/data1')
    ap.add_argument('--manifest_dir',
                    default='/root/data/datasets/data1/.nanobanana_sample_manifest')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--drop_step', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr_after_drop', type=float, default=5e-5)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--crop_size', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--eval_every', type=int, default=1000)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--limit_steps', type=int, default=0,
                    help='debug: stop early (0 = full budget)')
    a = ap.parse_args(_CLI)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_dir = os.path.join(a.out_root, 'V2_%s_s%d' % (a.arm.capitalize(), a.seed))
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    print('=== V2-%s  run_dir=%s' % (a.arm, run_dir))

    torch.manual_seed(a.seed)
    tr_rows = build_rows(a.out_root, 'train')
    ev_rows = build_rows(a.out_root, 'eval')
    train_cache = os.path.join(a.out_root, 'cache_n0_train')
    eval_cache = os.path.join(a.out_root, 'cache_n0_eval')
    print('rows: train %d, eval %d' % (len(tr_rows), len(ev_rows)))

    refiner = Refiner().to(device)
    if count_params(refiner) != 78724:
        raise SystemExit('refiner has %d params, expected 78724'
                         % count_params(refiner))
    # The two arms of one seed MUST share one initialisation (plan D1), but two
    # different *seeds* must not: a seed replication has to vary the init as well
    # as the data draw, otherwise it only re-tests data-order sensitivity. Seed
    # 42 keeps its original filename so the completed run stays bit-identical.
    init_name = ('refiner_init.pt' if a.seed == 42
                 else 'refiner_init_s%d.pt' % a.seed)
    init_path = os.path.join(a.out_root, init_name)
    if os.path.isfile(init_path):
        refiner.load_state_dict(torch.load(init_path, map_location=device))
        print('loaded shared %s' % init_name)
    else:
        torch.save(refiner.state_dict(), init_path)
        print('saved %s' % init_name)
    c_out = refiner.c_out
    if float(c_out.weight.abs().max()) != 0 or float(c_out.bias.abs().max()) != 0:
        raise SystemExit('C_out is not zero-initialised')

    vgg = Vgg19(requires_grad=False).to(device).eval()
    rec = ReconstructionLoss('l1')
    per = PerceptualLoss()
    smooth = IlluminationSmoothnessLoss()
    color = ColorConstancyLoss()

    ds = LocalRefineTrainSet(tr_rows, train_cache, crop_size=a.crop_size, seed=a.seed)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                    num_workers=a.workers, drop_last=True,
                    worker_init_fn=_worker_init)
    opt = torch.optim.Adam(refiner.parameters(), lr=a.lr, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=0)

    steps = a.limit_steps or a.steps
    log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w', encoding='utf-8')
    hist = []

    def run_eval(step):
        print('--- eval @ step %d' % step)
        res = evaluate_conditions(
            refiner, ev_rows, eval_cache,
            [('N0', 'bypass'), ('V2_%s' % a.arm, a.arm), ('mismatch', 'mismatch')],
            device)
        hist.append(dict(step=step, summary=res))
        torch.save(dict(refiner=refiner.state_dict(), optimizer=opt.state_dict(),
                        global_step=step, arm=a.arm, config=vars(a)),
                   os.path.join(run_dir, 'checkpoint_%05d.pt' % step))
        refiner.train()

    refiner.train()
    run_eval(0)
    step = 0
    while step < steps:
        for batch in dl:
            if step >= steps:
                break
            step += 1
            lr = a.lr if step <= a.drop_step else a.lr_after_drop
            for g in opt.param_groups:
                g['lr'] = lr
            y0 = batch['y0'].to(device)
            hr = batch['high'].to(device)
            r_eff = y0 if a.arm == 'self' else batch['nano'].to(device)
            sr, aux = refiner(y0.detach(), r_eff.detach())
            l_rec = rec(sr, hr)
            l_per = per(vgg((sr + 1.) / 2.), vgg((hr + 1.) / 2.))
            l_sm = smooth(sr)
            l_co = color(sr)
            loss = l_rec + 0.1 * l_per + 1.0 * l_sm + 0.5 * l_co
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum()
                                         for p in refiner.parameters()
                                         if p.grad is not None)))
            opt.step()

            if step % a.log_every == 0 or step == 1:
                row = dict(step=step, lr=lr, loss=float(loss.detach()),
                           rec=float(l_rec.detach()), per=float(l_per.detach()),
                           smooth=float(l_sm.detach()), color=float(l_co.detach()),
                           grad_norm=gnorm,
                           delta_absmean=float(aux['delta'].abs().mean()),
                           delta_absmax=float(aux['delta'].abs().max()),
                           gate_mean=float(aux['gate'].mean()),
                           gate_min=float(aux['gate'].min()),
                           gate_max=float(aux['gate'].max()),
                           oob=float(sr.abs().gt(1).float().mean()))
                log_f.write(json.dumps(row) + '\n')
                log_f.flush()
                if step % (a.log_every * 5) == 0 or step == 1:
                    print('  step %d/%d lr=%.2e loss=%.5f rec=%.5f '
                          '|grad|=%.2e delta=%.4f gate=[%.3f,%.3f] oob=%.4f%%'
                          % (step, steps, lr, row['loss'], row['rec'], gnorm,
                             row['delta_absmean'], row['gate_min'],
                             row['gate_max'], row['oob'] * 100))
            if step in (a.drop_step - 1, a.drop_step, a.drop_step + 1, a.drop_step + 2):
                print('  LR check at update %d: %.3e' % (step, lr))
            if step % a.eval_every == 0 or step == steps:
                run_eval(step)

    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(vars(a), refiner_params=count_params(refiner),
                       refiner_init_sha256=sha256(init_path),
                       train_cache=train_cache, eval_cache=eval_cache),
                  f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, 'eval_history.json'), 'w', encoding='utf-8') as f:
        json.dump(hist, f, indent=2)
    log_f.close()
    print('=== done V2-%s in %.1f min' % (a.arm, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
