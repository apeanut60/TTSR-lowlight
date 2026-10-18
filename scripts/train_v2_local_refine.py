#!/usr/bin/env python
"""v2 H1/H2: train LocalRefine on frozen TTSR-P2 Y0 cache (Nano or Self)."""

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
from loss.loss_enhance import (  # noqa: E402
    ColorConstancyLoss, IlluminationSmoothnessLoss, PerceptualLoss,
    ReconstructionLoss,
)
from local_refine_runtime import (  # noqa: E402
    evaluate_conditions, read_manifest_csv, sha256, verify_cache,
)
from model.LocalRefine import Refiner, count_params  # noqa: E402
from model.Vgg19 import Vgg19  # noqa: E402

EXP = '/root/data/experiments/TTSR-lowlight-v2'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['self', 'nano'])
    ap.add_argument('--out_root', default=EXP)
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
    ap.add_argument('--expect_train_n', type=int, default=500,
                    help='0 = skip train-size check (e.g. LOLv2 n=639)')
    a = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = 'H1_nano' if a.arm == 'nano' else 'H2_self'
    run_dir = os.path.join(a.out_root, 'checkpoints', '%s_s%d' % (run_name, a.seed))
    existing = [f for f in ('train.jsonl', 'config.json')
                if os.path.isfile(os.path.join(run_dir, f))]
    if os.path.isdir(run_dir):
        existing += [f for f in os.listdir(run_dir) if f.startswith('checkpoint_')]
    if existing and not a.allow_overwrite:
        raise SystemExit('refusing overwrite %s (%s); pass --allow_overwrite'
                         % (run_dir, existing[:3]))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(a.out_root, 'logs'), exist_ok=True)
    t0 = time.time()
    print('=== %s arm=%s dir=%s' % (run_name, a.arm, run_dir))

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    tr_rows = read_manifest_csv(os.path.join(a.out_root, 'manifests',
                                             'manifests_train.csv'))
    ev_rows = read_manifest_csv(os.path.join(a.out_root, 'manifests',
                                             'manifests_eval.csv'))
    train_cache = os.path.join(a.out_root, 'cache_p2_train')
    eval_cache = os.path.join(a.out_root, 'cache_p2_eval')
    print('rows train=%d eval=%d' % (len(tr_rows), len(ev_rows)))
    if a.expect_train_n and len(tr_rows) != a.expect_train_n:
        raise SystemExit('expected %d train rows, got %d' % (
            a.expect_train_n, len(tr_rows)))
    train_meta = verify_cache(train_cache)
    eval_meta = verify_cache(eval_cache)
    print('P2 cache sha %s' % train_meta['_base_sha'][:16])

    refiner = Refiner().to(device)
    nparam = count_params(refiner)
    if nparam != 78724:
        raise SystemExit('refiner params %d != 78724' % nparam)
    init_path = os.path.join(a.out_root, 'checkpoints', 'refiner_init_s%d.pt' % a.seed)
    os.makedirs(os.path.dirname(init_path), exist_ok=True)
    if os.path.isfile(init_path):
        refiner.load_state_dict(torch.load(init_path, map_location=device))
        print('loaded shared init', init_path)
    else:
        torch.save(refiner.state_dict(), init_path)
        print('saved shared init', init_path)
    if float(refiner.c_out.weight.abs().max()) != 0 \
            or float(refiner.c_out.bias.abs().max()) != 0:
        raise SystemExit('C_out not zero-init')

    vgg = Vgg19(requires_grad=False).to(device).eval()
    rec = ReconstructionLoss('l1')
    per = PerceptualLoss()
    smooth = IlluminationSmoothnessLoss()
    color = ColorConstancyLoss()

    ds = LocalRefineTrainSet(tr_rows, train_cache, crop_size=a.crop_size,
                             seed=a.seed)
    loader_gen = torch.Generator()
    loader_gen.manual_seed(a.seed)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                    num_workers=a.workers, drop_last=True,
                    worker_init_fn=_worker_init, generator=loader_gen)
    opt = torch.optim.Adam(refiner.parameters(), lr=a.lr, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=0)

    steps = a.limit_steps or a.steps
    log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w', encoding='utf-8')
    hist = []

    def run_eval(step):
        print('--- eval @ step %d' % step)
        res = evaluate_conditions(
            refiner, ev_rows, eval_cache,
            [('H0_P2', 'bypass'), ('arm', a.arm), ('mismatch', 'mismatch')],
            device)
        # summarize means
        def mean_psnr(label):
            return float(np.mean([d['psnr_rgb'] for d in res[label].values()]))
        summary = {l: mean_psnr(l) for l, _ in
                   [('H0_P2', 0), ('arm', 0), ('mismatch', 0)]}
        hist.append(dict(step=step, mean_psnr=summary))
        ckpt = os.path.join(run_dir, 'checkpoint_%05d.pt' % step)
        torch.save(dict(refiner=refiner.state_dict(), optimizer=opt.state_dict(),
                        global_step=step, arm=a.arm, config=vars(a),
                        mean_psnr=summary), ckpt)
        with open(os.path.join(run_dir, 'eval_step_%05d.json' % step), 'w') as f:
            json.dump(dict(step=step, mean_psnr=summary, per_image=res), f)
        print('  means', summary)
        refiner.train()

    # step-0 identity smoke
    refiner.eval()
    with torch.no_grad():
        y = torch.randn(1, 3, 128, 128, device=device)
        r = torch.randn(1, 3, 128, 128, device=device)
        sr, _ = refiner(y, r)
        if not torch.allclose(sr, y, atol=1e-6):
            raise SystemExit('step0 identity failed: max|sr-y|=%s'
                             % float((sr - y).abs().max()))
        print('step0 identity OK')
    # grad smoke
    refiner.train()
    y = torch.randn(2, 3, 128, 128, device=device)
    r = torch.randn(2, 3, 128, 128, device=device)
    hr = torch.randn(2, 3, 128, 128, device=device)
    sr, _ = refiner(y, r)
    loss = rec(sr, hr)
    loss.backward()
    if refiner.c_out.weight.grad is None \
            or float(refiner.c_out.weight.grad.abs().sum()) == 0:
        raise SystemExit('c_out has no gradient')
    opt.zero_grad(set_to_none=True)
    print('grad smoke OK')

    run_eval(0)
    step = 0
    data_pass = 0
    while step < steps:
        ds.set_data_pass(data_pass)
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
            if not torch.isfinite(loss):
                raise SystemExit('non-finite loss at step %d' % step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.sqrt(sum(
                (p.grad.detach() ** 2).sum()
                for p in refiner.parameters() if p.grad is not None)))
            opt.step()
            if step % a.log_every == 0 or step == 1:
                row = dict(step=step, data_pass=data_pass, lr=lr,
                           loss=float(loss.detach()),
                           rec=float(l_rec.detach()),
                           per=float(l_per.detach()),
                           smooth=float(l_sm.detach()),
                           color=float(l_co.detach()),
                           grad_norm=gnorm,
                           delta_absmean=float(aux['delta'].abs().mean()),
                           gate_mean=float(aux['gate'].mean()))
                log_f.write(json.dumps(row) + '\n')
                log_f.flush()
                print('  step %d/%d loss=%.5f rec=%.5f |g|=%.2e'
                      % (step, steps, row['loss'], row['rec'], gnorm))
            if step % a.eval_every == 0 or step == steps:
                run_eval(step)
        if step < steps:
            data_pass += 1

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(dict(vars(a), run_name=run_name,
                       refiner_params=nparam,
                       refiner_init_sha256=sha256(init_path),
                       train_cache=train_cache, eval_cache=eval_cache,
                       base_checkpoint_sha256=train_meta['_base_sha'],
                       data_passes=data_pass + 1), f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, 'eval_history.json'), 'w') as f:
        json.dump(hist, f, indent=2)
    log_f.close()
    print('=== done %s in %.1f min' % (run_name, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
