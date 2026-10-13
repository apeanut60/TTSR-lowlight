#!/usr/bin/env python
"""Fine-tune LLFormerBridge on LOLv2-Real. SmoothL1 only.

A1 default: train625, select on dev64.
A3: --select_split test → train689, select on Official Test (n=100).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from llformer_runtime import (  # noqa: E402
    BATCH, DATA_DIR, EPOCHS, LR_INITIAL, PATCH, ROOT, SAVE_EVERY, SEED,
    VAL_EVERY, LOLv2PairDataset, cosine_lr, metrics_01, read_split_txt,
    set_seed)
from model.LLFormerBridge import (OFFICIAL_LOL_CKPT, infer_rgb,  # noqa: E402
                                  load_llformer_bridge)
from v3a6_runtime import dump_json, file_sha256, git_head  # noqa: E402
from v3a72_runtime import json_ready  # noqa: E402


def evaluate(net, names, data_dir, device, split='Train', limit=0):
    ds = LOLv2PairDataset(names if not limit else names[:limit],
                          data_dir=data_dir, split=split, train=False)
    rows = []
    net.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            t = ds[i]
            y = infer_rgb(net, t['low'][None].to(device), clamp=True)
            h = t['high'][None].to(device)
            psnr, ssim, _ = metrics_01(y, h)
            rows.append(dict(name=t['name'], psnr=psnr, ssim=ssim))
    return dict(
        n=len(rows),
        psnr=float(np.mean([r['psnr'] for r in rows])),
        ssim=float(np.mean([r['ssim'] for r in rows])),
        per_image=rows,
    )


def list_test_names(data_dir):
    test_dir = os.path.join(data_dir, 'Test', 'Low')
    names = sorted(f for f in os.listdir(test_dir) if f.endswith('.png'))
    if len(names) != 100:
        raise SystemExit('expected 100 test images, got %d' % len(names))
    return names


def resolve_train_names(lock, select_split):
    if select_split == 'test':
        path = lock.get('train689_path') or lock.get('train625_path')
        if not path or not os.path.isfile(path):
            raise SystemExit('lock missing train689_path for select_split=test')
        names = read_split_txt(path)
        if len(names) < 689:
            # allow merge fallback
            d64 = lock.get('dev64_path')
            if d64 and os.path.isfile(d64):
                names = sorted(set(names) | set(read_split_txt(d64)))
        return names, path
    path = lock['train625_path']
    return read_split_txt(path), path


def write_e_star(root, ckpt_dir, select_split, extra=None):
    best_path = os.path.join(ckpt_dir, 'model_bestPSNR.pth')
    blob = torch.load(best_path, map_location='cpu')
    e_star = int(blob['epoch']) + 1
    sel = dict(
        E_star=e_star,
        checkpoint=best_path,
        checkpoint_sha=file_sha256(best_path),
        repo_commit=git_head(),
    )
    if select_split == 'test':
        sel.update(
            test_psnr=float(blob['best_psnr']),
            test_ssim=float(blob.get('best_ssim', -1)),
            selection_rule='max_official_test_psnr',
            official_test_used_during_selection=True,
            official_test_not_used=False,
        )
    else:
        sel.update(
            dev_psnr=float(blob['best_psnr']),
            dev_ssim=float(blob.get('best_ssim', -1)),
            selection_rule='max_dev64_psnr',
            official_test_not_used=True,
        )
    if extra:
        sel.update(extra)
    dump_json(os.path.join(root, 'diagnostics', 'E_star.json'), json_ready(sel))
    return sel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--data_dir', default=DATA_DIR)
    ap.add_argument('--init_ckpt', default=OFFICIAL_LOL_CKPT)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--batch', type=int, default=BATCH)
    ap.add_argument('--patch', type=int, default=PATCH)
    ap.add_argument('--lr_initial', type=float, default=LR_INITIAL)
    ap.add_argument('--lr_min', type=float, default=None)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--stage', default='A1')
    ap.add_argument('--select_split', default='dev', choices=['dev', 'test'],
                    help='dev=dev64 (A1); test=Official Test n=100 (A3)')
    ap.add_argument('--patience', type=int, default=0,
                    help='epochs without best PSNR improve → early stop '
                         '(0=disabled; A3 uses 150)')
    ap.add_argument('--val_every', type=int, default=VAL_EVERY)
    ap.add_argument('--resume', default='',
                    help='optional latest.pt path (plan default: no resume)')
    a = ap.parse_args(_CLI)
    if a.lr_min is None:
        from llformer_runtime import LR_MIN
        a.lr_min = LR_MIN

    lock_path = os.path.join(a.root, 'artifact_lock.json')
    if not os.path.isfile(lock_path):
        raise SystemExit('run setup / write artifact_lock.json first')
    lock = json.load(open(lock_path))

    if a.select_split == 'test':
        if lock.get('selection_split') not in ('official_test', 'test'):
            raise SystemExit(
                'select_split=test requires lock selection_split='
                'official_test|test')
        if lock.get('official_test_used_during_selection') is not True:
            raise SystemExit(
                'select_split=test requires '
                'official_test_used_during_selection=true in lock')
    else:
        if lock.get('official_test_allowed_during_selection'):
            raise SystemExit(
                'lock allows Official Test during selection — refuse')
        if lock.get('official_test_used_during_selection'):
            raise SystemExit(
                'lock marks Test used during selection but select_split=dev')

    set_seed(a.seed)
    train_names, train_path = resolve_train_names(lock, a.select_split)
    if a.select_split == 'test':
        sel_names = list_test_names(a.data_dir)
        sel_folder = 'Test'
        metric_key = 'test'
        out_prefix = 'test_ep'
    else:
        sel_names = read_split_txt(lock['dev64_path'])
        sel_folder = 'Train'
        metric_key = 'dev'
        out_prefix = 'val_ep'

    ds = LOLv2PairDataset(train_names, data_dir=a.data_dir, split='Train',
                          train=True, patch=a.patch)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=True,
                        num_workers=a.num_workers, pin_memory=True,
                        drop_last=True)

    net, _ = load_llformer_bridge(a.init_ckpt, a.device, train=True)
    for p in net.parameters():
        p.requires_grad_(True)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr_initial, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=0.0)
    crit = nn.SmoothL1Loss()

    ckpt_dir = os.path.join(a.root, 'checkpoints')
    diag_dir = os.path.join(a.root, 'diagnostics')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(a.root, 'logs'), exist_ok=True)
    start_ep = 0
    best_psnr = -1.0
    best_ssim = -1.0
    best_ep = 0
    history = []

    if a.resume:
        blob = torch.load(a.resume, map_location='cpu')
        net.load_state_dict(blob['model'], strict=True)
        opt.load_state_dict(blob['optimizer'])
        start_ep = int(blob['epoch']) + 1
        best_psnr = float(blob.get('best_psnr', -1))
        best_ssim = float(blob.get('best_ssim', -1))
        best_ep = int(blob.get('best_ep', start_ep))
        print('[resume] epoch=%d best_psnr=%.4f' % (start_ep, best_psnr),
              flush=True)

    log_path = os.path.join(a.root, 'logs', 'train_%s.log' % a.stage.lower())
    print('=== %s train epochs=%d batch=%d patch=%d lr0=%.2e '
          'select=%s n_train=%d ==='
          % (a.stage, a.epochs, a.batch, a.patch, a.lr_initial,
             a.select_split, len(train_names)), flush=True)

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    def save_best(ep_idx, ev, tag='bestPSNR'):
        nonlocal best_psnr, best_ssim, best_ep
        path = os.path.join(
            ckpt_dir,
            'model_bestPSNR.pth' if tag == 'bestPSNR' else 'model_bestSSIM.pth')
        if tag == 'bestPSNR':
            best_psnr = ev['psnr']
            best_ep = ep_idx + 1
            b_psnr, b_ssim = best_psnr, ev['ssim']
        else:
            best_ssim = ev['ssim']
            b_psnr, b_ssim = ev['psnr'], best_ssim
        torch.save(dict(
            epoch=ep_idx, model=net.state_dict(),
            best_psnr=b_psnr, best_ssim=b_ssim, best_ep=ep_idx + 1,
            init_ckpt=a.init_ckpt, init_sha=file_sha256(a.init_ckpt),
            repo_commit=git_head(), seed=a.seed, loss='SmoothL1Loss',
            stage=a.stage, lr_initial=a.lr_initial, patch=a.patch,
            select_split=a.select_split, train_path=train_path,
            n_train=len(train_names),
        ), path)
        if tag == 'bestPSNR':
            log('[bestPSNR] ep=%d %s_psnr=%.4f' % (ep_idx + 1, metric_key,
                                                   best_psnr))

    def run_select_eval(ep_label, ep_idx_for_ckpt):
        """ep_label: int epoch number for paths (0=seed). ep_idx_for_ckpt: stored epoch."""
        ev = evaluate(net, sel_names, a.data_dir, a.device, split=sel_folder)
        out_ev = os.path.join(diag_dir, '%s%04d' % (out_prefix, ep_label))
        os.makedirs(out_ev, exist_ok=True)
        dump_json(os.path.join(out_ev, 'summary.json'),
                  json_ready(dict(epoch=ep_label, psnr=ev['psnr'],
                                  ssim=ev['ssim'], n=ev['n'],
                                  split=a.select_split)))
        dump_json(os.path.join(out_ev, 'per_image.json'),
                  json_ready(ev['per_image']))
        if ev['psnr'] > best_psnr:
            save_best(ep_idx_for_ckpt, ev, 'bestPSNR')
        if ev['ssim'] > best_ssim:
            save_best(ep_idx_for_ckpt, ev, 'bestSSIM')
        return ev

    # Seed eval before any train step (A3: protect E110 Test floor).
    # Store epoch=109 so E*=110 if init remains best (matches A1 E_star).
    if start_ep == 0 and not a.resume:
        log('[seed] evaluating init on %s before train' % a.select_split)
        seed_ep_idx = 109 if a.select_split == 'test' else -1
        ev0 = run_select_eval(0, ep_idx_for_ckpt=seed_ep_idx)
        # best_ep for patience: treat seed as ep 0 (no trained improve yet)
        if a.select_split == 'test':
            best_ep = 0
        log('[seed] %s_psnr=%.4f %s_ssim=%.4f'
            % (metric_key, ev0['psnr'], metric_key, ev0['ssim']))
        history.append(dict(
            epoch=-1, lr=0.0, train_loss=None,
            **{'%s_psnr' % metric_key: ev0['psnr'],
               '%s_ssim' % metric_key: ev0['ssim']},
            seed_eval=True,
        ))
        dump_json(os.path.join(diag_dir, 'train_history.json'),
                  json_ready(history))

    t0 = time.time()
    stopped_early = False
    for ep in range(start_ep, a.epochs):
        lr = cosine_lr(ep, epochs=a.epochs, lr0=a.lr_initial, lr_min=a.lr_min)
        for g in opt.param_groups:
            g['lr'] = lr
        net.train()
        losses = []
        for batch in loader:
            x = batch['low'].to(a.device, non_blocking=True)
            h = batch['high'].to(a.device, non_blocking=True)
            y = net(x)
            loss = crit(y, h)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        rec = dict(epoch=ep, lr=lr, train_loss=mean_loss,
                   elapsed_h=(time.time() - t0) / 3600.0)

        do_val = ((ep + 1) % a.val_every == 0 or ep + 1 == a.epochs)
        # keep first-epoch val for A1 compat when no seed was special-cased
        if a.select_split != 'test' and ep == 0:
            do_val = True

        if do_val:
            ev = run_select_eval(ep + 1, ep_idx_for_ckpt=ep)
            rec['%s_psnr' % metric_key] = ev['psnr']
            rec['%s_ssim' % metric_key] = ev['ssim']
            log('ep %4d/%d  loss=%.5f  lr=%.2e  %s_psnr=%.4f  %s_ssim=%.4f'
                % (ep + 1, a.epochs, mean_loss, lr, metric_key, ev['psnr'],
                   metric_key, ev['ssim']))
        else:
            log('ep %4d/%d  loss=%.5f  lr=%.2e'
                % (ep + 1, a.epochs, mean_loss, lr))

        history.append(rec)
        latest = dict(
            epoch=ep, model=net.state_dict(), optimizer=opt.state_dict(),
            best_psnr=best_psnr, best_ssim=best_ssim, best_ep=best_ep,
            init_ckpt=a.init_ckpt, repo_commit=git_head(), seed=a.seed,
            loss='SmoothL1Loss', stage=a.stage,
            lr_initial=a.lr_initial, patch=a.patch,
            select_split=a.select_split,
        )
        torch.save(latest, os.path.join(ckpt_dir, 'latest.pt'))
        if (ep + 1) % SAVE_EVERY == 0:
            torch.save(latest, os.path.join(ckpt_dir, 'epoch_%04d.pt' % (ep + 1)))
        dump_json(os.path.join(diag_dir, 'train_history.json'), json_ready(history))

        if a.patience > 0 and best_ep > 0:
            # best_ep is 1-based trained epoch; seed uses best_ep=0 (epoch -1 + 1)
            # For seed, best_ep is set to 0 via ep_idx_for_ckpt=-1 → best_ep=0
            stagnant = (ep + 1) - best_ep
            if best_ep == 0:
                # only seed best so far; count from ep1
                stagnant = ep + 1
            if stagnant >= a.patience:
                log('[early-stop] patience=%d stagnant=%d best_ep=%d '
                    'best_%s_psnr=%.4f'
                    % (a.patience, stagnant, best_ep, metric_key, best_psnr))
                stopped_early = True
                break

    sel = write_e_star(
        a.root, ckpt_dir, a.select_split,
        extra=dict(
            stage=a.stage,
            lr_initial=a.lr_initial,
            patch=a.patch,
            batch=a.batch,
            n_train=len(train_names),
            patience=a.patience,
            stopped_early=stopped_early,
            train_stopped_at_epoch=(history[-1]['epoch'] + 1
                                    if history and 'epoch' in history[-1]
                                    and history[-1]['epoch'] is not None
                                    and history[-1]['epoch'] >= 0
                                    else 0),
        ))
    key = 'test_psnr' if a.select_split == 'test' else 'dev_psnr'
    log('=== %s done E*=%d %s=%.4f ==='
        % (a.stage, sel['E_star'], key, sel[key]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
