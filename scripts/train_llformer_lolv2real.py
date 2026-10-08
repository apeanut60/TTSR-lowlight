#!/usr/bin/env python
"""Stage A1: fine-tune LLFormerBridge on LOLv2-Real train625. SmoothL1 only."""

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


def evaluate(net, names, data_dir, device, limit=0):
    ds = LOLv2PairDataset(names if not limit else names[:limit],
                          data_dir=data_dir, split='Train', train=False)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--data_dir', default=DATA_DIR)
    ap.add_argument('--init_ckpt', default=OFFICIAL_LOL_CKPT)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--batch', type=int, default=BATCH)
    ap.add_argument('--patch', type=int, default=PATCH)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--resume', default='',
                    help='optional latest.pt path (plan default: no resume)')
    a = ap.parse_args(_CLI)

    lock_path = os.path.join(a.root, 'artifact_lock.json')
    if not os.path.isfile(lock_path):
        raise SystemExit('run setup_llformer_lolv2real.py first')
    lock = json.load(open(lock_path))
    if lock.get('official_test_allowed_during_selection'):
        raise SystemExit('lock allows Official Test during selection — refuse')

    set_seed(a.seed)
    train_names = read_split_txt(lock['train625_path'])
    dev_names = read_split_txt(lock['dev64_path'])
    ds = LOLv2PairDataset(train_names, data_dir=a.data_dir, split='Train',
                          train=True, patch=a.patch)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=True,
                        num_workers=a.num_workers, pin_memory=True,
                        drop_last=True)

    net, _ = load_llformer_bridge(a.init_ckpt, a.device, train=True)
    for p in net.parameters():
        p.requires_grad_(True)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=LR_INITIAL, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=0.0)
    crit = nn.SmoothL1Loss()

    ckpt_dir = os.path.join(a.root, 'checkpoints')
    diag_dir = os.path.join(a.root, 'diagnostics')
    os.makedirs(ckpt_dir, exist_ok=True)
    start_ep = 0
    best_psnr = -1.0
    best_ssim = -1.0
    history = []

    if a.resume:
        blob = torch.load(a.resume, map_location='cpu')
        net.load_state_dict(blob['model'], strict=True)
        opt.load_state_dict(blob['optimizer'])
        start_ep = int(blob['epoch']) + 1
        best_psnr = float(blob.get('best_psnr', -1))
        best_ssim = float(blob.get('best_ssim', -1))
        print('[resume] epoch=%d best_psnr=%.4f' % (start_ep, best_psnr), flush=True)

    log_path = os.path.join(a.root, 'logs', 'train_a1.log')
    print('=== A1 train epochs=%d batch=%d patch=%d ==='
          % (a.epochs, a.batch, a.patch), flush=True)

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    t0 = time.time()
    for ep in range(start_ep, a.epochs):
        lr = cosine_lr(ep, epochs=a.epochs)
        for g in opt.param_groups:
            g['lr'] = lr
        net.train()
        losses = []
        for batch in loader:
            x = batch['low'].to(a.device, non_blocking=True)
            h = batch['high'].to(a.device, non_blocking=True)
            # patches are multiple of 16 (128)
            y = net(x)
            loss = crit(y, h)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        rec = dict(epoch=ep, lr=lr, train_loss=mean_loss,
                   elapsed_h=(time.time() - t0) / 3600.0)

        if (ep + 1) % VAL_EVERY == 0 or ep == 0 or ep + 1 == a.epochs:
            ev = evaluate(net, dev_names, a.data_dir, a.device)
            rec['dev_psnr'] = ev['psnr']
            rec['dev_ssim'] = ev['ssim']
            out_ev = os.path.join(diag_dir, 'val_ep%04d' % (ep + 1))
            os.makedirs(out_ev, exist_ok=True)
            dump_json(os.path.join(out_ev, 'summary.json'),
                      json_ready(dict(epoch=ep + 1, psnr=ev['psnr'],
                                      ssim=ev['ssim'], n=ev['n'])))
            dump_json(os.path.join(out_ev, 'per_image.json'),
                      json_ready(ev['per_image']))
            if ev['psnr'] > best_psnr:
                best_psnr = ev['psnr']
                torch.save(dict(
                    epoch=ep, model=net.state_dict(),
                    best_psnr=best_psnr, best_ssim=ev['ssim'],
                    init_ckpt=a.init_ckpt, init_sha=file_sha256(a.init_ckpt),
                    repo_commit=git_head(), seed=a.seed, loss='SmoothL1Loss',
                    stage='A1',
                ), os.path.join(ckpt_dir, 'model_bestPSNR.pth'))
                log('[bestPSNR] ep=%d psnr=%.4f' % (ep + 1, best_psnr))
            if ev['ssim'] > best_ssim:
                best_ssim = ev['ssim']
                torch.save(dict(
                    epoch=ep, model=net.state_dict(),
                    best_psnr=ev['psnr'], best_ssim=best_ssim,
                    init_ckpt=a.init_ckpt, init_sha=file_sha256(a.init_ckpt),
                    repo_commit=git_head(), seed=a.seed, loss='SmoothL1Loss',
                    stage='A1',
                ), os.path.join(ckpt_dir, 'model_bestSSIM.pth'))
            log('ep %4d/%d  loss=%.5f  lr=%.2e  dev_psnr=%.4f  dev_ssim=%.4f'
                % (ep + 1, a.epochs, mean_loss, lr, ev['psnr'], ev['ssim']))
        else:
            log('ep %4d/%d  loss=%.5f  lr=%.2e'
                % (ep + 1, a.epochs, mean_loss, lr))

        history.append(rec)
        latest = dict(
            epoch=ep, model=net.state_dict(), optimizer=opt.state_dict(),
            best_psnr=best_psnr, best_ssim=best_ssim,
            init_ckpt=a.init_ckpt, repo_commit=git_head(), seed=a.seed,
            loss='SmoothL1Loss', stage='A1',
        )
        torch.save(latest, os.path.join(ckpt_dir, 'latest.pt'))
        if (ep + 1) % SAVE_EVERY == 0:
            torch.save(latest, os.path.join(ckpt_dir, 'epoch_%04d.pt' % (ep + 1)))
        dump_json(os.path.join(diag_dir, 'train_history.json'), json_ready(history))

    # lock E*
    best_path = os.path.join(ckpt_dir, 'model_bestPSNR.pth')
    blob = torch.load(best_path, map_location='cpu')
    e_star = int(blob['epoch']) + 1
    sel = dict(
        E_star=e_star,
        checkpoint=best_path,
        checkpoint_sha=file_sha256(best_path),
        dev_psnr=float(blob['best_psnr']),
        dev_ssim=float(blob.get('best_ssim', -1)),
        repo_commit=git_head(),
        selection_rule='max_dev64_psnr',
        official_test_not_used=True,
    )
    dump_json(os.path.join(a.root, 'diagnostics', 'E_star.json'), json_ready(sel))
    log('=== A1 done E*=%d dev_psnr=%.4f ===' % (e_star, sel['dev_psnr']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
