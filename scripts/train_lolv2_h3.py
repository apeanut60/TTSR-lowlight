#!/usr/bin/env python
"""LOLv2-Real H3 FT with frozen P2. RGB MSE, random 192 crops.

P2 stays at the illum-only epoch-1 snapshot. H3 starts from Huawei H3@2000.
Train = Train triplets that have nanobanana_ref_v2. Eval = Official Test + nano v3.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from local_refine_runtime import read_manifest_csv, sha256  # noqa: E402
from loss.loss_enhance import MSSSIMLoss, ReconstructionLoss  # noqa: E402
from train_nikon_domain import (  # noqa: E402
    SAVE_STEPS, ScheduleCropSet, eval_full, load_h3_from, load_p2_from,
    load_schedule,
)
from train_v24_end2end import (  # noqa: E402
    LV2_UNUSED, compute_loss, configure_p2, grad_norm, h3_forward,
    p2_forward, param_delta, snapshot,
)

DATA = '/root/data/datasets/lol-v2-real'
EXP = '/root/data/experiments/TTSR-lowlight-lolv2real'
P2_CKPT = (
    '/root/data/experiments/TTSR-lowlight-lolv2real/'
    'A_illum_only_from_huawei/model/model_00001.pt')
H3_CKPT = (
    '/root/data/experiments/TTSR-lowlight-v2/checkpoints/'
    'H3_v3b0_s42/checkpoint_02000.pt')
COLS = [
    'sample_id', 'camera', 'split', 'low_path', 'gt_path', 'high_path',
    'nano_path', 'low_hw', 'gt_hw', 'nano_hw', 'low_dtype', 'gt_dtype',
    'nano_dtype', 'decode_status',
]


def hw_of(path):
    with Image.open(path) as im:
        w, h = im.size
    return '%dx%d' % (w, h)


def pngs(directory):
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith('.png') and not f.startswith('.'))


def build_manifests(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_low = os.path.join(DATA, 'Train/Low')
    train_gt = os.path.join(DATA, 'Train/Normal')
    train_nano = os.path.join(DATA, 'Train/nanobanana_ref_v2')
    test_low = os.path.join(DATA, 'Test/Low')
    test_gt = os.path.join(DATA, 'Test/Normal')
    test_nano = os.path.join(DATA, 'Test/nanobanana_ref_v3')

    def one(name, split, low_dir, gt_dir, nano_dir):
        gt_name = name.replace('low', 'normal')
        low = os.path.join(low_dir, name)
        high = os.path.join(gt_dir, gt_name)
        nano = os.path.join(nano_dir, name)
        if not (os.path.isfile(high) and os.path.isfile(nano)):
            return None
        low_hw = hw_of(low)
        return dict(
            sample_id=os.path.splitext(name)[0], camera='LOLv2Real', split=split,
            low_path=low, gt_path=high, high_path=high, nano_path=nano,
            low_hw=low_hw, gt_hw=hw_of(high), nano_hw=hw_of(nano),
            low_dtype='uint8', gt_dtype='uint8', nano_dtype='uint8',
            decode_status='ok',
        )

    train_rows = []
    for name in pngs(train_low):
        row = one(name, 'train', train_low, train_gt, train_nano)
        if row is not None:
            train_rows.append(row)
    eval_rows = []
    for name in pngs(test_low):
        row = one(name, 'eval', test_low, test_gt, test_nano)
        if row is None:
            raise SystemExit('official test missing pair: ' + name)
        eval_rows.append(row)
    if len(eval_rows) != 100:
        raise SystemExit('expected 100 official test, got %d' % len(eval_rows))
    train_ids = {r['sample_id'] for r in train_rows}
    eval_ids = {r['sample_id'] for r in eval_rows}
    if train_ids & eval_ids:
        raise SystemExit('train/eval overlap')

    def write(path, rows):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)

    write(os.path.join(out_dir, 'manifests_train.csv'), train_rows)
    write(os.path.join(out_dir, 'manifests_eval.csv'), eval_rows)
    print('manifests train=%d eval=%d' % (len(train_rows), len(eval_rows)))
    return len(train_rows)


def build_schedule(manifest_dir, seed, crop, steps):
    tr = read_manifest_csv(os.path.join(manifest_dir, 'manifests_train.csv'))
    n = len(tr)
    epochs = int(np.ceil(steps / float(n)))
    rng = np.random.RandomState(seed)
    out = os.path.join(manifest_dir, 'random_crop_s%d.csv' % seed)
    rows = []
    step = 0
    for ep in range(epochs):
        order = rng.permutation(n)
        for sample_idx in order:
            if step >= steps:
                break
            step += 1
            row = tr[int(sample_idx)]
            W, H = map(int, row['low_hw'].split('x'))
            if H < crop or W < crop:
                raise SystemExit('crop too large for %s' % row['sample_id'])
            rows.append(dict(
                step=step, epoch=ep, sample_idx=int(sample_idx),
                sample_id=row['sample_id'],
                crop_y=int(rng.randint(0, H - crop + 1)),
                crop_x=int(rng.randint(0, W - crop + 1)),
                flip_h=0, flip_v=0, rotation=0, H=H, W=W, crop=crop,
            ))
        if step >= steps:
            break
    cols = ['step', 'epoch', 'sample_idx', 'sample_id', 'crop_y', 'crop_x',
            'flip_h', 'flip_v', 'rotation', 'H', 'W', 'crop']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    by_id = defaultdict(set)
    for r in rows:
        by_id[r['sample_id']].add((r['crop_y'], r['crop_x']))
    mean_u = float(np.mean([len(v) for v in by_id.values()])) if by_id else 0
    print('schedule %s steps=%d mean_unique_crops=%.2f' % (out, len(rows), mean_u))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--crop', type=int, default=192)
    ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--out_root', default=EXP)
    ap.add_argument('--p2_ckpt', default=P2_CKPT)
    ap.add_argument('--h3_ckpt', default=H3_CKPT)
    ap.add_argument('--allow_overwrite', action='store_true')
    a = ap.parse_args()

    man_dir = os.path.join(a.out_root, 'manifests_h3')
    build_manifests(man_dir)
    schedule_path = build_schedule(man_dir, a.seed, a.crop, a.steps)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = 'H3_frozenP2_mse_s%d' % a.seed
    run_dir = os.path.join(a.out_root, 'checkpoints', run_name)
    if os.path.isdir(run_dir) and not a.allow_overwrite:
        if any(f.startswith('checkpoint_') for f in os.listdir(run_dir)):
            raise SystemExit('refusing overwrite ' + run_dir)
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    p2, p2_args = load_p2_from(a.p2_ckpt, device, joint=False)
    enc, mat, head = load_h3_from(a.h3_ckpt, device, chunk=a.chunk, trainable=True)
    rec = ReconstructionLoss('l1')
    mss = MSSSIMLoss().to(device)
    groups = [
        dict(params=list(enc.parameters()), lr=1e-6, name='encoder'),
        dict(params=list(mat.parameters()), lr=2e-6, name='matcher'),
        dict(params=list(head.parameters()), lr=1e-5, name='head'),
    ]
    opt = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    n_h3 = sum(p.numel() for p in list(enc.parameters()) + list(mat.parameters()) + list(head.parameters()) if p.requires_grad)
    print('frozen P2 %s' % a.p2_ckpt)
    print('H3 init %s trainable=%d' % (a.h3_ckpt, n_h3))

    tr = read_manifest_csv(os.path.join(man_dir, 'manifests_train.csv'))
    ev = read_manifest_csv(os.path.join(man_dir, 'manifests_eval.csv'))
    sch = load_schedule(schedule_path)[:a.steps]
    ds = ScheduleCropSet(sch, {r['sample_id']: r for r in tr})
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=a.workers)

    log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w')
    hist = []
    peak_mem = 0.0
    t0 = time.time()
    snap_enc = snapshot(enc)
    snap_head = snapshot(head)

    def save_eval(step):
        nonlocal peak_mem
        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1 << 20))
        print('--- eval @', step, flush=True)
        summary, per = eval_full(p2, p2_args, enc, mat, head, ev, device, with_h3=True)
        print('  Y PSNR=%.4f SSIM=%.4f | P2-only=%.4f' % (
            summary['psnr'], summary['ssim'], summary['p2_only_psnr']), flush=True)
        with open(os.path.join(run_dir, 'eval_step_%05d.json' % step), 'w') as f:
            json.dump(dict(step=step, summary=summary, per_image=per), f)
        hist.append(dict(step=step, summary=summary))
        payload = dict(
            p2=p2.state_dict(),
            encoder=enc.state_dict(), match=mat.state_dict(), head=head.state_dict(),
            global_step=step, seed=a.seed, loss='mse',
            joint_p2=False, train_h3=True,
            lr_groups={g['name']: g['lr'] for g in groups},
            p2_init=a.p2_ckpt, h3_init=a.h3_ckpt,
            p2_init_sha256=sha256(a.p2_ckpt), h3_init_sha256=sha256(a.h3_ckpt),
            config=vars(a), mean_psnr=summary,
            n_h3_trainable=n_h3, lv2_unused=list(LV2_UNUSED),
        )
        torch.save(payload, os.path.join(run_dir, 'checkpoint_%05d.pt' % step))
        configure_p2(p2, joint=False)
        for m in (enc, mat, head):
            m.train()
        return summary

    save_eval(0)
    step = 0
    for batch in dl:
        step += 1
        low = batch['low'].to(device)
        gt = batch['high'].to(device)
        nano = batch['nano'].to(device)
        y0 = p2_forward(p2, low, nano, joint=False)
        y, _ = h3_forward(enc, mat, head, y0, nano)
        loss, parts = compute_loss(y, gt, 'mse', rec, mss)
        if not torch.isfinite(loss):
            raise SystemExit('non-finite at step %d' % step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g_h = grad_norm(head)
        opt.step()
        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1 << 20))
        if step % a.log_every == 0 or step == 1:
            row = dict(
                step=step, loss=float(loss.detach()), **parts,
                g_head=g_h,
                du_enc=param_delta(snap_enc, enc),
                du_head=param_delta(snap_head, head),
                peak_mem_mb=peak_mem, sample_id=batch['sample_id'][0],
            )
            log_f.write(json.dumps(row) + '\n')
            log_f.flush()
            print('  step %d/%d loss=%.5f mse=%.5f gH=%.2e mem=%.0fMB' % (
                step, a.steps, row['loss'], row['mse'], g_h, peak_mem), flush=True)
        if step in SAVE_STEPS:
            save_eval(step)

    if a.steps not in SAVE_STEPS and a.steps != 0:
        save_eval(a.steps)

    best = None
    for h in hist:
        if best is None or h['summary']['psnr'] > best['summary']['psnr']:
            best = h
    if best is not None:
        with open(os.path.join(run_dir, 'best.json'), 'w') as f:
            json.dump(best, f, indent=2)
        print('BEST step=%d PSNR=%.4f P2-only=%.4f' % (
            best['step'], best['summary']['psnr'], best['summary']['p2_only_psnr']))

    cfg = dict(
        vars(a), run_name=run_name, joint=False, train_h3=True, loss='mse',
        p2_init_sha256=sha256(a.p2_ckpt), h3_init_sha256=sha256(a.h3_ckpt),
        n_h3_trainable=n_h3,
        lr_groups={g['name']: g['lr'] for g in groups},
        peak_mem_mb=peak_mem, elapsed_min=(time.time() - t0) / 60,
        protocol='lolv2real-frozenP2-random-crop-%d' % a.crop,
        n_train=len(tr), n_eval=len(ev), best=best,
        train_nano='Train/nanobanana_ref_v2', eval_nano='Test/nanobanana_ref_v3',
    )
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, 'eval_history.json'), 'w') as f:
        json.dump(hist, f, indent=2)
    log_f.close()
    print('=== done %s in %.1f min ===' % (run_name, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
