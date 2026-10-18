#!/usr/bin/env python
"""LOLv2-Real domain FT: P0 eval / L1 P2-only / L2 frozen-P2+H3.

Protocol mirrors Nikon N1/N2 with LOLv2 hygiene:
  - RGB MSE, random 192 crops
  - Full P2_CONFIG (M1 + lv2 branch_bypass)
  - Dense eval + early stop
  - No LLFormer, no GT-degrade, no exposure loss
"""
from __future__ import annotations

import argparse
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

from eval_nikon_export import channel_metrics, to_uint8  # noqa: E402
from local_refine_runtime import read_manifest_csv, sha256, _read_eval_tensor  # noqa: E402
from loss.loss_enhance import MSSSIMLoss, ReconstructionLoss  # noqa: E402
from model.LocalRefine import LocalSoftMatch, SharedEncoder  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion  # noqa: E402
from train_nikon_domain import (  # noqa: E402
    ScheduleCropSet, load_h3_from, load_p2_from, load_schedule,
)
from train_v24_end2end import (  # noqa: E402
    LV2_UNUSED, compute_loss, configure_p2, grad_norm, h3_forward,
    p2_forward, param_delta, snapshot,
)
from v2_p2_runtime import tiled_p2  # noqa: E402


def build_h3_scratch(device, chunk=4096):
    enc = SharedEncoder(32).to(device)
    mat = LocalSoftMatch(32, 4, 0.1, chunk=chunk).to(device)
    head = V3B0ResidualFusion(96).to(device)
    for m in (enc, mat, head):
        for p in m.parameters():
            p.requires_grad_(True)
        m.train()
    return enc, mat, head

DATA = '/root/data/datasets/lol-v2-real'
EXP = '/root/data/experiments/TTSR-lowlight-lolv2real'
HUAWEI_P2 = (
    '/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/'
    'model/model_00004.pt')
ILLUM_P2 = (
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
        return dict(
            sample_id=os.path.splitext(name)[0], camera='LOLv2Real', split=split,
            low_path=low, gt_path=high, high_path=high, nano_path=nano,
            low_hw=hw_of(low), gt_hw=hw_of(high), nano_hw=hw_of(nano),
            low_dtype='uint8', gt_dtype='uint8', nano_dtype='uint8',
            decode_status='ok',
        )

    train_rows, eval_rows = [], []
    for name in pngs(train_low):
        row = one(name, 'train', train_low, train_gt, train_nano)
        if row is not None:
            train_rows.append(row)
    for name in pngs(test_low):
        row = one(name, 'eval', test_low, test_gt, test_nano)
        if row is None:
            raise SystemExit('official test missing pair: ' + name)
        eval_rows.append(row)
    if len(eval_rows) != 100:
        raise SystemExit('expected 100 official test, got %d' % len(eval_rows))

    def write(path, rows):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)

    write(os.path.join(out_dir, 'manifests_train.csv'), train_rows)
    write(os.path.join(out_dir, 'manifests_eval.csv'), eval_rows)
    print('manifests train=%d eval=%d' % (len(train_rows), len(eval_rows)), flush=True)
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
    print('schedule %s steps=%d mean_unique_crops=%.2f' % (out, len(rows), mean_u), flush=True)
    return out


@torch.no_grad()
def eval_rgb(p2, args, enc, mat, head, rows, device, with_h3=False):
    was = p2.training
    p2.eval()
    was_h = None
    if with_h3:
        was_h = (enc.training, mat.training, head.training)
        enc.eval(); mat.eval(); head.eval()
    per = []
    for row in rows:
        low = _read_eval_tensor(row['low_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=low.shape[-2:])
        gt = _read_eval_tensor(row['high_path'], device)
        y0 = tiled_p2(p2, args, low, low, nano, nano)
        if with_h3:
            y, _ = h3_forward(enc, mat, head, y0, nano)
        else:
            y = y0
        m = channel_metrics(to_uint8(y), to_uint8(gt))
        mp = channel_metrics(to_uint8(y0), to_uint8(gt))
        per.append(dict(
            sample_id=row['sample_id'], **m,
            p2_psnr_rgb=mp['psnr_rgb'], p2_ssim_rgb=mp['ssim_rgb'],
        ))
    p2.train(was)
    if with_h3:
        enc.train(was_h[0]); mat.train(was_h[1]); head.train(was_h[2])
    keys = [
        'psnr_rgb', 'ssim_rgb', 'psnr_R', 'psnr_G', 'psnr_B',
        'ssim_R', 'ssim_G', 'ssim_B', 'p2_psnr_rgb', 'p2_ssim_rgb',
    ]
    summary = {k: float(np.mean([r[k] for r in per])) for k in keys}
    summary['n'] = len(per)
    if with_h3:
        deltas = [r['psnr_rgb'] - r['p2_psnr_rgb'] for r in per]
        summary['median_delta'] = float(np.median(deltas))
        summary['win'] = int(sum(1 for d in deltas if d > 0))
    return summary, per


def run_p0(device, man_dir, out_root):
    ev = read_manifest_csv(os.path.join(man_dir, 'manifests_eval.csv'))
    out = os.path.join(out_root, 'p0_full_p2config')
    os.makedirs(out, exist_ok=True)
    results = {}
    for label, path in (
        ('huawei_00004', HUAWEI_P2),
        ('illum_e1', ILLUM_P2),
    ):
        print('=== P0 %s ===' % label, flush=True)
        p2, args = load_p2_from(path, device, joint=False)
        summary, per = eval_rgb(p2, args, None, None, None, ev, device, with_h3=False)
        results[label] = dict(
            ckpt=path, sha256=sha256(path), summary=summary)
        with open(os.path.join(out, '%s.json' % label), 'w') as f:
            json.dump(dict(summary=summary, per_image=per, ckpt=path), f, indent=2)
        print('  PSNR_RGB=%.4f SSIM_RGB=%.4f R/G/B=%.3f/%.3f/%.3f' % (
            summary['psnr_rgb'], summary['ssim_rgb'],
            summary['psnr_R'], summary['psnr_G'], summary['psnr_B']), flush=True)
        del p2
        torch.cuda.empty_cache()
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote', os.path.join(out, 'summary.json'), flush=True)
    return results


def train_arm(a, device, man_dir):
    joint = a.arm == 'P2'
    train_h3 = a.arm == 'A0'
    scratch = train_h3 and a.h3_init == 'scratch'
    if a.arm == 'P2':
        run_name = 'L1_p2_only_mse_s%d' % a.seed
        p2_ckpt = a.p2_ckpt
        h3_ckpt = None
    elif scratch:
        run_name = 'L2b_frozenP2_h3scratch_mse_s%d' % a.seed
        p2_ckpt = a.p2_ckpt
        h3_ckpt = None
    else:
        run_name = 'L2_frozenP2_mse_s%d' % a.seed
        p2_ckpt = a.p2_ckpt
        h3_ckpt = a.h3_ckpt

    run_dir = os.path.join(a.out_root, 'checkpoints', run_name)
    if os.path.isdir(run_dir) and not a.allow_overwrite:
        if any(f.startswith('checkpoint_') for f in os.listdir(run_dir)):
            raise SystemExit('refusing overwrite ' + run_dir)
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    p2, p2_args = load_p2_from(p2_ckpt, device, joint=joint)
    enc = mat = head = None
    if train_h3:
        if scratch:
            enc, mat, head = build_h3_scratch(device, chunk=a.chunk)
        else:
            enc, mat, head = load_h3_from(h3_ckpt, device, chunk=a.chunk, trainable=True)

    rec = ReconstructionLoss('l1')
    mss = MSSSIMLoss().to(device)
    groups = []
    if joint:
        groups.append(dict(
            params=[p for p in p2.parameters() if p.requires_grad],
            lr=a.p2_lr, name='p2'))
    if train_h3:
        # Scratch needs ~v3b0 rates; Huawei-FT keeps the tiny transfer LRs.
        if scratch:
            enc_lr, match_lr, head_lr = 1e-4, 1e-4, 1e-4
        else:
            enc_lr, match_lr, head_lr = 1e-6, 2e-6, 1e-5
        groups.append(dict(params=list(enc.parameters()), lr=enc_lr, name='encoder'))
        groups.append(dict(params=list(mat.parameters()), lr=match_lr, name='matcher'))
        groups.append(dict(params=list(head.parameters()), lr=head_lr, name='head'))
    opt = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    tr = read_manifest_csv(os.path.join(man_dir, 'manifests_train.csv'))
    ev = read_manifest_csv(os.path.join(man_dir, 'manifests_eval.csv'))
    schedule_path = a.schedule or os.path.join(man_dir, 'random_crop_s%d.csv' % a.seed)
    if not os.path.isfile(schedule_path):
        schedule_path = build_schedule(man_dir, a.seed, a.crop, a.steps)
    sch = load_schedule(schedule_path)[:a.steps]
    ds = ScheduleCropSet(sch, {r['sample_id']: r for r in tr})
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=a.workers)

    n_p2 = sum(p.numel() for p in p2.parameters() if p.requires_grad)
    print('arm=%s joint=%s train_h3=%s h3_init=%s p2_lr=%g p2_train=%d steps=%d' % (
        a.arm, joint, train_h3, a.h3_init if train_h3 else 'N/A',
        a.p2_lr, n_p2, a.steps), flush=True)
    print('p2=%s\nh3=%s' % (p2_ckpt, h3_ckpt if h3_ckpt else 'scratch'), flush=True)
    if train_h3:
        print('h3_lr enc/match/head = %g / %g / %g' % (
            groups[-3]['lr'], groups[-2]['lr'], groups[-1]['lr']), flush=True)

    log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w')
    hist = []
    peak_mem = 0.0
    t0 = time.time()
    snap_p2 = snapshot(p2)
    snap_enc = snapshot(enc) if enc is not None else {}
    snap_head = snapshot(head) if head is not None else {}
    best = None
    no_improve = 0

    def save_eval(step):
        nonlocal peak_mem, best, no_improve
        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1 << 20))
        print('--- eval @', step, flush=True)
        summary, per = eval_rgb(
            p2, p2_args, enc, mat, head, ev, device, with_h3=train_h3)
        key = 'psnr_rgb'
        print('  Y PSNR_RGB=%.4f SSIM_RGB=%.4f | P2-only=%.4f' % (
            summary['psnr_rgb'], summary['ssim_rgb'], summary['p2_psnr_rgb']), flush=True)
        with open(os.path.join(run_dir, 'eval_step_%05d.json' % step), 'w') as f:
            json.dump(dict(step=step, summary=summary, per_image=per), f)
        hist.append(dict(step=step, summary=summary))
        payload = dict(
            p2=p2.state_dict(),
            global_step=step, arm=a.arm, seed=a.seed, loss='mse',
            joint_p2=joint, train_h3=train_h3,
            lr_groups={g['name']: g['lr'] for g in groups},
            p2_init=p2_ckpt, h3_init=('scratch' if scratch else h3_ckpt),
            p2_init_sha256=sha256(p2_ckpt),
            config=vars(a), mean_psnr=summary,
            n_p2_trainable=n_p2, lv2_unused=list(LV2_UNUSED),
        )
        if enc is not None:
            payload.update(
                encoder=enc.state_dict(), match=mat.state_dict(), head=head.state_dict())
            if h3_ckpt:
                payload['h3_init_sha256'] = sha256(h3_ckpt)
        torch.save(payload, os.path.join(run_dir, 'checkpoint_%05d.pt' % step))
        if a.arm == 'P2':
            torch.save(p2.state_dict(),
                       os.path.join(run_dir, 'p2_weights_%05d.pt' % step))
        configure_p2(p2, joint)
        if train_h3:
            for m in (enc, mat, head):
                m.train()

        improved = best is None or summary[key] > best['summary'][key]
        if improved:
            best = dict(step=step, summary=summary)
            no_improve = 0
            with open(os.path.join(run_dir, 'best.json'), 'w') as f:
                json.dump(best, f, indent=2)
            print('  * new best %.4f @%d' % (summary[key], step), flush=True)
        else:
            no_improve += 1
            print('  no improve (%d/%d)' % (no_improve, a.patience), flush=True)
        return summary, no_improve >= a.patience and step > 0

    stop = False
    _, stop = save_eval(0)
    if stop:
        print('early stop immediately (unexpected)', flush=True)

    step = 0
    for batch in dl:
        if stop:
            break
        step += 1
        low = batch['low'].to(device)
        gt = batch['high'].to(device)
        nano = batch['nano'].to(device)
        y0 = p2_forward(p2, low, nano, joint=joint)
        if train_h3:
            y, _ = h3_forward(enc, mat, head, y0, nano)
        else:
            y = y0
        loss, parts = compute_loss(y, gt, 'mse', rec, mss)
        if not torch.isfinite(loss):
            raise SystemExit('non-finite at step %d' % step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g_p2 = grad_norm(p2)
        g_h = grad_norm(head) if head is not None else 0.0
        opt.step()
        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1 << 20))
        if step % a.log_every == 0 or step == 1:
            row = dict(
                step=step, loss=float(loss.detach()), **parts,
                g_p2=g_p2, g_head=g_h,
                du_p2=param_delta(snap_p2, p2),
                du_enc=param_delta(snap_enc, enc) if enc else 0.0,
                du_head=param_delta(snap_head, head) if head else 0.0,
                peak_mem_mb=peak_mem, sample_id=batch['sample_id'][0],
            )
            log_f.write(json.dumps(row) + '\n')
            log_f.flush()
            print('  step %d/%d loss=%.5f mse=%.5f gP2=%.2e gH=%.2e' % (
                step, a.steps, row['loss'], row['mse'], g_p2, g_h), flush=True)
        if step % a.eval_every == 0:
            _, stop = save_eval(step)
            if stop:
                print('early stop at step %d' % step, flush=True)
                break

    if step not in {h['step'] for h in hist} and step > 0 and not stop:
        save_eval(step)

    if best is not None:
        print('BEST step=%d PSNR_RGB=%.4f P2-only=%.4f' % (
            best['step'], best['summary']['psnr_rgb'],
            best['summary']['p2_psnr_rgb']), flush=True)

    cfg = dict(
        vars(a), run_name=run_name, joint=joint, train_h3=train_h3,
        p2_init_sha256=sha256(p2_ckpt),
        n_p2_trainable=n_p2,
        lr_groups={g['name']: g['lr'] for g in groups},
        peak_mem_mb=peak_mem, elapsed_min=(time.time() - t0) / 60,
        protocol='lolv2real-domain-random-crop-%d' % a.crop,
        n_train=len(tr), n_eval=len(ev), best=best,
        stopped_step=step,
    )
    if train_h3 and h3_ckpt:
        cfg['h3_init_sha256'] = sha256(h3_ckpt)
    cfg['h3_init'] = 'scratch' if scratch else (h3_ckpt or None)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, 'eval_history.json'), 'w') as f:
        json.dump(hist, f, indent=2)
    log_f.close()
    print('=== done %s in %.1f min ===' % (run_name, (time.time() - t0) / 60), flush=True)
    return best, run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['P0', 'P2', 'A0'])
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--crop', type=int, default=192)
    ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--patience', type=int, default=3,
                    help='stop after this many consecutive evals without new best')
    ap.add_argument('--out_root', default=EXP)
    ap.add_argument('--p2_ckpt', default=HUAWEI_P2)
    ap.add_argument('--h3_ckpt', default=H3_CKPT)
    ap.add_argument('--h3_init', choices=['huawei', 'scratch'], default='huawei',
                    help='A0 only: load Huawei H3 or train H3 from random init')
    ap.add_argument('--p2_lr', type=float, default=2e-6)
    ap.add_argument('--allow_overwrite', action='store_true')
    ap.add_argument('--schedule', default='')
    a = ap.parse_args()

    if a.arm == 'A0' and a.h3_init == 'huawei':
        a.steps = min(a.steps, 1000)
    # scratch H3 needs more steps; default keep caller --steps (2000)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    man_dir = os.path.join(a.out_root, 'manifests_h3')
    if not os.path.isfile(os.path.join(man_dir, 'manifests_train.csv')):
        build_manifests(man_dir)
    else:
        print('reuse manifests in', man_dir, flush=True)

    if a.arm == 'P0':
        run_p0(device, man_dir, a.out_root)
    else:
        if a.arm == 'P2' and not os.path.isfile(
                os.path.join(man_dir, 'random_crop_s%d.csv' % a.seed)):
            build_schedule(man_dir, a.seed, a.crop, a.steps)
        train_arm(a, device, man_dir)


if __name__ == '__main__':
    main()
