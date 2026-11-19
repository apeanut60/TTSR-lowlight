#!/usr/bin/env python
"""P2-only training with RetinexFormer-like recipe on LSRW camera splits.

Huawei-only or Nikon-only. MSE, geometric augs, mixup, long cosine schedule,
fit/val early stop. Official Eval is final-only (no GT-mean).

Does not change P2 structure / P2_CONFIG. Uses live tiled_p2 for eval.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from dataset.data1_localrefine import read_rgb_model_tensor  # noqa: E402
from eval_nikon_export import channel_metrics, to_uint8  # noqa: E402
from local_refine_runtime import read_manifest_csv, sha256, _read_eval_tensor  # noqa: E402
from lolv2_split import prepare_fit_val  # noqa: E402
from train_nikon_domain import load_p2_from  # noqa: E402
from train_v24_end2end import configure_p2, p2_forward  # noqa: E402
from v2_p2_runtime import tiled_p2  # noqa: E402

DEFAULT_P2 = (
    '/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/'
    'model/model_00004.pt'
)
CAMERA_ROOTS = {
    'Huawei': '/root/data/experiments/TTSR-lowlight-v2',
    'Nikon': '/root/data/experiments/TTSR-lowlight-v27-nikon',
    # Full nano coverage on train+test (unlike LSRW Huawei nano500).
    'LOLv2Real': '/root/data/experiments/TTSR-lowlight-lolv2real/manifests_h3',
}
# man_dir defaults: LOLv2Real root already points at manifests_h3
CAMERA_OUT = {
    'Huawei': 'TTSR-lowlight-p2-official-recipe-huawei',
    'Nikon': 'TTSR-lowlight-p2-official-recipe-nikon',
    'LOLv2Real': 'TTSR-lowlight-p2-official-recipe-lolv2real',
}


class OnlineCropSet(Dataset):
    """Random crop + geometric augs on Low/GT/Nano triplets in [-1,1]."""

    def __init__(self, rows, crop, seed):
        self.rows = list(rows)
        self.crop = int(crop)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def _geom(self, *tensors):
        # identical transform on all
        if self.rng.random() < 0.5:
            tensors = [torch.flip(t, dims=[-1]) for t in tensors]
        if self.rng.random() < 0.5:
            tensors = [torch.flip(t, dims=[-2]) for t in tensors]
        k = self.rng.randint(0, 3)
        if k:
            tensors = [torch.rot90(t, k, dims=[-2, -1]) for t in tensors]
        return tensors

    def __getitem__(self, idx):
        row = self.rows[idx]
        low = read_rgb_model_tensor(row['low_path'])
        high = read_rgb_model_tensor(row['high_path'])
        h, w = low.shape[-2:]
        if tuple(high.shape[-2:]) != (h, w):
            high = F.interpolate(
                high[None], size=(h, w), mode='bilinear', align_corners=False)[0]
        nano = read_rgb_model_tensor(row['nano_path'], size=(h, w))
        ch = min(self.crop, h, w)
        top = 0 if h == ch else self.rng.randint(0, h - ch)
        left = 0 if w == ch else self.rng.randint(0, w - ch)
        sl = (slice(None), slice(top, top + ch), slice(left, left + ch))
        low, high, nano = self._geom(low[sl], high[sl], nano[sl])
        return dict(
            low=low, high=high, nano=nano, sample_id=row['sample_id'],
            idx=idx)


def mixup_batch(low, gt, nano, beta, use_identity, rng):
    """Batch mixup; same lambda for Low/GT/Nano. Tensors [B,C,H,W] in [-1,1]."""
    if beta <= 0:
        return low, gt, nano, 1.0
    if use_identity and rng.random() < 0.5:
        return low, gt, nano, 1.0
    lam = float(np.random.beta(beta, beta))
    perm = torch.randperm(low.size(0), device=low.device)
    low = lam * low + (1.0 - lam) * low[perm]
    gt = lam * gt + (1.0 - lam) * gt[perm]
    nano = lam * nano + (1.0 - lam) * nano[perm]
    return low, gt, nano, lam


def cosine_lr(step, total, base_lr, eta_min, period1=46000):
    """Two-cycle schedule mirroring official RetinexFormer YAML."""
    if step < period1:
        # near-constant high LR (official eta_mins[0] ≈ base)
        return max(base_lr, eta_min)
    t = step - period1
    t_max = max(1, total - period1)
    return eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * t / t_max))


@torch.no_grad()
def eval_p2_rgb(p2, args, rows, device):
    was = p2.training
    p2.eval()
    per = []
    for row in rows:
        low = _read_eval_tensor(row['low_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=low.shape[-2:])
        gt = _read_eval_tensor(row['high_path'], device)
        y0 = tiled_p2(p2, args, low, low, nano, nano)
        m = channel_metrics(to_uint8(y0), to_uint8(gt))
        per.append(dict(sample_id=row['sample_id'], **m))
    p2.train(was)
    keys = [
        'psnr_rgb', 'ssim_rgb', 'psnr_R', 'psnr_G', 'psnr_B',
        'ssim_R', 'ssim_G', 'ssim_B',
    ]
    summary = {k: float(np.mean([r[k] for r in per])) for k in keys}
    summary['n'] = len(per)
    return summary, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', required=True,
                    choices=['Huawei', 'Nikon', 'LOLv2Real'])
    ap.add_argument('--steps', type=int, default=150000)
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--crop', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--accum', type=int, default=8,
                    help='grad accumulation to mimic official batch 8')
    ap.add_argument('--lr', type=float, default=1e-5,
                    help='peak LR; default small FT LR from model_00004')
    ap.add_argument('--eta_min', type=float, default=1e-7)
    ap.add_argument('--period1', type=int, default=46000)
    ap.add_argument('--mixup_beta', type=float, default=1.2)
    ap.add_argument('--no_mixup_identity', action='store_true')
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--eval_every', type=int, default=1000)
    ap.add_argument('--patience', type=int, default=15,
                    help='early stop after this many evals without val gain')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--p2_ckpt', default=DEFAULT_P2)
    ap.add_argument('--from_scratch', action='store_true',
                    help='Ignore --p2_ckpt; randomly init P2 (official-style)')
    ap.add_argument('--out_root', default='')
    ap.add_argument('--man_dir', default='')
    ap.add_argument('--allow_overwrite', action='store_true')
    ap.add_argument('--skip_final_test', action='store_true')
    a = ap.parse_args()

    base = CAMERA_ROOTS[a.camera]
    a.out_root = a.out_root or os.path.join(
        '/root/data/experiments', CAMERA_OUT[a.camera])
    if a.camera == 'LOLv2Real':
        a.man_dir = a.man_dir or base
    else:
        a.man_dir = a.man_dir or os.path.join(base, 'manifests')
    run_name = ('P2_official_mse_mixup_scratch_s%d' % a.seed
                if a.from_scratch else
                'P2_official_mse_mixup_s%d' % a.seed)
    run_dir = os.path.join(a.out_root, 'checkpoints', run_name)
    if os.path.isdir(run_dir) and not a.allow_overwrite:
        if any(f.startswith('checkpoint_') for f in os.listdir(run_dir)):
            raise SystemExit('refusing overwrite ' + run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(a.out_root, 'logs'), exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    train_all = read_manifest_csv(os.path.join(a.man_dir, 'manifests_train.csv'))
    if a.camera in ('Huawei', 'Nikon'):
        train_all = [r for r in train_all if r.get('camera', a.camera) == a.camera]
    # LOLv2Real: keep all train rows (camera field is LOLv2Real)
    if not train_all:
        raise SystemExit('empty train for ' + a.camera)
    split_dir = os.path.join(a.out_root, 'manifests')
    fit, val, split_meta = prepare_fit_val(
        train_all, split_dir, a.seed, a.val_ratio)
    print('split fit=%d val=%d isolation=%s' % (
        split_meta['n_fit'], split_meta['n_val'],
        split_meta.get('scene_isolation')), flush=True)
    test = read_manifest_csv(os.path.join(a.man_dir, 'manifests_eval.csv'))
    print('camera=%s fit=%d val=%d test=%d crop=%d accum=%d lr=%g steps=%d' % (
        a.camera, len(fit), len(val), len(test), a.crop, a.accum, a.lr, a.steps),
          flush=True)

    if a.from_scratch:
        from v2_p2_runtime import build_args, apply_p2_cfg
        from model import TTSREnhance
        p2_args = build_args()
        p2_args.data1_camera = a.camera
        p2 = TTSREnhance.TTSREnhance(p2_args).to(device)
        apply_p2_cfg(p2)
        configure_p2(p2, joint=True)
        init_tag = 'from_scratch'
        init_sha = None
        print('P2 randomly initialized (from_scratch)', flush=True)
    else:
        p2, p2_args = load_p2_from(a.p2_ckpt, device, joint=True)
        configure_p2(p2, joint=True)
        init_tag = a.p2_ckpt
        init_sha = sha256(a.p2_ckpt)
    params = [p for p in p2.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print('P2 trainable params=%d init=%s' % (n_train, init_tag), flush=True)
    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.9, 0.999))

    ds = OnlineCropSet(fit, a.crop, a.seed)
    dl = DataLoader(
        ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
        drop_last=True, persistent_workers=a.workers > 0)
    mix_rng = random.Random(a.seed + 7)

    log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w')
    hist = []
    best = None
    bad = 0
    t0 = time.time()
    step = 0
    data_iter = iter(dl)
    opt.zero_grad(set_to_none=True)

    def run_eval(tag_step, rows, name):
        summary, per = eval_p2_rgb(p2, p2_args, rows, device)
        print('%s PSNR_RGB=%.4f SSIM_RGB=%.4f R/G/B=%.3f/%.3f/%.3f n=%d' % (
            name, summary['psnr_rgb'], summary['ssim_rgb'],
            summary['psnr_R'], summary['psnr_G'], summary['psnr_B'],
            summary['n']), flush=True)
        with open(os.path.join(run_dir, 'eval_%s_%05d.json' % (name, tag_step)),
                  'w') as f:
            json.dump(dict(step=tag_step, summary=summary, per_image=per), f)
        return summary

    print('--- eval @ 0 (val) ---', flush=True)
    s0 = run_eval(0, val, 'val')
    hist.append(dict(step=0, summary=s0))
    best = dict(step=0, summary=s0)
    torch.save(dict(
        p2=p2.state_dict(), global_step=0, config=vars(a),
        p2_init=init_tag, p2_init_sha256=init_sha,
        mean_psnr=s0), os.path.join(run_dir, 'checkpoint_00000.pt'))

    while step < a.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        step += 1
        lr = cosine_lr(step, a.steps, a.lr, a.eta_min, a.period1)
        for g in opt.param_groups:
            g['lr'] = lr

        low = batch['low'].to(device)
        gt = batch['high'].to(device)
        nano = batch['nano'].to(device)
        low, gt, nano, lam = mixup_batch(
            low, gt, nano, a.mixup_beta, not a.no_mixup_identity, mix_rng)

        # batch>1: forward each sample (P2 path assumes B-compatible)
        losses = []
        for i in range(low.size(0)):
            y0 = p2_forward(p2, low[i:i + 1], nano[i:i + 1], joint=True)
            losses.append(F.mse_loss(y0, gt[i:i + 1]))
        loss = torch.stack(losses).mean() / a.accum
        if not torch.isfinite(loss):
            raise SystemExit('non-finite at step %d' % step)
        loss.backward()

        if step % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % a.log_every == 0 or step == 1:
            row = dict(
                step=step, loss=float(loss.detach() * a.accum), lr=lr, lam=lam,
                sample_id=batch['sample_id'][0])
            log_f.write(json.dumps(row) + '\n'); log_f.flush()
            print('  step %d/%d loss=%.5f lr=%.3e lam=%.3f' % (
                step, a.steps, row['loss'], lr, lam), flush=True)

        if step % a.eval_every == 0:
            print('--- eval @ %d (val) ---' % step, flush=True)
            summary = run_eval(step, val, 'val')
            hist.append(dict(step=step, summary=summary))
            ckpt = os.path.join(run_dir, 'checkpoint_%05d.pt' % step)
            torch.save(dict(
                p2=p2.state_dict(), global_step=step, config=vars(a),
                p2_init=init_tag, p2_init_sha256=init_sha,
                mean_psnr=summary), ckpt)
            torch.save(p2.state_dict(),
                       os.path.join(run_dir, 'p2_weights_%05d.pt' % step))
            if summary['psnr_rgb'] > best['summary']['psnr_rgb'] + 1e-4:
                best = dict(step=step, summary=summary)
                bad = 0
                torch.save(dict(
                    p2=p2.state_dict(), global_step=step, config=vars(a),
                    p2_init=init_tag, p2_init_sha256=init_sha,
                    mean_psnr=summary),
                    os.path.join(run_dir, 'checkpoint_best.pt'))
                print('BEST step=%d PSNR_RGB=%.4f' % (
                    step, summary['psnr_rgb']), flush=True)
            else:
                bad += 1
                print('no improve %d/%d (best %.4f @ %d)' % (
                    bad, a.patience, best['summary']['psnr_rgb'], best['step']),
                      flush=True)
                if bad >= a.patience:
                    print('early stop at step %d' % step, flush=True)
                    break

    # final official test with best ckpt
    best_path = os.path.join(run_dir, 'checkpoint_best.pt')
    if os.path.isfile(best_path):
        ck = torch.load(best_path, map_location=device)
        p2.load_state_dict(ck['p2'])
        configure_p2(p2, joint=True)
        print('loaded best step=%d for final test' % ck['global_step'], flush=True)
    final = None
    if not a.skip_final_test:
        print('--- FINAL Eval (official test) ---', flush=True)
        final = run_eval(best['step'], test, 'test')
        with open(os.path.join(run_dir, 'final_test.json'), 'w') as f:
            json.dump(dict(best_step=best['step'], summary=final), f, indent=2)

    cfg = dict(
        vars(a), run_name=run_name, n_fit=len(fit), n_val=len(val),
        n_test=len(test), n_p2_trainable=n_train,
        p2_init=init_tag, p2_init_sha256=init_sha,
        best=best, final_test=final,
        elapsed_min=(time.time() - t0) / 60,
        protocol='p2-official-recipe-mse-mixup-no-gt-mean',
        gt_mean=False,
    )
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(hist, f)
    print('WROTE', run_dir, 'best=%.4f @ %d' % (
        best['summary']['psnr_rgb'], best['step']), flush=True)
    log_f.close()


if __name__ == '__main__':
    main()
