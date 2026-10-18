#!/usr/bin/env python
"""LOLv2-Syn: freeze P2 backbone, train only illumination heads on full images.

Why: S1c@400 Official Test is 20.576; per-channel oracle affine on the same
outputs reaches 24.74. Remaining gap is global tone, not texture. Crop MSE on
the full 4.8M params cannot see eval-time image stats for GlobalIllumHead /
RefIllumTransfer. This arm:
  - freezes everything except MainNet.global_illum_head + MainNet.ref_illum
  - trains with full-image RGB MSE (nano ref unchanged)
  - forward matches eval: tiled backbone (no grad) → live illum heads once
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
from model.query_illum_align import estimate_query_align_maps  # noqa: E402
from train_nikon_domain import load_p2_from  # noqa: E402
from train_v24_end2end import grad_norm, param_delta, snapshot  # noqa: E402
from v2_p2_runtime import apply_p2_cfg, tiled_p2  # noqa: E402

EXP = '/root/data/experiments/TTSR-lowlight-lolv2syn'
S1C = (
    '/root/data/experiments/TTSR-lowlight-lolv2syn/checkpoints/'
    'S1c_nano_lr5e6_fromS1b1100_s44/checkpoint_00400.pt')


def configure_illum_only(model):
    """Train only GlobalIllumHead + RefIllumTransfer; freeze the rest."""
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for name, p in model.named_parameters():
        if 'global_illum_head' in name or 'ref_illum' in name:
            p.requires_grad_(True)
            trainable.append(name)
    model.train()
    # Backbone tiles must stay in eval/BN-free mode; heads stay train.
    model.eval()
    model.MainNet.global_illum_head.train()
    model.MainNet.ref_illum.train()
    return trainable


@torch.no_grad()
def tiled_backbone(model, args, lr, lrsr, ref, refsr):
    """Same as tiled_p2 but without illum heads (pre-tone residual)."""
    apply_p2_cfg(model)
    tile = args.tile_size
    overlap = max(0, min(args.tile_overlap, tile // 2))
    g, lut, _ = estimate_query_align_maps(
        lrsr, ref, alpha=args.query_illum_alpha,
        gamma_min=args.query_gamma_min, gamma_max=args.query_gamma_max,
        bins=args.query_hist_bins)
    fwd = dict(
        query_gamma_override=g, query_hist_lut_override=lut,
        stage_mode_lv3=args.stage_mode_lv3,
        stage_mode_lv2=args.stage_mode_lv2,
        stage_mode_lv1=args.stage_mode_lv1,
    )
    _, _, H, W = lr.shape

    def one(y0, y1, x0, x1):
        return model(
            lr=lr[:, :, y0:y1, x0:x1],
            lrsr=lrsr[:, :, y0:y1, x0:x1],
            ref=ref[:, :, y0:y1, x0:x1],
            refsr=refsr[:, :, y0:y1, x0:x1],
            apply_illum=False, apply_ref_illum=False, **fwd)[0]

    if H <= tile and W <= tile:
        return one(0, H, 0, W)
    stride = max(1, tile - overlap)
    nh = max(1, (H - overlap + stride - 1) // stride)
    nw = max(1, (W - overlap + stride - 1) // stride)
    acc = torch.zeros(1, 3, H, W, device=lr.device)
    wt = torch.zeros(1, 1, H, W, device=lr.device)
    my = torch.ones(tile, device=lr.device)
    mx = torch.ones(tile, device=lr.device)
    if overlap > 0:
        ramp = 0.5 - 0.5 * torch.cos(
            torch.linspace(0, math.pi, overlap, device=lr.device))
        my[:overlap] = ramp
        my[-overlap:] = ramp.flip(0)
        mx[:overlap] = ramp
        mx[-overlap:] = ramp.flip(0)
    feather = (my[:, None] * mx[None, :]).view(1, 1, tile, tile)
    for ih in range(nh):
        y0 = min(ih * stride, H - tile)
        for iw in range(nw):
            x0 = min(iw * stride, W - tile)
            t = one(y0, y0 + tile, x0, x0 + tile)
            acc[:, :, y0:y0 + tile, x0:x0 + tile] += t * feather
            wt[:, :, y0:y0 + tile, x0:x0 + tile] += feather
    return acc / wt.clamp(min=1e-8)


def illum_forward(model, args, low, nano, live=True):
    """Backbone no-grad; illum heads live if live=True."""
    yb = tiled_backbone(model, args, low, low, nano, nano)
    if live:
        y = model.apply_ref_illum(yb, low, nano)
        y = model.apply_illum_head(y, low)
    else:
        with torch.no_grad():
            y = model.apply_ref_illum(yb, low, nano)
            y = model.apply_illum_head(y, low)
    return y, yb


class FullImageSet(Dataset):
    def __init__(self, rows, order):
        self.rows = rows
        self.order = list(order)

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        row = self.rows[int(self.order[idx])]
        low = read_rgb_model_tensor(row['low_path'])
        high = read_rgb_model_tensor(row['high_path'])
        h, w = low.shape[-2:]
        if tuple(high.shape[-2:]) != (h, w):
            high = F.interpolate(
                high[None], size=(h, w), mode='bilinear', align_corners=False)[0]
        nano = read_rgb_model_tensor(row['nano_path'], size=(h, w))
        return dict(low=low, high=high, nano=nano, sample_id=row['sample_id'])


@torch.no_grad()
def eval_rgb(p2, args, rows, device):
    was_g = p2.MainNet.global_illum_head.training
    was_r = p2.MainNet.ref_illum.training
    p2.eval()
    per = []
    for row in rows:
        low = _read_eval_tensor(row['low_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=low.shape[-2:])
        gt = _read_eval_tensor(row['high_path'], device)
        y = tiled_p2(p2, args, low, low, nano, nano)
        m = channel_metrics(to_uint8(y), to_uint8(gt))
        per.append(dict(sample_id=row['sample_id'], **m))
    p2.MainNet.global_illum_head.train(was_g)
    p2.MainNet.ref_illum.train(was_r)
    keys = [
        'psnr_rgb', 'ssim_rgb', 'psnr_R', 'psnr_G', 'psnr_B',
        'ssim_R', 'ssim_G', 'ssim_B',
    ]
    summary = {k: float(np.mean([r[k] for r in per])) for k in keys}
    summary['n'] = len(per)
    return summary, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p2_ckpt', default=S1C)
    ap.add_argument('--out_root', default=EXP)
    ap.add_argument('--run_name', default='')
    ap.add_argument('--seed', type=int, default=45)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--allow_overwrite', action='store_true')
    a = ap.parse_args()

    run_name = a.run_name or ('S1d_illum_full_mse_fromS1c400_s%d' % a.seed)
    run_dir = os.path.join(a.out_root, 'checkpoints', run_name)
    if os.path.isdir(run_dir) and not a.allow_overwrite:
        if any(f.startswith('checkpoint_') for f in os.listdir(run_dir)):
            raise SystemExit('refusing overwrite ' + run_dir)
    os.makedirs(run_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    man_dir = os.path.join(a.out_root, 'manifests_h3')
    tr = read_manifest_csv(os.path.join(man_dir, 'manifests_train.csv'))
    ev = read_manifest_csv(os.path.join(man_dir, 'manifests_eval.csv'))

    p2, p2_args = load_p2_from(a.p2_ckpt, device, joint=False)
    names = configure_illum_only(p2)
    params = [p for p in p2.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
    print('illum_params=%d tensors=%d names=%s' % (
        n_train, len(names), names[:6] + (['...'] if len(names) > 6 else [])),
          flush=True)
    print('p2=%s lr=%g steps=%d' % (a.p2_ckpt, a.lr, a.steps), flush=True)

    # Build a step schedule: reshuffled full-image epochs until steps filled.
    rng = np.random.RandomState(a.seed)
    order = []
    while len(order) < a.steps:
        order.extend(rng.permutation(len(tr)).tolist())
    order = order[:a.steps]
    ds = FullImageSet(tr, order)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=a.workers)

    log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w')
    hist = []
    best = None
    no_improve = 0
    peak_mem = 0.0
    t0 = time.time()
    snap = snapshot(p2)

    def save_eval(step):
        nonlocal peak_mem, best, no_improve
        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1 << 20))
        print('--- eval @', step, flush=True)
        summary, per = eval_rgb(p2, p2_args, ev, device)
        print('  PSNR_RGB=%.4f SSIM_RGB=%.4f R/G/B=%.3f/%.3f/%.3f' % (
            summary['psnr_rgb'], summary['ssim_rgb'],
            summary['psnr_R'], summary['psnr_G'], summary['psnr_B']), flush=True)
        with open(os.path.join(run_dir, 'eval_step_%05d.json' % step), 'w') as f:
            json.dump(dict(step=step, summary=summary, per_image=per), f)
        hist.append(dict(step=step, summary=summary))
        payload = dict(
            p2=p2.state_dict(),
            global_step=step, arm='illum', seed=a.seed, loss='mse',
            lr=a.lr, p2_init=a.p2_ckpt, p2_init_sha256=sha256(a.p2_ckpt),
            config=vars(a), mean_psnr=summary, n_illum_trainable=n_train,
            illum_param_names=names,
        )
        torch.save(payload, os.path.join(run_dir, 'checkpoint_%05d.pt' % step))
        torch.save(p2.state_dict(),
                   os.path.join(run_dir, 'p2_weights_%05d.pt' % step))
        configure_illum_only(p2)

        improved = best is None or summary['psnr_rgb'] > best['summary']['psnr_rgb']
        if improved:
            best = dict(step=step, summary=summary)
            no_improve = 0
            with open(os.path.join(run_dir, 'best.json'), 'w') as f:
                json.dump(best, f, indent=2)
            print('  * new best %.4f @%d' % (summary['psnr_rgb'], step), flush=True)
        else:
            no_improve += 1
            print('  no improve (%d/%d)' % (no_improve, a.patience), flush=True)
        return no_improve >= a.patience and step > 0

    stop = save_eval(0)
    step = 0
    for batch in dl:
        if stop:
            break
        step += 1
        low = batch['low'].to(device)
        gt = batch['high'].to(device)
        nano = batch['nano'].to(device)
        y, _ = illum_forward(p2, p2_args, low, nano, live=True)
        loss = F.mse_loss(y, gt)
        if not torch.isfinite(loss):
            raise SystemExit('non-finite at step %d' % step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g = grad_norm(p2)
        opt.step()
        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1 << 20))
        if step % a.log_every == 0 or step == 1:
            row = dict(
                step=step, loss=float(loss.detach()), mse=float(loss.detach()),
                g_illum=g, du_p2=param_delta(snap, p2),
                peak_mem_mb=peak_mem, sample_id=batch['sample_id'][0],
            )
            log_f.write(json.dumps(row) + '\n')
            log_f.flush()
            print('  step %d/%d loss=%.5f g=%.2e' % (
                step, a.steps, row['loss'], g), flush=True)
        if step % a.eval_every == 0:
            stop = save_eval(step)
            if stop:
                print('early stop at step %d' % step, flush=True)
                break

    if step not in {h['step'] for h in hist} and step > 0 and not stop:
        save_eval(step)

    if best is not None:
        print('BEST step=%d PSNR_RGB=%.4f' % (
            best['step'], best['summary']['psnr_rgb']), flush=True)
    cfg = dict(
        vars(a), run_name=run_name, n_illum_trainable=n_train,
        illum_param_names=names, p2_init_sha256=sha256(a.p2_ckpt),
        peak_mem_mb=peak_mem, elapsed_min=(time.time() - t0) / 60,
        protocol='lolv2syn-illum-full-image',
        n_train=len(tr), n_eval=len(ev), best=best, stopped_step=step,
    )
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, 'eval_history.json'), 'w') as f:
        json.dump(hist, f, indent=2)
    log_f.close()
    print('=== done %s in %.1f min ===' % (run_name, (time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
