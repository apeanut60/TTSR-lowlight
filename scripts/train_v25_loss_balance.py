#!/usr/bin/env python
"""v2.5 joint P2+H3: L = L1 + λ(1-MS-SSIM), λ ∈ {1.0, 0.3, 0.1}.

Deterministic md5 crops (fixes v2.4 unstable hash()). No MSE in train loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from local_refine_runtime import (  # noqa: E402
    EVAL_QUERY_CHUNK, metrics, read_manifest_csv, sha256, _read_eval_tensor,
)
from loss.loss_enhance import MSSSIMLoss, ReconstructionLoss  # noqa: E402
from model.LocalRefine import LocalSoftMatch, SharedEncoder  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion, count_params  # noqa: E402
from train_v24_end2end import (  # noqa: E402
    H3_CKPT, LV2_UNUSED, V2_EXP, configure_p2, grad_norm, h3_forward,
    load_h3, load_p2, p2_forward, param_delta, snapshot,
)
from v2_p2_runtime import CKPT as P2_CKPT, tiled_p2  # noqa: E402

EXP = '/root/data/experiments/TTSR-lowlight-v25'
SAVE_STEPS = (0, 250, 500, 1000, 1500, 2000)

ARM_META = {
    'W0': dict(run='W0_l1_ms1_s%d', lambda_ms=1.0),
    'W1': dict(run='W1_l1_ms03_s%d', lambda_ms=0.3),
    'W2': dict(run='W2_l1_ms01_s%d', lambda_ms=0.1),
}


def stable_crop_xy(sample_id, seed, H, W, crop):
    dig = hashlib.md5(('%d|%s' % (seed, sample_id)).encode()).hexdigest()
    top = int(dig[:8], 16) % (H - crop + 1)
    left = int(dig[8:16], 16) % (W - crop + 1)
    return top, left


class CropSetMD5(Dataset):
    def __init__(self, rows, crop, seed):
        self.rows = rows
        self.crop = int(crop)
        self.seed = int(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        low = read_rgb_model_tensor(row['low_path'])
        high = read_rgb_model_tensor(row['high_path'])
        h, w = low.shape[-2:]
        if tuple(high.shape[-2:]) != (h, w):
            high = F.interpolate(
                high[None], size=(h, w), mode='bilinear', align_corners=False)[0]
        nano = read_rgb_model_tensor(row['nano_path'], size=(h, w))
        ch = self.crop
        top, left = stable_crop_xy(row['sample_id'], self.seed, h, w, ch)
        sl = (slice(None), slice(top, top + ch), slice(left, left + ch))
        return dict(low=low[sl], high=high[sl], nano=nano[sl],
                    sample_id=row['sample_id'])


def compute_l1_mss(y, gt, rec, mss, lam):
    l1 = rec(y, gt)
    ms = mss(y, gt)
    mse = F.mse_loss(y, gt)
    loss = l1 + float(lam) * ms
    with torch.no_grad():
        # quick PSNR/SSIM on crop for logging only
        p, s, _ = metrics(y, gt)
    return loss, dict(
        l1=float(l1.detach()), ms_ssim=float(ms.detach()),
        mse=float(mse.detach()), psnr=float(p), ssim=float(s),
        total=float(loss.detach()), lambda_ms=float(lam))


@torch.no_grad()
def eval_full(p2, args, enc, mat, head, rows, device, rec, mss):
    was = (p2.training, enc.training, mat.training, head.training)
    p2.eval(); enc.eval(); mat.eval(); head.eval()
    per = {}
    for row in rows:
        low = _read_eval_tensor(row['low_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=low.shape[-2:])
        gt = _read_eval_tensor(row['high_path'], device)
        y0 = tiled_p2(p2, args, low, low, nano, nano)
        y, _ = h3_forward(enc, mat, head, y0, nano)
        p, s, m = metrics(y, gt)
        p2p, p2s, _ = metrics(y0, gt)
        per[row['sample_id']] = dict(
            psnr=p, ssim=s, mse=m,
            l1=float(rec(y, gt).item()),
            ms_ssim=float(mss(y, gt).item()),
            p2_psnr=p2p, p2_ssim=p2s)
    p2.train(was[0]); enc.train(was[1]); mat.train(was[2]); head.train(was[3])
    def mean(k):
        return float(np.mean([v[k] for v in per.values()]))
    summary = {k: mean(k) for k in
               ('psnr', 'ssim', 'mse', 'l1', 'ms_ssim', 'p2_psnr', 'p2_ssim')}
    deltas = [v['psnr'] - v['p2_psnr'] for v in per.values()]
    summary.update(
        median_delta=float(np.median(deltas)),
        p10_delta=float(np.percentile(deltas, 10)),
        worst_delta=float(np.min(deltas)),
        n_lt_m1=int(sum(1 for d in deltas if d < -1)),
        win=int(sum(1 for d in deltas if d > 0)),
        loss_n=int(sum(1 for d in deltas if d < 0)),
    )
    return summary, per


@torch.no_grad()
def eval_crop_probe(p2, enc, mat, head, rows, lock, device, rec, mss):
    was = (p2.training, enc.training, mat.training, head.training)
    p2.eval(); enc.eval(); mat.eval(); head.eval()
    crop = int(lock['crop'])
    crops = lock['crops']
    per = {}
    for row in rows:
        sid = row['sample_id']
        low = read_rgb_model_tensor(row['low_path'])
        high = read_rgb_model_tensor(row['high_path'])
        h, w = low.shape[-2:]
        if tuple(high.shape[-2:]) != (h, w):
            high = F.interpolate(high[None], size=(h, w), mode='bilinear',
                                 align_corners=False)[0]
        nano = read_rgb_model_tensor(row['nano_path'], size=(h, w))
        c = crops[sid]
        top, left = c['top'], c['left']
        sl = (slice(None), slice(top, top + crop), slice(left, left + crop))
        low_c = low[sl][None].to(device)
        nano_c = nano[sl][None].to(device)
        gt_c = high[sl][None].to(device)
        y0 = p2_forward(p2, low_c, nano_c, joint=False)
        y, _ = h3_forward(enc, mat, head, y0, nano_c)
        p, s, m = metrics(y, gt_c)
        p2p, p2s, _ = metrics(y0, gt_c)
        per[sid] = dict(
            psnr=float(p), ssim=float(s), mse=float(m),
            l1=float(rec(y, gt_c).item()),
            ms_ssim=float(mss(y, gt_c).item()),
            p2_psnr=float(p2p), p2_ssim=float(p2s))
    p2.train(was[0]); enc.train(was[1]); mat.train(was[2]); head.train(was[3])
    def mean(k):
        return float(np.mean([v[k] for v in per.values()]))
    return {k: mean(k) for k in
            ('psnr', 'ssim', 'mse', 'l1', 'ms_ssim', 'p2_psnr', 'p2_ssim')}, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=list(ARM_META))
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--crop', type=int, default=192)
    ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--out_root', default=EXP)
    ap.add_argument('--allow_overwrite', action='store_true')
    ap.add_argument('--limit_steps', type=int, default=0)
    ap.add_argument('--skip_eval', action='store_true')
    a = ap.parse_args()

    meta = ARM_META[a.arm]
    lam = meta['lambda_ms']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = meta['run'] % a.seed
    run_dir = os.path.join(a.out_root, 'checkpoints', run_name)
    if os.path.isdir(run_dir) and not a.allow_overwrite:
        if any(f.startswith('checkpoint_') for f in os.listdir(run_dir)):
            raise SystemExit('refusing overwrite ' + run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(a.out_root, 'logs'), exist_ok=True)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    p2, p2_args = load_p2(device, joint=True)
    enc, mat, head, _ = load_h3(device, chunk=a.chunk)
    rec = ReconstructionLoss('l1')
    mss = MSSSIMLoss().to(device)

    groups = [
        dict(params=[p for p in p2.parameters() if p.requires_grad], lr=1e-6, name='p2'),
        dict(params=list(enc.parameters()), lr=1e-6, name='encoder'),
        dict(params=list(mat.parameters()), lr=2e-6, name='matcher'),
        dict(params=list(head.parameters()), lr=1e-5, name='head'),
    ]
    opt = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    tr = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_train.csv'))
    ev = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_eval.csv'))
    lock = json.load(open(os.path.join(a.out_root, 'probes/train80_lock.json')))
    by_id = {r['sample_id']: r for r in tr}
    probe_rows = [by_id[sid] for sid in lock['sample_ids']]

    ds = CropSetMD5(tr, crop=a.crop, seed=a.seed)
    gen = torch.Generator(); gen.manual_seed(a.seed)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=a.workers,
                    generator=gen)

    print('arm=%s λ=%.2f crop=%d p2_train=%d joint=True md5-crops' % (
        a.arm, lam, a.crop,
        sum(p.numel() for p in p2.parameters() if p.requires_grad)))

    # step0 identity
    with torch.no_grad():
        b0 = ds[0]
        y0 = p2_forward(p2, b0['low'][None].to(device),
                        b0['nano'][None].to(device), joint=False)
        f0 = enc((y0 + 1) * 0.5)
        d = head(f0, mat(f0, enc((b0['nano'][None].to(device) + 1) * 0.5)),
                 y0.shape[-2:])
        head2 = V3B0ResidualFusion(96).to(device).eval()
        head2.load_state_dict(torch.load(H3_CKPT, map_location=device)['head'])
        d2 = head2(f0, mat(f0, enc((b0['nano'][None].to(device) + 1) * 0.5)),
                   y0.shape[-2:])
        assert torch.allclose(d, d2, atol=1e-5)
    print('step0 H3 identity OK')
    configure_p2(p2, joint=True)
    for m in (enc, mat, head):
        m.train()

    log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w')
    hist = []
    peak_mem = 0.0
    t0 = time.time()
    steps = a.limit_steps or a.steps
    snap_p2, snap_enc, snap_mat, snap_head = (
        snapshot(p2), snapshot(enc), snapshot(mat), snapshot(head))

    def save_eval(step):
        nonlocal peak_mem
        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1 << 20))
        summary = None
        tr_sum = None
        if not a.skip_eval:
            print('--- eval @', step)
            summary, per = eval_full(
                p2, p2_args, enc, mat, head, ev, device, rec, mss)
            tr_sum, tr_per = eval_crop_probe(
                p2, enc, mat, head, probe_rows, lock, device, rec, mss)
            # Full-res Train80 only at endpoints (tiled cost)
            tr80_full = None
            if step in (0, 1000, 2000) or step == steps:
                print('  train80 full-res...', flush=True)
                tr80_full, _ = eval_full(
                    p2, p2_args, enc, mat, head, probe_rows, device, rec, mss)
            print('  Eval PSNR=%.3f SSIM=%.4f | Train80-crop PSNR=%.3f SSIM=%.4f | '
                  'P2only-E=%.3f%s' % (
                      summary['psnr'], summary['ssim'],
                      tr_sum['psnr'], tr_sum['ssim'], summary['p2_psnr'],
                      (' | Train80-full=%.3f' % tr80_full['psnr']) if tr80_full else ''),
                  flush=True)
            with open(os.path.join(run_dir, 'eval_step_%05d.json' % step), 'w') as f:
                json.dump(dict(
                    step=step, eval30=summary, per_image=per,
                    train80_crop=tr_sum, train80_crop_per=tr_per,
                    train80_full=tr80_full,
                    gap_psnr=tr_sum['psnr'] - summary['psnr'],
                    gap_ssim=tr_sum['ssim'] - summary['ssim'],
                ), f)
            hist.append(dict(step=step, eval30=summary, train80_crop=tr_sum,
                             train80_full=tr80_full))
        ckpt = os.path.join(run_dir, 'checkpoint_%05d.pt' % step)
        torch.save(dict(
            p2=p2.state_dict(), encoder=enc.state_dict(),
            match=mat.state_dict(), head=head.state_dict(),
            optimizer=opt.state_dict(), global_step=step, arm=a.arm,
            seed=a.seed, lambda_ms=lam, loss='L1+λ(1-MS-SSIM)',
            lr_groups={g['name']: g['lr'] for g in groups},
            p2_init=P2_CKPT, h3_init=H3_CKPT,
            p2_init_sha256=sha256(P2_CKPT), h3_init_sha256=sha256(H3_CKPT),
            config=vars(a), mean_psnr=summary, train80_crop=tr_sum,
            crop_scheme='md5(seed|sample_id)',
        ), ckpt)
        configure_p2(p2, joint=True)
        for m in (enc, mat, head):
            m.train()
        return summary

    save_eval(0)
    step = 0
    data_pass = 0
    while step < steps:
        for batch in dl:
            if step >= steps:
                break
            step += 1
            low = batch['low'].to(device)
            gt = batch['high'].to(device)
            nano = batch['nano'].to(device)
            y0 = p2_forward(p2, low, nano, joint=True)
            y, _ = h3_forward(enc, mat, head, y0, nano)
            loss, parts = compute_l1_mss(y, gt, rec, mss, lam)
            if not torch.isfinite(loss):
                raise SystemExit('non-finite at step %d' % step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            g_p2, g_e, g_m, g_h = (
                grad_norm(p2), grad_norm(enc), grad_norm(mat), grad_norm(head))
            opt.step()
            if torch.cuda.is_available():
                peak_mem = max(peak_mem,
                               torch.cuda.max_memory_allocated() / (1 << 20))
            if step % a.log_every == 0 or step == 1:
                row = dict(
                    step=step, data_pass=data_pass, **parts,
                    g_p2=g_p2, g_enc=g_e, g_mat=g_m, g_head=g_h,
                    du_p2=param_delta(snap_p2, p2),
                    du_enc=param_delta(snap_enc, enc),
                    du_mat=param_delta(snap_mat, mat),
                    du_head=param_delta(snap_head, head),
                    peak_mem_mb=peak_mem,
                    sample_id=batch['sample_id'][0],
                )
                log_f.write(json.dumps(row) + '\n'); log_f.flush()
                print('  step %d/%d L=%.4f L1=%.4f MS=%.4f PSNR=%.2f '
                      'gP2=%.2e mem=%.0fMB' % (
                          step, steps, row['total'], row['l1'], row['ms_ssim'],
                          row['psnr'], g_p2, peak_mem))
            if step in SAVE_STEPS and step != 0:
                save_eval(step)
        data_pass += 1

    if steps not in SAVE_STEPS and steps != 0:
        save_eval(steps)

    cfg = dict(
        vars(a), run_name=run_name, lambda_ms=lam,
        p2_init_sha256=sha256(P2_CKPT), h3_init_sha256=sha256(H3_CKPT),
        peak_mem_mb=peak_mem, data_passes=data_pass,
        elapsed_min=(time.time() - t0) / 60,
        protocol='crop-%d-md5' % a.crop,
        loss_recipe='L1 + %.2f*(1-MS-SSIM)' % lam,
        note='Retrain W0 (not reuse A3): crop hash scheme changed to md5',
    )
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, 'eval_history.json'), 'w') as f:
        json.dump(hist, f, indent=2)
    log_f.close()
    print('=== done %s in %.1f min peak=%.0fMB ===' % (
        run_name, (time.time() - t0) / 60, peak_mem))


if __name__ == '__main__':
    main()
