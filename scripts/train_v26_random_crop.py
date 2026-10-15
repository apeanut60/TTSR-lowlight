#!/usr/bin/env python
"""v2.6 joint P2+H3 with dynamic random 192 crops (precomputed schedule).

C1: P2 LR=1e-6.  C2: P2 LR=2e-7.  Loss = L1 + 1.0*(1-MS-SSIM).
"""
from __future__ import annotations

import argparse
import csv
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
    metrics, read_manifest_csv, sha256, _read_eval_tensor,
)
from loss.loss_enhance import MSSSIMLoss, ReconstructionLoss  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion  # noqa: E402
from train_v24_end2end import (  # noqa: E402
    H3_CKPT, configure_p2, grad_norm, h3_forward, load_h3, load_p2,
    p2_forward, param_delta, snapshot,
)
from train_v25_loss_balance import (  # noqa: E402
    compute_l1_mss, eval_crop_probe, eval_full, stable_crop_xy,
)
from v2_p2_runtime import CKPT as P2_CKPT  # noqa: E402

EXP = '/root/data/experiments/TTSR-lowlight-v26'
V2_EXP = '/root/data/experiments/TTSR-lowlight-v2'
V25 = '/root/data/experiments/TTSR-lowlight-v25'
SAVE_STEPS = (0, 250, 500, 1000, 1500, 2000)

ARM_META = {
    'C1': dict(run='C1_random_p2lr1e6_s%d', p2_lr=1e-6),
    'C2': dict(run='C2_random_p2lr2e7_s%d', p2_lr=2e-7),
}


class ScheduleCropSet(Dataset):
    """Index = training step-1; returns precomputed random crop (no shuffle)."""

    def __init__(self, schedule_rows, train_by_id):
        self.schedule = schedule_rows
        self.train_by_id = train_by_id

    def __len__(self):
        return len(self.schedule)

    def __getitem__(self, idx):
        sch = self.schedule[idx]
        row = self.train_by_id[sch['sample_id']]
        low = read_rgb_model_tensor(row['low_path'])
        high = read_rgb_model_tensor(row['high_path'])
        h, w = low.shape[-2:]
        if tuple(high.shape[-2:]) != (h, w):
            high = F.interpolate(
                high[None], size=(h, w), mode='bilinear', align_corners=False)[0]
        nano = read_rgb_model_tensor(row['nano_path'], size=(h, w))
        top = int(sch['crop_y']); left = int(sch['crop_x'])
        ch = int(sch['crop'])
        sl = (slice(None), slice(top, top + ch), slice(left, left + ch))
        return dict(
            low=low[sl], high=high[sl], nano=nano[sl],
            sample_id=sch['sample_id'],
            crop_y=top, crop_x=left, sched_step=int(sch['step']))


def load_schedule(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda r: int(r['step']))
    return rows


def ensure_train80_lock(out_root):
    """Reuse v2.5 fixed md5 Train80 probe for fitting curves."""
    src = os.path.join(V25, 'probes/train80_lock.json')
    dst_dir = os.path.join(out_root, 'probes')
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, 'train80_lock.json')
    if os.path.isfile(src) and not os.path.isfile(dst):
        import shutil
        shutil.copy2(src, dst)
    if not os.path.isfile(dst):
        # rebuild
        tr = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_train.csv'))
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(len(tr), size=80, replace=False).tolist())
        probe = [tr[i] for i in idx]
        crops = {}
        for row in probe:
            W, H = map(int, row['low_hw'].split('x'))
            top, left = stable_crop_xy(row['sample_id'], 42, H, W, 192)
            crops[row['sample_id']] = dict(top=top, left=left, h=H, w=W)
        lock = dict(n=80, seed=42, crop=192, indices=idx,
                    sample_ids=[r['sample_id'] for r in probe],
                    crops=crops, crop_hash='md5(seed|sample_id)')
        json.dump(lock, open(dst, 'w'), indent=2)
    return json.load(open(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=list(ARM_META))
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--out_root', default=EXP)
    ap.add_argument('--schedule', default=None)
    ap.add_argument('--allow_overwrite', action='store_true')
    ap.add_argument('--limit_steps', type=int, default=0)
    ap.add_argument('--skip_eval', action='store_true')
    a = ap.parse_args()

    meta = ARM_META[a.arm]
    p2_lr = meta['p2_lr']
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

    sched_path = a.schedule or os.path.join(
        a.out_root, 'manifests/random_crop_s42.csv')
    schedule = load_schedule(sched_path)
    steps = a.limit_steps or a.steps
    schedule = schedule[:steps]

    tr = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_train.csv'))
    ev = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_eval.csv'))
    by_id = {r['sample_id']: r for r in tr}
    lock = ensure_train80_lock(a.out_root)
    probe_rows = [by_id[sid] for sid in lock['sample_ids']]

    ds = ScheduleCropSet(schedule, by_id)
    # shuffle=False — order is the schedule
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=a.workers)

    p2, p2_args = load_p2(device, joint=True)
    enc, mat, head, _ = load_h3(device, chunk=a.chunk)
    rec = ReconstructionLoss('l1')
    mss = MSSSIMLoss().to(device)

    groups = [
        dict(params=[p for p in p2.parameters() if p.requires_grad],
             lr=p2_lr, name='p2'),
        dict(params=list(enc.parameters()), lr=1e-6, name='encoder'),
        dict(params=list(mat.parameters()), lr=2e-6, name='matcher'),
        dict(params=list(head.parameters()), lr=1e-5, name='head'),
    ]
    opt = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    print('arm=%s p2_lr=%.1e steps=%d schedule=%s' % (
        a.arm, p2_lr, steps, sched_path), flush=True)

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
        assert not b0['low'].requires_grad
    print('step0 H3 identity OK; first crop y=%d x=%d id=%s' % (
        int(b0['crop_y']), int(b0['crop_x']), b0['sample_id']), flush=True)
    configure_p2(p2, joint=True)
    for m in (enc, mat, head):
        m.train()

    log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w')
    hist = []
    peak_mem = 0.0
    t0 = time.time()
    snap_p2, snap_enc, snap_mat, snap_head = (
        snapshot(p2), snapshot(enc), snapshot(mat), snapshot(head))

    def save_eval(step):
        nonlocal peak_mem
        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1 << 20))
        summary = tr_sum = tr80_full = None
        if not a.skip_eval:
            print('--- eval @', step, flush=True)
            summary, per = eval_full(
                p2, p2_args, enc, mat, head, ev, device, rec, mss)
            tr_sum, tr_per = eval_crop_probe(
                p2, enc, mat, head, probe_rows, lock, device, rec, mss)
            tr80_full = None
            if step in (0, 1000, 2000) or step == steps:
                print('  train80 full-res...', flush=True)
                tr80_full, _ = eval_full(
                    p2, p2_args, enc, mat, head, probe_rows, device, rec, mss)
            print('  Eval PSNR=%.3f SSIM=%.4f | Train80-crop=%.3f/%.4f | '
                  'P2only=%.3f%s' % (
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
                ), f)
            hist.append(dict(step=step, eval30=summary, train80_crop=tr_sum,
                             train80_full=tr80_full))
        ckpt = os.path.join(run_dir, 'checkpoint_%05d.pt' % step)
        torch.save(dict(
            p2=p2.state_dict(), encoder=enc.state_dict(),
            match=mat.state_dict(), head=head.state_dict(),
            optimizer=opt.state_dict(), global_step=step, arm=a.arm,
            seed=a.seed, p2_lr=p2_lr, loss='L1+1.0*(1-MS-SSIM)',
            lr_groups={g['name']: g['lr'] for g in groups},
            p2_init=P2_CKPT, h3_init=H3_CKPT,
            p2_init_sha256=sha256(P2_CKPT), h3_init_sha256=sha256(H3_CKPT),
            schedule=sched_path, config=vars(a),
            mean_psnr=summary, train80_crop=tr_sum,
            crop_scheme='random_schedule_s42',
        ), ckpt)
        configure_p2(p2, joint=True)
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
        y0 = p2_forward(p2, low, nano, joint=True)
        if step == 1:
            assert y0.requires_grad, 'Y0 must require grad for joint FT'
        y, _ = h3_forward(enc, mat, head, y0, nano)
        loss, parts = compute_l1_mss(y, gt, rec, mss, lam=1.0)
        if not torch.isfinite(loss):
            raise SystemExit('non-finite at step %d' % step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g_p2 = grad_norm(p2)
        g_e = grad_norm(enc); g_m = grad_norm(mat); g_h = grad_norm(head)
        if step == 1 and g_p2 <= 0:
            raise SystemExit('P2 grad norm zero at step 1')
        opt.step()
        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1 << 20))
        if step % a.log_every == 0 or step == 1:
            row = dict(
                step=step, **parts,
                g_p2=g_p2, g_enc=g_e, g_mat=g_m, g_head=g_h,
                du_p2=param_delta(snap_p2, p2),
                du_enc=param_delta(snap_enc, enc),
                du_mat=param_delta(snap_mat, mat),
                du_head=param_delta(snap_head, head),
                peak_mem_mb=peak_mem,
                sample_id=batch['sample_id'][0],
                crop_y=int(batch['crop_y'][0]), crop_x=int(batch['crop_x'][0]),
                y0_requires_grad=True,
            )
            log_f.write(json.dumps(row) + '\n'); log_f.flush()
            print('  step %d/%d L=%.4f L1=%.4f MS=%.4f PSNR=%.2f '
                  'gP2=%.2e crop=(%d,%d) mem=%.0fMB' % (
                      step, steps, row['total'], row['l1'], row['ms_ssim'],
                      row['psnr'], g_p2, row['crop_y'], row['crop_x'], peak_mem),
                  flush=True)
        if step in SAVE_STEPS and step != 0:
            save_eval(step)
        if step >= steps:
            break

    if steps not in SAVE_STEPS and steps != 0:
        save_eval(steps)

    cfg = dict(
        vars(a), run_name=run_name, p2_lr=p2_lr,
        p2_init_sha256=sha256(P2_CKPT), h3_init_sha256=sha256(H3_CKPT),
        peak_mem_mb=peak_mem, elapsed_min=(time.time() - t0) / 60,
        protocol='random-crop-192-schedule',
        loss_recipe='L1 + 1.0*(1-MS-SSIM)',
        schedule=sched_path,
    )
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, 'eval_history.json'), 'w') as f:
        json.dump(hist, f, indent=2)
    log_f.close()
    print('=== done %s in %.1f min peak=%.0fMB ===' % (
        run_name, (time.time() - t0) / 60, peak_mem), flush=True)


if __name__ == '__main__':
    main()
