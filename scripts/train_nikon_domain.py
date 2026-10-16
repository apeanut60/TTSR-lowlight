#!/usr/bin/env python
"""Nikon domain FT stages: P2-only / A0 (frozen P2+H3) / A2 (joint MSE).

Uses random-crop schedule (v2.6 hygiene) and RGB MSE for PSNR path.
P2-only also supports L1+MS-SSIM via --loss.
"""
from __future__ import annotations

import argparse
import csv
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
from model.LocalRefine import LocalSoftMatch, SharedEncoder  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion, count_params  # noqa: E402
from model.query_illum_align import estimate_query_align_maps  # noqa: E402
from train_v24_end2end import (  # noqa: E402
    LV2_UNUSED, compute_loss, configure_p2, grad_norm, h3_forward,
    p2_forward, param_delta, snapshot,
)
from v2_p2_runtime import apply_p2_cfg, build_args, tiled_p2  # noqa: E402

EXP = '/root/data/experiments/TTSR-lowlight-v27-nikon'
DEFAULT_P2 = ('/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei/'
              'model/model_00004.pt')
DEFAULT_H3 = ('/root/data/experiments/TTSR-lowlight-v2/checkpoints/'
              'H3_v3b0_s42/checkpoint_02000.pt')
SAVE_STEPS = (0, 250, 500, 1000, 1500, 2000)

ARM_META = {
    'P2': dict(run='N1_p2_only_%s_s%d', joint=True, train_h3=False),
    'A0': dict(run='N2_frozenP2_mse_s%d', joint=False, train_h3=True),
    'A2': dict(run='N3_jointP2_mse_s%d', joint=True, train_h3=True),
}


class ScheduleCropSet(Dataset):
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
            sample_id=sch['sample_id'])


def load_schedule(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda r: int(r['step']))
    return rows


def load_p2_from(path, device, joint):
    args = build_args()
    args.data1_camera = 'Nikon'
    from model import TTSREnhance
    model = TTSREnhance.TTSREnhance(args).to(device)
    state = torch.load(path, map_location=device)
    # accept either raw state_dict or joint/stage ckpt with 'p2' key
    if isinstance(state, dict) and 'p2' in state:
        state = state['p2']
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    cur = model.state_dict()
    n = 0
    for k, v in state.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
            n += 1
    model.load_state_dict(cur)
    apply_p2_cfg(model)
    configure_p2(model, joint=joint)
    print('Loaded P2 %d tensors from %s joint=%s' % (n, path, joint))
    return model, args


def load_h3_from(path, device, chunk=4096, trainable=True):
    ck = torch.load(path, map_location=device)
    enc = SharedEncoder(32).to(device)
    mat = LocalSoftMatch(32, 4, 0.1, chunk=chunk).to(device)
    head = V3B0ResidualFusion(96).to(device)
    if 'encoder' in ck:
        enc.load_state_dict(ck['encoder'])
        mat.load_state_dict(ck['match'])
        head.load_state_dict(ck['head'])
    else:
        raise SystemExit('H3 ckpt missing encoder/match/head: ' + path)
    for m in (enc, mat, head):
        for p in m.parameters():
            p.requires_grad_(trainable)
        m.train(trainable)
        if not trainable:
            m.eval()
    return enc, mat, head


@torch.no_grad()
def eval_full(p2, args, enc, mat, head, rows, device, with_h3=True):
    was = (p2.training,)
    p2.eval()
    if with_h3:
        was_h = (enc.training, mat.training, head.training)
        enc.eval(); mat.eval(); head.eval()
    per = {}
    for row in rows:
        low = _read_eval_tensor(row['low_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=low.shape[-2:])
        gt = _read_eval_tensor(row['high_path'], device)
        y0 = tiled_p2(p2, args, low, low, nano, nano)
        if with_h3:
            y, _ = h3_forward(enc, mat, head, y0, nano)
        else:
            y = y0
        p, s, m = metrics(y, gt)
        p2p, p2s, _ = metrics(y0, gt)
        per[row['sample_id']] = dict(
            psnr=p, ssim=s, mse=m, p2_psnr=p2p, p2_ssim=p2s)
    p2.train(was[0])
    if with_h3:
        enc.train(was_h[0]); mat.train(was_h[1]); head.train(was_h[2])
    deltas = [v['psnr'] - v['p2_psnr'] for v in per.values()]
    summary = dict(
        psnr=float(np.mean([v['psnr'] for v in per.values()])),
        ssim=float(np.mean([v['ssim'] for v in per.values()])),
        p2_only_psnr=float(np.mean([v['p2_psnr'] for v in per.values()])),
        p2_only_ssim=float(np.mean([v['p2_ssim'] for v in per.values()])),
        median_delta=float(np.median(deltas)) if deltas else 0.0,
        n_lt_m1=int(sum(1 for d in deltas if d < -1)),
        win=int(sum(1 for d in deltas if d > 0)),
        n=len(per),
    )
    return summary, per


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
    ap.add_argument('--p2_ckpt', default=DEFAULT_P2)
    ap.add_argument('--h3_ckpt', default=DEFAULT_H3)
    ap.add_argument('--loss', default='mse', choices=['mse', 'l1msssim'])
    ap.add_argument('--p2_lr', type=float, default=1e-6)
    ap.add_argument('--allow_overwrite', action='store_true')
    ap.add_argument('--limit_steps', type=int, default=0)
    ap.add_argument('--skip_eval', action='store_true')
    ap.add_argument('--schedule', default='')
    a = ap.parse_args()

    meta = ARM_META[a.arm]
    joint = meta['joint']
    train_h3 = meta['train_h3']
    # P2-only never trains H3; force loss label into run name
    if a.arm == 'P2':
        run_name = meta['run'] % (a.loss, a.seed)
        joint = True  # train P2
        train_h3 = False
    else:
        run_name = meta['run'] % a.seed
        a.loss = 'mse'  # A0/A2 forced MSE per plan

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    p2, p2_args = load_p2_from(a.p2_ckpt, device, joint=joint)
    enc = mat = head = None
    if train_h3 or a.arm != 'P2':
        enc, mat, head = load_h3_from(
            a.h3_ckpt, device, chunk=a.chunk, trainable=train_h3)
    elif a.arm == 'P2':
        # dummy H3 not used; keep None
        pass

    rec = ReconstructionLoss('l1')
    mss = MSSSIMLoss().to(device)

    groups = []
    if joint:
        p2_params = [p for p in p2.parameters() if p.requires_grad]
        groups.append(dict(params=p2_params, lr=a.p2_lr, name='p2'))
    if train_h3:
        groups.append(dict(params=list(enc.parameters()), lr=1e-6, name='encoder'))
        groups.append(dict(params=list(mat.parameters()), lr=2e-6, name='matcher'))
        groups.append(dict(params=list(head.parameters()), lr=1e-5, name='head'))
    if not groups:
        raise SystemExit('no trainable groups')
    opt = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    n_p2_train = sum(p.numel() for p in p2.parameters() if p.requires_grad)
    print('arm=%s joint=%s train_h3=%s loss=%s p2_lr=%g p2_train=%d' % (
        a.arm, joint, train_h3, a.loss, a.p2_lr, n_p2_train))
    print('p2_init=%s\nh3_init=%s' % (a.p2_ckpt, a.h3_ckpt if train_h3 else 'N/A'))

    tr = read_manifest_csv(os.path.join(a.out_root, 'manifests/manifests_train.csv'))
    ev = read_manifest_csv(os.path.join(a.out_root, 'manifests/manifests_eval.csv'))
    assert len(ev) == 20, len(ev)
    schedule_path = a.schedule or os.path.join(
        a.out_root, 'manifests/random_crop_s%d.csv' % a.seed)
    sch = load_schedule(schedule_path)
    steps = a.limit_steps or a.steps
    sch = sch[:steps]
    train_by_id = {r['sample_id']: r for r in tr}
    ds = ScheduleCropSet(sch, train_by_id)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=a.workers)

    log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w')
    hist = []
    peak_mem = 0.0
    t0 = time.time()
    snap_p2 = snapshot(p2)
    snap_enc = snapshot(enc) if enc is not None else {}
    snap_head = snapshot(head) if head is not None else {}

    def save_eval(step):
        nonlocal peak_mem
        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1 << 20))
        summary = None
        if not a.skip_eval:
            print('--- eval @', step)
            summary, per = eval_full(
                p2, p2_args, enc, mat, head, ev, device,
                with_h3=train_h3 and enc is not None)
            print('  Y PSNR=%.3f SSIM=%.4f | P2-only=%.3f' % (
                summary['psnr'], summary['ssim'], summary['p2_only_psnr']))
            with open(os.path.join(run_dir, 'eval_step_%05d.json' % step), 'w') as f:
                json.dump(dict(step=step, summary=summary, per_image=per), f)
            hist.append(dict(step=step, summary=summary))
        payload = dict(
            p2=p2.state_dict(),
            global_step=step, arm=a.arm, seed=a.seed, loss=a.loss,
            joint_p2=joint, train_h3=train_h3,
            lr_groups={g['name']: g['lr'] for g in groups},
            p2_init=a.p2_ckpt, h3_init=a.h3_ckpt if train_h3 else None,
            p2_init_sha256=sha256(a.p2_ckpt),
            config=vars(a), mean_psnr=summary,
            n_p2_trainable=n_p2_train, lv2_unused=list(LV2_UNUSED),
        )
        if enc is not None:
            payload.update(encoder=enc.state_dict(), match=mat.state_dict(),
                           head=head.state_dict())
            payload['h3_init_sha256'] = sha256(a.h3_ckpt)
        ckpt = os.path.join(run_dir, 'checkpoint_%05d.pt' % step)
        torch.save(payload, ckpt)
        # also dump P2-only weights for easy N2 init
        if a.arm == 'P2':
            torch.save(p2.state_dict(),
                       os.path.join(run_dir, 'p2_weights_%05d.pt' % step))
        configure_p2(p2, joint)
        if train_h3:
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
        y0 = p2_forward(p2, low, nano, joint=joint)
        if train_h3:
            y, _ = h3_forward(enc, mat, head, y0, nano)
        else:
            y = y0
        loss, parts = compute_loss(y, gt, a.loss, rec, mss)
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
            log_f.write(json.dumps(row) + '\n'); log_f.flush()
            print('  step %d/%d loss=%.5f mse=%.5f gP2=%.2e gH=%.2e mem=%.0fMB' % (
                step, steps, row['loss'], row['mse'], g_p2, g_h, peak_mem))
        if step in SAVE_STEPS and step != 0:
            save_eval(step)

    if steps not in SAVE_STEPS and steps != 0:
        save_eval(steps)

    # pick best by Eval PSNR
    best = None
    for h in hist:
        if best is None or h['summary']['psnr'] > best['summary']['psnr']:
            best = h
    if best is not None:
        with open(os.path.join(run_dir, 'best.json'), 'w') as f:
            json.dump(best, f, indent=2)
        print('BEST step=%d PSNR=%.4f P2-only=%.4f' % (
            best['step'], best['summary']['psnr'],
            best['summary']['p2_only_psnr']))

    cfg = dict(
        vars(a), run_name=run_name, joint=joint, train_h3=train_h3,
        p2_init_sha256=sha256(a.p2_ckpt),
        n_p2_trainable=n_p2_train,
        lr_groups={g['name']: g['lr'] for g in groups},
        peak_mem_mb=peak_mem, elapsed_min=(time.time() - t0) / 60,
        protocol='nikon-random-crop-%d' % a.crop,
        n_train=len(tr), n_eval=len(ev), best=best,
    )
    if train_h3:
        cfg['h3_init_sha256'] = sha256(a.h3_ckpt)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, 'eval_history.json'), 'w') as f:
        json.dump(hist, f, indent=2)
    log_f.close()
    print('=== done %s in %.1f min ===' % (run_name, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
