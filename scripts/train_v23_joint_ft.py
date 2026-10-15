#!/usr/bin/env python
"""v2.3 joint FT: H0 head-only / H1 matcher+head / H2 encoder+matcher+head.

Init = H3@checkpoint_02000. P2 frozen (Y0 cache). Loss = RGB MSE.
H1/H2 keep live differentiable Matcher (and Encoder for H2); no detached F0/T
for trainable modules.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from dataset.data1_localrefine import read_rgb_model_tensor  # noqa: E402
from local_refine_runtime import (  # noqa: E402
    EVAL_QUERY_CHUNK, load_cache, metrics, read_manifest_csv, sha256,
    _read_eval_tensor,
)
from model.LocalRefine import LocalSoftMatch, SharedEncoder  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion, count_params  # noqa: E402

H3_CKPT = ('/root/data/experiments/TTSR-lowlight-v2/checkpoints/'
           'H3_v3b0_s42/checkpoint_02000.pt')
V2_EXP = '/root/data/experiments/TTSR-lowlight-v2'
EXP = '/root/data/experiments/TTSR-lowlight-v23'
SAVE_STEPS = (0, 250, 500, 750, 1000)

ARM_NAMES = {
    'H0': 'H0_head_only_s%d',
    'H1': 'H1_matcher_head_s%d',
    'H2': 'H2_encoder_matcher_head_s%d',
}


class FullImageSet(Dataset):
    def __init__(self, rows, cache_dir, crop=0):
        self.rows = rows
        self.cache_dir = cache_dir
        self.crop = int(crop)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        y0 = torch.from_numpy(np.ascontiguousarray(load_cache(self.cache_dir, row)))
        high = read_rgb_model_tensor(row['high_path'])
        h, w = y0.shape[-2:]
        if tuple(high.shape[-2:]) != (h, w):
            high = torch.nn.functional.interpolate(
                high[None], size=(h, w), mode='bilinear', align_corners=False)[0]
        nano = read_rgb_model_tensor(row['nano_path'], size=(h, w))
        if self.crop > 0:
            ch = self.crop
            if h < ch or w < ch:
                raise RuntimeError('crop %d too large for %dx%d' % (ch, h, w))
            # deterministic crop from sample_id hash + global seed handled outside
            rng = np.random.RandomState(hash(row['sample_id']) % (2 ** 31 - 1))
            top = int(rng.randint(0, h - ch + 1))
            left = int(rng.randint(0, w - ch + 1))
            y0 = y0[:, top:top + ch, left:left + ch]
            high = high[:, top:top + ch, left:left + ch]
            nano = nano[:, top:top + ch, left:left + ch]
        return dict(y0=y0, high=high, nano=nano, sample_id=row['sample_id'])


def load_models(device, chunk=EVAL_QUERY_CHUNK):
    ck = torch.load(H3_CKPT, map_location=device)
    enc = SharedEncoder(32).to(device)
    mat = LocalSoftMatch(32, 4, 0.1, chunk=chunk).to(device)
    head = V3B0ResidualFusion(96).to(device)
    enc.load_state_dict(ck['encoder'])
    mat.load_state_dict(ck['match'])
    head.load_state_dict(ck['head'])
    return enc, mat, head, ck


def configure_trainable(arm, enc, mat, head):
    """Set requires_grad and return Adam param groups + trainable name list."""
    for p in enc.parameters():
        p.requires_grad_(False)
    for p in mat.parameters():
        p.requires_grad_(False)
    for p in head.parameters():
        p.requires_grad_(False)

    groups = []
    trainable = []
    if arm in ('H0', 'H1', 'H2'):
        for p in head.parameters():
            p.requires_grad_(True)
        groups.append(dict(params=list(head.parameters()), lr=1e-5,
                           name='head'))
        trainable.append('V3B0ResidualFusion')
    if arm in ('H1', 'H2'):
        for p in mat.parameters():
            p.requires_grad_(True)
        groups.append(dict(params=list(mat.parameters()), lr=2e-6,
                           name='matcher'))
        trainable.append('LocalSoftMatch')
    if arm == 'H2':
        for p in enc.parameters():
            p.requires_grad_(True)
        groups.append(dict(params=list(enc.parameters()), lr=1e-6,
                           name='encoder'))
        trainable.append('SharedEncoder')

    enc.train(arm == 'H2')
    mat.train(arm in ('H1', 'H2'))
    head.train()
    return groups, trainable


def forward_h3(enc, mat, head, y0, nano, arm, use_ckpt=False):
    """Y = Y0 + D. Differentiable path depends on arm."""
    y0_in = (y0 + 1.) * 0.5
    nano_in = (nano + 1.) * 0.5

    if arm == 'H0':
        with torch.no_grad():
            f0 = enc(y0_in)
            fr = enc(nano_in)
            t = mat(f0, fr)
        f0 = f0.detach()
        t = t.detach()
        d = head(f0, t, y0.shape[-2:])
    elif arm == 'H1':
        with torch.no_grad():
            f0 = enc(y0_in)
            fr = enc(nano_in)
        f0 = f0.detach().requires_grad_(False)
        fr = fr.detach().requires_grad_(False)
        # Matcher must be live
        if use_ckpt:
            t = checkpoint(mat, f0, fr, use_reentrant=False)
        else:
            t = mat(f0, fr)
        d = head(f0, t, y0.shape[-2:])
    else:  # H2
        if use_ckpt:
            def _enc_y(x):
                return enc(x)
            def _enc_r(x):
                return enc(x)
            f0 = checkpoint(_enc_y, y0_in, use_reentrant=False)
            fr = checkpoint(_enc_r, nano_in, use_reentrant=False)
            t = checkpoint(mat, f0, fr, use_reentrant=False)
        else:
            f0 = enc(y0_in)
            fr = enc(nano_in)
            t = mat(f0, fr)
        d = head(f0, t, y0.shape[-2:])
    return y0 + d, d, f0, t


@torch.no_grad()
def eval_full(enc, mat, head, rows, cache, device):
    was = (enc.training, mat.training, head.training)
    enc.eval(); mat.eval(); head.eval()
    per = {}
    for row in rows:
        y0 = torch.from_numpy(np.ascontiguousarray(
            load_cache(cache, row)))[None].to(device)
        gt = _read_eval_tensor(row['high_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=y0.shape[-2:])
        f0 = enc((y0 + 1.) * 0.5)
        fr = enc((nano + 1.) * 0.5)
        t = mat(f0, fr)
        d = head(f0, t, y0.shape[-2:])
        y = y0 + d
        p, s, m = metrics(y, gt)
        p2, s2, m2 = metrics(y0, gt)
        per[row['sample_id']] = dict(
            psnr=p, ssim=s, mse=m,
            p2_psnr=p2, p2_ssim=s2, p2_mse=m2)
    enc.train(was[0]); mat.train(was[1]); head.train(was[2])
    mean_psnr = float(np.mean([v['psnr'] for v in per.values()]))
    mean_ssim = float(np.mean([v['ssim'] for v in per.values()]))
    mean_p2 = float(np.mean([v['p2_psnr'] for v in per.values()]))
    deltas = [v['psnr'] - v['p2_psnr'] for v in per.values()]
    summary = dict(
        psnr=mean_psnr, ssim=mean_ssim, p2_psnr=mean_p2,
        median_delta=float(np.median(deltas)),
        mean_delta=float(np.mean(deltas)),
        p10_delta=float(np.percentile(deltas, 10)),
        worst_delta=float(np.min(deltas)),
        n_lt_m1=int(sum(1 for d in deltas if d < -1.0)),
        win=int(sum(1 for d in deltas if d > 0)),
        loss_n=int(sum(1 for d in deltas if d < 0)),
    )
    return summary, per


def grad_norm(module):
    sq = 0.0
    n = 0
    for p in module.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            sq += float(g.pow(2).sum())
            n += 1
    return float(np.sqrt(sq)) if n else 0.0


def param_snapshot(module):
    return {n: p.detach().cpu().clone()
            for n, p in module.named_parameters()}


def param_delta_norm(before, module):
    sq = 0.0
    rel = 0.0
    for n, p in module.named_parameters():
        d = (p.detach().cpu() - before[n]).pow(2).sum()
        sq += float(d)
        base = before[n].pow(2).sum().clamp_min(1e-12)
        rel += float(d / base)
    return float(np.sqrt(sq)), float(rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['H0', 'H1', 'H2'])
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--out_root', default=EXP)
    ap.add_argument('--chunk', type=int, default=EVAL_QUERY_CHUNK)
    ap.add_argument('--crop', type=int, default=0,
                    help='If >0, use random crop of this size (all arms must match)')
    ap.add_argument('--use_ckpt', action='store_true',
                    help='activation checkpoint encoder/matcher')
    ap.add_argument('--allow_overwrite', action='store_true')
    ap.add_argument('--limit_steps', type=int, default=0)
    ap.add_argument('--skip_eval', action='store_true')
    a = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = ARM_NAMES[a.arm] % a.seed
    run_dir = os.path.join(a.out_root, 'checkpoints', run_name)
    if os.path.isdir(run_dir) and not a.allow_overwrite:
        if any(f.startswith('checkpoint_') for f in os.listdir(run_dir)):
            raise SystemExit('refusing overwrite ' + run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(a.out_root, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(a.out_root, 'metrics'), exist_ok=True)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    enc, mat, head, ck0 = load_models(device, chunk=a.chunk)
    groups, trainable = configure_trainable(a.arm, enc, mat, head)
    opt = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    n_enc = count_params(enc)
    n_mat = count_params(mat)
    n_head = count_params(head)
    n_train = sum(p.numel() for g in groups for p in g['params'])
    print('arm=%s trainable=%s params_train=%d (enc=%d mat=%d head=%d) '
          'chunk=%d crop=%d ckpt=%s init=%s' % (
              a.arm, trainable, n_train, n_enc, n_mat, n_head,
              a.chunk, a.crop, a.use_ckpt, H3_CKPT))

    tr = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_train.csv'))
    ev = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_eval.csv'))
    assert len(tr) == 500 and len(ev) == 30
    ds = FullImageSet(tr, os.path.join(V2_EXP, 'cache_p2_train'), crop=a.crop)
    gen = torch.Generator(); gen.manual_seed(a.seed)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=a.workers,
                    generator=gen)

    # step-0 weight identity
    head.eval(); enc.eval(); mat.eval()
    with torch.no_grad():
        row0 = tr[0]
        y0 = torch.from_numpy(np.ascontiguousarray(
            load_cache(os.path.join(V2_EXP, 'cache_p2_train'), row0)))[None].to(device)
        nano = _read_eval_tensor(row0['nano_path'], device, size=y0.shape[-2:])
        f0 = enc((y0 + 1) * 0.5); fr = enc((nano + 1) * 0.5)
        d = head(f0, mat(f0, fr), y0.shape[-2:])
        head2 = V3B0ResidualFusion(96).to(device).eval()
        head2.load_state_dict(ck0['head'])
        d2 = head2(f0, mat(f0, fr), y0.shape[-2:])
        assert torch.allclose(d, d2, atol=1e-6)
    print('step0 weight identity OK')
    configure_trainable(a.arm, enc, mat, head)  # restore train flags

    log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w')
    hist = []
    peak_mem = 0.0
    t0 = time.time()
    steps = a.limit_steps or a.steps

    def save_eval(step):
        nonlocal peak_mem
        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1 << 20))
        summary, per = None, None
        if not a.skip_eval:
            print('--- eval @', step)
            summary, per = eval_full(
                enc, mat, head, ev, os.path.join(V2_EXP, 'cache_p2_eval'), device)
            print('  PSNR=%.3f SSIM=%.4f Δmed=%.3f n<-1=%d' % (
                summary['psnr'], summary['ssim'], summary['median_delta'],
                summary['n_lt_m1']))
            with open(os.path.join(run_dir, 'eval_step_%05d.json' % step), 'w') as f:
                json.dump(dict(step=step, summary=summary, per_image=per), f)
            hist.append(dict(step=step, summary=summary))
        ckpt = os.path.join(run_dir, 'checkpoint_%05d.pt' % step)
        torch.save(dict(
            head=head.state_dict(),
            encoder=enc.state_dict(),
            match=mat.state_dict(),
            optimizer=opt.state_dict(),
            global_step=step,
            arm=a.arm,
            seed=a.seed,
            trainable=trainable,
            lr_groups={g['name']: g['lr'] for g in groups},
            h3_init=H3_CKPT,
            h3_init_sha256=sha256(H3_CKPT),
            config=vars(a),
            mean_psnr=summary,
        ), ckpt)
        configure_trainable(a.arm, enc, mat, head)
        return summary

    save_eval(0)

    # capture param snapshot after step0 save for update norms
    snap_enc = param_snapshot(enc)
    snap_mat = param_snapshot(mat)
    snap_head = param_snapshot(head)

    step = 0
    data_pass = 0
    while step < steps:
        for batch in dl:
            if step >= steps:
                break
            step += 1
            y0 = batch['y0'].to(device)
            gt = batch['high'].to(device)
            nano = batch['nano'].to(device)
            y, d, _, _ = forward_h3(
                enc, mat, head, y0, nano, a.arm, use_ckpt=a.use_ckpt)
            loss = torch.nn.functional.mse_loss(y, gt)
            if not torch.isfinite(loss):
                raise SystemExit('non-finite at step %d' % step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            g_enc = grad_norm(enc)
            g_mat = grad_norm(mat)
            g_head = grad_norm(head)
            opt.step()
            if torch.cuda.is_available():
                peak_mem = max(peak_mem,
                               torch.cuda.max_memory_allocated() / (1 << 20))
            if step % a.log_every == 0 or step == 1:
                du_e, rel_e = param_delta_norm(snap_enc, enc)
                du_m, rel_m = param_delta_norm(snap_mat, mat)
                du_h, rel_h = param_delta_norm(snap_head, head)
                row = dict(
                    step=step, data_pass=data_pass, loss=float(loss.detach()),
                    g_enc=g_enc, g_mat=g_mat, g_head=g_head,
                    du_enc=du_e, du_mat=du_m, du_head=du_h,
                    rel_enc=rel_e, rel_mat=rel_m, rel_head=rel_h,
                    peak_mem_mb=peak_mem,
                    sample_id=batch['sample_id'][0] if isinstance(
                        batch['sample_id'], (list, tuple)) else str(batch['sample_id'][0]),
                )
                log_f.write(json.dumps(row) + '\n'); log_f.flush()
                print('  step %d/%d loss=%.6f gE=%.3e gM=%.3e gH=%.3e mem=%.0fMB' % (
                    step, steps, row['loss'], g_enc, g_mat, g_head, peak_mem))
            if step in SAVE_STEPS and step != 0:
                save_eval(step)
        data_pass += 1

    if steps not in SAVE_STEPS and steps != 0:
        save_eval(steps)

    cfg = dict(
        vars(a), run_name=run_name, h3_init_sha256=sha256(H3_CKPT),
        n_encoder=n_enc, n_matcher=n_mat, n_head=n_head, n_trainable=n_train,
        trainable=trainable, lr_groups={g['name']: g['lr'] for g in groups},
        peak_mem_mb=peak_mem, data_passes=data_pass,
        elapsed_min=(time.time() - t0) / 60,
        protocol=('crop-%d' % a.crop) if a.crop else 'full-image',
    )
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, 'eval_history.json'), 'w') as f:
        json.dump(hist, f, indent=2)
    log_f.close()
    print('=== done %s in %.1f min peak_mem=%.0fMB ===' % (
        run_name, (time.time() - t0) / 60, peak_mem))


if __name__ == '__main__':
    main()
