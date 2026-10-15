#!/usr/bin/env python
"""v2.2 Full-Image DC-aware fine-tune of V3B0ResidualFusion head only.

V1: λ_DC=1 (≡ full RGB MSE). V2: λ_DC=2.
Init = H3@checkpoint_02000. Encoder/Matcher/P2 frozen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
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
from model.v22_dc_loss import dc_aware_residual_loss, mean_hw, split_dc  # noqa: E402

H3_CKPT = ('/root/data/experiments/TTSR-lowlight-v2/checkpoints/'
           'H3_v3b0_s42/checkpoint_02000.pt')
V2_EXP = '/root/data/experiments/TTSR-lowlight-v2'
EXP = '/root/data/experiments/TTSR-lowlight-v22'
SAVE_STEPS = (0, 250, 500, 750, 1000)


class FullImageSet(Dataset):
    """Full-image Y0 / GT / Nano — no crop. Variable HxW → batch_size must be 1."""

    def __init__(self, rows, cache_dir):
        self.rows = rows
        self.cache_dir = cache_dir

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
        return dict(y0=y0, high=high, nano=nano, sample_id=row['sample_id'])


def load_models(device):
    ck = torch.load(H3_CKPT, map_location=device)
    enc = SharedEncoder(32).to(device).eval()
    mat = LocalSoftMatch(32, 4, 0.1, chunk=EVAL_QUERY_CHUNK).to(device).eval()
    head = V3B0ResidualFusion(96).to(device)
    enc.load_state_dict(ck['encoder'])
    mat.load_state_dict(ck['match'])
    head.load_state_dict(ck['head'])
    for p in list(enc.parameters()) + list(mat.parameters()):
        p.requires_grad_(False)
    enc.eval(); mat.eval()
    return enc, mat, head, ck


@torch.no_grad()
def eval_full_centered(enc, mat, head, rows, cache, device):
    head.eval()
    out = {'Full': {}, 'Centered': {}, 'P2': {}}
    for row in rows:
        y0 = torch.from_numpy(np.ascontiguousarray(
            load_cache(cache, row)))[None].to(device)
        gt = _read_eval_tensor(row['high_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=y0.shape[-2:])
        f0 = enc((y0 + 1.) * 0.5)
        fr = enc((nano + 1.) * 0.5)
        t = mat(f0, fr)
        d = head(f0, t, y0.shape[-2:])
        y_full = y0 + d
        _, ds = split_dc(d)
        y_cent = y0 + ds
        for label, pred in [('P2', y0), ('Full', y_full), ('Centered', y_cent)]:
            p, s, m = metrics(pred, gt)
            out[label][row['sample_id']] = dict(psnr=p, ssim=s, mse=m)
    def mean_psnr(lab):
        return float(np.mean([v['psnr'] for v in out[lab].values()]))
    def mean_ssim(lab):
        return float(np.mean([v['ssim'] for v in out[lab].values()]))
    summary = {lab: dict(psnr=mean_psnr(lab), ssim=mean_ssim(lab))
               for lab in out}
    # worst deltas Full vs P2
    deltas = []
    for sid in out['P2']:
        dlt = out['Full'][sid]['psnr'] - out['P2'][sid]['psnr']
        deltas.append((sid, dlt))
    deltas.sort(key=lambda x: x[1])
    summary['worst5_full_vs_p2'] = deltas[:5]
    summary['best5_full_vs_p2'] = deltas[-5:][::-1]
    summary['n_lt_m1'] = sum(1 for _, d in deltas if d < -1.0)
    summary['median_delta'] = float(np.median([d for _, d in deltas]))
    summary['mean_delta'] = float(np.mean([d for _, d in deltas]))
    return summary, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['V1', 'V2'])
    ap.add_argument('--lambda_dc', type=float, default=None)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--out_root', default=EXP)
    ap.add_argument('--allow_overwrite', action='store_true')
    ap.add_argument('--limit_steps', type=int, default=0)
    a = ap.parse_args()
    if a.lambda_dc is None:
        a.lambda_dc = 1.0 if a.arm == 'V1' else 2.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = ('V1_full_mse_s%d' % a.seed if a.arm == 'V1'
                else 'V2_full_dc2_s%d' % a.seed)
    run_dir = os.path.join(a.out_root, 'checkpoints', run_name)
    if os.path.isdir(run_dir) and not a.allow_overwrite:
        if any(f.startswith('checkpoint_') for f in os.listdir(run_dir)):
            raise SystemExit('refusing overwrite ' + run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(a.out_root, 'logs'), exist_ok=True)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    enc, mat, head, ck0 = load_models(device)
    print('arm=%s λDC=%.1f head_params=%d init=%s' % (
        a.arm, a.lambda_dc, count_params(head), H3_CKPT))

    tr = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_train.csv'))
    ev = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_eval.csv'))
    assert len(tr) == 500 and len(ev) == 30
    ds = FullImageSet(tr, os.path.join(V2_EXP, 'cache_p2_train'))
    gen = torch.Generator(); gen.manual_seed(a.seed)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=a.workers,
                    generator=gen)
    opt = torch.optim.Adam(head.parameters(), lr=a.lr, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=0)

    # step-0 identity vs H3@2000 on a probe batch
    head.eval()
    with torch.no_grad():
        row0 = tr[0]
        y0 = torch.from_numpy(np.ascontiguousarray(
            load_cache(os.path.join(V2_EXP, 'cache_p2_train'), row0)))[None].to(device)
        nano = _read_eval_tensor(row0['nano_path'], device, size=y0.shape[-2:])
        f0 = enc((y0 + 1) * 0.5); fr = enc((nano + 1) * 0.5)
        d = head(f0, mat(f0, fr), y0.shape[-2:])
        # reload fresh head from ckpt for compare
        head2 = V3B0ResidualFusion(96).to(device).eval()
        head2.load_state_dict(ck0['head'])
        d2 = head2(f0, mat(f0, fr), y0.shape[-2:])
        assert torch.allclose(d, d2, atol=1e-6)
    print('step0 weight identity OK')

    log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w')
    hist = []
    peak_mem = 0.0
    t0 = time.time()
    steps = a.limit_steps or a.steps

    def save_eval(step):
        nonlocal peak_mem
        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1 << 20))
        print('--- eval @', step)
        summary, per = eval_full_centered(
            enc, mat, head, ev, os.path.join(V2_EXP, 'cache_p2_eval'), device)
        print('  Full=%.3f Centered=%.3f P2=%.3f' % (
            summary['Full']['psnr'], summary['Centered']['psnr'],
            summary['P2']['psnr']))
        ckpt = os.path.join(run_dir, 'checkpoint_%05d.pt' % step)
        torch.save(dict(
            head=head.state_dict(), encoder=enc.state_dict(), match=mat.state_dict(),
            global_step=step, arm=a.arm, lambda_dc=a.lambda_dc,
            lr=a.lr, seed=a.seed, h3_init=H3_CKPT,
            mean_psnr=summary, config=vars(a),
        ), ckpt)
        with open(os.path.join(run_dir, 'eval_step_%05d.json' % step), 'w') as f:
            json.dump(dict(step=step, summary=summary, per_image=per), f)
        hist.append(dict(step=step, summary=summary))
        head.train()
        return summary

    save_eval(0)
    step = 0
    data_pass = 0
    head.train()
    while step < steps:
        for batch in dl:
            if step >= steps:
                break
            step += 1
            y0 = batch['y0'].to(device)
            gt = batch['high'].to(device)
            nano = batch['nano'].to(device)
            with torch.no_grad():
                f0 = enc((y0 + 1.) * 0.5)
                fr = enc((nano + 1.) * 0.5)
                t = mat(f0, fr)
            d = head(f0, t, y0.shape[-2:])
            d_tgt = (gt - y0).detach()
            loss, l_sp, l_dc, _, _ = dc_aware_residual_loss(
                d, d_tgt, lambda_dc=a.lambda_dc)
            if not torch.isfinite(loss):
                raise SystemExit('non-finite at step %d' % step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if torch.cuda.is_available():
                peak_mem = max(peak_mem,
                               torch.cuda.max_memory_allocated() / (1 << 20))
            if step % a.log_every == 0 or step == 1:
                row = dict(step=step, data_pass=data_pass, lr=a.lr,
                           loss=float(loss.detach()),
                           l_sp=float(l_sp.detach()),
                           l_dc=float(l_dc.detach()),
                           m_norm=float(mean_hw(d).reshape(-1).norm()))
                log_f.write(json.dumps(row) + '\n'); log_f.flush()
                print('  step %d/%d loss=%.6f sp=%.6f dc=%.6f' % (
                    step, steps, row['loss'], row['l_sp'], row['l_dc']))
            if step in SAVE_STEPS and step != 0:
                save_eval(step)
        data_pass += 1

    if steps not in SAVE_STEPS and steps != 0:
        save_eval(steps)

    cfg = dict(vars(a), run_name=run_name, h3_init_sha256=sha256(H3_CKPT),
               head_params=count_params(head), peak_mem_mb=peak_mem,
               data_passes=data_pass, elapsed_min=(time.time() - t0) / 60,
               note='full-image; batch=1; frozen enc/match; true HW DC mean')
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, 'eval_history.json'), 'w') as f:
        json.dump(hist, f, indent=2)
    log_f.close()
    print('=== done %s in %.1f min peak_mem=%.0fMB ===' % (
        run_name, (time.time() - t0) / 60, peak_mem))


if __name__ == '__main__':
    main()
