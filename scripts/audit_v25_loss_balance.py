#!/usr/bin/env python
"""v2.5 Phase A: A3 fitting audit + L1 vs MS-SSIM gradient balance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from dataset.data1_localrefine import read_rgb_model_tensor  # noqa: E402
from local_refine_runtime import (  # noqa: E402
    metrics, read_manifest_csv, _read_eval_tensor,
)
from loss.loss_enhance import MSSSIMLoss, ReconstructionLoss  # noqa: E402
from train_v24_end2end import (  # noqa: E402
    configure_p2, h3_forward, load_h3, load_p2, p2_forward,
)
from v2_p2_runtime import tiled_p2  # noqa: E402

V24 = '/root/data/experiments/TTSR-lowlight-v24'
V25 = '/root/data/experiments/TTSR-lowlight-v25'
A3 = os.path.join(V24, 'checkpoints/A3_jointP2_l1ssim_s42')
STEPS = [0, 250, 500, 1000, 1500, 2000]


def stable_crop_xy(sample_id, seed, H, W, crop):
    """Deterministic crop (md5) — used for locked probes / v2.5 training."""
    dig = hashlib.md5(('%d|%s' % (seed, sample_id)).encode()).hexdigest()
    u1 = int(dig[:8], 16); u2 = int(dig[8:16], 16)
    top = u1 % (H - crop + 1)
    left = u2 % (W - crop + 1)
    return top, left


def load_a3_ckpt(step, device):
    path = os.path.join(A3, 'checkpoint_%05d.pt' % step)
    ck = torch.load(path, map_location=device)
    p2, args = load_p2(device, joint=False)
    p2.load_state_dict(ck['p2']); configure_p2(p2, joint=False)
    enc, mat, head, _ = load_h3(device)
    enc.load_state_dict(ck['encoder'])
    mat.load_state_dict(ck['match'])
    head.load_state_dict(ck['head'])
    p2.eval(); enc.eval(); mat.eval(); head.eval()
    return p2, args, enc, mat, head, path


def flat_grads(module):
    grads = []
    for p in module.parameters():
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad.detach().reshape(-1))
    if not grads:
        return None
    return torch.cat(grads)


def grad_stats(g1, gs, lambdas=(1.0, 0.3, 0.1)):
    if g1 is None or gs is None:
        return None
    n1 = float(g1.norm())
    ns = float(gs.norm())
    cos = float(F.cosine_similarity(g1[None], gs[None]).item()) if n1 > 0 and ns > 0 else 0.0
    out = dict(norm_l1=n1, norm_mss=ns, cosine=cos, r={})
    for lam in lambdas:
        out['r'][str(lam)] = (lam * ns) / (n1 + 1e-12)
    return out


@torch.no_grad()
def eval_full_subset(p2, args, enc, mat, head, rows, device, rec, mss):
    per = {}
    for row in rows:
        low = _read_eval_tensor(row['low_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=low.shape[-2:])
        gt = _read_eval_tensor(row['high_path'], device)
        y0 = tiled_p2(p2, args, low, low, nano, nano)
        y, _ = h3_forward(enc, mat, head, y0, nano)
        p, s, m = metrics(y, gt)
        p2p, p2s, _ = metrics(y0, gt)
        l1 = float(rec(y, gt).item())
        ms = float(mss(y, gt).item())
        per[row['sample_id']] = dict(
            psnr=p, ssim=s, mse=m, l1=l1, ms_ssim=ms,
            p2_psnr=p2p, p2_ssim=p2s)
    def mean(k):
        return float(np.mean([v[k] for v in per.values()]))
    summary = {k: mean(k) for k in
               ('psnr', 'ssim', 'mse', 'l1', 'ms_ssim', 'p2_psnr', 'p2_ssim')}
    return summary, per


@torch.no_grad()
def eval_crop_probe(p2, enc, mat, head, rows, crops, crop, device, rec, mss):
    """Fixed 192 crops on locked train samples (online P2, no tile)."""
    per = {}
    for row in rows:
        sid = row['sample_id']
        c = crops[sid]
        low = read_rgb_model_tensor(row['low_path'])
        high = read_rgb_model_tensor(row['high_path'])
        h, w = low.shape[-2:]
        if tuple(high.shape[-2:]) != (h, w):
            high = F.interpolate(high[None], size=(h, w), mode='bilinear',
                                 align_corners=False)[0]
        nano = read_rgb_model_tensor(row['nano_path'], size=(h, w))
        top, left = c['top'], c['left']
        # Prefer locked coords; fall back if HW mismatch
        if h != c['h'] or w != c['w']:
            top, left = stable_crop_xy(sid, 42, h, w, crop)
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
    def mean(k):
        return float(np.mean([v[k] for v in per.values()]))
    summary = {k: mean(k) for k in
               ('psnr', 'ssim', 'mse', 'l1', 'ms_ssim', 'p2_psnr', 'p2_ssim')}
    return summary, per


def gradient_balance(device, probe_row, crop_xy, crop=192):
    """At H3@2000+P2 init (step0), compare ∇L1 vs ∇(1-MS-SSIM)."""
    p2, _ = load_p2(device, joint=True)
    enc, mat, head, _ = load_h3(device, chunk=1024)
    # use init weights (already loaded)
    rec = ReconstructionLoss('l1')
    mss = MSSSIMLoss().to(device)
    low = read_rgb_model_tensor(probe_row['low_path'])
    high = read_rgb_model_tensor(probe_row['high_path'])
    h, w = low.shape[-2:]
    if tuple(high.shape[-2:]) != (h, w):
        high = F.interpolate(high[None], size=(h, w), mode='bilinear',
                             align_corners=False)[0]
    nano = read_rgb_model_tensor(probe_row['nano_path'], size=(h, w))
    top, left = crop_xy
    sl = (slice(None), slice(top, top + crop), slice(left, left + crop))
    low_c = low[sl][None].to(device)
    nano_c = nano[sl][None].to(device)
    gt_c = high[sl][None].to(device)

    modules = dict(p2=p2, encoder=enc, matcher=mat, head=head)

    def run_one(kind):
        for m in modules.values():
            m.zero_grad(set_to_none=True)
        y0 = p2_forward(p2, low_c, nano_c, joint=True)
        y, _ = h3_forward(enc, mat, head, y0, nano_c)
        if kind == 'l1':
            loss = rec(y, gt_c)
        else:
            loss = mss(y, gt_c)
        loss.backward()
        return float(loss.detach()), {n: flat_grads(m) for n, m in modules.items()}

    l1_val, g1 = run_one('l1')
    ms_val, gs = run_one('mss')
    per_mod = {}
    for name in modules:
        per_mod[name] = grad_stats(g1[name], gs[name])
    return dict(l1=l1_val, ms_ssim_loss=ms_val, modules=per_mod,
                sample_id=probe_row['sample_id'], crop=list(crop_xy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--full_train_steps', default='0,1000,2000',
                    help='comma steps for optional full Train500 (expensive)')
    a = ap.parse_args()
    device = a.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(os.path.join(V25, 'diagnostics'), exist_ok=True)
    os.makedirs(os.path.join(V25, 'metrics'), exist_ok=True)

    lock = json.load(open(os.path.join(V25, 'probes/train80_lock.json')))
    tr_all = read_manifest_csv(
        '/root/data/experiments/TTSR-lowlight-v2/manifests/manifests_train.csv')
    by_id = {r['sample_id']: r for r in tr_all}
    probe_rows = [by_id[sid] for sid in lock['sample_ids']]
    crops = {}
    for row in probe_rows:
        # manifest low_hw is WxH
        W, H = map(int, row['low_hw'].split('x'))
        top, left = stable_crop_xy(row['sample_id'], 42, H, W, 192)
        crops[row['sample_id']] = dict(top=top, left=left, h=H, w=W)
    lock['crops'] = crops
    lock['crop_hash'] = 'md5(seed|sample_id)'
    json.dump(lock, open(os.path.join(V25, 'probes/train80_lock.json'), 'w'), indent=2)
    print('locked crops for %d probes' % len(crops), flush=True)

    ev = read_manifest_csv(
        '/root/data/experiments/TTSR-lowlight-v2/manifests/manifests_eval.csv')
    rec = ReconstructionLoss('l1')
    mss = MSSSIMLoss().to(device)

    fitting = []
    full_steps = {int(x) for x in a.full_train_steps.split(',') if x.strip() != ''}

    for st in STEPS:
        print('=== A3@%d ===' % st, flush=True)
        p2, args, enc, mat, head, path = load_a3_ckpt(st, device)
        # Eval30 PSNR/SSIM from v2.4 logs; L1/MS only at selected steps
        evj = os.path.join(A3, 'eval_step_%05d.json' % st)
        ev_data = json.load(open(evj))
        ev_sum = ev_data['summary']
        ev_out = dict(
            psnr=ev_sum['psnr'], ssim=ev_sum['ssim'],
            p2_psnr=ev_sum.get('p2_only_psnr', ev_sum.get('p2_psnr')),
            n_lt_m1=ev_sum.get('n_lt_m1'))
        if st in full_steps or st in (0, 2000):
            print('  Eval30 L1/MS-SSIM recompute...', flush=True)
            ev_lm, _ = eval_full_subset(
                p2, args, enc, mat, head, ev, device, rec, mss)
            ev_out.update(l1=ev_lm['l1'], ms_ssim=ev_lm['ms_ssim'],
                          mse=ev_lm['mse'], p2_ssim=ev_lm['p2_ssim'])

        print('  train80 crop192...', flush=True)
        tr_crop, _ = eval_crop_probe(
            p2, enc, mat, head, probe_rows, crops, 192, device, rec, mss)

        row = dict(step=st, ckpt=path, eval30=ev_out, train80_crop192=tr_crop)
        if st in full_steps:
            print('  train80 full-res tiled...', flush=True)
            tr80, _ = eval_full_subset(
                p2, args, enc, mat, head, probe_rows, device, rec, mss)
            row['train80_full'] = tr80
            print('  Train80-full PSNR=%.3f' % tr80['psnr'], flush=True)

        fitting.append(row)
        print('  Eval30 PSNR=%.3f SSIM=%.4f | Crop PSNR=%.3f SSIM=%.4f L1=%.4f MS=%.4f' % (
            ev_out['psnr'], ev_out['ssim'],
            tr_crop['psnr'], tr_crop['ssim'], tr_crop['l1'], tr_crop['ms_ssim']),
            flush=True)
        del p2, enc, mat, head
        torch.cuda.empty_cache()

    # Gradient balance at init on first probe crop
    print('=== gradient balance @ init ===')
    gbal = gradient_balance(
        device, probe_rows[0],
        (crops[probe_rows[0]['sample_id']]['top'],
         crops[probe_rows[0]['sample_id']]['left']))
    for mod, st in gbal['modules'].items():
        if st is None:
            print(' ', mod, 'NO GRAD'); continue
        print(' ', mod, '‖g1‖=%.3e ‖gs‖=%.3e cos=%.3f r(1)=%.3f r(0.3)=%.3f r(0.1)=%.3f' % (
            st['norm_l1'], st['norm_mss'], st['cosine'],
            st['r']['1.0'], st['r']['0.3'], st['r']['0.1']))

    out = dict(fitting=fitting, gradient_balance=gbal,
               note='Train80/Train500 are NOT holdout; Eval30 is development set. '
                    'A3 crops used unstable hash(); v2.5 locks md5 crops.')
    path = os.path.join(V25, 'diagnostics/a3_fitting_grad_audit.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote', path)


if __name__ == '__main__':
    main()
