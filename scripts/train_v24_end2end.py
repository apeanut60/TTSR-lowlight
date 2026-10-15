#!/usr/bin/env python
"""v2.4 end-to-end P2+H3 FT: A0/A1 frozen-P2, A2/A3 joint-P2; MSE vs L1+MS-SSIM.

Unified 192x192 crop training (Full-Image joint OOM). Eval stays full-image tiled.
"""
from __future__ import annotations

import argparse
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

from dataset.data1_localrefine import read_rgb_model_tensor  # noqa: E402
from local_refine_runtime import (  # noqa: E402
    EVAL_QUERY_CHUNK, metrics, read_manifest_csv, sha256, _read_eval_tensor,
)
from loss.loss_enhance import MSSSIMLoss, ReconstructionLoss  # noqa: E402
from model.LocalRefine import LocalSoftMatch, SharedEncoder  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion, count_params  # noqa: E402
from model.query_illum_align import estimate_query_align_maps  # noqa: E402
from v2_p2_runtime import (  # noqa: E402
    CKPT as P2_CKPT, apply_p2_cfg, build_args, tiled_p2,
)

H3_CKPT = ('/root/data/experiments/TTSR-lowlight-v2/checkpoints/'
           'H3_v3b0_s42/checkpoint_02000.pt')
V2_EXP = '/root/data/experiments/TTSR-lowlight-v2'
EXP = '/root/data/experiments/TTSR-lowlight-v24'
SAVE_STEPS = (0, 250, 500, 1000, 1500, 2000)

ARM_META = {
    'A0': dict(run='A0_frozenP2_mse_s%d', joint=False, loss='mse'),
    'A1': dict(run='A1_frozenP2_l1ssim_s%d', joint=False, loss='l1msssim'),
    'A2': dict(run='A2_jointP2_mse_s%d', joint=True, loss='mse'),
    'A3': dict(run='A3_jointP2_l1ssim_s%d', joint=True, loss='l1msssim'),
}

# lv2 bypass: these heads never participate in forward
LV2_UNUSED = ('MainNet.conv22_head.weight', 'MainNet.conv22_head.bias')


class CropSet(Dataset):
    """Aligned Low / Nano / GT crops. Deterministic per (seed, sample_id)."""

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
        if h < ch or w < ch:
            raise RuntimeError('crop %d > %dx%d for %s' % (ch, h, w, row['sample_id']))
        rng = np.random.RandomState(
            (self.seed * 1000003 + (hash(row['sample_id']) % 1000000007)) % (2 ** 31 - 1))
        top = int(rng.randint(0, h - ch + 1))
        left = int(rng.randint(0, w - ch + 1))
        sl = (slice(None), slice(top, top + ch), slice(left, left + ch))
        return dict(
            low=low[sl], high=high[sl], nano=nano[sl],
            sample_id=row['sample_id'])


def load_p2(device, joint):
    args = build_args()
    from model import TTSREnhance
    model = TTSREnhance.TTSREnhance(args).to(device)
    state = torch.load(P2_CKPT, map_location=device)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    cur = model.state_dict()
    for k, v in state.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
    model.load_state_dict(cur)
    apply_p2_cfg(model)
    configure_p2(model, joint=joint)
    return model, args


def configure_p2(model, joint):
    for n, p in model.named_parameters():
        is_lte = 'LTE' in n
        is_lv2 = n in LV2_UNUSED
        p.requires_grad_(bool(joint) and (not is_lte) and (not is_lv2))
    model.train(bool(joint))
    if not joint:
        model.eval()


def load_h3(device, chunk=EVAL_QUERY_CHUNK):
    ck = torch.load(H3_CKPT, map_location=device)
    enc = SharedEncoder(32).to(device)
    mat = LocalSoftMatch(32, 4, 0.1, chunk=chunk).to(device)
    head = V3B0ResidualFusion(96).to(device)
    enc.load_state_dict(ck['encoder'])
    mat.load_state_dict(ck['match'])
    head.load_state_dict(ck['head'])
    for m in (enc, mat, head):
        for p in m.parameters():
            p.requires_grad_(True)
        m.train()
    return enc, mat, head, ck


def p2_forward(p2, low, nano, joint):
    """Online P2 crop forward. Query maps under no_grad (non-param); backbone live if joint."""
    with torch.no_grad():
        g, lut, _ = estimate_query_align_maps(
            low, nano, alpha=0.75, gamma_min=0.3, gamma_max=1.0, bins=256)
    fwd = dict(
        query_gamma_override=g, query_hist_lut_override=lut,
        stage_mode_lv3='normal', stage_mode_lv2='branch_bypass',
        stage_mode_lv1='normal')
    if joint:
        y0 = p2(lr=low, lrsr=low, ref=nano, refsr=nano,
                apply_illum=False, apply_ref_illum=False, **fwd)[0]
        y0 = p2.apply_ref_illum(y0, low, nano)
        y0 = p2.apply_illum_head(y0, low)
    else:
        with torch.no_grad():
            y0 = p2(lr=low, lrsr=low, ref=nano, refsr=nano,
                    apply_illum=False, apply_ref_illum=False, **fwd)[0]
            y0 = p2.apply_ref_illum(y0, low, nano)
            y0 = p2.apply_illum_head(y0, low)
        y0 = y0.detach()
    return y0


def h3_forward(enc, mat, head, y0, nano):
    f0 = enc((y0 + 1.) * 0.5)
    fr = enc((nano + 1.) * 0.5)
    t = mat(f0, fr)
    d = head(f0, t, y0.shape[-2:])
    return y0 + d, d


def compute_loss(y, gt, kind, rec, mss):
    mse = F.mse_loss(y, gt)
    l1 = rec(y, gt)
    ms = mss(y, gt)
    if kind == 'mse':
        loss = mse
    else:
        loss = l1 + ms  # ms_ssim_w = 1.0
    return loss, dict(mse=float(mse.detach()), l1=float(l1.detach()),
                      ms_ssim=float(ms.detach()))


def grad_norm(module, only_trainable=True):
    sq = 0.0
    n = 0
    for p in module.parameters():
        if only_trainable and not p.requires_grad:
            continue
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum())
            n += 1
    return float(np.sqrt(sq)) if n else 0.0


def param_delta(before, module):
    sq = 0.0
    for n, p in module.named_parameters():
        if n not in before:
            continue
        sq += float((p.detach().cpu() - before[n]).pow(2).sum())
    return float(np.sqrt(sq))


def snapshot(module):
    return {n: p.detach().cpu().clone() for n, p in module.named_parameters()}


@torch.no_grad()
def eval_full(p2, args, enc, mat, head, rows, device):
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
            psnr=p, ssim=s, mse=m, p2_psnr=p2p, p2_ssim=p2s)
    p2.train(was[0]); enc.train(was[1]); mat.train(was[2]); head.train(was[3])
    deltas = [v['psnr'] - v['p2_psnr'] for v in per.values()]
    summary = dict(
        psnr=float(np.mean([v['psnr'] for v in per.values()])),
        ssim=float(np.mean([v['ssim'] for v in per.values()])),
        p2_only_psnr=float(np.mean([v['p2_psnr'] for v in per.values()])),
        p2_only_ssim=float(np.mean([v['p2_ssim'] for v in per.values()])),
        median_delta=float(np.median(deltas)),
        p10_delta=float(np.percentile(deltas, 10)),
        worst_delta=float(np.min(deltas)),
        n_lt_m1=int(sum(1 for d in deltas if d < -1)),
        win=int(sum(1 for d in deltas if d > 0)),
        loss_n=int(sum(1 for d in deltas if d < 0)),
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
    ap.add_argument('--allow_overwrite', action='store_true')
    ap.add_argument('--limit_steps', type=int, default=0)
    ap.add_argument('--skip_eval', action='store_true')
    a = ap.parse_args()

    meta = ARM_META[a.arm]
    joint = meta['joint']
    loss_kind = meta['loss']
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

    p2, p2_args = load_p2(device, joint=joint)
    enc, mat, head, _ = load_h3(device, chunk=a.chunk)
    rec = ReconstructionLoss('l1')
    mss = MSSSIMLoss().to(device)

    groups = []
    if joint:
        p2_params = [p for p in p2.parameters() if p.requires_grad]
        groups.append(dict(params=p2_params, lr=1e-6, name='p2'))
    groups.append(dict(params=list(enc.parameters()), lr=1e-6, name='encoder'))
    groups.append(dict(params=list(mat.parameters()), lr=2e-6, name='matcher'))
    groups.append(dict(params=list(head.parameters()), lr=1e-5, name='head'))
    opt = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    n_p2_all = count_params(p2)
    n_p2_train = sum(p.numel() for p in p2.parameters() if p.requires_grad)
    n_h3 = count_params(enc) + count_params(mat) + count_params(head)
    print('arm=%s joint=%s loss=%s crop=%d p2_train=%d/%d h3=%d init_p2=%s init_h3=%s' % (
        a.arm, joint, loss_kind, a.crop, n_p2_train, n_p2_all, n_h3, P2_CKPT, H3_CKPT))

    tr = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_train.csv'))
    ev = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_eval.csv'))
    assert len(tr) == 500 and len(ev) == 30
    ds = CropSet(tr, crop=a.crop, seed=a.seed)
    gen = torch.Generator(); gen.manual_seed(a.seed)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=a.workers,
                    generator=gen)

    # step0 identity: H3 head matches ckpt on a crop
    enc.eval(); mat.eval(); head.eval()
    with torch.no_grad():
        batch0 = ds[0]
        low = batch0['low'][None].to(device)
        nano = batch0['nano'][None].to(device)
        y0 = p2_forward(p2, low, nano, joint=False)
        d = head(enc((y0 + 1) * 0.5),
                 mat(enc((y0 + 1) * 0.5), enc((nano + 1) * 0.5)),
                 y0.shape[-2:])
        head2 = V3B0ResidualFusion(96).to(device).eval()
        head2.load_state_dict(torch.load(H3_CKPT, map_location=device)['head'])
        d2 = head2(enc((y0 + 1) * 0.5),
                   mat(enc((y0 + 1) * 0.5), enc((nano + 1) * 0.5)),
                   y0.shape[-2:])
        assert torch.allclose(d, d2, atol=1e-5)
    print('step0 H3 weight identity OK')
    configure_p2(p2, joint)
    for m in (enc, mat, head):
        m.train()

    log_f = open(os.path.join(run_dir, 'train.jsonl'), 'w')
    hist = []
    peak_mem = 0.0
    t0 = time.time()
    steps = a.limit_steps or a.steps
    snap_p2 = snapshot(p2)
    snap_enc = snapshot(enc)
    snap_mat = snapshot(mat)
    snap_head = snapshot(head)

    def save_eval(step):
        nonlocal peak_mem
        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1 << 20))
        summary = None
        if not a.skip_eval:
            print('--- eval @', step)
            summary, per = eval_full(p2, p2_args, enc, mat, head, ev, device)
            print('  Y PSNR=%.3f SSIM=%.4f | P2-only=%.3f n<-1=%d' % (
                summary['psnr'], summary['ssim'], summary['p2_only_psnr'],
                summary['n_lt_m1']))
            with open(os.path.join(run_dir, 'eval_step_%05d.json' % step), 'w') as f:
                json.dump(dict(step=step, summary=summary, per_image=per), f)
            hist.append(dict(step=step, summary=summary))
        ckpt = os.path.join(run_dir, 'checkpoint_%05d.pt' % step)
        torch.save(dict(
            p2=p2.state_dict(),
            encoder=enc.state_dict(),
            match=mat.state_dict(),
            head=head.state_dict(),
            optimizer=opt.state_dict(),
            global_step=step,
            arm=a.arm,
            seed=a.seed,
            loss=loss_kind,
            joint_p2=joint,
            lr_groups={g['name']: g['lr'] for g in groups},
            p2_init=P2_CKPT,
            h3_init=H3_CKPT,
            p2_init_sha256=sha256(P2_CKPT),
            h3_init_sha256=sha256(H3_CKPT),
            config=vars(a),
            mean_psnr=summary,
            n_p2_trainable=n_p2_train,
            lv2_unused=list(LV2_UNUSED),
        ), ckpt)
        configure_p2(p2, joint)
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
            y0 = p2_forward(p2, low, nano, joint=joint)
            y, _ = h3_forward(enc, mat, head, y0, nano)
            loss, parts = compute_loss(y, gt, loss_kind, rec, mss)
            if not torch.isfinite(loss):
                raise SystemExit('non-finite at step %d' % step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            g_p2 = grad_norm(p2)
            g_e = grad_norm(enc)
            g_m = grad_norm(mat)
            g_h = grad_norm(head)
            opt.step()
            if torch.cuda.is_available():
                peak_mem = max(peak_mem,
                               torch.cuda.max_memory_allocated() / (1 << 20))
            if step % a.log_every == 0 or step == 1:
                row = dict(
                    step=step, data_pass=data_pass, loss=float(loss.detach()),
                    **parts,
                    g_p2=g_p2, g_enc=g_e, g_mat=g_m, g_head=g_h,
                    du_p2=param_delta(snap_p2, p2),
                    du_enc=param_delta(snap_enc, enc),
                    du_mat=param_delta(snap_mat, mat),
                    du_head=param_delta(snap_head, head),
                    peak_mem_mb=peak_mem,
                    y0_requires_grad=bool(y0.requires_grad),
                    sample_id=batch['sample_id'][0],
                )
                log_f.write(json.dumps(row) + '\n'); log_f.flush()
                print('  step %d/%d loss=%.5f mse=%.5f l1=%.5f mss=%.5f '
                      'gP2=%.2e gH=%.2e mem=%.0fMB' % (
                          step, steps, row['loss'], row['mse'], row['l1'],
                          row['ms_ssim'], g_p2, g_h, peak_mem))
            if step in SAVE_STEPS and step != 0:
                save_eval(step)
        data_pass += 1

    if steps not in SAVE_STEPS and steps != 0:
        save_eval(steps)

    cfg = dict(
        vars(a), run_name=run_name, joint=joint, loss=loss_kind,
        p2_init_sha256=sha256(P2_CKPT), h3_init_sha256=sha256(H3_CKPT),
        n_p2_all=n_p2_all, n_p2_trainable=n_p2_train, n_h3=n_h3,
        lr_groups={g['name']: g['lr'] for g in groups},
        peak_mem_mb=peak_mem, data_passes=data_pass,
        elapsed_min=(time.time() - t0) / 60,
        protocol='crop-%d' % a.crop,
        loss_recipe=('mse' if loss_kind == 'mse'
                     else 'L1 + 1.0*(1-MS-SSIM) [audited P2 recipe]'),
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
