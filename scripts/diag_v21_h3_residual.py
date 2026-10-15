#!/usr/bin/env python
"""v2.1 zero-training H3 residual failure diagnosis (Huawei Eval30)."""
from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from local_refine_runtime import (  # noqa: E402
    EVAL_QUERY_CHUNK, load_cache, metrics, mismatch_permutation,
    read_manifest_csv, _read_eval_tensor,
)
from model.LocalRefine import LocalSoftMatch, SharedEncoder  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion  # noqa: E402

EXP = '/root/data/experiments/TTSR-lowlight-v2'
CKPT = os.path.join(EXP, 'checkpoints/H3_v3b0_s42/checkpoint_02000.pt')
CACHE = os.path.join(EXP, 'cache_p2_eval')
OUT_DIAG = os.path.join(ROOT, 'diagnostics/v2')
OUT_VIS = os.path.join(OUT_DIAG, 'h3_diag_2053')
FOCUS_WORST = 'Huawei/2053.jpg'


def to_u8(t):
    if t.dim() == 4:
        t = t[0]
    return ((t.detach().float().cpu() + 1) * 127.5).clamp(0, 255).byte().permute(
        1, 2, 0).numpy()


def load_h3(device):
    ck = torch.load(CKPT, map_location=device)
    enc = SharedEncoder(32).to(device).eval()
    mat = LocalSoftMatch(32, 4, 0.1, chunk=EVAL_QUERY_CHUNK).to(device).eval()
    head = V3B0ResidualFusion(96).to(device).eval()
    enc.load_state_dict(ck['encoder'])
    mat.load_state_dict(ck['match'])
    head.load_state_dict(ck['head'])
    for p in list(enc.parameters()) + list(mat.parameters()) + list(head.parameters()):
        p.requires_grad_(False)
    return enc, mat, head


@torch.no_grad()
def h3_delta(enc, mat, head, y0, r_eff):
    """Return ΔRGB from frozen H3 given Y0 and second-path image."""
    f0 = enc((y0 + 1.) * 0.5)
    fr = enc((r_eff + 1.) * 0.5)
    t = mat(f0, fr)
    return head(f0, t, y0.shape[-2:])


@torch.no_grad()
def h3_delta_zero_t(enc, head, y0):
    """Zero-T: T=0 feature, still uses F0 from Y0 (no Nano match)."""
    f0 = enc((y0 + 1.) * 0.5)
    t = torch.zeros_like(f0)
    return head(f0, t, y0.shape[-2:])


def spat_mean(d):
    """Per-channel spatial mean -> [1,3,1,1]."""
    return d.mean(dim=(-2, -1), keepdim=True)


def freq_energy(d):
    """Low/high energy of mean-over-RGB residual map."""
    m = d[0].mean(0, keepdim=True)[None]  # [1,1,H,W]
    k = torch.ones(1, 1, 5, 5, device=d.device) / 25.0
    low = F.conv2d(m, k, padding=2)
    high = m - low
    e_low = float((low ** 2).mean())
    e_high = float((high ** 2).mean())
    return e_low, e_high, e_low / (e_low + e_high + 1e-12)


def mse_decomp(y0, gt, d):
    """Identity: ΔMSE = mean(2*(Y0-GT)*D + D^2)."""
    err0 = y0 - gt
    term_cross = float((2.0 * err0 * d).mean())
    term_d2 = float((d * d).mean())
    mse_p2 = float((err0 ** 2).mean())
    mse_h3 = float(((y0 + d - gt) ** 2).mean())
    return dict(
        mse_p2=mse_p2, mse_h3=mse_h3, dmse=mse_h3 - mse_p2,
        term_cross=term_cross, term_d2=term_d2,
        check=abs((mse_h3 - mse_p2) - (term_cross + term_d2)),
    )


def channel_stats(y0, y3, gt, d):
    out = {}
    for i, ch in enumerate('RGB'):
        e_p2 = float((y0[0, i] - gt[0, i]).mean())
        e_h3 = float((y3[0, i] - gt[0, i]).mean())
        dm = float(d[0, i].mean())
        ds_std = float((d[0, i] - dm).std())
        out[ch] = dict(
            p2_mean_signed_err=e_p2,
            h3_mean_signed_err=e_h3,
            d_mean=dm,
            d_spatial_std=ds_std,
        )
    e_mean = float((spat_mean(d) ** 2).mean())  # broadcast energy approx
    # proper energies over full tensor
    d_mean = spat_mean(d)
    d_sp = d - d_mean
    e_m = float((d_mean ** 2).mean())  # equals mean of channel means squared
    # spatial mean of d_mean**2 across H,W is just d_mean**2 (constant map)
    # Use sum of squares ratio over all elements for energy share:
    sum_m = float((d_mean.expand_as(d) ** 2).sum())
    sum_s = float((d_sp ** 2).sum())
    sum_t = sum_m + sum_s
    out['energy'] = dict(
        e_mean_frac=sum_m / (sum_t + 1e-12),
        e_spatial_frac=sum_s / (sum_t + 1e-12),
        sum_sq_mean=sum_m, sum_sq_spatial=sum_s,
    )
    return out


@torch.no_grad()
def analyze_one(enc, mat, head, row, rows, mismatch, device):
    y0 = torch.from_numpy(np.ascontiguousarray(load_cache(CACHE, row)))[None].to(device)
    gt = _read_eval_tensor(row['high_path'], device)
    nano = _read_eval_tensor(row['nano_path'], device, size=y0.shape[-2:])
    # Correct Nano residual
    d = h3_delta(enc, mat, head, y0, nano)
    y3 = y0 + d
    d_mean = spat_mean(d)
    d_sp = d - d_mean
    mean_only = y0 + d_mean
    centered = y0 + d_sp

    # ablations of second path only
    d_self = h3_delta(enc, mat, head, y0, y0)
    idx = next(i for i, r in enumerate(rows)
               if r['sample_id'] == row['sample_id'])
    mis_row = rows[mismatch[idx]]
    mis_nano = _read_eval_tensor(mis_row['nano_path'], device, size=y0.shape[-2:])
    d_mis = h3_delta(enc, mat, head, y0, mis_nano)
    d_zt = h3_delta_zero_t(enc, head, y0)

    variants = {
        'P2': y0,
        'H3': y3,
        'MeanOnly': mean_only,
        'Centered': centered,
        'H3_Self': y0 + d_self,
        'H3_Mismatch': y0 + d_mis,
        'H3_ZeroT': y0 + d_zt,
        'Bypass': y0,
    }
    met = {}
    for k, v in variants.items():
        p, s, m = metrics(v, gt)
        met[k] = dict(psnr=p, ssim=s, mse=m)

    ch = channel_stats(y0, y3, gt, d)
    decomp = mse_decomp(y0, gt, d)
    decomp_mean = mse_decomp(y0, gt, d_mean.expand_as(d))
    decomp_sp = mse_decomp(y0, gt, d_sp)
    e_low, e_high, r_low = freq_energy(d)

    return dict(
        sample_id=row['sample_id'],
        metrics=met,
        d_mean_rgb=[float(d_mean[0, i, 0, 0]) for i in range(3)],
        channel=ch,
        decomp_full=decomp,
        decomp_meanonly=decomp_mean,
        decomp_centered=decomp_sp,
        freq=dict(e_low=e_low, e_high=e_high, ratio_low=r_low),
        tensors=dict(y0=y0, y3=y3, gt=gt, d=d, d_mean=d_mean, d_sp=d_sp,
                     mean_only=mean_only, centered=centered),
    )


def viz_2053(pack, path):
    os.makedirs(path, exist_ok=True)
    y0, y3, gt = pack['tensors']['y0'], pack['tensors']['y3'], pack['tensors']['gt']
    d = pack['tensors']['d']
    mean_only = pack['tensors']['mean_only']
    centered = pack['tensors']['centered']
    d_mean = pack['tensors']['d_mean']

    # amplify residual for display
    amp = 8.0
    d_vis = (d * amp).clamp(-1, 1)
    # per-channel mean residual as solid color patches
    dm = d_mean[0, :, 0, 0].cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    panels = [
        (axes[0, 0], to_u8(y0), 'P2 (Y0)'),
        (axes[0, 1], to_u8(y3), 'H3'),
        (axes[0, 2], to_u8(gt), 'GT'),
        (axes[0, 3], to_u8(d_vis), 'D x%.0f (amp)' % amp),
        (axes[1, 0], to_u8(mean_only), 'MeanOnly'),
        (axes[1, 1], to_u8(centered), 'Centered'),
    ]
    for ax, im, title in panels:
        ax.imshow(im)
        ax.set_title(title)
        ax.axis('off')
    # mean residual bar
    ax = axes[1, 2]
    ax.bar(['R', 'G', 'B'], dm, color=['#c44', '#4a4', '#44c'])
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title('D_mean RGB')
    ax.set_ylabel('Δ ([-1,1] scale)')
    # abs residual heatmap
    ax = axes[1, 3]
    heat = d[0].abs().mean(0).cpu().numpy()
    im = ax.imshow(heat, cmap='magma')
    ax.set_title('|D| mean-RGB')
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('2053 H3 residual diagnosis  P2=%.3f  H3=%.3f  MeanOnly=%.3f  Centered=%.3f' % (
        pack['metrics']['P2']['psnr'], pack['metrics']['H3']['psnr'],
        pack['metrics']['MeanOnly']['psnr'], pack['metrics']['Centered']['psnr']))
    fig.tight_layout()
    fig.savefig(os.path.join(path, '2053_overview.png'), dpi=140)
    plt.close(fig)

    # also save individual PNGs
    for name, t in [('P2', y0), ('H3', y3), ('GT', gt), ('MeanOnly', mean_only),
                    ('Centered', centered), ('D_amp8', d_vis)]:
        from imageio import imwrite
        imwrite(os.path.join(path, '%s.png' % name), to_u8(t))


def flat_row(pack):
    m = pack['metrics']
    ch = pack['channel']
    e = ch['energy']
    f = pack['freq']
    df = pack['decomp_full']
    dm = pack['decomp_meanonly']
    ds = pack['decomp_centered']
    return dict(
        sample_id=pack['sample_id'],
        psnr_P2=m['P2']['psnr'], ssim_P2=m['P2']['ssim'],
        psnr_H3=m['H3']['psnr'], ssim_H3=m['H3']['ssim'],
        psnr_MeanOnly=m['MeanOnly']['psnr'], ssim_MeanOnly=m['MeanOnly']['ssim'],
        psnr_Centered=m['Centered']['psnr'], ssim_Centered=m['Centered']['ssim'],
        psnr_H3_Self=m['H3_Self']['psnr'],
        psnr_H3_Mismatch=m['H3_Mismatch']['psnr'],
        psnr_H3_ZeroT=m['H3_ZeroT']['psnr'],
        d_mean_R=pack['d_mean_rgb'][0],
        d_mean_G=pack['d_mean_rgb'][1],
        d_mean_B=pack['d_mean_rgb'][2],
        p2_bias_R=ch['R']['p2_mean_signed_err'],
        p2_bias_G=ch['G']['p2_mean_signed_err'],
        p2_bias_B=ch['B']['p2_mean_signed_err'],
        h3_bias_R=ch['R']['h3_mean_signed_err'],
        h3_bias_G=ch['G']['h3_mean_signed_err'],
        h3_bias_B=ch['B']['h3_mean_signed_err'],
        d_spatial_std_R=ch['R']['d_spatial_std'],
        d_spatial_std_G=ch['G']['d_spatial_std'],
        d_spatial_std_B=ch['B']['d_spatial_std'],
        e_mean_frac=e['e_mean_frac'],
        e_spatial_frac=e['e_spatial_frac'],
        freq_e_low=f['e_low'], freq_e_high=f['e_high'], freq_ratio_low=f['ratio_low'],
        mse_p2=df['mse_p2'], mse_h3=df['mse_h3'], dmse=df['dmse'],
        term_cross=df['term_cross'], term_d2=df['term_d2'],
        dmse_meanonly=dm['dmse'], dmse_centered=ds['dmse'],
        identity_check=df['check'],
        delta_psnr_h3=m['H3']['psnr'] - m['P2']['psnr'],
        delta_psnr_meanonly=m['MeanOnly']['psnr'] - m['P2']['psnr'],
        delta_psnr_centered=m['Centered']['psnr'] - m['P2']['ssim'] and (
            m['Centered']['psnr'] - m['P2']['psnr']),
    )


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)
    os.makedirs(OUT_DIAG, exist_ok=True)
    rows = read_manifest_csv(os.path.join(EXP, 'manifests/manifests_eval.csv'))
    mismatch = mismatch_permutation(rows)
    enc, mat, head = load_h3(device)

    packs = []
    for i, row in enumerate(rows):
        pack = analyze_one(enc, mat, head, row, rows, mismatch, device)
        # drop heavy tensors except for 2053
        if row['sample_id'] == FOCUS_WORST:
            viz_2053(pack, OUT_VIS)
            focus = {k: v for k, v in pack.items() if k != 'tensors'}
            with open(os.path.join(OUT_DIAG, 'h3_diag_2053.json'), 'w') as f:
                json.dump(focus, f, indent=2)
        # strip tensors for aggregate
        light = {k: v for k, v in pack.items() if k != 'tensors'}
        packs.append(light)
        print('[%02d] %s H3-P2=%+.3f MeanOnly=%+.3f Centered=%+.3f' % (
            i, row['sample_id'],
            pack['metrics']['H3']['psnr'] - pack['metrics']['P2']['psnr'],
            pack['metrics']['MeanOnly']['psnr'] - pack['metrics']['P2']['psnr'],
            pack['metrics']['Centered']['psnr'] - pack['metrics']['P2']['psnr']))

    # CSV
    flat = []
    for pack in packs:
        # rebuild flat without tensors
        m = pack['metrics']
        ch = pack['channel']
        e = ch['energy']
        f = pack['freq']
        df = pack['decomp_full']
        dm = pack['decomp_meanonly']
        ds = pack['decomp_centered']
        flat.append(dict(
            sample_id=pack['sample_id'],
            psnr_P2=m['P2']['psnr'], ssim_P2=m['P2']['ssim'],
            psnr_H3=m['H3']['psnr'], ssim_H3=m['H3']['ssim'],
            psnr_MeanOnly=m['MeanOnly']['psnr'], ssim_MeanOnly=m['MeanOnly']['ssim'],
            psnr_Centered=m['Centered']['psnr'], ssim_Centered=m['Centered']['ssim'],
            psnr_H3_Self=m['H3_Self']['psnr'],
            psnr_H3_Mismatch=m['H3_Mismatch']['psnr'],
            psnr_H3_ZeroT=m['H3_ZeroT']['psnr'],
            d_mean_R=pack['d_mean_rgb'][0],
            d_mean_G=pack['d_mean_rgb'][1],
            d_mean_B=pack['d_mean_rgb'][2],
            p2_bias_R=ch['R']['p2_mean_signed_err'],
            p2_bias_G=ch['G']['p2_mean_signed_err'],
            p2_bias_B=ch['B']['p2_mean_signed_err'],
            h3_bias_R=ch['R']['h3_mean_signed_err'],
            h3_bias_G=ch['G']['h3_mean_signed_err'],
            h3_bias_B=ch['B']['h3_mean_signed_err'],
            d_spatial_std_R=ch['R']['d_spatial_std'],
            d_spatial_std_G=ch['G']['d_spatial_std'],
            d_spatial_std_B=ch['B']['d_spatial_std'],
            e_mean_frac=e['e_mean_frac'],
            e_spatial_frac=e['e_spatial_frac'],
            freq_e_low=f['e_low'], freq_e_high=f['e_high'],
            freq_ratio_low=f['ratio_low'],
            mse_p2=df['mse_p2'], mse_h3=df['mse_h3'], dmse=df['dmse'],
            term_cross=df['term_cross'], term_d2=df['term_d2'],
            dmse_meanonly=dm['dmse'], dmse_centered=ds['dmse'],
            identity_check=df['check'],
            delta_psnr_h3=m['H3']['psnr'] - m['P2']['psnr'],
            delta_psnr_meanonly=m['MeanOnly']['psnr'] - m['P2']['psnr'],
            delta_psnr_centered=m['Centered']['psnr'] - m['P2']['psnr'],
            delta_psnr_self=m['H3_Self']['psnr'] - m['P2']['psnr'],
            delta_psnr_mismatch=m['H3_Mismatch']['psnr'] - m['P2']['psnr'],
            delta_psnr_zerot=m['H3_ZeroT']['psnr'] - m['P2']['psnr'],
        ))

    csv_path = os.path.join(OUT_DIAG, 'h3_residual_decomposition.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader(); w.writerows(flat)

    # aggregates
    def mean_key(k):
        return float(np.mean([r[k] for r in flat]))

    summary = dict(
        n=len(flat),
        mean_psnr=dict(
            P2=mean_key('psnr_P2'), H3=mean_key('psnr_H3'),
            MeanOnly=mean_key('psnr_MeanOnly'), Centered=mean_key('psnr_Centered'),
            H3_Self=mean_key('psnr_H3_Self'),
            H3_Mismatch=mean_key('psnr_H3_Mismatch'),
            H3_ZeroT=mean_key('psnr_H3_ZeroT'),
        ),
        mean_delta_vs_P2=dict(
            H3=mean_key('delta_psnr_h3'),
            MeanOnly=mean_key('delta_psnr_meanonly'),
            Centered=mean_key('delta_psnr_centered'),
            Self=mean_key('delta_psnr_self'),
            Mismatch=mean_key('delta_psnr_mismatch'),
            ZeroT=mean_key('delta_psnr_zerot'),
        ),
        mean_e_mean_frac=mean_key('e_mean_frac'),
        mean_freq_ratio_low=mean_key('freq_ratio_low'),
        mean_dmse=mean_key('dmse'),
        mean_term_cross=mean_key('term_cross'),
        mean_term_d2=mean_key('term_d2'),
        mean_dmse_meanonly=mean_key('dmse_meanonly'),
        mean_dmse_centered=mean_key('dmse_centered'),
        worst5=sorted(flat, key=lambda r: r['delta_psnr_h3'])[:5],
        best5=sorted(flat, key=lambda r: -r['delta_psnr_h3'])[:5],
        focus_2053=next(r for r in flat if '2053' in r['sample_id']),
        ckpt=CKPT,
        device=device,
    )
    # trim worst/best for json
    for key in ('worst5', 'best5'):
        summary[key] = [
            dict(sample_id=r['sample_id'], delta_psnr_h3=r['delta_psnr_h3'],
                 psnr_H3=r['psnr_H3'], psnr_P2=r['psnr_P2'],
                 psnr_MeanOnly=r['psnr_MeanOnly'],
                 psnr_Centered=r['psnr_Centered'],
                 e_mean_frac=r['e_mean_frac'],
                 d_mean_RGB=[r['d_mean_R'], r['d_mean_G'], r['d_mean_B']],
                 dmse=r['dmse'], term_cross=r['term_cross'], term_d2=r['term_d2'])
            for r in summary[key]
        ]
    with open(os.path.join(OUT_DIAG, 'h3_diag_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in (
        'mean_psnr', 'mean_delta_vs_P2', 'mean_e_mean_frac',
        'mean_freq_ratio_low', 'mean_dmse', 'mean_term_cross', 'mean_term_d2',
        'mean_dmse_meanonly', 'mean_dmse_centered')}, indent=2))
    print('wrote', csv_path)


if __name__ == '__main__':
    main()
