#!/usr/bin/env python
"""v2.2 Phase A: Full-Image DC diagnosis (zero training) on Train500 + Eval30."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from local_refine_runtime import (  # noqa: E402
    EVAL_QUERY_CHUNK, load_cache, metrics, read_manifest_csv, _read_eval_tensor,
)
from model.LocalRefine import LocalSoftMatch, SharedEncoder  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion  # noqa: E402
from model.v22_dc_loss import mean_hw, split_dc  # noqa: E402

H3_CKPT = ('/root/data/experiments/TTSR-lowlight-v2/checkpoints/'
           'H3_v3b0_s42/checkpoint_02000.pt')
V2_EXP = '/root/data/experiments/TTSR-lowlight-v2'
OUT = os.path.join(ROOT, 'diagnostics/v22')
FOCUS_BAD = ['2053', '2059', '2055', '2058', '2042']
FOCUS_GOOD = ['2052', '2057', '2044', '2061', '2056']


def load_h3(device):
    ck = torch.load(H3_CKPT, map_location=device)
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
def predict_d(enc, mat, head, y0, nano):
    f0 = enc((y0 + 1.) * 0.5)
    fr = enc((nano + 1.) * 0.5)
    t = mat(f0, fr)
    return head(f0, t, y0.shape[-2:])


def analyze_split(enc, mat, head, rows, cache, device, tag):
    flat = []
    for i, row in enumerate(rows):
        y0 = torch.from_numpy(np.ascontiguousarray(
            load_cache(cache, row)))[None].to(device)
        gt = _read_eval_tensor(row['high_path'], device)
        nano = _read_eval_tensor(row['nano_path'], device, size=y0.shape[-2:])
        d = predict_d(enc, mat, head, y0, nano)
        y3 = y0 + d
        m, ds = split_dc(d)
        e0 = y0 - gt
        e = mean_hw(e0)
        target = gt - y0
        m_tgt = mean_hw(target)
        centered = y0 + ds

        p_p2, s_p2, mse_p2 = metrics(y0, gt)
        p_h3, s_h3, mse_h3 = metrics(y3, gt)
        p_c, s_c, _ = metrics(centered, gt)

        cross = float((2.0 * e0 * d).mean())
        quad = float((d * d).mean())
        dmse = float((mse_h3 - mse_p2) if False else
                     ((y3 - gt) ** 2).mean() - ((y0 - gt) ** 2).mean())
        # use float tensors for mse
        mse_p2_t = float(((y0 - gt) ** 2).mean())
        mse_h3_t = float(((y3 - gt) ** 2).mean())
        dmse = mse_h3_t - mse_p2_t

        sum_m = float((m.expand_as(d) ** 2).sum())
        sum_s = float((ds ** 2).sum())
        e_mean_frac = sum_m / (sum_m + sum_s + 1e-12)

        mv = m[0, :, 0, 0].cpu().numpy()
        ev = e[0, :, 0, 0].cpu().numpy()
        mtv = m_tgt[0, :, 0, 0].cpu().numpy()
        dot_em = float(np.dot(ev, mv))
        m_norm = float(np.linalg.norm(mv))

        # harmful if D_mean aligns with existing P2 bias (same sign product > 0)
        harmful_dc = bool(dot_em > 0 and m_norm > 1e-4)

        flat.append(dict(
            sample_id=row['sample_id'], split=tag,
            psnr_P2=p_p2, ssim_P2=s_p2,
            psnr_H3=p_h3, ssim_H3=s_h3,
            psnr_Centered=p_c, ssim_Centered=s_c,
            delta_h3=p_h3 - p_p2, delta_centered=p_c - p_p2,
            d_mean_R=float(mv[0]), d_mean_G=float(mv[1]), d_mean_B=float(mv[2]),
            m_tgt_R=float(mtv[0]), m_tgt_G=float(mtv[1]), m_tgt_B=float(mtv[2]),
            e0_mean_R=float(ev[0]), e0_mean_G=float(ev[1]), e0_mean_B=float(ev[2]),
            dot_e_m=dot_em, m_norm=m_norm,
            e_mean_frac=e_mean_frac, e_spatial_frac=1.0 - e_mean_frac,
            dmse=dmse, term_cross=cross, term_d2=quad,
            identity_err=abs(dmse - (cross + quad)),
            harmful_dc=int(harmful_dc),
            excessive=int(m_norm > float(np.linalg.norm(mtv)) * 1.5
                          and m_norm > 0.02),
        ))
        if (i + 1) % 50 == 0 or i == 0:
            print('[%s %d/%d] %s ΔH3=%+.3f' % (
                tag, i + 1, len(rows), row['sample_id'], p_h3 - p_p2))
    return flat


@torch.no_grad()
def crop_full_compare(enc, mat, head, rows, cache, device, n=20, crop=128, seed=0):
    """Compare full-then-crop vs crop-then-H3 on Train samples."""
    rng = np.random.RandomState(seed)
    # pick diverse: first n with preference to harmful/extreme later filled
    picks = list(range(min(n, len(rows))))
    out = []
    for i in picks:
        row = rows[i]
        y0 = torch.from_numpy(np.ascontiguousarray(
            load_cache(cache, row)))[None].to(device)
        nano = _read_eval_tensor(row['nano_path'], device, size=y0.shape[-2:])
        _, _, H, W = y0.shape
        if H < crop or W < crop:
            continue
        oy = int(rng.randint(0, H - crop + 1))
        ox = int(rng.randint(0, W - crop + 1))

        d_full = predict_d(enc, mat, head, y0, nano)
        d_fc = d_full[:, :, oy:oy + crop, ox:ox + crop]

        y0c = y0[:, :, oy:oy + crop, ox:ox + crop]
        nc = nano[:, :, oy:oy + crop, ox:ox + crop]
        d_c = predict_d(enc, mat, head, y0c, nc)

        mae = float((d_fc - d_c).abs().mean())
        # cosine
        a = d_fc.reshape(-1); b = d_c.reshape(-1)
        cos = float((a * b).sum() / (a.norm() * b.norm() + 1e-8))
        m_fc = mean_hw(d_fc)[0, :, 0, 0]
        m_c = mean_hw(d_c)[0, :, 0, 0]
        mean_diff = float((m_fc - m_c).abs().mean())
        # border vs center
        border = torch.zeros_like(d_fc)
        bsz = 8
        border[:, :, :bsz, :] = 1
        border[:, :, -bsz:, :] = 1
        border[:, :, :, :bsz] = 1
        border[:, :, :, -bsz:] = 1
        interior = 1 - border
        mae_b = float(((d_fc - d_c).abs() * border).sum() / (border.sum() + 1e-8))
        mae_i = float(((d_fc - d_c).abs() * interior).sum() / (interior.sum() + 1e-8))

        out.append(dict(
            sample_id=row['sample_id'], oy=oy, ox=ox,
            mae=mae, cosine=cos, mean_abs_diff=mean_diff,
            mae_border=mae_b, mae_interior=mae_i,
            d_full_m_norm=float(mean_hw(d_full).reshape(-1).norm()),
            d_crop_m_norm=float(m_c.norm()),
        ))
    return out


def summarize(flat, tag):
    def m(k):
        return float(np.mean([r[k] for r in flat]))
    pos = sum(1 for r in flat if r['delta_h3'] > 0.01)
    neg = sum(1 for r in flat if r['delta_h3'] < -0.01)
    harm = sum(r['harmful_dc'] for r in flat)
    excess = sum(r['excessive'] for r in flat)
    return dict(
        tag=tag, n=len(flat),
        mean_delta_h3=m('delta_h3'),
        mean_delta_centered=m('delta_centered'),
        mean_m_norm=m('m_norm'),
        mean_dot_e_m=m('dot_e_m'),
        mean_e_mean_frac=m('e_mean_frac'),
        frac_positive_h3=pos / len(flat),
        frac_negative_h3=neg / len(flat),
        frac_harmful_dc=harm / len(flat),
        frac_excessive=excess / len(flat),
        mean_identity_err=m('identity_err'),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_limit', type=int, default=0)
    ap.add_argument('--skip_train', action='store_true')
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(OUT, exist_ok=True)
    print('device', device)
    enc, mat, head = load_h3(device)

    # Reproduce H3@2000 on Eval30 quickly
    ev = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_eval.csv'))
    tr = read_manifest_csv(os.path.join(V2_EXP, 'manifests/manifests_train.csv'))
    if a.train_limit:
        tr = tr[:a.train_limit]

    flat_ev = analyze_split(
        enc, mat, head, ev, os.path.join(V2_EXP, 'cache_p2_eval'), device, 'eval')
    print('Eval30 H3 mean PSNR', float(np.mean([r['psnr_H3'] for r in flat_ev])))

    flat_tr = []
    if not a.skip_train:
        flat_tr = analyze_split(
            enc, mat, head, tr, os.path.join(V2_EXP, 'cache_p2_train'),
            device, 'train')

    # crop/full
    crop_rows = crop_full_compare(
        enc, mat, head, tr if flat_tr else tr[:20],
        os.path.join(V2_EXP, 'cache_p2_train'), device, n=20)

    def write_csv(path, rows):
        if not rows:
            return
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    write_csv(os.path.join(OUT, 'eval30_dc_analysis.csv'), flat_ev)
    write_csv(os.path.join(OUT, 'train500_dc_analysis.csv'), flat_tr)
    write_csv(os.path.join(OUT, 'crop_full_comparison.csv'), crop_rows)

    summary = dict(
        eval=summarize(flat_ev, 'eval'),
        train=summarize(flat_tr, 'train') if flat_tr else None,
        crop_full=dict(
            n=len(crop_rows),
            mean_mae=float(np.mean([r['mae'] for r in crop_rows])) if crop_rows else None,
            mean_cosine=float(np.mean([r['cosine'] for r in crop_rows])) if crop_rows else None,
            mean_abs_mean_diff=float(np.mean([r['mean_abs_diff'] for r in crop_rows])) if crop_rows else None,
            mean_mae_border=float(np.mean([r['mae_border'] for r in crop_rows])) if crop_rows else None,
            mean_mae_interior=float(np.mean([r['mae_interior'] for r in crop_rows])) if crop_rows else None,
        ),
        focus_eval={
            sid: next((r for r in flat_ev if sid in r['sample_id']), None)
            for sid in FOCUS_BAD + FOCUS_GOOD
        },
        h3_ckpt=H3_CKPT,
    )
    # trim focus
    for k, v in list(summary['focus_eval'].items()):
        if v is None:
            continue
        summary['focus_eval'][k] = {kk: v[kk] for kk in (
            'sample_id', 'psnr_P2', 'psnr_H3', 'psnr_Centered', 'delta_h3',
            'delta_centered', 'd_mean_R', 'd_mean_G', 'd_mean_B',
            'm_tgt_R', 'm_tgt_G', 'm_tgt_B', 'dot_e_m', 'm_norm',
            'e_mean_frac', 'harmful_dc')}

    with open(os.path.join(OUT, 'dc_distribution_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(dict(eval=summary['eval'], train=summary['train'],
                          crop_full=summary['crop_full']), indent=2))


if __name__ == '__main__':
    main()
