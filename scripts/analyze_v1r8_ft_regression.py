#!/usr/bin/env python3
"""Phase C — FT regression analysis from existing logs/checkpoints (no training)."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'diagnostics' / 'v1r8' / 'ft_regression'
OUT.mkdir(parents=True, exist_ok=True)

V1R3 = Path('/root/data/experiments/retry/TTSR-lowlight-v1r3-nano-Huawei')
C0 = Path('/root/data/experiments/retry/v1r6-ft-C0-control')
C1 = Path('/root/data/experiments/retry/v1r6-ft-C1-gamma075')
B2 = Path('/root/data/experiments/retry/v1r7-ft-B2-b050')
BASE_CKPT = V1R3 / 'model' / 'model_00004.pt'


def _parse_train_avgs(log_path):
    rows = []
    if not log_path.is_file():
        return rows
    text = log_path.read_text(errors='ignore')
    for m in re.finditer(
            r'epoch (\d+) train_avg\s+rec_loss: ([0-9.eE+-]+)\s+ms_ssim_loss: ([0-9.eE+-]+)',
            text):
        rows.append({
            'epoch': int(m.group(1)),
            'rec_loss': float(m.group(2)),
            'ms_ssim_loss': float(m.group(3)),
        })
    return rows


def _parse_eval_psnr(log_path):
    rows = []
    if not log_path.is_file():
        return rows
    text = log_path.read_text(errors='ignore')
    # Epoch N evaluation ... then PSNR line
    epochs = [int(x) for x in re.findall(r'Epoch (\d+) evaluation process', text)]
    psnrs = re.findall(
        r'PSNR: ([0-9.]+)\s+SSIM: ([0-9.]+).*?\(PSNR/SSIM=RGB\)', text)
    # Also capture Current epoch learning rate
    lrs = [float(x) for x in re.findall(
        r'Current epoch learning rate: ([0-9.eE+-]+)', text)]
    for i, (p, s) in enumerate(psnrs):
        rows.append({
            'epoch': epochs[i] if i < len(epochs) else i + 1,
            'psnr': float(p), 'ssim': float(s),
            'lr': lrs[i] if i < len(lrs) else None,
        })
    return rows


def _ckpt_delta(base_path, ft_path):
    a = torch.load(base_path, map_location='cpu')
    b = torch.load(ft_path, map_location='cpu')
    if any(k.startswith('module.') for k in a):
        a = {k.replace('module.', '', 1): v for k, v in a.items()}
    if any(k.startswith('module.') for k in b):
        b = {k.replace('module.', '', 1): v for k, v in b.items()}
    keys = sorted(set(a) & set(b))
    changed = []
    frozen_like = []
    for k in keys:
        if a[k].shape != b[k].shape:
            continue
        da = (a[k].float() - b[k].float()).abs()
        mx = float(da.max().item())
        mean = float(da.mean().item())
        if mx < 1e-8:
            frozen_like.append(k)
        else:
            changed.append({'key': k, 'max_abs': mx, 'mean_abs': mean})
    changed.sort(key=lambda x: -x['max_abs'])
    # Prefix buckets
    buckets = {}
    for c in changed:
        pref = c['key'].split('.')[0]
        buckets.setdefault(pref, {'n': 0, 'max_abs': 0.0})
        buckets[pref]['n'] += 1
        buckets[pref]['max_abs'] = max(buckets[pref]['max_abs'], c['max_abs'])
    return {
        'n_shared': len(keys),
        'n_unchanged': len(frozen_like),
        'n_changed': len(changed),
        'top_changed': changed[:15],
        'buckets': buckets,
        'lte_unchanged': all(k.startswith('LTE.') for k in frozen_like) or
        all((a[k] - b[k]).abs().max().item() < 1e-8
            for k in keys if k.startswith('LTE.')),
    }


def _per_image_traj(dir_):
    d = Path(dir_) / 'per_image_eval'
    if not d.is_dir():
        return None
    out = {}
    for ep in range(1, 6):
        p = d / ('data1_nanobanana_e%05d.json' % ep)
        if not p.is_file():
            continue
        j = json.loads(p.read_text())
        out[ep] = {
            'psnr_mean': j['psnr_rgb_mean'],
            'ssim_mean': j['ssim_rgb_mean'],
            'per_image': j['per_image'],
        }
    return out


def main():
    report = {
        'baseline_ckpt': str(BASE_CKPT),
        'baseline_exists': BASE_CKPT.is_file(),
        'recipes': {
            'v1r3_original': {
                'script': 'train/train_v1r3_nano_continue.sh',
                'lr': 1e-5, 'epochs': 10, 'crop': 192,
                'freeze_lte': True, 'loss': 'L1+MS-SSIM',
                'seed': 'not set in script',
                'init': 'v1r2 e4',
            },
            'C0_continue': {
                'script': 'train/train_v1r6_ft_c0.sh',
                'lr': 3e-6, 'epochs': 5, 'crop': 192,
                'freeze_lte': True, 'loss': 'L1+MS-SSIM',
                'seed': 42, 'align': False,
                'init': 'v1r3e4',
                'optimizer_state': 'fresh Adam (no restore from v1r3)',
                'scheduler': 'StepLR decay=999999 → constant lr',
            },
            'C1_gamma_ft': {
                'script': 'train/train_v1r6_ft_c1.sh',
                'same_as_C0_except': 'query_illum_align=True alpha=0.75',
                'seed': 42,
            },
            'B2_hist_ft': {
                'script': 'train/train_v1r7_ft_b2.sh',
                'same_as_C0_except': 'align+alpha=0.75 beta=0.50',
                'seed': 42,
            },
        },
        'missing_evidence': [],
    }

    # Logs
    for tag, path in [
        ('v1r3', V1R3 / 'train_v1r3_nano_Huawei.log'),
        ('C0', C0 / 'train_v1r6_ft_C0.log'),
        ('C1', C1 / 'train_v1r6_ft_C1.log'),
        ('B2', B2 / 'train_v1r7_ft_B2_b050.log'),
    ]:
        if not path.is_file():
            report['missing_evidence'].append('log missing: %s' % path)
            continue
        report.setdefault('train_loss', {})[tag] = _parse_train_avgs(path)
        report.setdefault('eval_traj', {})[tag] = _parse_eval_psnr(path)

    # Checkpoint deltas e5 vs base
    for tag, path in [
        ('C0e5', C0 / 'model' / 'model_00005.pt'),
        ('C1e5', C1 / 'model' / 'model_00005.pt'),
        ('B2e5', B2 / 'model' / 'model_00005.pt'),
    ]:
        if not path.is_file():
            report['missing_evidence'].append('ckpt missing: %s' % path)
            continue
        report.setdefault('ckpt_delta_vs_v1r3e4', {})[tag] = _ckpt_delta(
            BASE_CKPT, path)

    # Per-image traj
    for tag, path in [('C0', C0), ('C1', C1), ('B2', B2)]:
        traj = _per_image_traj(path)
        if traj is None:
            report['missing_evidence'].append('per_image_eval missing: %s' % path)
        else:
            # How many images drop vs epoch1
            e1 = {r['idx']: r['psnr_rgb'] for r in traj[1]['per_image']}
            e5 = {r['idx']: r['psnr_rgb'] for r in traj[5]['per_image']}
            drops = sum(1 for i in e1 if e5[i] < e1[i] - 0.01)
            gains = sum(1 for i in e1 if e5[i] > e1[i] + 0.01)
            report.setdefault('per_image_traj_summary', {})[tag] = {
                'psnr_e1': traj[1]['psnr_mean'],
                'psnr_e5': traj[5]['psnr_mean'],
                'ssim_e1': traj[1]['ssim_mean'],
                'ssim_e5': traj[5]['ssim_mean'],
                'n_up_e5_vs_e1': gains,
                'n_down_e5_vs_e1': drops,
                'vs_baseline_e5': traj[5]['psnr_mean'] - 20.162,
            }

    # Stochastic comparability note
    report['stochastic_comparability'] = {
        'shared_seed': 42,
        'dataloader_generator_seeded': True,
        'worker_init_fn': True,
        'cudnn_deterministic': True,
        'caveat': (
            'Same seed + Generator makes C0/C1/B2 share the same shuffle order '
            'and numpy/torch RNG stream at process start, but Query gamma/hist '
            'path adds extra torch ops during __getitem__/forward, which can '
            'desynchronize subsequent RNG draws relative to C0. Independent Ref '
            'crop uses python random / np.random inside transforms — seeded via '
            'worker_init but not guaranteed bit-identical across align on/off.'
        ),
    }

    # SSIM up / PSNR down pattern
    for tag in ('C0', 'C1', 'B2'):
        pe = report.get('per_image_traj_summary', {}).get(tag)
        if pe:
            pe['ssim_improved_vs_baseline'] = pe['ssim_e5'] > 0.6324
            pe['psnr_regressed_vs_baseline'] = pe['psnr_e5'] < 20.162

    out_path = OUT / 'ft_regression_analysis.json'
    out_path.write_text(json.dumps(report, indent=2))
    print('Wrote', out_path)
    print('missing:', report['missing_evidence'])
    if 'ckpt_delta_vs_v1r3e4' in report:
        for k, v in report['ckpt_delta_vs_v1r3e4'].items():
            print(k, 'changed', v['n_changed'], 'unchanged', v['n_unchanged'],
                  'buckets', v['buckets'])


if __name__ == '__main__':
    main()
