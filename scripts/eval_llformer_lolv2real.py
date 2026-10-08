#!/usr/bin/env python
"""Evaluate a LLFormerBridge ckpt on locked dev64 or Official Test.

Official Test is blocked unless --allow_official_test and E* is locked.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from llformer_runtime import (  # noqa: E402
    ROOT, DATA_DIR, LOLv2PairDataset, decide_go, lock_architecture_fields,
    metrics_01, read_split_txt, RETINEX_TEST_PSNR)
from model.LLFormerBridge import (BRIDGE_VERSION, LLFormerBridge, infer_rgb,  # noqa: E402
                                  load_into_llformer)
from v3a6_runtime import dump_json, file_sha256, git_head  # noqa: E402
from v3a72_runtime import json_ready  # noqa: E402


def load_trained(path, device):
    blob = torch.load(path, map_location='cpu')
    net = LLFormerBridge().to(device).eval()
    sd = blob['model'] if isinstance(blob, dict) and 'model' in blob else blob
    # trained saves are bare state_dict under 'model'
    if any(k.startswith('module.') for k in sd):
        from model.LLFormerBridge import strip_module_prefix
        sd = strip_module_prefix(sd)
    net.load_state_dict(sd, strict=True)
    for p in net.parameters():
        p.requires_grad_(False)
    return net, blob


def eval_names(net, names, data_dir, split, device, with_lpips=False):
    ds = LOLv2PairDataset(names, data_dir=data_dir, split=split, train=False)
    lpips_net = None
    if with_lpips:
        import lpips
        lpips_net = lpips.LPIPS(net='alex').to(device).eval()
        for p in lpips_net.parameters():
            p.requires_grad_(False)
    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            t = ds[i]
            y = infer_rgb(net, t['low'][None].to(device), clamp=True)
            h = t['high'][None].to(device)
            psnr, ssim, _ = metrics_01(y, h)
            rec = dict(name=t['name'], psnr=psnr, ssim=ssim)
            if lpips_net is not None:
                rec['lpips'] = float(lpips_net(y * 2 - 1, h * 2 - 1).mean())
            rows.append(rec)
            if (i + 1) % 10 == 0 or i + 1 == len(ds):
                print('  %d/%d' % (i + 1, len(ds)), flush=True)
    out = dict(
        n=len(rows),
        psnr=float(np.mean([r['psnr'] for r in rows])),
        ssim=float(np.mean([r['ssim'] for r in rows])),
        per_image=rows,
    )
    if with_lpips:
        out['lpips'] = float(np.mean([r['lpips'] for r in rows]))
    return out


def write_canonical(root, e_star_blob, ckpt_path, test_summary, lock):
    ep = int(e_star_blob['E_star'])
    out_path = os.path.join(root, 'canonical',
                            'llformer_lolv2real_base_E%d.pth' % ep)
    trained = torch.load(ckpt_path, map_location='cpu')
    meta = lock_architecture_fields()
    meta.update(
        stage='LLFORMER_LOLV2REAL_BASE_CANONICAL',
        epoch=ep,
        repo_commit=git_head(),
        train_split_sha=lock['train625_sha'],
        dev_split_sha=lock['dev64_sha'],
        init_checkpoint_sha=lock['official_lol_checkpoint_sha256'],
        source_checkpoint=ckpt_path,
        source_checkpoint_sha=file_sha256(ckpt_path),
        bridge_version=BRIDGE_VERSION,
        dev_psnr=float(e_star_blob['dev_psnr']),
        dev_ssim=float(e_star_blob['dev_ssim']),
        test_psnr=float(test_summary['psnr']),
        test_ssim=float(test_summary['ssim']),
        test_lpips=float(test_summary.get('lpips', -1)),
        decision=test_summary['decision'],
        retinex_anchor=RETINEX_TEST_PSNR,
    )
    torch.save(dict(model=trained['model'], meta=meta), out_path)
    dump_json(os.path.join(root, 'canonical', 'canonical_meta.json'),
              json_ready(meta))
    print('wrote canonical', out_path, flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--data_dir', default=DATA_DIR)
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--split', default='dev', choices=['dev', 'official_test'])
    ap.add_argument('--with_lpips', action='store_true')
    ap.add_argument('--allow_official_test', action='store_true')
    ap.add_argument('--write_canonical', action='store_true')
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    e_path = os.path.join(a.root, 'diagnostics', 'E_star.json')
    ckpt = a.ckpt
    if not ckpt:
        if os.path.isfile(e_path):
            ckpt = json.load(open(e_path))['checkpoint']
        else:
            ckpt = os.path.join(a.root, 'checkpoints', 'model_bestPSNR.pth')

    if a.split == 'official_test':
        if not a.allow_official_test:
            raise SystemExit('Official Test blocked; pass --allow_official_test '
                             'after E* locked')
        if not os.path.isfile(e_path):
            raise SystemExit('missing E_star.json — select E* first')
        test_dir = os.path.join(a.data_dir, 'Test', 'Low')
        names = sorted(f for f in os.listdir(test_dir) if f.endswith('.png'))
        if len(names) != 100:
            raise SystemExit('expected 100 test images, got %d' % len(names))
        split_folder = 'Test'
        out_dir = os.path.join(a.root, 'diagnostics', 'official_test')
    else:
        names = read_split_txt(lock['dev64_path'])
        split_folder = 'Train'
        out_dir = os.path.join(a.root, 'diagnostics', 'eval_dev')

    print('=== eval split=%s n=%d ckpt=%s ===' % (a.split, len(names), ckpt),
          flush=True)
    net, blob = load_trained(ckpt, a.device)
    ev = eval_names(net, names, a.data_dir, split_folder, a.device,
                    with_lpips=a.with_lpips or a.split == 'official_test')
    summary = dict(
        split=a.split, ckpt=ckpt, ckpt_sha=file_sha256(ckpt),
        repo_commit=git_head(), **{k: ev[k] for k in ('n', 'psnr', 'ssim')
                                   if k in ev},
    )
    if 'lpips' in ev:
        summary['lpips'] = ev['lpips']
    if a.split == 'official_test':
        summary['retinex_anchor_psnr'] = RETINEX_TEST_PSNR
        summary['delta_vs_retinex'] = summary['psnr'] - RETINEX_TEST_PSNR
        summary['decision'] = decide_go(summary['psnr'])

    os.makedirs(out_dir, exist_ok=True)
    dump_json(os.path.join(out_dir, 'summary.json'), json_ready(summary))
    dump_json(os.path.join(out_dir, 'per_image.json'), json_ready(ev['per_image']))
    print('══ %s ══' % a.split)
    print('  PSNR %.4f  SSIM %.4f%s'
          % (summary['psnr'], summary['ssim'],
             ('  LPIPS %.4f' % summary['lpips']) if 'lpips' in summary else ''))
    if a.split == 'official_test':
        print('  vs Retinex %.3f → %+.4f  decision=%s'
              % (RETINEX_TEST_PSNR, summary['delta_vs_retinex'],
                 summary['decision']))
        if a.write_canonical:
            write_canonical(a.root, json.load(open(e_path)), ckpt, summary, lock)
    print('wrote', out_dir, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
