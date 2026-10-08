#!/usr/bin/env python
"""Stage A0: official LOL Bridge zero-shot on LOLv2-Real dev64."""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from llformer_runtime import (ROOT, DATA_DIR, LOLv2PairDataset, metrics_01,  # noqa: E402
                              read_split_txt)
from model.LLFormerBridge import (OFFICIAL_LOL_CKPT, infer_rgb,  # noqa: E402
                                  load_llformer_bridge)
from v3a6_runtime import dump_json  # noqa: E402
from v3a72_runtime import json_ready  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--data_dir', default=DATA_DIR)
    ap.add_argument('--ckpt', default=OFFICIAL_LOL_CKPT)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--split', default='dev', choices=['dev', 'train_probe'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--with_lpips', action='store_true')
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    names = read_split_txt(lock['dev64_path'] if a.split == 'dev'
                           else lock['train625_path'])
    if a.limit:
        names = names[:a.limit]
    ds = LOLv2PairDataset(names, data_dir=a.data_dir, split='Train', train=False)
    net, _ = load_llformer_bridge(a.ckpt, a.device, train=False)

    lpips_net = None
    if a.with_lpips:
        import lpips
        lpips_net = lpips.LPIPS(net='alex').to(a.device).eval()
        for p in lpips_net.parameters():
            p.requires_grad_(False)

    rows = []
    print('=== A0 zero-shot n=%d ===' % len(names), flush=True)
    with torch.no_grad():
        for i in range(len(ds)):
            t = ds[i]
            x = t['low'][None].to(a.device)
            h = t['high'][None].to(a.device)
            y = infer_rgb(net, x, clamp=True)
            psnr, ssim, _ = metrics_01(y, h)
            rec = dict(name=t['name'], psnr=psnr, ssim=ssim)
            if lpips_net is not None:
                # LPIPS expects [-1,1]
                rec['lpips'] = float(lpips_net(y * 2 - 1, h * 2 - 1).mean())
            rows.append(rec)
            if (i + 1) % 8 == 0 or i + 1 == len(ds):
                print('  %d/%d  last_psnr=%.3f' % (i + 1, len(ds), psnr), flush=True)

    summary = dict(
        stage='A0_zero_shot',
        split=a.split,
        n=len(rows),
        ckpt=a.ckpt,
        psnr=float(np.mean([r['psnr'] for r in rows])),
        ssim=float(np.mean([r['ssim'] for r in rows])),
    )
    if a.with_lpips:
        summary['lpips'] = float(np.mean([r['lpips'] for r in rows]))

    out = os.path.join(a.root, 'diagnostics', 'a0_zero_shot')
    os.makedirs(out, exist_ok=True)
    dump_json(os.path.join(out, 'summary.json'), json_ready(summary))
    dump_json(os.path.join(out, 'per_image.json'), json_ready(rows))
    print('══ A0 zero-shot ══')
    print('  PSNR %.4f  SSIM %.4f%s'
          % (summary['psnr'], summary['ssim'],
             ('  LPIPS %.4f' % summary['lpips']) if 'lpips' in summary else ''))
    print('wrote', out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
