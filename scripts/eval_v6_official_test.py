#!/usr/bin/env python
"""Official Test for V6 after Case (does not flip Case)."""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from dataset.lolv2real_v3a import TrainSet, pairs_from_manifest  # noqa: E402
from local_refine_runtime import metrics as _metrics  # noqa: E402
from model.V6LLFormerRefBridge import (INJECTION_POINT, base_forward_m11,  # noqa: E402
                                       load_frozen_llformer_base, tiled_v6_forward)
from model.V6Model import V6Model  # noqa: E402
from option import parser as option_parser  # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head  # noqa: E402
from v3a72_runtime import json_ready  # noqa: E402
from v6_runtime import ARM_A0, ARM_A1, ROOT, SRC, require_ckpt_blob_v6  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--step', type=int, default=30000)
    ap.add_argument('--allow_official_test', action='store_true')
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    if not a.allow_official_test:
        raise SystemExit('pass --allow_official_test after Case locked')

    base = load_frozen_llformer_base(lock['base_ckpt'], a.device)
    path = os.path.join(a.root, ARM_A1, 'checkpoints', 'ckpt_%06d.pt' % a.step)
    blob = torch.load(path, map_location='cpu')
    require_ckpt_blob_v6(
        blob, arm=ARM_A1, step=a.step, init_sha=lock['v6_init_sha'],
        repo_commit=lock['repo_commit'], injection_point=INJECTION_POINT,
        formal=True)
    model = V6Model().to(a.device).eval()
    model.load_state_dict(blob['model'], strict=True)

    import lpips
    lpips_net = lpips.LPIPS(net='alex').to(a.device).eval()
    for p in lpips_net.parameters():
        p.requires_grad_(False)

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = lock['reference_variant']
    test_manifest = os.path.join(a.src_root, 'manifests', 'test.csv')
    pairs = pairs_from_manifest(test_manifest)
    y0_cache = os.path.join(a.root, lock['cache_name'], 'test')
    ds = TrainSet(ns, crop_size=0, pairs=pairs, y0_cache=y0_cache,
                  mismatch_map={}, split='Test')

    rows = []
    print('=== Official Test V6 n=%d ===' % len(pairs), flush=True)
    with torch.no_grad():
        for i in range(len(pairs)):
            name, lr, hr, ref, y0, _ = ds._load(i)
            x = lr[None].to(a.device)
            ht = hr[None].to(a.device)
            y0t = y0[None].to(a.device)
            rt = ref[None].to(a.device)
            y_base = base_forward_m11(base, x)
            y_v6 = tiled_v6_forward(base, model, x, y0t, rt)
            rec = dict(name=name)
            for tag, y in ((ARM_A0, y_base), (ARM_A1, y_v6), ('Y0_cache', y0t)):
                psnr, ssim, _ = _metrics(y, ht)
                rec['psnr_%s' % tag] = float(psnr)
                rec['ssim_%s' % tag] = float(ssim)
                rec['lpips_%s' % tag] = float(lpips_net(y, ht).mean())
            rows.append(rec)
            if (i + 1) % 10 == 0:
                print('  %d/%d' % (i + 1, len(pairs)), flush=True)

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    summary = dict(
        split='official_Test', n=len(rows), step=a.step,
        repo_commit=git_head(), ckpt=path, ckpt_sha=file_sha256(path),
    )
    for tag in (ARM_A0, ARM_A1, 'Y0_cache'):
        summary[tag] = dict(
            psnr=mean('psnr_%s' % tag),
            ssim=mean('ssim_%s' % tag),
            lpips=mean('lpips_%s' % tag),
        )
    summary['V6_minus_Base'] = dict(
        psnr=summary[ARM_A1]['psnr'] - summary[ARM_A0]['psnr'],
        ssim=summary[ARM_A1]['ssim'] - summary[ARM_A0]['ssim'],
        lpips=summary[ARM_A1]['lpips'] - summary[ARM_A0]['lpips'],
    )
    out = os.path.join(a.root, 'diagnostics', 'official_test_%06d' % a.step)
    os.makedirs(out, exist_ok=True)
    dump_json(os.path.join(out, 'summary.json'), json_ready(summary))
    dump_json(os.path.join(out, 'per_image.json'), json_ready(rows))
    print('══ Official Test ══')
    for tag in (ARM_A0, ARM_A1):
        s = summary[tag]
        print('  %-20s PSNR %.4f SSIM %.4f LPIPS %.4f'
              % (tag, s['psnr'], s['ssim'], s['lpips']))
    d = summary['V6_minus_Base']
    print('  V6−Base %+ .4f / %+ .4f / %+ .4f' % (d['psnr'], d['ssim'], d['lpips']))
    print('wrote', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
