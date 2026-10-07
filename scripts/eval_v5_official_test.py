#!/usr/bin/env python
"""One-shot LOL-v2-real Official Test scores: Base / B0 / V5@30k.

PSNR (RGB), SSIM (Y), LPIPS (alex). Correct Nano-v2 refs only.
User-requested; does not change formal Case verdict (still Case E on dev).
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

from dataset.lolv2real_v3a import TrainSet, pairs_from_manifest          # noqa: E402
from local_refine_runtime import EVAL_QUERY_CHUNK, metrics as _metrics  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion                  # noqa: E402
from model.V5Model import V5Model                                       # noqa: E402
from model.V5RetinexBridge import (INJECTION_POINT, load_frozen_retinex_mainnet,  # noqa: E402
                                   tiled_v5_forward)
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_proposal                                 # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head               # noqa: E402
from v3a72_runtime import json_ready                                    # noqa: E402
from v3b_runtime import match_features, proposal_core                   # noqa: E402
from v5_runtime import ARM_A0, ARM_A1, require_ckpt_blob_v5             # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
ROOT = '/root/data/experiments/v5_aligned_ref'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--step', type=int, default=30000)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--allow_eval_code_drift', action='store_true')
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    live = git_head()
    if live != lock['repo_commit'] and not a.allow_eval_code_drift:
        raise SystemExit('eval repo HEAD drift: live=%s lock=%s' % (live, lock['repo_commit']))
    if lock.get('reference_variant') != 'nanobanana_ref_v2':
        raise SystemExit('expected nanobanana_ref_v2, got %r' % lock.get('reference_variant'))

    test_manifest = os.path.join(a.src_root, 'manifests', 'test.csv')
    y0_cache = os.path.join(a.src_root, lock.get('cache_name', 'cache_y0_lolbase'), 'test')
    if not os.path.isdir(y0_cache):
        raise SystemExit('missing Y0 test cache: %s' % y0_cache)

    mainnet = load_frozen_retinex_mainnet(
        lock['base_ckpt'], lock['base_run_dir'], a.device)

    wrapper = load_proposal(lock['proposal_ckpt'], a.device)
    core = proposal_core(wrapper)
    core.match.chunk = EVAL_QUERY_CHUNK
    for p in wrapper.parameters():
        p.requires_grad_(False)
    wrapper.eval()
    b0_blob = torch.load(lock['b0_head_ckpt'], map_location='cpu')
    b0_head = V3B0ResidualFusion(in_ch=96).to(a.device).eval()
    b0_head.load_state_dict(b0_blob['model'], strict=True)
    for p in b0_head.parameters():
        p.requires_grad_(False)

    path = os.path.join(a.root, ARM_A1, 'checkpoints', 'ckpt_%06d.pt' % a.step)
    blob = torch.load(path, map_location='cpu')
    require_ckpt_blob_v5(
        blob, arm=ARM_A1, step=a.step, init_sha=lock['v5_init_sha'],
        repo_commit=lock['repo_commit'], injection_point=INJECTION_POINT,
        formal=True)
    v5 = V5Model().to(a.device).eval()
    v5.load_state_dict(blob['model'], strict=True)

    import lpips
    lpips_net = lpips.LPIPS(net='alex').to(a.device).eval()
    for p in lpips_net.parameters():
        p.requires_grad_(False)

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = lock['reference_variant']
    pairs = pairs_from_manifest(test_manifest)
    if a.limit:
        pairs = pairs[:a.limit]
    ds = TrainSet(ns, crop_size=0, pairs=pairs, y0_cache=y0_cache,
                  mismatch_map={}, split='Test')

    rows = []
    print('=== Official Test n=%d step=%d ref=%s ==='
          % (len(pairs), a.step, lock['reference_variant']), flush=True)
    with torch.no_grad():
        for i in range(len(pairs)):
            name, lr, hr, ref, y0, _mis = ds._load(i)
            if y0 is None:
                raise SystemExit('Y0 miss %s' % name)
            x = lr[None].to(a.device)
            y0t = y0[None].to(a.device)
            ht = hr[None].to(a.device)
            rt = ref[None].to(a.device)
            rec = dict(name=name)
            y_base = y0t
            f0, tt = match_features(wrapper, y0t, rt)
            y_b0 = y0t + b0_head(f0, tt, y0t.shape[-2:], E=None)
            y_v5 = tiled_v5_forward(mainnet, v5, x, y0t, rt)
            for tag, y in (('Base', y_base), (ARM_A0, y_b0), (ARM_A1, y_v5)):
                psnr, ssim, _mse = _metrics(y, ht)
                rec['psnr_%s' % tag] = float(psnr)
                rec['ssim_%s' % tag] = float(ssim)
                rec['lpips_%s' % tag] = float(lpips_net(y, ht).mean())
            rows.append(rec)
            if (i + 1) % 10 == 0 or i + 1 == len(pairs):
                print('  %d/%d' % (i + 1, len(pairs)), flush=True)

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    summary = {
        'split': 'official_Test',
        'n': len(rows),
        'step': a.step,
        'reference_variant': lock['reference_variant'],
        'repo_commit': live,
        'Base': dict(psnr=mean('psnr_Base'), ssim=mean('ssim_Base'),
                     lpips=mean('lpips_Base')),
        'Frozen_B0': dict(psnr=mean('psnr_%s' % ARM_A0),
                          ssim=mean('ssim_%s' % ARM_A0),
                          lpips=mean('lpips_%s' % ARM_A0)),
        'V5': dict(psnr=mean('psnr_%s' % ARM_A1),
                   ssim=mean('ssim_%s' % ARM_A1),
                   lpips=mean('lpips_%s' % ARM_A1)),
        'V5_minus_B0': dict(
            psnr=mean('psnr_%s' % ARM_A1) - mean('psnr_%s' % ARM_A0),
            ssim=mean('ssim_%s' % ARM_A1) - mean('ssim_%s' % ARM_A0),
            lpips=mean('lpips_%s' % ARM_A1) - mean('lpips_%s' % ARM_A0),
        ),
        'V5_minus_Base': dict(
            psnr=mean('psnr_%s' % ARM_A1) - mean('psnr_Base'),
            ssim=mean('ssim_%s' % ARM_A1) - mean('ssim_Base'),
            lpips=mean('lpips_%s' % ARM_A1) - mean('lpips_Base'),
        ),
        'artifact_sha': dict(
            test_manifest=file_sha256(test_manifest),
            y0_meta=file_sha256(os.path.join(y0_cache, 'metadata.json')),
            base=file_sha256(lock['base_ckpt']),
            b0=file_sha256(lock['b0_head_ckpt']),
            v5_ckpt=file_sha256(path),
        ),
        'note': 'One-shot Official Test; formal V5 Case remains Case E on dev64.',
    }
    out_dir = os.path.join(a.root, 'diagnostics', 'official_test_%06d' % a.step)
    dump_json(os.path.join(out_dir, 'summary.json'), json_ready(summary))
    dump_json(os.path.join(out_dir, 'per_image.json'), json_ready(rows))

    print()
    print('══ LOL-v2-real Official Test (n=%d, V5@%d, Nano-v2) ══' % (len(rows), a.step))
    print('  %-12s %10s %10s %10s' % ('method', 'PSNR', 'SSIM', 'LPIPS'))
    for tag, key in (('Base', 'Base'), ('Frozen_B0', 'Frozen_B0'), ('V5', 'V5')):
        s = summary[key]
        print('  %-12s %10.4f %10.4f %10.4f'
              % (tag, s['psnr'], s['ssim'], s['lpips']))
    print('  V5−B0       %+10.4f %+10.4f %+10.4f'
          % (summary['V5_minus_B0']['psnr'],
             summary['V5_minus_B0']['ssim'],
             summary['V5_minus_B0']['lpips']))
    print('wrote', out_dir, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
