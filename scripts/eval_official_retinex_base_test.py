#!/usr/bin/env python
"""Compare our Base vs official RetinexFormer LOL_v2_real.pth on Official Test.

Does NOT retrain B0/V5. Reports PSNR/SSIM/LPIPS for:
  Base_ours / Base_official
  B0/V5 on our-Y0 (training Base) and on official live Y0 + official MainNet
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
from model.RetinexRefMainNet import RetinexRefMainNet                   # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion                  # noqa: E402
from model.V5Model import V5Model                                       # noqa: E402
from model.V5RetinexBridge import (INJECTION_POINT, load_frozen_retinex_mainnet,  # noqa: E402
                                   tiled_bridge_decode, tiled_v5_forward)
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_proposal                                 # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head               # noqa: E402
from v3a72_runtime import json_ready                                    # noqa: E402
from v3b_runtime import match_features, proposal_core                   # noqa: E402
from v5_runtime import ARM_A0, ARM_A1, require_ckpt_blob_v5             # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
ROOT = '/root/data/experiments/v5_aligned_ref'
OFFICIAL = ('/root/projects/TTSR-lowlight/pretrain_model/retinexformer/'
            'LOL_v2_real.pth')


def load_official_retinex_mainnet(path, device='cuda'):
    """Official BasicSR-style {'params': body.0.*} → RetinexRefMainNet."""
    blob = torch.load(path, map_location='cpu')
    params = blob['params'] if isinstance(blob, dict) and 'params' in blob else blob
    mapped = {}
    for k, v in params.items():
        if k.startswith('body.0.'):
            mapped[k[len('body.0.'):]] = v
        else:
            mapped[k] = v
    net = RetinexRefMainNet(
        n_feat=40, num_blocks=(1, 2, 2), level=2, use_global_illum=False)
    msd = net.state_dict()
    loaded = 0
    for k, v in mapped.items():
        if k in msd and tuple(msd[k].shape) == tuple(v.shape):
            msd[k] = v
            loaded += 1
        else:
            raise SystemExit('official key mismatch: %s' % k)
    # estimator+denoiser must be fully covered (122 tensors)
    need = [k for k in msd if k.startswith(('estimator.', 'denoiser.'))
            and 'ref_adapter' not in k]
    miss = [k for k in need if k not in mapped]
    if miss:
        raise SystemExit('official incomplete: %s' % miss[:6])
    if loaded != 122:
        raise SystemExit('expected 122 official tensors, got %d' % loaded)
    net.load_state_dict(msd, strict=True)
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    print('[official] loaded %d tensors from %s' % (loaded, path), flush=True)
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--official_ckpt', default=OFFICIAL)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--step', type=int, default=30000)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--with_b0_v5', action='store_true',
                    help='report B0/V5 on our-Y0 and on official Base')
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    test_manifest = os.path.join(a.src_root, 'manifests', 'test.csv')
    y0_cache = os.path.join(a.src_root, lock.get('cache_name', 'cache_y0_lolbase'),
                            'test')

    our_main = load_frozen_retinex_mainnet(
        lock['base_ckpt'], lock['base_run_dir'], a.device)
    off_main = load_official_retinex_mainnet(a.official_ckpt, a.device)

    wrapper = b0_head = v5 = lpips_net = None
    if a.with_b0_v5:
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
        path = os.path.join(a.root, ARM_A1, 'checkpoints',
                            'ckpt_%06d.pt' % a.step)
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
    print('=== Official Test Base swap n=%d ===' % len(pairs), flush=True)
    with torch.no_grad():
        for i in range(len(pairs)):
            name, lr, hr, ref, y0, _ = ds._load(i)
            x = lr[None].to(a.device)
            ht = hr[None].to(a.device)
            y0t = y0[None].to(a.device)
            rt = ref[None].to(a.device)
            # live our Base (bridge, no ref) vs cache sanity
            y_ours_live = tiled_bridge_decode(our_main, x, delta_fn=None)
            y_off = tiled_bridge_decode(off_main, x, delta_fn=None)
            rec = dict(name=name,
                       d_cache_live=float((y_ours_live - y0t).abs().max()))
            for tag, y in (('Base_cache', y0t),
                           ('Base_ours_live', y_ours_live),
                           ('Base_official', y_off)):
                psnr, ssim, _ = _metrics(y, ht)
                rec['psnr_%s' % tag] = float(psnr)
                rec['ssim_%s' % tag] = float(ssim)
                rec['lpips_%s' % tag] = float(lpips_net(y, ht).mean())
            if a.with_b0_v5:
                # our Base Y0 (training-time cache)
                f0, tt = match_features(wrapper, y0t, rt)
                y_b0 = y0t + b0_head(f0, tt, y0t.shape[-2:], E=None)
                y_v5 = tiled_v5_forward(our_main, v5, x, y0t, rt)
                # official Base: live Y0 + official MainNet in V5 bridge
                f0o, tto = match_features(wrapper, y_off, rt)
                y_b0_off = y_off + b0_head(f0o, tto, y_off.shape[-2:], E=None)
                y_v5_off = tiled_v5_forward(off_main, v5, x, y_off, rt)
                for tag, y in (
                        (ARM_A0, y_b0), (ARM_A1, y_v5),
                        ('%s_official_base' % ARM_A0, y_b0_off),
                        ('%s_official_base' % ARM_A1, y_v5_off),
                ):
                    psnr, ssim, _ = _metrics(y, ht)
                    rec['psnr_%s' % tag] = float(psnr)
                    rec['ssim_%s' % tag] = float(ssim)
                    rec['lpips_%s' % tag] = float(lpips_net(y, ht).mean())
            rows.append(rec)
            if (i + 1) % 10 == 0 or i + 1 == len(pairs):
                print('  %d/%d' % (i + 1, len(pairs)), flush=True)

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    tags = ['Base_cache', 'Base_ours_live', 'Base_official']
    if a.with_b0_v5:
        tags += [
            ARM_A0, ARM_A1,
            '%s_official_base' % ARM_A0,
            '%s_official_base' % ARM_A1,
        ]
    summary = {
        'split': 'official_Test',
        'n': len(rows),
        'official_ckpt': a.official_ckpt,
        'official_sha256': file_sha256(a.official_ckpt),
        'our_base_ckpt': lock['base_ckpt'],
        'repo_commit': git_head(),
        'cache_vs_live_maxabs_mean': mean('d_cache_live'),
        'note': ('*_official_base = live official Y0; B0 residual / V5 refine '
                 'heads unchanged (trained on our Base). No retrain.'),
    }
    for tag in tags:
        summary[tag] = dict(
            psnr=mean('psnr_%s' % tag),
            ssim=mean('ssim_%s' % tag),
            lpips=mean('lpips_%s' % tag),
        )
    summary['official_minus_ours_live'] = dict(
        psnr=summary['Base_official']['psnr'] - summary['Base_ours_live']['psnr'],
        ssim=summary['Base_official']['ssim'] - summary['Base_ours_live']['ssim'],
        lpips=summary['Base_official']['lpips'] - summary['Base_ours_live']['lpips'],
    )
    if a.with_b0_v5:
        for arm in (ARM_A0, ARM_A1):
            off = '%s_official_base' % arm
            summary['%s_off_minus_our' % arm] = dict(
                psnr=summary[off]['psnr'] - summary[arm]['psnr'],
                ssim=summary[off]['ssim'] - summary[arm]['ssim'],
                lpips=summary[off]['lpips'] - summary[arm]['lpips'],
            )

    out_dir = os.path.join(a.root, 'diagnostics', 'official_base_test')
    dump_json(os.path.join(out_dir, 'summary.json'), json_ready(summary))
    dump_json(os.path.join(out_dir, 'per_image.json'), json_ready(rows))

    print()
    print('══ Official Test: our Base vs official RetinexFormer ══')
    print('  %-28s %10s %10s %10s' % ('method', 'PSNR', 'SSIM', 'LPIPS'))
    for tag in tags:
        s = summary[tag]
        print('  %-28s %10.4f %10.4f %10.4f'
              % (tag, s['psnr'], s['ssim'], s['lpips']))
    d = summary['official_minus_ours_live']
    print('  %-28s %+10.4f %+10.4f %+10.4f'
          % ('official−ours Base', d['psnr'], d['ssim'], d['lpips']))
    if a.with_b0_v5:
        for arm in (ARM_A0, ARM_A1):
            d = summary['%s_off_minus_our' % arm]
            print('  %-28s %+10.4f %+10.4f %+10.4f'
                  % ('%s off−our' % arm, d['psnr'], d['ssim'], d['lpips']))
    print('  cache vs live max|Δ| mean = %.3e' % summary['cache_vs_live_maxabs_mean'])
    print('wrote', out_dir, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
