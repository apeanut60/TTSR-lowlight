#!/usr/bin/env python
"""V6.0 eval: frozen LLFormer Base vs V6 Ref + Ref dependence. Verdict at 30k."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import metrics as _metrics  # noqa: E402
from model.V6LLFormerRefBridge import (INJECTION_POINT, base_forward_m11,  # noqa: E402
                                       load_frozen_llformer_base, tiled_v6_forward)
from model.V6Model import V6Model  # noqa: E402
from option import parser as option_parser  # noqa: E402
from v3a5_pipeline import load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import STATES  # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head, nanmean, require_ckpt  # noqa: E402
from v3a72_runtime import json_ready  # noqa: E402
from v3b2_runtime import pair_delta_stats_by_state  # noqa: E402
from v5_runtime import feat_energy, flow_stats, r_delta, residual_offset_stats  # noqa: E402
from v6_runtime import (ARM_A0, ARM_A1, ROOT, SRC, require_ckpt_blob_v6,  # noqa: E402
                        safety_plus, verdict_v6)

REF_MODES = ('normal', 'self', 'zero', 'shuffled')


def _ref_for_mode(t, mode, donor_r):
    if mode == 'normal':
        return t['R']
    if mode == 'self':
        return t['Y0']
    if mode == 'zero':
        return torch.zeros_like(t['R'])
    if mode == 'shuffled':
        return donor_r
    raise SystemExit('unknown ref mode %r' % mode)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--full_dep', action='store_true')
    ap.add_argument('--verdict', action='store_true')
    ap.add_argument('--allow_eval_code_drift', action='store_true')
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    if lock.get('official_test_allowed') is not False:
        raise SystemExit('official test forbidden')
    live = git_head()
    if live != lock['repo_commit'] and not a.allow_eval_code_drift:
        raise SystemExit('eval repo HEAD drift: live=%s lock=%s' % (live, lock['repo_commit']))
    live_sha = dict(
        split=file_sha256(lock['split_json']),
        mismatch_dev=file_sha256(lock['mismatch_dev']),
        cache_meta=file_sha256(lock['cache_metadata_path']),
        base=file_sha256(lock['base_ckpt']),
    )
    want = dict(
        split=lock['split_sha256'],
        mismatch_dev=lock['mismatch_dev_sha256'],
        cache_meta=lock['cache_metadata_sha256'],
        base=lock['base_ckpt_sha256'],
    )
    bad = [k for k in want if live_sha[k] != want[k]]
    if bad:
        raise SystemExit('eval live artifact SHA fail: %s' % bad)

    base = load_frozen_llformer_base(lock['base_ckpt'], a.device)

    path = os.path.join(a.root, ARM_A1, 'checkpoints', 'ckpt_%06d.pt' % a.step)
    model = None
    if os.path.isfile(path):
        require_ckpt(path, formal=True)
        blob = torch.load(path, map_location='cpu')
        require_ckpt_blob_v6(
            blob, arm=ARM_A1, step=a.step, init_sha=lock['v6_init_sha'],
            repo_commit=lock['repo_commit'], injection_point=INJECTION_POINT,
            formal=True)
        model = V6Model().to(a.device).eval()
        model.load_state_dict(blob['model'], strict=True)
    if a.verdict and model is None:
        raise SystemExit('--verdict needs V6 checkpoint')

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = lock['reference_variant']
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       lock['split_json'])
    mmap = json.load(open(lock['mismatch_dev']))
    ds = make_dataset(
        ns, splits['dev'],
        os.path.join(a.root, lock['cache_name'], 'refiner_train'), mmap)
    n = len(splits['dev']) if not a.limit else min(a.limit, len(splits['dev']))
    modes = REF_MODES if a.full_dep else ('normal',)

    lpips_net = None
    if a.verdict:
        try:
            import lpips
            lpips_net = lpips.LPIPS(net='alex').to(a.device).eval()
            for p in lpips_net.parameters():
                p.requires_grad_(False)
        except Exception as e:
            print('LPIPS skipped: %s' % e, flush=True)

    rows = []
    print('=== V6.0 eval @%d n=%d V6=%s full_dep=%s ==='
          % (a.step, n, model is not None, a.full_dep), flush=True)
    with torch.no_grad():
        for i in range(n):
            t_next = sample_tensors(ds, (i + 1) % n, 'correct', a.device)
            donor_r = t_next['R']
            for state in STATES:
                t = sample_tensors(ds, i, state, a.device)
                y_base = base_forward_m11(base, t['X'])
                rec = dict(
                    name=t['name'], state=state,
                    psnr_base=float(_metrics(y_base, t['H'])[0]),
                    ssim_base=float(_metrics(y_base, t['H'])[1]),
                    psnr_y0_cache=float(_metrics(t['Y0'], t['H'])[0]),
                )
                rec['psnr_%s' % ARM_A0] = rec['psnr_base']
                rec['ssim_%s' % ARM_A0] = rec['ssim_base']
                if lpips_net is not None:
                    rec['lpips_base'] = float(lpips_net(y_base, t['H']).mean())
                    rec['lpips_%s' % ARM_A0] = rec['lpips_base']
                if model is not None:
                    for mode in modes:
                        r = _ref_for_mode(t, mode, donor_r)
                        if mode == 'normal':
                            y, aux = tiled_v6_forward(
                                base, model, t['X'], t['Y0'], r, collect_aux=True)
                        else:
                            y = tiled_v6_forward(base, model, t['X'], t['Y0'], r)
                            aux = None
                        rec['psnr_%s_%s' % (ARM_A1, mode)] = float(
                            _metrics(y, t['H'])[0])
                        rec['ssim_%s_%s' % (ARM_A1, mode)] = float(
                            _metrics(y, t['H'])[1])
                        if mode == 'normal' and aux is not None:
                            rec.update(flow_stats(aux['match_flow_h4']))
                            rec['sim_max'] = float(aux['sim_max'].mean())
                            rec['margin'] = float(aux['margin'].mean())
                            rec['entropy'] = float(aux['entropy'].mean())
                            rec.update({('off_%s' % k): v for k, v in
                                        residual_offset_stats(
                                            aux['residual_offset']).items()})
                            r2, md2 = r_delta(aux['delta_d2'], aux['d2'])
                            r1, md1 = r_delta(aux['h2_delta_d1'], aux['d1'])
                            rec['r_D2'] = r2
                            rec['r_D1'] = r1
                            rec['dD2_mean'] = md2
                            rec['dD1_mean'] = md1
                            rec['flow_def_l1'] = float(
                                (aux['flow_feat'] - aux['deform_feat']).abs().mean())
                            for tag, ten in (('R2', aux['tex']['h4']),
                                             ('Warp2', aux['flow_feat']),
                                             ('DCN2', aux['deform_feat']),
                                             ('A2', aux['aligned_h4'])):
                                en = feat_energy(ten)
                                rec['hf_%s' % tag] = en['hf']
                                rec['grad_%s' % tag] = en['grad']
                            if lpips_net is not None:
                                rec['lpips_%s' % ARM_A1] = float(
                                    lpips_net(y, t['H']).mean())
                rows.append(rec)
            if (i + 1) % 8 == 0 or i + 1 == n:
                print('  %d/%d' % (i + 1, n), flush=True)

    def by_state(key):
        out = {}
        for s in STATES:
            out[s] = nanmean([r[key] for r in rows if r['state'] == s and key in r])
        out['mean'] = nanmean([r[key] for r in rows if key in r])
        return out

    table = dict(Base=by_state('psnr_base'), Frozen_Base=by_state('psnr_%s' % ARM_A0))
    ssim_table = dict(Base=by_state('ssim_base'),
                      Frozen_Base=by_state('ssim_%s' % ARM_A0))
    if model is not None:
        table[ARM_A1] = by_state('psnr_%s_normal' % ARM_A1)
        ssim_table[ARM_A1] = by_state('ssim_%s_normal' % ARM_A1)
        table['V6_minus_Base'] = {
            s: table[ARM_A1][s] - table['Frozen_Base'][s]
            for s in list(STATES) + ['mean']
        }
        for mode in modes:
            if mode != 'normal':
                table['%s_%s' % (ARM_A1, mode)] = by_state(
                    'psnr_%s_%s' % (ARM_A1, mode))

    lpips_table = {}
    if lpips_net is not None:
        lpips_table = dict(Base=by_state('lpips_base'),
                           Frozen_Base=by_state('lpips_%s' % ARM_A0))
        if model is not None:
            lpips_table[ARM_A1] = by_state('lpips_%s' % ARM_A1)

    diag = {}
    if model is not None and rows and 'disp_mean' in rows[0]:
        for k in ('disp_mean', 'disp_p50', 'disp_p90', 'disp_p95', 'boundary_hit',
                  'sim_max', 'margin', 'entropy', 'off_dP_mean', 'off_dP_p90',
                  'r_D2', 'r_D1', 'flow_def_l1',
                  'hf_R2', 'hf_Warp2', 'hf_DCN2', 'hf_A2'):
            if any(k in r for r in rows):
                diag[k] = by_state(k)

    safety = {}
    for tag, key in ((ARM_A0, 'psnr_%s' % ARM_A0),
                     (ARM_A1, 'psnr_%s_normal' % ARM_A1)):
        if not any(key in r for r in rows):
            continue
        deltas = {s: [] for s in STATES}
        names = {s: [] for s in STATES}
        for r in rows:
            deltas[r['state']].append(r[key] - r['psnr_base'])
            names[r['state']].append('%s|%s' % (r['name'], r['state']))
        safety[tag] = {s: safety_plus(deltas[s], names[s]) for s in STATES}

    out_dir = os.path.join(a.root, 'diagnostics', 'eval_%06d' % a.step)
    os.makedirs(out_dir, exist_ok=True)
    dump_json(os.path.join(out_dir, 'per_image.json'), json_ready(rows))
    dump_json(os.path.join(out_dir, 'summary_psnr.json'), json_ready(table))
    dump_json(os.path.join(out_dir, 'summary_ssim.json'), json_ready(ssim_table))
    dump_json(os.path.join(out_dir, 'safety.json'), json_ready(safety))
    if diag:
        dump_json(os.path.join(out_dir, 'align_stats.json'), json_ready(diag))
    if lpips_table:
        dump_json(os.path.join(out_dir, 'summary_lpips.json'), json_ready(lpips_table))

    if a.verdict and model is not None:
        dep = dict(normal=table[ARM_A1]['mean'], self=float('nan'),
                   zero=float('nan'), shuffled=float('nan'))
        if a.full_dep:
            dep['self'] = table['%s_self' % ARM_A1]['mean']
            dep['zero'] = table['%s_zero' % ARM_A1]['mean']
            dep['shuffled'] = table['%s_shuffled' % ARM_A1]['mean']
        deltas_by = {s: [] for s in STATES}
        for r in rows:
            deltas_by[r['state']].append(
                r['psnr_%s_normal' % ARM_A1] - r['psnr_%s' % ARM_A0])
        align_ok = bool(diag and float(diag.get('disp_mean', {}).get('mean', 0)) > 0.5)
        verdict = verdict_v6(
            table['Frozen_Base'], table[ARM_A1],
            safety[ARM_A0], safety[ARM_A1], dep,
            lpips_base=lpips_table.get('Frozen_Base') if lpips_table else None,
            lpips_v6=lpips_table.get(ARM_A1) if lpips_table else None,
            pair_stats=pair_delta_stats_by_state(deltas_by),
            align_ok=align_ok)
        verdict['step'] = a.step
        verdict['eval_repo_commit'] = live
        dump_json(os.path.join(out_dir, 'verdict.json'), json_ready(verdict))
        print('VERDICT', verdict['label'],
              'ΔV6-Base=%.3f' % verdict['delta_v6_base'],
              'next=', verdict['next_step'], flush=True)
        for s in STATES:
            print('  %s  Base=%.3f V6=%.3f d=%.3f' % (
                s, table['Frozen_Base'][s], table[ARM_A1][s],
                table['V6_minus_Base'][s]), flush=True)
        if a.full_dep:
            print('  ref normal/self/zero/shuf = %.3f / %.3f / %.3f / %.3f' % (
                dep['normal'], dep['self'], dep['zero'], dep['shuffled']),
                flush=True)
    else:
        print('Frozen_Base mean=%.3f' % table['Frozen_Base']['mean'], flush=True)
        if model is not None:
            print('ARM', ARM_A1, 'mean=%.3f' % table[ARM_A1]['mean'], flush=True)
    print('wrote', out_dir, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
