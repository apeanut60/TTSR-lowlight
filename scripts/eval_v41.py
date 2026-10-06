#!/usr/bin/env python
"""V4.1a eval: frozen B0 vs grounded-FR A1 + grounding ablations."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import EVAL_QUERY_CHUNK, metrics as _metrics  # noqa: E402
from model.V3BResidualFusion import V3B0ResidualFusion                  # noqa: E402
from model.V4RefGrounder import (FORWARD_DIR, GROUND_DIR, RefGrounder,  # noqa: E402
                                 frozen_pair_and_tlow, grounded_forward)
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import load_proposal, load_rows, make_dataset, sample_tensors  # noqa: E402
from v3a5_runtime import STATES, state_dict_sha                         # noqa: E402
from v3a6_runtime import dump_json, file_sha256, git_head, nanmean, require_ckpt  # noqa: E402
from v3a72_runtime import json_ready                                    # noqa: E402
from v3b_runtime import proposal_core                                   # noqa: E402
from v3b2_runtime import pair_delta_stats_by_state                      # noqa: E402
from v41_runtime import (ARM_A0, ARM_A1, dist_to_f0, feat_delta_stats,  # noqa: E402
                         require_ckpt_blob_v41, safety_plus, t_change_stats,
                         verdict_v41)

SRC = '/root/data/experiments/v3a1_lolv2real'
ROOT = '/root/data/experiments/v41_ground_then_transfer'
TLOW_MODES = ('normal', 'self', 'zero', 'shuffled')


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
        proposal=file_sha256(lock['proposal_ckpt']),
        split=file_sha256(lock['split_json']),
        mismatch_dev=file_sha256(lock['mismatch_dev']),
        cache_meta=file_sha256(lock['cache_metadata_path']),
        b0_ckpt=file_sha256(lock['b0_head_ckpt']),
    )
    want = dict(
        proposal=lock['proposal_sha256'],
        split=lock['split_sha256'],
        mismatch_dev=lock['mismatch_dev_sha256'],
        cache_meta=lock['cache_metadata_sha256'],
        b0_ckpt=lock['b0_head_ckpt_sha256'],
    )
    bad = [k for k in want if live_sha[k] != want[k]]
    if bad:
        raise SystemExit('eval live artifact SHA fail: %s' % bad)

    wrapper = load_proposal(lock['proposal_ckpt'], a.device)
    core = proposal_core(wrapper)
    core.match.chunk = EVAL_QUERY_CHUNK
    for p in wrapper.parameters():
        p.requires_grad_(False)
    wrapper.eval()

    b0_blob = torch.load(lock['b0_head_ckpt'], map_location='cpu')
    b0_head = V3B0ResidualFusion(in_ch=96).to(a.device).eval()
    b0_head.load_state_dict(b0_blob['model'], strict=True)
    if state_dict_sha(b0_blob['model']) != lock['b0_head_state_sha']:
        raise SystemExit('B0 head state sha drift')
    for p in b0_head.parameters():
        p.requires_grad_(False)

    grounder = None
    path = os.path.join(a.root, ARM_A1, 'checkpoints', 'ckpt_%06d.pt' % a.step)
    if os.path.isfile(path):
        require_ckpt(path, formal=True)
        blob = torch.load(path, map_location='cpu')
        require_ckpt_blob_v41(
            blob, arm=ARM_A1, step=a.step, init_sha=lock['grounder_init_sha'],
            proposal_sha=lock['proposal_sha256'], repo_commit=lock['repo_commit'],
            b0_head_state_sha=lock['b0_head_state_sha'],
            ground_direction=GROUND_DIR, forward_direction=FORWARD_DIR, formal=True)
        grounder = RefGrounder().to(a.device).eval()
        grounder.load_state_dict(blob['model'], strict=True)
    if a.verdict and grounder is None:
        raise SystemExit('--verdict needs A1 checkpoint')

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = lock['reference_variant']
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       lock['split_json'])
    mmap = json.load(open(lock['mismatch_dev']))
    ds = make_dataset(
        ns, splits['dev'],
        os.path.join(a.src_root, lock.get('cache_name', 'cache_y0_lolbase'),
                     'refiner_train'), mmap)
    n = len(splits['dev']) if not a.limit else min(a.limit, len(splits['dev']))
    modes = TLOW_MODES if a.full_dep else ('normal',)

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
    print('=== V4.1a eval @%d n=%d A1=%s full_dep=%s ==='
          % (a.step, n, grounder is not None, a.full_dep), flush=True)
    with torch.no_grad():
        for i in range(n):
            t_next = sample_tensors(ds, (i + 1) % n, 'correct', a.device)
            y0_donor = t_next['Y0']
            for state in STATES:
                t = sample_tensors(ds, i, state, a.device)
                rec = dict(name=t['name'], state=state,
                           psnr_base=float(_metrics(t['Y0'], t['H'])[0]),
                           ssim_base=float(_metrics(t['Y0'], t['H'])[1]),
                           psnr_raw_ref=float(_metrics(t['R'], t['H'])[0]))
                f0, fr, t_low, t_raw = frozen_pair_and_tlow(wrapper, t['Y0'], t['R'])
                y_b0 = t['Y0'] + b0_head(f0, t_raw, t['Y0'].shape[-2:], E=None)
                rec['psnr_%s' % ARM_A0] = float(_metrics(y_b0, t['H'])[0])
                rec['ssim_%s' % ARM_A0] = float(_metrics(y_b0, t['H'])[1])
                if lpips_net is not None:
                    rec['lpips_base'] = float(lpips_net(t['Y0'], t['H']).mean())
                    rec['lpips_%s' % ARM_A0] = float(lpips_net(y_b0, t['H']).mean())
                if grounder is not None:
                    for mode in modes:
                        f0m, frm, tlm, trm = frozen_pair_and_tlow(
                            wrapper, t['Y0'], t['R'],
                            y0_donor=y0_donor if mode == 'shuffled' else None,
                            tlow_mode=mode)
                        y, dfr, frs, tstar = grounded_forward(
                            core.match, b0_head, f0m, frm, tlm, t['Y0'], grounder)
                        rec['psnr_%s_%s' % (ARM_A1, mode)] = float(_metrics(y, t['H'])[0])
                        rec['ssim_%s_%s' % (ARM_A1, mode)] = float(_metrics(y, t['H'])[1])
                        if mode == 'normal':
                            rec.update(feat_delta_stats(dfr, frm))
                            rec.update(dist_to_f0(frm, frs, f0m))
                            rec.update(t_change_stats(trm, tstar))
                            if lpips_net is not None:
                                rec['lpips_%s' % ARM_A1] = float(lpips_net(y, t['H']).mean())
                rows.append(rec)
            if (i + 1) % 8 == 0 or i + 1 == n:
                print('  %d/%d' % (i + 1, n), flush=True)

    def by_state(key):
        out = {}
        for s in STATES:
            out[s] = nanmean([r[key] for r in rows if r['state'] == s and key in r])
        out['mean'] = nanmean([r[key] for r in rows if key in r])
        return out

    table = dict(Base=by_state('psnr_base'), raw_ref=by_state('psnr_raw_ref'),
                 Frozen_B0=by_state('psnr_%s' % ARM_A0))
    if 'frozen_b0_psnr' in lock:
        table['old_B0'] = {k: float(v) for k, v in lock['frozen_b0_psnr'].items()}
    ssim_table = dict(Base=by_state('ssim_base'), Frozen_B0=by_state('ssim_%s' % ARM_A0))
    if grounder is not None:
        table[ARM_A1] = by_state('psnr_%s_normal' % ARM_A1)
        ssim_table[ARM_A1] = by_state('ssim_%s_normal' % ARM_A1)
        table['A1_minus_B0'] = {
            s: table[ARM_A1][s] - table['Frozen_B0'][s]
            for s in list(STATES) + ['mean']
        }
        for mode in modes:
            if mode != 'normal':
                table['%s_%s' % (ARM_A1, mode)] = by_state('psnr_%s_%s' % (ARM_A1, mode))
    lpips_table = {}
    if lpips_net is not None:
        lpips_table = dict(Base=by_state('lpips_base'),
                           Frozen_B0=by_state('lpips_%s' % ARM_A0))
        if grounder is not None:
            lpips_table[ARM_A1] = by_state('lpips_%s' % ARM_A1)

    feat = {}
    if grounder is not None and rows and 'dFR_mean' in rows[0]:
        for k in ('dFR_mean', 'dFR_p50', 'dFR_p90', 'dFR_max', 'FR_mean', 'r_R',
                  'd_before', 'd_after', 'dT_mean', 'dT_p90', 'cos_T'):
            feat[k] = by_state(k)

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
    dump_json(os.path.join(out_dir, 'summary_psnr.json'), json_ready(table))
    dump_json(os.path.join(out_dir, 'summary_ssim.json'), json_ready(ssim_table))
    dump_json(os.path.join(out_dir, 'safety.json'), json_ready(safety))
    if feat:
        dump_json(os.path.join(out_dir, 'ground_stats.json'), json_ready(feat))
    if lpips_table:
        dump_json(os.path.join(out_dir, 'summary_lpips.json'), json_ready(lpips_table))

    if a.verdict and grounder is not None:
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
        verdict = verdict_v41(
            table['Frozen_B0'], table[ARM_A1],
            safety[ARM_A0], safety[ARM_A1], dep,
            pair_stats=pair_delta_stats_by_state(deltas_by))
        verdict['step'] = a.step
        verdict['eval_repo_commit'] = live
        dump_json(os.path.join(out_dir, 'verdict.json'), json_ready(verdict))
        print('VERDICT', verdict['label'], 'ΔA1-B0=%.3f' % verdict['delta_a1_b0'],
              'next=', verdict['next_step'], flush=True)
        for s in STATES:
            print('  %s  B0=%.3f A1=%.3f d=%.3f' % (
                s, table['Frozen_B0'][s], table[ARM_A1][s],
                table['A1_minus_B0'][s]), flush=True)
        if a.full_dep:
            print('  ground normal/self/zero/shuf = %.3f / %.3f / %.3f / %.3f' % (
                dep['normal'], dep['self'], dep['zero'], dep['shuffled']), flush=True)
    else:
        print('Frozen_B0 mean=%.3f' % table['Frozen_B0']['mean'], flush=True)
        if grounder is not None:
            print('ARM', ARM_A1, 'mean=%.3f' % table[ARM_A1]['mean'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
