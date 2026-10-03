#!/usr/bin/env python
"""V3-A.6 eval: A0 vs A1 on dev64 + constant-q sweep + formal verdict."""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import metrics as _metrics                    # noqa: E402
from model.V3A6DecisionVerifier import build_v3a6_model                  # noqa: E402
from option import parser as option_parser                              # noqa: E402
from v3a5_pipeline import (correction, load_proposal, load_rows,        # noqa: E402
                           make_dataset, sample_tensors)
from v3a5_runtime import (STATES, action_optimal_target, block_energy,  # noqa: E402
                          energy_mask, expand_gate, prepare_geometry,
                          recovery, target_geometry)
from v3a5c_runtime import aggregate_pair_metrics, pair_gate_bundle      # noqa: E402
from v3a6_runtime import (ARMS, CONSTANT_Q, EVAL_STEPS, OBJECTIVES,     # noqa: E402
                          constant_q_psnr, dump_json, nanmean,
                          require_ckpt, verdict_v3a6)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
ROOT = '/root/data/experiments/v3a6_decision_gate'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'
REF_H, REF_W = 400, 600


@torch.no_grad()
def eval_arm(model, ds, n, geom, thr, proposal, device, limit=0):
    n = n if not limit else min(limit, n)
    by_state = {s: [] for s in STATES}
    const_by_state = {s: {q: [] for q in CONSTANT_Q} for s in STATES}
    q_means_by_state = {s: [] for s in STATES}
    for i in range(n):
        for state in STATES:
            t = sample_tensors(ds, i, state, device)
            D, _ = correction(proposal.proposal, t['Y0'], t['R'])
            tgt = action_optimal_target(t['Y0'], t['H'], D, geom)
            mask = energy_mask(block_energy(D, geom), thr)
            q_v = model(t['X'], t['Y0'], t['R'], geom=geom)
            q_full = expand_gate(q_v, geom)
            bundle = pair_gate_bundle(q_v, tgt['q_grid'], mask)
            y_hat = t['Y0'] + q_full * D
            y_r1 = t['Y0'] + D
            y_ao = t['Y0'] + tgt['q_full'] * D
            p_base = _metrics(t['Y0'], t['H'])[0]
            p_r1 = _metrics(y_r1, t['H'])[0]
            p_hat = _metrics(y_hat, t['H'])[0]
            p_ao = _metrics(y_ao, t['H'])[0]
            rec = recovery(p_hat, p_r1, p_ao)
            bundle.update(dict(
                PSNR=float(p_hat), PSNR_Base=float(p_base), PSNR_R1=float(p_r1),
                PSNR_AO64=float(p_ao),
                Recovery64=float(rec) if rec is not None else float('nan'),
                name=t['name'], state=state,
                q_mean_img=float(q_v.float().mean()),
                q_std_img=float(q_v.float().std()),
            ))
            by_state[state].append(bundle)
            q_means_by_state[state].append(float(q_v.float().mean()))
            for q in CONSTANT_Q:
                const_by_state[state][q].append(
                    constant_q_psnr(t['Y0'], t['H'], D, q, _metrics))
        if (i + 1) % 16 == 0 or i + 1 == n:
            print('    %d/%d' % (i + 1, n), flush=True)

    out = {}
    const_mean = {}
    const_best = {}
    gate = {}
    for s in STATES:
        agg = aggregate_pair_metrics(by_state[s])
        agg['PSNR'] = nanmean([r['PSNR'] for r in by_state[s]])
        agg['PSNR_Base'] = nanmean([r['PSNR_Base'] for r in by_state[s]])
        agg['PSNR_R1'] = nanmean([r['PSNR_R1'] for r in by_state[s]])
        agg['PSNR_AO64'] = nanmean([r['PSNR_AO64'] for r in by_state[s]])
        agg['Recovery64'] = nanmean([r['Recovery64'] for r in by_state[s]])
        out[s] = agg
        const_mean[s] = {str(q): nanmean(const_by_state[s][q]) for q in CONSTANT_Q}
        best_q = max(CONSTANT_Q, key=lambda q: nanmean(const_by_state[s][q]))
        const_best[s] = nanmean(const_by_state[s][best_q])
        const_mean[s]['best_q'] = float(best_q)
        const_mean[s]['best_psnr'] = const_best[s]
        gate[s] = dict(
            q_mean=nanmean([r['q_mean_img'] for r in by_state[s]]),
            q_std=nanmean([r['q_std_img'] for r in by_state[s]]),
            frac0=agg.get('frac_qv_le_05'),
            frac1=agg.get('frac_qv_ge_95'),
            corr=agg.get('masked_corr'),
            masked_MAE=agg.get('masked_MAE'),
            decision_accuracy=agg.get('decision_accuracy'),
            img_qmean_std=float(np.std(q_means_by_state[s]))
            if len(q_means_by_state[s]) > 1 else float('nan'),
        )
    return out, const_mean, const_best, gate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--src_root', default=SRC)
    ap.add_argument('--v4_root', default=V4)
    ap.add_argument('--data_dir', default='/root/data/datasets/lol-v2-real')
    ap.add_argument('--variant', default='nanobanana_ref_v2')
    ap.add_argument('--cache_name', default='cache_y0_lolbase')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--formal', action='store_true', default=True)
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    thr = float(lock['energy_threshold'])
    geom = prepare_geometry(target_geometry(REF_H, REF_W, 'g64'), a.device)
    proposal = load_proposal(os.path.join(a.src_root, R1_CK), a.device)

    ns = option_parser.parse_args([])
    ns.dataset_dir = a.data_dir
    ns.v3a_ref_variant = a.variant
    splits = load_rows(os.path.join(a.src_root, 'manifests', 'refiner_train.csv'),
                       os.path.join(a.v4_root, 'splits', 'split.json'))
    mmap = json.load(open(os.path.join(a.v4_root, 'mappings',
                                       'mismatch_dev_64.json')))
    ds = make_dataset(
        ns, splits['dev'],
        os.path.join(a.src_root, a.cache_name, 'refiner_train'), mmap)

    results = {}
    for arm in ARMS:
        ckpt = os.path.join(a.root, arm, 'checkpoints',
                            'ckpt_%06d.pt' % a.step)
        require_ckpt(ckpt, formal=a.formal)
        blob = torch.load(ckpt, map_location='cpu')
        if int(blob.get('step', -1)) != int(a.step):
            raise SystemExit('ckpt step mismatch %s' % ckpt)
        model = build_v3a6_model(arm).to(a.device).eval()
        model.load_state_dict(blob['model'], strict=True)
        print('=== eval %s @%d ===' % (arm, a.step), flush=True)
        out, const_m, const_b, gate = eval_arm(
            model, ds, len(splits['dev']), geom, thr, proposal, a.device,
            limit=a.limit)
        results[arm] = dict(
            by_state=out, constant_q=const_m, const_best=const_b, gate=gate)

    # shared Base/R1/AO from either arm (same proposal)
    base = {s: results['A0_qstar']['by_state'][s]['PSNR_Base'] for s in STATES}
    r1 = {s: results['A0_qstar']['by_state'][s]['PSNR_R1'] for s in STATES}
    ao = {s: results['A0_qstar']['by_state'][s]['PSNR_AO64'] for s in STATES}
    a0_psnr = {s: results['A0_qstar']['by_state'][s]['PSNR'] for s in STATES}
    a1_psnr = {s: results['A1_decision_mse']['by_state'][s]['PSNR'] for s in STATES}
    # const_best: max over constant-q per state (same for both; use A0 sweep)
    const_best = results['A0_qstar']['const_best']

    gate_a1 = results['A1_decision_mse']['gate']
    gate_stats = dict(
        q_std_mean=nanmean([gate_a1[s]['q_std'] for s in STATES]),
        state_qmean_std=float(np.std([gate_a1[s]['q_mean'] for s in STATES])),
        per_state=gate_a1,
    )
    verdict = verdict_v3a6(a0_psnr, a1_psnr, base, r1, const_best, gate_stats)
    verdict['step'] = a.step

    out_dir = os.path.join(a.root, 'diagnostics', 'eval_%06d' % a.step)
    dump_json(os.path.join(out_dir, 'dev_results.json'), results)
    dump_json(os.path.join(out_dir, 'verdict.json'), verdict)
    dump_json(os.path.join(out_dir, 'summary_psnr.json'), dict(
        Base=base, R1=r1, AO64=ao, A0=a0_psnr, A1=a1_psnr,
        const_best=const_best,
        constant_q_A0=results['A0_qstar']['constant_q'],
        constant_q_A1=results['A1_decision_mse']['constant_q'],
    ))

    print('VERDICT', verdict['label'], 'next=', verdict['next_step'], flush=True)
    print('  meaning:', verdict['meaning'], flush=True)
    print('  mean Δ(A1-A0)=%.3f  beat_cq=%s  q_collapsed=%s'
          % (verdict['delta_a1_a0'], verdict['beat_constant_q'],
             verdict['q_collapsed']), flush=True)
    for s in STATES:
        print('  %s  Base=%.3f R1=%.3f cq*=%.3f A0=%.3f A1=%.3f AO=%.3f'
              % (s, base[s], r1[s], const_best[s], a0_psnr[s], a1_psnr[s],
                 ao[s]), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
