#!/usr/bin/env python
"""V3-A.7 eval: A0=V3A6 decision_mse vs A1=utility BCE on dev64."""

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
from v3a6_runtime import (CONSTANT_Q, constant_q_psnr, file_sha256,     # noqa: E402
                          hard_verify_lock, nanmean, require_ckpt)
from v3a7_runtime import (FORMAL_LOCK_KEYS, accept_target, block_utility,  # noqa: E402
                          deployable_global_constant, dump_json,
                          verdict_v3a7)

SRC = '/root/data/experiments/v3a1_lolv2real'
V4 = '/root/data/experiments/v3a4_lolv2real'
ROOT = '/root/data/experiments/v3a7_utility_gate'
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
            U = block_utility(t['Y0'], t['H'], D, geom)
            t_acc = accept_target(U)
            m = mask.bool()
            if m.any():
                pred = (q_v > 0.5)
                acc = float((pred == t_acc.bool())[m].float().mean())
                pos = float(t_acc[m].float().mean())
            else:
                acc, pos = float('nan'), float('nan')
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
                accept_acc=acc, utility_pos_frac=pos,
            ))
            by_state[state].append(bundle)
            q_means_by_state[state].append(float(q_v.float().mean()))
            for q in CONSTANT_Q:
                const_by_state[state][q].append(
                    constant_q_psnr(t['Y0'], t['H'], D, q, _metrics))
        if (i + 1) % 16 == 0 or i + 1 == n:
            print('    %d/%d' % (i + 1, n), flush=True)

    out, const_mean, const_best, gate = {}, {}, {}, {}
    for s in STATES:
        agg = aggregate_pair_metrics(by_state[s])
        agg['PSNR'] = nanmean([r['PSNR'] for r in by_state[s]])
        agg['PSNR_Base'] = nanmean([r['PSNR_Base'] for r in by_state[s]])
        agg['PSNR_R1'] = nanmean([r['PSNR_R1'] for r in by_state[s]])
        agg['PSNR_AO64'] = nanmean([r['PSNR_AO64'] for r in by_state[s]])
        agg['Recovery64'] = nanmean([r['Recovery64'] for r in by_state[s]])
        agg['accept_acc'] = nanmean([r['accept_acc'] for r in by_state[s]])
        agg['utility_pos_frac'] = nanmean(
            [r['utility_pos_frac'] for r in by_state[s]])
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
            accept_acc=agg['accept_acc'],
            utility_pos_frac=agg['utility_pos_frac'],
            img_qmean_std=float(np.std(q_means_by_state[s]))
            if len(q_means_by_state[s]) > 1 else float('nan'),
        )
    return out, const_mean, const_best, gate


def load_arm(ckpt, device):
    require_ckpt(ckpt, formal=True)
    blob = torch.load(ckpt, map_location='cpu')
    model = build_v3a6_model('A1_decision_mse').to(device).eval()
    model.load_state_dict(blob['model'], strict=True)
    return model, blob


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
    a = ap.parse_args(_CLI)

    lock = json.load(open(os.path.join(a.root, 'artifact_lock.json')))
    missing = [k for k in FORMAL_LOCK_KEYS if k not in lock]
    if missing:
        raise SystemExit('artifact_lock missing keys: %s' % missing)
    hard_verify_lock(lock, dict(
        proposal_sha256=file_sha256(lock['proposal_ckpt']),
        split_sha256=file_sha256(lock['split_json']),
        mismatch_train_sha256=file_sha256(lock['mismatch_train']),
        mismatch_dev_sha256=file_sha256(lock['mismatch_dev']),
        official_test_allowed=False,
        architecture='V3A5D2Verifier.A1_multiscale',
        geometry='g64',
        bottleneck=64,
        reference_variant=a.variant,
    ), formal=True)
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

    specs = {
        'A0_decision_mse': lock['a0_checkpoints'][str(a.step)],
        'A1_utility_bce': os.path.join(
            a.root, 'A1_utility_bce', 'checkpoints', 'ckpt_%06d.pt' % a.step),
    }
    results = {}
    for arm, ckpt in specs.items():
        print('=== eval %s @%d (%s) ===' % (arm, a.step, ckpt), flush=True)
        model, blob = load_arm(ckpt, a.device)
        if int(blob.get('step', -1)) != int(a.step):
            raise SystemExit('ckpt step mismatch %s' % ckpt)
        if arm == 'A1_utility_bce':
            if blob.get('arm') not in (None, 'A1_utility_bce'):
                raise SystemExit('ckpt arm mismatch %s' % blob.get('arm'))
            if blob.get('objective') not in (None, 'utility_accept_bce'):
                raise SystemExit('ckpt objective mismatch')
            if blob.get('init_sha') and blob['init_sha'] != lock.get('init_sha'):
                raise SystemExit('ckpt init_sha mismatch')
            if (blob.get('proposal_sha')
                    and blob['proposal_sha'] != lock.get('proposal_sha256')):
                raise SystemExit('ckpt proposal_sha mismatch')
        out, const_m, const_b, gate = eval_arm(
            model, ds, len(splits['dev']), geom, thr, proposal, a.device,
            limit=a.limit)
        results[arm] = dict(
            by_state=out, constant_q=const_m, const_best=const_b, gate=gate,
            ckpt=ckpt)

    base = {s: results['A0_decision_mse']['by_state'][s]['PSNR_Base']
            for s in STATES}
    r1 = {s: results['A0_decision_mse']['by_state'][s]['PSNR_R1'] for s in STATES}
    ao = {s: results['A0_decision_mse']['by_state'][s]['PSNR_AO64'] for s in STATES}
    a0_psnr = {s: results['A0_decision_mse']['by_state'][s]['PSNR'] for s in STATES}
    a1_psnr = {s: results['A1_utility_bce']['by_state'][s]['PSNR'] for s in STATES}
    const_best = results['A0_decision_mse']['const_best']
    gate_a1 = results['A1_utility_bce']['gate']
    gate_stats = dict(
        q_std_mean=nanmean([gate_a1[s]['q_std'] for s in STATES]),
        state_qmean_std=float(np.std([gate_a1[s]['q_mean'] for s in STATES])),
        per_state=gate_a1,
    )
    gq = deployable_global_constant(results['A0_decision_mse']['constant_q'])
    verdict = verdict_v3a7(
        a0_psnr, a1_psnr, base, r1, const_best, gate_stats, global_const=gq)
    verdict['step'] = a.step
    verdict['a0_is'] = 'V3-A.6 A1_decision_mse'
    verdict['a1_is'] = 'utility_accept_bce'
    verdict['const_best_per_state'] = const_best

    out_dir = os.path.join(a.root, 'diagnostics', 'eval_%06d' % a.step)
    dump_json(os.path.join(out_dir, 'dev_results.json'), results)
    dump_json(os.path.join(out_dir, 'verdict.json'), verdict)
    dump_json(os.path.join(out_dir, 'summary_psnr.json'), dict(
        Base=base, R1=r1, AO64=ao, A0=a0_psnr, A1=a1_psnr,
        const_best=const_best,
        constant_q_A0=results['A0_decision_mse']['constant_q'],
        constant_q_A1=results['A1_utility_bce']['constant_q'],
    ))

    print('VERDICT', verdict['label'], 'next=', verdict['next_step'], flush=True)
    print('  meaning:', verdict['meaning'], flush=True)
    print('  mean Δ(A1-A0)=%.3f  beat_global_const=%s  q_collapsed=%s'
          % (verdict['delta_a1_a0'], verdict['beat_global_constant'],
             verdict['q_collapsed']), flush=True)
    for s in STATES:
        print('  %s  Base=%.3f R1=%.3f cq*=%.3f A0=%.3f A1=%.3f AO=%.3f accU=%.3f'
              % (s, base[s], r1[s], const_best[s], a0_psnr[s], a1_psnr[s],
                 ao[s], gate_a1[s]['accept_acc']), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
