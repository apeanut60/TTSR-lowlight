"""Eval helper for V3-A.5D1 tiny arms (passes cached evidence_g64)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from local_refine_runtime import metrics
from v3a5_pipeline import sample_tensors
from v3a5_runtime import STATES, expand_gate
from v3a5c_runtime import aggregate_pair_metrics, pair_gate_bundle


@torch.no_grad()
def eval_all_pairs_d1(model, ds, name_to_i, pairs, entries, geom, thr, device,
                      out_weight=0.0) -> Dict:
    model.eval()
    rows = []
    qv_by_state = {s: [] for s in STATES}
    for p in pairs:
        ent = entries[p['key']]
        t = sample_tensors(ds, name_to_i[p['name']], p['state'], device)
        D = ent['D'].to(device).float()
        q_star = ent['q_grid'].to(device).float()
        mask = ent['mask'].to(device)
        ev = None
        if 'evidence_g64' in ent:
            ev = ent['evidence_g64'].to(device).float()
        q_v = model(t['X'], t['Y0'], t['R'], geom=geom, evidence_g64=ev)
        q_full = expand_gate(q_v, geom)
        bundle = pair_gate_bundle(q_v, q_star, mask)
        y_hat = t['Y0'] + q_full * D
        y_r1 = t['Y0'] + D
        y_ao = t['Y0'] + ent['q_full'].to(device).float() * D
        psnr = metrics(y_hat, t['H'])[0]
        psnr_r1 = metrics(y_r1, t['H'])[0]
        psnr_ao = metrics(y_ao, t['H'])[0]
        psnr_base = metrics(t['Y0'], t['H'])[0]
        denom = psnr_ao - psnr_r1
        recovery = ((psnr - psnr_r1) / denom) if abs(denom) > 1e-8 else float('nan')
        bundle.update(dict(
            PSNR=float(psnr), PSNR_R1=float(psnr_r1), PSNR_AO64=float(psnr_ao),
            PSNR_Base=float(psnr_base), Recovery64=float(recovery),
            name=p['name'], state=p['state'], out_weight=float(out_weight),
            energy_threshold=float(thr),
        ))
        rows.append(bundle)
        qv_by_state[p['state']].append(bundle)

    overall = aggregate_pair_metrics(rows)
    by_state = {s: aggregate_pair_metrics(qv_by_state[s]) for s in STATES}
    means = [float(entries[p['key']]['q_grid'].float().mean()) for p in pairs]
    global_mean = float(np.mean(means)) if means else 0.5
    return dict(
        overall=overall,
        by_state=by_state,
        baselines=dict(global_mean_qstar=global_mean),
        n_pairs=len(rows),
        per_pair=rows,
    )
