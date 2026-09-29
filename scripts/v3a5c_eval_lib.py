"""Shared evaluate-all-pairs helper for V3-A.5C train/eval scripts."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from local_refine_runtime import metrics
from v3a5_pipeline import sample_tensors
from v3a5_runtime import STATES, expand_gate
from v3a5c_runtime import aggregate_pair_metrics, pair_gate_bundle


@torch.no_grad()
def eval_all_pairs(model, ds, name_to_i, pairs, entries, geom, thr, device,
                   out_weight=0.1) -> Dict:
    """Full tiny-train overfit metrics over every image-state pair."""
    model.eval()
    rows = []
    qv_by_state = {s: [] for s in STATES}
    for p in pairs:
        ent = entries[p['key']]
        t = sample_tensors(ds, name_to_i[p['name']], p['state'], device)
        D = ent['D'].to(device).float()
        q_star = ent['q_grid'].to(device).float()
        mask = ent['mask'].to(device)
        q_v = model(t['X'], t['Y0'], t['R'], geom=geom)
        q_full = expand_gate(q_v, geom)
        bundle = pair_gate_bundle(q_v, q_star, mask)
        # output metrics
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
    # baselines on stacked gates
    qv = torch.stack([torch.as_tensor(r.get('_qv', 0)) for r in []], dim=0) \
        if False else None
    # constant-0.5 / mean-q* baselines via recompute on stored q_star
    const05 = _baseline_from_entries(entries, pairs, 0.5)
    # mean target over all pairs
    means = []
    for p in pairs:
        means.append(float(entries[p['key']]['q_grid'].float().mean()))
    global_mean = float(np.mean(means)) if means else 0.5
    const_mean = _baseline_from_entries(entries, pairs, global_mean)
    per_image = _per_image_scalar_baseline(entries, pairs)

    return dict(
        overall=overall,
        by_state=by_state,
        baselines=dict(
            const_0_5=const05,
            const_global_mean=const_mean,
            per_image_scalar=per_image,
            global_mean_qstar=global_mean,
        ),
        n_pairs=len(rows),
        per_pair=[{k: v for k, v in r.items()
                   if k not in ('q_v',)} for r in rows],
    )


def _baseline_from_entries(entries, pairs, const) -> Dict:
    rows = []
    for p in pairs:
        ent = entries[p['key']]
        q_star = ent['q_grid'].float()
        mask = ent['mask']
        q_v = torch.full_like(q_star, float(const))
        rows.append(pair_gate_bundle(q_v, q_star, mask))
    return aggregate_pair_metrics(rows)


def _per_image_scalar_baseline(entries, pairs) -> Dict:
    rows = []
    for p in pairs:
        ent = entries[p['key']]
        q_star = ent['q_grid'].float()
        mask = ent['mask']
        m = float(q_star.mean())
        q_v = torch.full_like(q_star, m)
        rows.append(pair_gate_bundle(q_v, q_star, mask))
    return aggregate_pair_metrics(rows)
