"""V3-A.4.3 runtime: fine-resolution saturation sweep on the nested base-H4 family.

Question: between ``G16`` and ``Block_H4`` (100x150 base cells on 400x600), where
does the oracle headroom actually saturate?

Everything from V3-A.4.2 is inherited UNCHANGED: the base partition is
``Block_H4``, coarse grids are built by grouping base cells (never by cutting
the pixel lattice), the primary denominator stays ``Block_H4``, and
``Spatial_H4_legacy`` stays an anchor outside the resolution curve.

What is new:

1. a **verified** nested level chain for the whole ladder. ``round(linspace)``
   on the base lattice happens to be strictly nested for 100/150 cells, but
   this must never be assumed (plan §16): every level is checked against the
   previous one, and any level that would drop a previous boundary is rebuilt
   by hierarchical refinement instead. The chain

       F_G1 subset F_G2 subset ... subset F_G64 subset F_BlockH4

   therefore holds by construction, so the adjacent marginal gains measure
   *resolution only* -- no boundary realignment can leak in.

2. saturation deltas between adjacent levels plus the plan's engineering bands
   (>=0.05 dB structural / 0.02-0.05 marginal / <0.02 negligible).

3. a reproduction check against the frozen V3-A.4.2 summary
   (G16 / Block_H4 / Spatial_H4_legacy / cap16).
"""

import json
import os

import numpy as np

from v3a42_runtime import _sha256, as_grid_pair, block_edges, git_state

ORACLE_DEF_VERSION = 'v3a43:fine-ladder-nested-from-block-h4'
LEVELS = (16, 32, 64)
# plan §12 engineering bands, in dB
BAND_STRUCTURAL, BAND_MARGINAL = 0.05, 0.02
MSE_TOL = 1e-9


# ── §13/§16 verified nested chain ───────────────────────────────────────────

def is_refinement(coarse_edges, fine_edges):
    """Every coarse boundary must also be a boundary of the finer partition."""
    return set(int(v) for v in coarse_edges) <= set(int(v) for v in fine_edges)


def linspace_index_edges(n_base, grid):
    """V3-A.4.2's rule, on the base-cell lattice: round(linspace(0, n, G+1))."""
    if grid <= 1:
        return np.array([0, int(n_base)], dtype=np.int64)
    idx = np.round(np.linspace(0.0, float(n_base),
                               int(grid) + 1)).astype(np.int64)
    idx = np.unique(np.clip(idx, 0, int(n_base)))
    if idx.size < 2 or idx[0] != 0 or idx[-1] != int(n_base):
        idx = np.array([0, int(n_base)], dtype=np.int64)
    return idx


def refine_index_edges(prev_idx, factor):
    """Split every previous block into `factor` near-equal parts.

    Keeps every previous boundary (that is the whole point), and never emits an
    empty block: a block of a single base cell is left alone.
    """
    factor = max(2, int(factor))
    out = [int(prev_idx[0])]
    for a, b in zip(prev_idx[:-1], prev_idx[1:]):
        a, b = int(a), int(b)
        if b - a < 2:
            out.append(b)
            continue
        cuts = np.unique(np.round(np.linspace(a, b, factor + 1)).astype(np.int64))
        out.extend(int(v) for v in cuts[1:])
    return np.array(sorted(set(out)), dtype=np.int64)


def nested_index_chain(n_base, levels):
    """-> (chain: level -> base-cell index edges, notes: level -> provenance).

    Level-by-level: try ``round(linspace)`` first (identical to V3-A.4.2, so the
    reproduction anchor is preserved bit-for-bit); if that would drop a boundary
    of the previous level, rebuild this level from the previous one instead.
    """
    levels = sorted(set(int(g) for g in levels))
    chain, notes = {}, {}
    prev_idx, prev_g = None, None
    for g in levels:
        idx = linspace_index_edges(n_base, g)
        if prev_idx is not None and not is_refinement(prev_idx, idx):
            factor = max(2, int(round(float(g) / float(max(prev_g, 1)))))
            idx = refine_index_edges(prev_idx, factor)
            notes[g] = 'hierarchical_refine_of_%d(x%d)' % (prev_g, factor)
        else:
            notes[g] = 'linspace_on_base_lattice'
        chain[g] = idx
        prev_idx, prev_g = idx, g
    return chain, notes


def nested_axis_chain(length, base_blocks, levels):
    """-> (level -> pixel edges, notes, scheme) for one axis."""
    ey_base = block_edges(length, base_blocks)
    n_base = len(ey_base) - 1
    idx_chain, notes = nested_index_chain(n_base, levels)
    chain = {g: ey_base[idx] for g, idx in idx_chain.items()}
    scheme = ('linspace_nested'
              if all(n.startswith('linspace') for n in notes.values())
              else 'hierarchical_refine')
    return chain, notes, scheme


def build_level_chain(height, width, base_grid, levels):
    """-> the full 2-D nested chain plus its containment audit (plan §15)."""
    levels = sorted(set(int(g) for g in levels))
    by, bx = as_grid_pair(base_grid)
    yc, yn, ys = nested_axis_chain(height, by, levels)
    xc, xn, xs = nested_axis_chain(width, bx, levels)
    containment = []
    for a, b in zip(levels[:-1], levels[1:]):
        containment.append(dict(coarse=a, fine=b,
                                y=is_refinement(yc[a], yc[b]),
                                x=is_refinement(xc[a], xc[b])))
    return dict(levels=levels, y=yc, x=xc, y_notes=yn, x_notes=xn,
                scheme_y=ys, scheme_x=xs, containment=containment,
                all_nested=all(c['y'] and c['x'] for c in containment))


def level_geometry_report(height, width, base_grid, chain):
    """The §16 artifact: actual block counts and pixel edges per level."""
    out = dict(reference_geometry=[int(height), int(width)],
               base_grid=[int(v) for v in as_grid_pair(base_grid)],
               scheme=dict(y=chain['scheme_y'], x=chain['scheme_x']),
               containment=chain['containment'], all_nested=chain['all_nested'],
               levels={})
    for g in chain['levels']:
        ey, ex = chain['y'][g], chain['x'][g]
        out['levels'][str(g)] = dict(
            nby=int(len(ey) - 1), nbx=int(len(ex) - 1),
            note_y=chain['y_notes'][g], note_x=chain['x_notes'][g],
            pixel_edges_y=[int(v) for v in ey],
            pixel_edges_x=[int(v) for v in ex],
            block_h=int(np.diff(ey).min()), block_w=int(np.diff(ex).min()),
            block_h_max=int(np.diff(ey).max()),
            block_w_max=int(np.diff(ex).max()))
    return out


# ── §10-§12 saturation metrics ──────────────────────────────────────────────

def classify_delta(delta_db):
    if delta_db >= BAND_STRUCTURAL:
        return 'structural'
    if delta_db >= BAND_MARGINAL:
        return 'marginal'
    return 'negligible'


def saturation_metrics(mean_psnr, levels, block_arm='Block_H4', start_arm=None):
    """Adjacent deltas of the resolution ladder, with their engineering band.

    The ladder is the *same-family* chain only; ``Spatial_H4_legacy`` must never
    appear here (its formulation differs, so it is not a resolution step).
    ``start_arm`` optionally prepends the no-correction arm 'R1'.
    """
    arms = ['G%d' % int(g) for g in sorted(int(v) for v in levels)] + [block_arm]
    if start_arm is not None:
        arms = [start_arm] + arms
    out = {}
    for a, b in zip(arms[:-1], arms[1:]):
        d = float(mean_psnr[b]) - float(mean_psnr[a])
        out['%s->%s' % (a, b)] = dict(delta=d, band=classify_delta(d))
    return out


# ── §21 reproduction against V3-A.4.2 ───────────────────────────────────────

def check_v3a42_reproduction(got_state, ref_state, tol_psnr=2e-3, tol_capture=1e-3):
    """Compare this sweep's shared arms against the frozen V3-A.4.2 summary.

    ``Spatial_H4_legacy`` is included because this round must also prove it is
    still the same historical anchor; it is *not* part of the resolution curve.
    """
    arms = ('G16', 'Block_H4', 'Spatial_H4_legacy')
    psnr = {}
    mism = []
    for arm in arms:
        got = float(got_state['mean_psnr'][arm])
        ref = float(ref_state['mean_psnr'][arm])
        d = abs(got - ref)
        psnr[arm] = dict(got=got, v3a42=ref, d=d)
        if d > tol_psnr:
            mism.append('psnr_%s' % arm)
    cap_got = got_state['capture']['G16'].get('capture')
    cap_ref = ref_state['capture']['G16'].get('capture')
    if cap_got is None or cap_ref is None:
        raise SystemExit('reproduction needs cap16 to be defined (H_BlockH4 > 0)')
    d_cap = abs(float(cap_got) - float(cap_ref))
    if d_cap > tol_capture:
        mism.append('capture_G16')
    return dict(psnr=psnr,
                capture_G16=dict(got=cap_got, v3a42=cap_ref, d=d_cap),
                tol_psnr=tol_psnr, tol_capture=tol_capture,
                ok=not mism, mismatches=mism)


def mse_chain_violations(mses, arms):
    """Continuous-MSE must be non-increasing along the nested ladder (§13)."""
    bad = []
    for a, b in zip(arms[:-1], arms[1:]):
        if mses[b] > mses[a] + MSE_TOL:
            bad.append(dict(finer=b, coarser=a,
                            mse_finer=mses[b], mse_coarser=mses[a],
                            diff=mses[b] - mses[a]))
    return bad


# ── §20 artifact lock ───────────────────────────────────────────────────────

def verify_v3a43_artifact_lock(root, src_root,
                               ckpt_name='R1_v2stable_naive_s42/checkpoint_03000.pt',
                               cache_name='cache_y0_lolbase',
                               v4_root='/root/data/experiments/v3a4_lolv2real'):
    path = os.path.join(root, 'artifact_lock.json')
    if not os.path.isfile(path):
        raise SystemExit('v3a43 artifact lock missing: %s' % path)
    lock = json.load(open(path, encoding='utf-8'))
    files = {
        'proposal_sha256': os.path.join(src_root, ckpt_name),
        'cache_metadata_sha256': os.path.join(src_root, cache_name,
                                              'refiner_train', 'metadata.json'),
        'manifest_sha256': os.path.join(src_root, 'manifests', 'refiner_train.csv'),
        'split_sha256': os.path.join(v4_root, 'splits', 'split.json'),
        'mismatch_train_sha256': os.path.join(v4_root, 'mappings',
                                              'mismatch_train_575.json'),
        'mismatch_dev_sha256': os.path.join(v4_root, 'mappings',
                                            'mismatch_dev_64.json'),
        'energy_stats_sha256': os.path.join(v4_root, 'action_stats', 'energy.json'),
    }
    bad = []
    for k, p in files.items():
        if k not in lock:
            raise SystemExit('v3a43 lock missing %s' % k)
        if not os.path.isfile(p):
            raise SystemExit('locked artifact not found: %s' % p)
        if _sha256(p) != lock[k]:
            bad.append(k)
    # the V3-A.4.2 summary IS an input this round: it is the reproduction anchor
    if 'v3a42_summary_sha256' not in lock:
        raise SystemExit('v3a43 lock missing v3a42_summary_sha256')
    v42 = lock.get('v3a42_summary_path')
    if not v42 or not os.path.isfile(v42):
        raise SystemExit('locked V3-A.4.2 summary not found: %s' % v42)
    if _sha256(v42) != lock['v3a42_summary_sha256']:
        bad.append('v3a42_summary_sha256')
    if bad:
        raise SystemExit('v3a43 lock mismatch on %s -- rerun '
                         'scripts/setup_v3a43_fine_resolution.py' % ', '.join(bad))
    if lock.get('oracle_def_version') != ORACLE_DEF_VERSION:
        raise SystemExit('oracle definition changed since the lock (%s vs %s)'
                         % (lock.get('oracle_def_version'), ORACLE_DEF_VERSION))
    head, dirty = git_state()
    if head and lock.get('repo_commit') and lock['repo_commit'] != head:
        raise SystemExit('v3a43 lock generated at %s but HEAD is %s -- rerun '
                         'scripts/setup_v3a43_fine_resolution.py'
                         % (lock['repo_commit'][:8], head[:8]))
    if dirty:
        print('[v3a43] WARNING: working tree is dirty (%d path(s), e.g. %s) -- '
              'the run cannot be attributed to %s alone'
              % (len(dirty), ', '.join(dirty[:3]), (head or 'HEAD')[:8]))
    return lock
