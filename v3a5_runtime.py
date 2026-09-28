"""V3-A.5 runtime: learnable G64 verifier -- target family, mask, fairness, verdict.

This round does NOT search a new oracle family. It reuses the V3-A.4.3 nested
base-H4 construction and only changes the *resolution of the gate target*:

    A0_dense_blockh4  q* on 100x150 base cells   (the dense control)
    A1_g64            q* on 64x64 nested blocks  (the regional arm)

Both are the same closed form on the same raw physical D / H / Y0:

    q_B* = clip( sum_{B,c}(H-Y0)D / (sum_{B,c}D^2 + eps), 0, 1 )

and both expand with the same nearest (piecewise-constant) block mapping --
never bilinear, never an AvgPool of a finer gate.
"""

import json
import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from v3a42_runtime import (block_edges, blockwise_action_optimal_gate, git_state,
                           h4_block_grid)
from v3a43_runtime import build_level_chain

STATES = ('correct', 'true_dark_g0.5', 'mismatch')
# §11 locked state weights, realised as sampling probabilities
STATE_WEIGHTS = {'correct': 1.0, 'true_dark_g0.5': 0.5, 'mismatch': 0.5}
STATE_PROBS = (0.5, 0.25, 0.25)
ARMS = ('A0_dense_blockh4', 'A1_g64')
ARM_MODE = {'A0_dense_blockh4': 'dense_block_h4', 'A1_g64': 'g64'}
MODE_ARM = {v: k for k, v in ARM_MODE.items()}
MODES = ('dense_block_h4', 'g64')
G64 = 64
ENERGY_PCTL = 10.0


# ── §4 target geometry (both modes come from ONE nested base-H4 family) ─────

def target_geometry(height, width, mode, factor=4):
    """-> dict describing one target parameterisation.

    ``edges`` are explicit pixel edges, ``index`` is the flat block id of every
    pixel (nearest expansion), ``shape`` is (nby, nbx).
    """
    if mode not in MODES:
        raise SystemExit('unknown target mode %r' % mode)
    base = h4_block_grid(height, width, factor)
    if mode == 'dense_block_h4':
        ey = block_edges(height, base[0])
        ex = block_edges(width, base[1])
        provenance = 'block_h4_base_cells'
    else:
        chain = build_level_chain(height, width, base, (G64,))
        ey, ex = chain['y'][G64], chain['x'][G64]
        if not chain['all_nested']:
            raise SystemExit('G64 is not a coarsening of Block_H4: %s'
                             % chain['containment'])
        provenance = 'g64_nested_from_block_h4'
    # The explicit chain edges are the authority. NOTE: the G64 chain edges are
    # NOT round(linspace(0, height, 65)) -- they are base-cell boundaries, so
    # the id map must be built from the edges, never re-derived from a grid size.
    idx, nby, nbx = _cached_index(int(height), int(width),
                                  tuple(int(v) for v in ey),
                                  tuple(int(v) for v in ex))
    return dict(mode=mode, factor=int(factor), base_grid=(int(base[0]), int(base[1])),
                edges=(np.asarray(ey, dtype=np.int64), np.asarray(ex, dtype=np.int64)),
                index=idx, shape=(int(nby), int(nbx)),
                provenance=provenance)


@lru_cache(maxsize=64)
def _cached_index(height, width, ey, ex):
    """Cached (read-only by convention) id map for one explicit partition."""
    idx = _index_from_edges(height, width, ey, ex)
    return idx, len(ey) - 1, len(ex) - 1


def _index_from_edges(height, width, ey, ex):
    ey = [int(v) for v in ey]
    ex = [int(v) for v in ex]
    nby, nbx = len(ey) - 1, len(ex) - 1
    idx = np.empty((height, width), dtype=np.int64)
    for i in range(nby):
        for j in range(nbx):
            idx[ey[i]:ey[i + 1], ex[j]:ex[j + 1]] = i * nbx + j
    return idx.reshape(-1)


def edges_are_nested(coarse, fine):
    return set(int(v) for v in coarse) <= set(int(v) for v in fine)


def geometry_report(height, width, geoms):
    """The §30 artifact: edges / block counts / nesting for every arm."""
    out = {}
    for mode, g in geoms.items():
        ey, ex = g['edges']
        out[mode] = dict(shape=list(g['shape']), base_grid=list(g['base_grid']),
                         provenance=g['provenance'],
                         pixel_edges_y=[int(v) for v in ey],
                         pixel_edges_x=[int(v) for v in ex],
                         block_h=int(np.diff(ey).min()), block_h_max=int(np.diff(ey).max()),
                         block_w=int(np.diff(ex).min()), block_w_max=int(np.diff(ex).max()))
    base = geoms['dense_block_h4']['edges']
    g64 = geoms['g64']['edges']
    out['nesting'] = dict(g64_within_block_h4=bool(edges_are_nested(g64[0], base[0])
                                                   and edges_are_nested(g64[1], base[1])))
    return out


# ── §4/§8 oracle target and its expansion ───────────────────────────────────

def action_optimal_target(y0, hr, d_correction, geom, eps=1e-8):
    """-> dict(q_grid [B,1,nby,nbx], q_full [B,1,H,W], Y_oracle, N_grid, Z_grid).

    The gate is solved on the RAW physical tensors at this geometry's blocks --
    explicitly not a pooled/resized version of another arm's target.
    """
    o = blockwise_action_optimal_gate(y0, hr, d_correction, geom['shape'],
                                      eps=eps, edges=geom['edges'])
    nby, nbx = geom['shape']
    return dict(q_grid=o['q_grid'].reshape(-1, 1, nby, nbx),
                q_full=o['q_full'], Y_oracle=o['Y_oracle'],
                N_grid=o['N_grid'], Z_grid=o['Z_grid'])


def expand_gate(q_map, geom):
    """[B,1,nby,nbx] -> [B,1,H,W] with the SAME nearest mapping as the target."""
    b = q_map.shape[0]
    idx = geom['index_t']
    return q_map.reshape(b, -1).gather(1, idx.unsqueeze(0).expand(b, -1)) \
        .reshape(b, 1, geom['height'], geom['width'])


def prepare_geometry(geom, device):
    """Attach the tensors that the training/eval loops need (device-resident)."""
    ey, ex = geom['edges']
    h = int(ey[-1])
    w = int(ex[-1])
    idx = np.asarray(geom['index'])
    if idx.size != h * w:
        idx = _index_from_edges(h, w, ey, ex)
    geom = dict(geom, height=h, width=w,
                index_t=torch.as_tensor(idx, dtype=torch.long, device=device))
    return geom


# ── §12 resolution-independent physical energy mask ─────────────────────────

def pixel_energy(d_correction):
    """Per-pixel physical correction energy (1/3) sum_c D^2 -> [B,1,H,W]."""
    return d_correction.pow(2).sum(dim=1, keepdim=True) / 3.0


def block_energy(d_correction, geom):
    """e_B = (1/(3|B|)) sum_{B,c} D^2, at this geometry's blocks -> [B,1,nby,nbx].

    The threshold is derived once from the Block_H4 cell distribution, so the
    criterion is resolution independent; each arm evaluates its OWN blocks
    against that same threshold (§12).
    """
    nby, nbx = geom['shape']
    return block_mean(pixel_energy(d_correction), geom).reshape(-1, 1, nby, nbx)


def block_mean(pixel_map, geom):
    """Block mean of a [B,1,H,W] per-pixel map with the geometry's blocks."""
    b = pixel_map.shape[0]
    flat = pixel_map.reshape(b, -1)
    nblk = geom['shape'][0] * geom['shape'][1]
    acc = torch.zeros(b, nblk, dtype=flat.dtype, device=flat.device)
    acc.index_add_(1, geom['index_t'], flat)
    cnt = torch.zeros(nblk, dtype=flat.dtype, device=flat.device)
    cnt.index_add_(0, geom['index_t'], torch.ones_like(flat[0]))
    return (acc / cnt.clamp(min=1.0).unsqueeze(0)).reshape(
        b, 1, geom['shape'][0], geom['shape'][1])


def energy_threshold(pooled_energies, pctl=ENERGY_PCTL):
    """p-th percentile of the pooled train cell energies (§12: p10)."""
    v = np.asarray([float(x) for x in pooled_energies], dtype=np.float64)
    if v.size == 0:
        raise SystemExit('no energies pooled for the threshold')
    return float(np.percentile(v, pctl))


def energy_mask(energies, threshold):
    """-> float mask [B,1,nby,nbx]; a block is valid when e_B >= threshold."""
    return (energies >= float(threshold)).to(energies.dtype)


# ── §13 loss ────────────────────────────────────────────────────────────────

def masked_smooth_l1(q_v, q_target, mask, beta=1.0):
    d = F.smooth_l1_loss(q_v, q_target, reduction='none', beta=beta)
    m = mask.to(d.dtype)
    return (m * d).sum() / m.sum().clamp(min=1.0)


def gate_output_loss(q_v_full, y0, hrd, d_correction):
    """§13 L_out = L1(Y0 + q_v D, H) -- the actual enhancement objective."""
    return F.l1_loss(y0 + q_v_full * d_correction, hrd)


def verifier_loss(q_v, q_target, mask, q_v_full, y0, hrd, d_correction,
                  out_weight=0.1, beta=1.0):
    """-> (total, gate_term, output_term); §13 fixes the weights at 1.0 / 0.1."""
    gate = masked_smooth_l1(q_v, q_target, mask, beta=beta)
    out = gate_output_loss(q_v_full, y0, hrd, d_correction)
    return gate + out_weight * out, gate, out


# ── §15 fairness ────────────────────────────────────────────────────────────

def snapshot_(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def bit_equal(a, b, keys=None):
    keys = keys or sorted(a)
    for k in keys:
        if a[k].shape != b[k].shape or not torch.equal(a[k], b[k]):
            return False
    return True


def parameter_l1_drift(before, after):
    """Total L1 drift of a state_dict snapshot; 0.0 means bit-identical."""
    tot = 0.0
    for k, v in before.items():
        tot += float((after[k].detach() - v).abs().sum())
    return tot


def state_dict_sha(state_dict):
    """Content hash of a state_dict (names + shapes + dtypes + bytes)."""
    import hashlib
    h = hashlib.sha256()
    for k in sorted(state_dict):
        t = state_dict[k].detach().to('cpu').contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


# ── §18/§20 evaluation ──────────────────────────────────────────────────────

def recovery(psnr_arm, psnr_r1, psnr_oracle):
    """§18 Recovery = (PSNR(V) - PSNR(R1)) / (PSNR(AO) - PSNR(R1))."""
    den = float(psnr_oracle) - float(psnr_r1)
    if den <= 0:
        return None
    return (float(psnr_arm) - float(psnr_r1)) / den


def gate_metrics(q_v, q_target, mask):
    """§19 gate prediction metrics on flattened maps (any leading shape)."""
    a = q_v.detach().float().reshape(-1)
    b = q_target.detach().float().reshape(-1)
    m = mask.detach().float().reshape(-1)
    d = (a - b).abs()
    out = dict(q_v_mean=float(a.mean()), q_v_std=float(a.std()),
               q_opt_mean=float(b.mean()), q_opt_std=float(b.std()),
               MAE=float(d.mean()), RMSE=float((a - b).pow(2).mean().sqrt()),
               frac_q0=float((a <= 0.01).float().mean()),
               frac_q1=float((a >= 0.99).float().mean()),
               mask_valid_frac=float(m.mean()))
    av, bv = a[m > 0], b[m > 0]
    if av.numel() >= 2 and float(av.std()) > 0 and float(bv.std()) > 0:
        out['masked_MAE'] = float((av - bv).abs().mean())
        out['masked_corr'] = float(torch.corrcoef(torch.stack([av, bv]))[0, 1])
    else:
        out['masked_MAE'] = float('nan')
        out['masked_corr'] = float('nan')
    if float(a.std()) > 0 and float(b.std()) > 0:
        out['corr'] = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    else:
        out['corr'] = float('nan')
    return out


def per_image_qmean_corr(q_v_list, q_opt_list):
    """§19: does the arm track per-image level, or real regional structure?

    Returns corr between the two per-image mean gates. A high value with a low
    spatial corr means the model only learned the image-level state.
    """
    a = np.asarray([float(v.mean()) for v in q_v_list], dtype=np.float64)
    b = np.asarray([float(v.mean()) for v in q_opt_list], dtype=np.float64)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def verdict_5a(dev, base_psnr):
    """§20 criteria. ``dev``: state -> dict(R1=, A0=, A1=). Reports, never decides."""
    corr = dev['correct']
    gain = {s: dev[s]['A1'] - dev[s]['A0'] for s in STATES}
    harmful = [s for s in ('true_dark_g0.5', 'mismatch')]
    mean_gain = float(np.mean(list(gain.values())))
    n_gain = sum(1 for v in gain.values() if v >= 0.05)
    return dict(
        # correct must not collapse (V3-A.4 C0 was ~ R1 - 0.06 dB)
        correct_not_worse=bool(corr['A1'] >= corr['R1'] - 0.02),
        correct_beats_r1=bool(corr['A1'] >= corr['R1']),
        # harmful references must be suppressed
        harmful_ge_base=bool(all(dev[s]['A1'] >= base_psnr for s in harmful)),
        harmful_ge_base_005=bool(all(dev[s]['A1'] >= base_psnr + 0.05
                                     for s in harmful)),
        # the primary causal criterion: G64 vs the dense control
        gain_by_state=gain, mean_gain=mean_gain,
        gain_rule_mean=bool(mean_gain >= 0.05),
        gain_rule_majority=bool(n_gain >= 2 and all(v >= -0.02 for v in gain.values())),
        gain_rule=bool(mean_gain >= 0.05
                       or (n_gain >= 2 and all(v >= -0.02 for v in gain.values()))),
        base_psnr=float(base_psnr))


# ── §30 artifact lock ───────────────────────────────────────────────────────

def verify_v3a5_artifact_lock(root, src_root,
                              ckpt_name='R1_v2stable_naive_s42/checkpoint_03000.pt',
                              cache_name='cache_y0_lolbase',
                              v4_root='/root/data/experiments/v3a4_lolv2real',
                              v43_root='/root/data/experiments/v3a43_fine_resolution'):
    from v3a42_runtime import _sha256
    path = os.path.join(root, 'artifact_lock.json')
    if not os.path.isfile(path):
        raise SystemExit('v3a5 artifact lock missing: %s' % path)
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
        'v3a43_summary_sha256': os.path.join(v43_root, 'oracle', 'summary.json'),
        'v3a43_nesting_sha256': os.path.join(v43_root, 'oracle', 'nesting.json'),
        'energy_stats_sha256': os.path.join(root, 'targets', 'energy_stats.json'),
    }
    bad = []
    for k, p in files.items():
        if k not in lock:
            raise SystemExit('v3a5 lock missing %s' % k)
        if not os.path.isfile(p):
            raise SystemExit('locked artifact not found: %s' % p)
        if _sha256(p) != lock[k]:
            bad.append(k)
    if bad:
        raise SystemExit('v3a5 lock mismatch on %s -- rerun '
                         'scripts/setup_v3a5_verifier.py' % ', '.join(bad))
    if lock.get('target_family') != 'nested_blockwise_action_optimal':
        raise SystemExit('unexpected target family %r' % lock.get('target_family'))
    head, _dirty = git_state()
    if head and lock.get('repo_commit') and lock['repo_commit'] != head:
        raise SystemExit('v3a5 lock generated at %s but HEAD is %s -- rerun '
                         'scripts/setup_v3a5_verifier.py'
                         % (lock['repo_commit'][:8], head[:8]))
    return lock


def check_worktree(limit, head=None, dirty=None):
    """Formal runs need a clean tree; smokes only warn (§30)."""
    if head is None or dirty is None:
        head, dirty = git_state()
    dirty = list(dirty or [])
    if not dirty:
        return head, dirty, None
    msg = ('working tree is dirty (%d path(s), e.g. %s)'
           % (len(dirty), ', '.join(dirty[:3])))
    if int(limit) == 0:
        raise SystemExit('%s -- a formal V3-A.5 run must be attributable to %s '
                         'alone; commit and rerun setup' % (msg, (head or 'HEAD')[:8]))
    return head, dirty, ('%s -- smoke run, NOT attributable to %s alone'
                         % (msg, (head or 'HEAD')[:8]))


def sample_states(n, seed, probs=STATE_PROBS):
    """Deterministic §11 sampling: correct 0.50 / dark 0.25 / mismatch 0.25."""
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(STATES), size=int(n), p=list(probs))
    return [STATES[i] for i in idx]
