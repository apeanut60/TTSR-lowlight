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
from v3a43_runtime import nested_index_chain

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

    ONE source of truth for three consumers (§29):

      * ``edges`` (pixel)  -> the oracle target q* and the gate expansion,
      * ``base_index_edges`` (H/4 base-cell lattice) -> the EXACT region pooling
        of the verifier feature map,
      * ``shape``          -> both, and the head's output grid.

    ``index`` is the flat block id of every pixel (nearest expansion).
    """
    if mode not in MODES:
        raise SystemExit('unknown target mode %r' % mode)
    base = h4_block_grid(height, width, factor)
    by, bx = int(base[0]), int(base[1])
    # native H/4 lattice: the verifier's common feature map lives exactly here
    ey_base = block_edges(height, by)
    ex_base = block_edges(width, bx)
    n_by, n_bx = len(ey_base) - 1, len(ex_base) - 1
    if mode == 'dense_block_h4':
        iy = np.arange(n_by + 1, dtype=np.int64)
        ix = np.arange(n_bx + 1, dtype=np.int64)
        provenance = 'block_h4_base_cells'
    else:
        # the same round(linspace) grouping on the base-cell lattice that
        # V3-A.4.3 used; the pixel edges are its image under ey_base
        iy = nested_index_chain(n_by, (G64,))[0][G64]
        ix = nested_index_chain(n_bx, (G64,))[0][G64]
        if set(int(v) for v in iy) > set(range(n_by + 1)) or \
                set(int(v) for v in ix) > set(range(n_bx + 1)):
            raise SystemExit('G64 index edges left the base-cell lattice')
        provenance = 'g64_nested_from_block_h4'
    ey = np.asarray(ey_base)[iy]
    ex = np.asarray(ex_base)[ix]
    # The explicit chain edges are the authority. NOTE: the G64 chain edges are
    # NOT round(linspace(0, height, 65)) -- they are base-cell boundaries, so
    # the id map must be built from the edges, never re-derived from a grid size.
    idx, nby, nbx = _cached_index(int(height), int(width),
                                  tuple(int(v) for v in ey),
                                  tuple(int(v) for v in ex))
    if (nby, nbx) != (len(iy) - 1, len(ix) - 1):
        raise SystemExit('pixel-edge partition (%d,%d) disagrees with the base-cell '
                         'partition (%d,%d)' % (nby, nbx, len(iy) - 1, len(ix) - 1))
    return dict(mode=mode, factor=int(factor), base_grid=(int(base[0]), int(base[1])),
                edges=(np.asarray(ey, dtype=np.int64), np.asarray(ex, dtype=np.int64)),
                base_index_edges=(np.asarray(iy, dtype=np.int64),
                                  np.asarray(ix, dtype=np.int64)),
                index=idx, shape=(int(nby), int(nbx)),
                native_shape=(int(n_by), int(n_bx)),
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
    out['native_feature_shape'] = list(geoms['g64']['native_shape'])
    for mode, g in geoms.items():
        ey, ex = g['edges']
        iy, ix = g['base_index_edges']
        out[mode] = dict(shape=list(g['shape']), base_grid=list(g['base_grid']),
                         native_shape=list(g['native_shape']),
                         provenance=g['provenance'],
                         base_index_edges_y=[int(v) for v in iy],
                         base_index_edges_x=[int(v) for v in ix],
                         pixel_edges_y=[int(v) for v in ey],
                         pixel_edges_x=[int(v) for v in ex],
                         block_h=int(np.diff(ey).min()), block_h_max=int(np.diff(ey).max()),
                         block_w=int(np.diff(ex).min()), block_w_max=int(np.diff(ex).max()))
    base = geoms['dense_block_h4']['base_index_edges']
    g64 = geoms['g64']['base_index_edges']
    # the dense arm's native support IS the Block_H4 partition, so a G64 block
    # must be a union of dense support cells -- the feature pooling and the
    # target therefore share one partition by construction
    out['nesting'] = dict(g64_within_block_h4=bool(
        edges_are_nested(g64[0], np.arange(len(base[0]))) and
        edges_are_nested(g64[1], np.arange(len(base[1])))))
    return out


# ── §5-§7 exact region pooling on the native H/4 lattice ────────────────────

@lru_cache(maxsize=64)
def _pool_index(native_h, native_w, iy, ix):
    """-> (idx [native_h*native_w] block id, counts [nblk]) on the base lattice."""
    idx = np.empty((native_h, native_w), dtype=np.int64)
    nby, nbx = len(iy) - 1, len(ix) - 1
    for i in range(nby):
        for j in range(nbx):
            idx[iy[i]:iy[i + 1], ix[j]:ix[j + 1]] = i * nbx + j
    idx = idx.reshape(-1)
    counts = np.bincount(idx, minlength=nby * nbx).astype(np.int64)
    if counts.min() < 1:
        raise SystemExit('pool geometry has an empty block')
    if int(counts.sum()) != native_h * native_w:
        raise SystemExit('pool geometry does not cover the native lattice')
    return idx, counts, nby, nbx


def pool_feature_by_geom(feature, geom):
    """Exact non-overlap block MEAN of a native-H/4 feature map (§6).

    ``feature`` is [B,C,native_h,native_w]; the partition is the SAME base-cell
    grouping that produced the oracle target, so the pooled tensor and q* share
    one geometry. Differentiable, exact, no Python block loop.
    """
    if tuple(feature.shape[-2:]) != tuple(geom['native_shape']):
        raise SystemExit('pool_feature_by_geom expects the native H/4 map %s, got %s'
                         % (tuple(geom['native_shape']), tuple(feature.shape[-2:])))
    nh, nw = int(geom['native_shape'][0]), int(geom['native_shape'][1])
    iy, ix = geom['base_index_edges']
    idx, counts, nby, nbx = _pool_index(nh, nw, tuple(int(v) for v in iy),
                                       tuple(int(v) for v in ix))
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=feature.device)
    cnt_t = torch.as_tensor(counts, dtype=feature.dtype, device=feature.device)
    b, c = feature.shape[0], feature.shape[1]
    flat = feature.reshape(b, c, -1)
    acc = feature.new_zeros((b, c, nby * nbx))
    acc.index_add_(2, idx_t, flat)
    return (acc / cnt_t.clamp(min=1.0)).reshape(b, c, nby, nbx)


def pool_geometry_audit(feature_h, feature_w, geom):
    """§7: no gap / no overlap / exact coverage of the native lattice."""
    iy, ix = geom['base_index_edges']
    idx, counts, nby, nbx = _pool_index(int(feature_h), int(feature_w),
                                       tuple(int(v) for v in iy),
                                       tuple(int(v) for v in ix))
    return dict(native_cells=int(feature_h) * int(feature_w), idx_size=int(idx.size),
                shape=[int(nby), int(nbx)], n_blocks=int(nby * nbx),
                min_block=int(counts.min()), max_block=int(counts.max()),
                total_cells=int(counts.sum()),
                all_cells_covered_once=bool(int(counts.sum()) == int(feature_h) * int(feature_w)),
                no_empty_block=bool(counts.min() >= 1))


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
    # §26: the V3-A.4.3 anchors are addressed THROUGH THE LOCK, so a run cannot
    # validate one tree while reading another
    v43 = lock.get('v3a43_root') or v43_root
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
        'v3a43_summary_sha256': os.path.join(v43, 'oracle', 'summary.json'),
        'v3a43_nesting_sha256': os.path.join(v43, 'oracle', 'nesting.json'),
        'v3a43_per_image_sha256': os.path.join(v43, 'oracle', 'per_image.csv'),
        'energy_stats_sha256': os.path.join(root, 'targets', 'energy_stats.json'),
        'geometry_sha256': os.path.join(root, 'targets', 'geometry.json'),
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


def validate_run_protocol(lock, formal, seed=None, steps=None, grad_accum=None,
                          variant=None, cache_name=None):
    """-> effective science parameters; hard-fails on a protocol change (§18-§24).

    A **formal** run (``limit 0``) takes seed / steps / grad_accum from the LOCK:
    a different CLI value is an error rather than a silent override, because the
    seed drives the image order, the state sequence and the shared init. A smoke
    run may vary those three, but ``variant`` / ``cache_name`` always come from
    the lock -- they select the actual reference set and Y0 cache, so changing
    them would silently change the experiment even in a smoke.
    """
    for name, got, want in (('--variant', variant, lock['reference_variant']),
                            ('--cache_name', cache_name, lock['cache_name'])):
        if got is not None and got != want:
            raise SystemExit('%s %r != the locked value %r -- the reference '
                             'variant / Y0 cache are part of the protocol'
                             % (name, got, want))
    eff = dict(formal=bool(formal), variant=lock['reference_variant'],
               cache_name=lock['cache_name'], seed=int(lock['seed']),
               steps=int(lock['steps']),
               grad_accum=int(lock['grad_accum_default']),
               factor=int(lock['block_h4_factor']),
               energy_threshold=float(lock['energy_threshold']),
               energy_pctl=float(lock['energy_pctl']))
    if formal:
        bad = []
        for name, got, want in (('--seed', seed, lock['seed']),
                                ('--steps', steps, lock['steps']),
                                ('--grad_accum', grad_accum,
                                 lock['grad_accum_default'])):
            if got is not None and int(got) != int(want):
                bad.append('%s %s != locked %s' % (name, got, want))
        if bad:
            raise SystemExit('a formal V3-A.5 run must use the locked protocol: '
                             + '; '.join(bad))
    else:
        for k, v in (('seed', seed), ('steps', steps), ('grad_accum', grad_accum)):
            if v is not None:
                eff[k] = int(v)
    return eff


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
