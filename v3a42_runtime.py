"""V3-A.4.2 runtime: TRUE blockwise gate-resolution oracle.

Question: how coarse can the gate be and still capture most of the H/4 spatial
oracle headroom?

The coarse gate must be the *optimum under a blockwise-constant parameterisation*,
not a smoothed version of the dense one. Averaging or resizing ``q_spatial``
produces a different (and non-optimal) gate -- that is explicitly forbidden.

For one block B the optimum is closed-form:

    q_B = clip( sum_{(x,y) in B, c} (H-Y0)*D  /  ( sum_B D^2 + eps ), 0, 1 )

and the gate is piecewise-constant (nearest block assignment, no interpolation).

The oracle minimises the *continuous* RGB MSE on the [-1,1] tensors. What the
project reports (`local_refine_runtime.metrics`) is PSNR on 8-bit rounded
images, which is not exactly the same objective; the difference is far below
the resolution effects measured here, but the distinction is kept explicit.
"""

import json
import os
import subprocess
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

ORACLE_DEF_VERSION = ('v3a42c:blockwise-piecewise-constant-nearest'
                      '+nested-base-h4-grid')


# ── §6 deterministic, gap-free block partition ──────────────────────────────

def block_edges(length, grid):
    """Edges via round(linspace(0, L, G+1)); deduplicated so every block has at
    least one pixel and the whole range is covered exactly once.

    Deliberately NOT ``L // G`` (which drops the remainder) and deliberately not
    adaptive-pool boundaries (which overlap when L is not divisible by G).
    """
    if grid <= 1:
        return np.array([0, int(length)])
    e = np.round(np.linspace(0.0, float(length), int(grid) + 1)).astype(np.int64)
    e = np.unique(e)                       # collapses to <G blocks if L < G
    if e.size < 2 or e[0] != 0 or e[-1] != int(length):
        e = np.array([0, int(length)])
    return e


def as_grid_pair(grid_size):
    """G -> (G, G); (Gy, Gx) -> (Gy, Gx). Both must be >= 1.

    A rectangular pair is needed for the H/4 endpoint: 400x600 gives a
    100 x 150 cell grid, which is H/4 on BOTH axes and therefore the same
    resolution as the legacy H/4 spatial oracle.
    """
    if isinstance(grid_size, (tuple, list)):
        gy, gx = int(grid_size[0]), int(grid_size[1])
    else:
        gy = gx = int(grid_size)
    return max(1, gy), max(1, gx)


def h4_block_grid(height, width, factor=4):
    """The (Gy, Gx) pair whose cells are one H/factor x W/factor sample."""
    return (max(1, int(height) // factor), max(1, int(width) // factor))


@lru_cache(maxsize=256)
def _index_map_from_edges(height, width, ey, ex):
    """Cached on the edge tuples; the returned map is read-only by convention."""
    nby, nbx = len(ey) - 1, len(ex) - 1
    idx = np.empty((height, width), dtype=np.int64)
    for i in range(nby):
        for j in range(nbx):
            idx[ey[i]:ey[i + 1], ex[j]:ex[j + 1]] = i * nbx + j
    return idx.reshape(-1), nby, nbx


def _index_map(height, width, ey, ex):
    return _index_map_from_edges(int(height), int(width),
                                 tuple(int(v) for v in ey),
                                 tuple(int(v) for v in ex))


def _block_index_map_cached(height, width, gy, gx):
    return _index_map(height, width, block_edges(height, gy), block_edges(width, gx))


def block_index_map(height, width, grid_size):
    """-> (idx [H*W] int64 with values 0..nby*nbx-1, nby, nbx).

    Every pixel is assigned exactly one block; no gaps, no overlap. The result
    is a private copy: the cached map must not become mutable shared state.
    ``grid_size`` is an int or a (Gy, Gx) pair.

    The edges come from the raw pixel lattice. Use
    ``block_index_map_from_base_grid`` when the coarse blocks must be exact
    unions of a finer base partition.
    """
    gy, gx = as_grid_pair(grid_size)
    idx, nby, nbx = _block_index_map_cached(int(height), int(width), gy, gx)
    return idx.copy(), nby, nbx


def coarse_edges_from_base(length, base_grid, coarse_grid):
    """Pixel edges of a coarse partition whose blocks are WHOLE base cells.

    ``round(linspace(0, n_base, G+1))`` picks base-cell boundaries, which are
    then mapped to pixel coordinates. Building the G grid independently on the
    pixel lattice instead would let a G block cut a base cell in half -- then a
    finer endpoint is not a superset of the coarser family and
    ``PSNR(G) > PSNR(finer)`` becomes possible for a pure *boundary alignment*
    reason rather than a resolution one.
    """
    ey_base = block_edges(length, base_grid)
    n_base = len(ey_base) - 1
    if coarse_grid <= 1:
        return np.array([0, int(length)])
    iy = np.round(np.linspace(0.0, float(n_base),
                              int(coarse_grid) + 1)).astype(np.int64)
    iy = np.unique(np.clip(iy, 0, n_base))
    if iy.size < 2 or iy[0] != 0 or iy[-1] != n_base:
        iy = np.array([0, n_base])
    return ey_base[iy]


def block_index_map_from_base_grid(height, width, coarse_grid, base_grid):
    """Same as ``block_index_map`` but every coarse block is a union of base
    cells: F_G is a subset of F_base, so MSE(F_base) <= MSE(F_G)."""
    by, bx = as_grid_pair(base_grid)
    cy, cx = as_grid_pair(coarse_grid)
    idx, nby, nbx = _index_map(height, width,
                               coarse_edges_from_base(height, by, cy),
                               coarse_edges_from_base(width, bx, cx))
    return idx.copy(), nby, nbx


# ── §4 blockwise oracle ─────────────────────────────────────────────────────

def blockwise_action_optimal_gate(y0, hr, d_correction, grid_size, eps=1e-8,
                                  base_grid=None):
    """-> dict(q_grid, q_full, Y_oracle, N_grid, Z_grid, nby, nbx).

    ``d_correction`` must already include the proposal gate (g_v2 * delta).
    ``grid_size`` is an int (square grid) or a (Gy, Gx) pair.
    ``base_grid`` -- when given, the blocks are unions of that finer grid's
    cells (see ``coarse_edges_from_base``); the default builds the partition
    directly on the pixel lattice.
    """
    if y0.shape != hr.shape or y0.shape != d_correction.shape:
        raise ValueError('y0/hr/D must share a shape, got %s %s %s'
                         % (tuple(y0.shape), tuple(hr.shape),
                            tuple(d_correction.shape)))
    b, _c, h, w = y0.shape
    N = ((hr - y0) * d_correction).sum(dim=1, keepdim=True)     # [B,1,H,W]
    Z = (d_correction ** 2).sum(dim=1, keepdim=True)
    gy, gx = as_grid_pair(grid_size)
    if base_grid is None:
        # hot path: read the cached (read-only) map instead of rebuilding it
        idx, nby, nbx = _block_index_map_cached(h, w, gy, gx)
    else:
        by, bx = as_grid_pair(base_grid)
        idx, nby, nbx = _index_map(h, w,
                                   coarse_edges_from_base(h, by, gy),
                                   coarse_edges_from_base(w, bx, gx))
    nblk = nby * nbx
    t = torch.as_tensor(idx, dtype=torch.long, device=y0.device)
    Nf = N.reshape(b, -1)
    Zf = Z.reshape(b, -1)
    N_grid = torch.zeros(b, nblk, dtype=N.dtype, device=y0.device)
    Z_grid = torch.zeros(b, nblk, dtype=Z.dtype, device=y0.device)
    N_grid.index_add_(1, t, Nf)
    Z_grid.index_add_(1, t, Zf)
    q_grid = (N_grid / (Z_grid + eps)).clamp(0.0, 1.0)
    q_full = q_grid.gather(1, t.unsqueeze(0).expand(b, -1)).reshape(b, 1, h, w)
    return dict(q_grid=q_grid, q_full=q_full, Y_oracle=y0 + q_full * d_correction,
                N_grid=N_grid, Z_grid=Z_grid, nby=nby, nbx=nbx)


def global_action_optimal_gate_reference(y0, hr, d_correction, eps=1e-8):
    """The V3-A.4.1 definition, kept here so grid=1 can be checked against it."""
    N = ((hr - y0) * d_correction).sum(dim=(1, 2, 3), keepdim=True)
    Z = (d_correction ** 2).sum(dim=(1, 2, 3), keepdim=True)
    return (N / (Z + eps)).clamp(0.0, 1.0), N, Z


def spatial_oracle_q4(y0, hr, d_correction, factor=4, eps=1e-8):
    """The V3-A.4.1 dense oracle *at its own resolution* H/factor.

    Returned separately from the upsampled gate so that gate statistics describe
    the map the oracle is actually parameterised by, not the interpolated one.
    """
    h, w = y0.shape[-2:]
    th, tw = max(1, h // factor), max(1, w // factor)
    y4 = F.interpolate(y0, size=(th, tw), mode='area')
    h4 = F.interpolate(hr, size=(th, tw), mode='area')
    d4 = F.interpolate(d_correction, size=(th, tw), mode='area')
    return (((h4 - y4) * d4).sum(dim=1, keepdim=True)
            / ((d4 ** 2).sum(dim=1, keepdim=True) + eps)).clamp(0.0, 1.0)


def spatial_action_optimal_gate(y0, hr, d_correction, factor=4, eps=1e-8):
    """The V3-A.4.1 H/4 upper bound (bilinear upsample of the H/4 oracle)."""
    h, w = y0.shape[-2:]
    q4 = spatial_oracle_q4(y0, hr, d_correction, factor=factor, eps=eps)
    q = F.interpolate(q4, size=(h, w), mode='bilinear', align_corners=False)
    return q


# ── §20 artifact lock ───────────────────────────────────────────────────────

def verify_v3a42_artifact_lock(root, src_root,
                               ckpt_name='R1_v2stable_naive_s42/checkpoint_03000.pt',
                               cache_name='cache_y0_lolbase',
                               v4_root='/root/data/experiments/v3a4_lolv2real'):
    path = os.path.join(root, 'artifact_lock.json')
    if not os.path.isfile(path):
        raise SystemExit('v3a42 artifact lock missing: %s' % path)
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
            raise SystemExit('v3a42 lock missing %s' % k)
        if not os.path.isfile(p):
            raise SystemExit('locked artifact not found: %s' % p)
        if _sha256(p) != lock[k]:
            bad.append(k)
    if bad:
        raise SystemExit('v3a42 lock mismatch on %s -- rerun '
                         'scripts/setup_v3a42_oracle.py' % ', '.join(bad))
    if lock.get('oracle_def_version') != ORACLE_DEF_VERSION:
        raise SystemExit('oracle definition changed since the lock (%s vs %s)'
                         % (lock.get('oracle_def_version'), ORACLE_DEF_VERSION))
    head, dirty = git_state()
    if head and lock.get('repo_commit') and lock['repo_commit'] != head:
        raise SystemExit('v3a42 lock generated at %s but HEAD is %s -- rerun '
                         'scripts/setup_v3a42_oracle.py'
                         % (lock['repo_commit'][:8], head[:8]))
    if dirty:
        # The lock pins repo_commit, so a dirty tree means the recorded commit
        # is NOT the code that ran. Not fatal (the smoke tests run from a dirty
        # tree by construction), but it must be visible in the run log.
        print('[v3a42] WARNING: working tree is dirty (%d path(s), e.g. %s) -- '
              'the run cannot be attributed to %s alone'
              % (len(dirty), ', '.join(dirty[:3]), (head or 'HEAD')[:8]))
    return lock


def git_state(repo_dir=None):
    """-> (HEAD sha or None, sorted dirty paths). Read-only provenance helper."""
    repo = repo_dir or os.path.dirname(os.path.abspath(__file__))
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo,
                                       stderr=subprocess.DEVNULL).decode().strip()
        out = subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo,
                                      stderr=subprocess.DEVNULL).decode()
    except Exception:                                            # noqa: BLE001
        return None, []
    dirty = sorted({l[3:].strip() for l in out.splitlines() if l.strip()})
    return head, dirty


def check_v3a41_reproduction(got_state, ref_state, tol_capture=1e-3,
                             tol_psnr=2e-3):
    """Compare one (split, state) slice against the V3-A.4.1 baseline.

    WHICH CAPTURE: V3-A.4.1 defined ``capture_global = H_global / H_spatial``
    where its spatial oracle is the LEGACY area-downsample + bilinear one.
    V3-A.4.2's primary ``capture`` uses the same-family Block_H4 denominator
    instead, so the two numbers are different by construction. The historical
    comparison must therefore use ``capture_legacy`` -- comparing ``capture``
    would hard-fail a correct sweep. ``denominator`` is recorded in the row so
    the artifact says which number was actually used.

    ``got_state`` is one entry of our ``summary['splits'][split]``;
    ``ref_state`` is one entry of the V3-A.4.1 ``global_vs_spatial.json``.
    """
    got_legacy = got_state['capture']['G1'].get('capture_legacy')
    if got_legacy is None:
        raise SystemExit('reproduction needs H_LegacyH4 > 0 to define '
                         'capture_legacy; got None')
    d_psnr = abs(got_state['mean_psnr']['G1'] - ref_state['Global_AO'])
    d_spat = abs(got_state['mean_psnr']['Spatial_H4_legacy']
                 - ref_state['Spatial_AO'])
    d_cap = abs(got_legacy - ref_state['capture_global'])
    mism = []
    if d_psnr > tol_psnr:
        mism.append('psnr_G1')
    if d_spat > tol_psnr:
        mism.append('psnr_Spatial_H4_legacy')
    if d_cap > tol_capture:
        mism.append('capture_legacy')
    return dict(
        denominator='capture_legacy',
        psnr_G1=got_state['mean_psnr']['G1'],
        psnr_G1_v3a41=ref_state['Global_AO'], d_psnr=d_psnr,
        psnr_Spatial_H4_legacy=got_state['mean_psnr']['Spatial_H4_legacy'],
        psnr_Spatial_H4_v3a41=ref_state['Spatial_AO'], d_psnr_spatial=d_spat,
        capture_G1_legacy=got_legacy,
        capture_v3a41=ref_state['capture_global'], d_capture=d_cap,
        # informational only -- NOT what the historical capture is compared to
        capture_G1_primary=got_state['capture']['G1'].get('capture'),
        H_S_block_h4=got_state.get('H_S_block_h4'),
        H_S_legacy_h4=got_state.get('H_S_legacy_h4'),
        tol_capture=tol_capture, tol_psnr=tol_psnr,
        ok=not mism, mismatches=mism)


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
