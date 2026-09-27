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

ORACLE_DEF_VERSION = ('v3a42b:blockwise-piecewise-constant-nearest'
                      '+block-h4-endpoint')


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
def _block_index_map_cached(height, width, gy, gx):
    ey = block_edges(height, gy)
    ex = block_edges(width, gx)
    nby, nbx = len(ey) - 1, len(ex) - 1
    idx = np.empty((height, width), dtype=np.int64)
    for i in range(nby):
        for j in range(nbx):
            idx[ey[i]:ey[i + 1], ex[j]:ex[j + 1]] = i * nbx + j
    # NOTE: this cached map is read-only *by convention*; `block_index_map`
    # hands out a copy so no caller can turn it into mutable shared state.
    return idx.reshape(-1), nby, nbx


def block_index_map(height, width, grid_size):
    """-> (idx [H*W] int64 with values 0..nby*nbx-1, nby, nbx).

    Every pixel is assigned exactly one block; no gaps, no overlap. The result
    is a private copy: the cached map must not become mutable shared state.
    ``grid_size`` is an int or a (Gy, Gx) pair.
    """
    gy, gx = as_grid_pair(grid_size)
    idx, nby, nbx = _block_index_map_cached(int(height), int(width), gy, gx)
    return idx.copy(), nby, nbx


# ── §4 blockwise oracle ─────────────────────────────────────────────────────

def blockwise_action_optimal_gate(y0, hr, d_correction, grid_size, eps=1e-8):
    """-> dict(q_grid, q_full, Y_oracle, N_grid, Z_grid, nby, nbx).

    ``d_correction`` must already include the proposal gate (g_v2 * delta).
    ``grid_size`` is an int (square grid) or a (Gy, Gx) pair.
    """
    if y0.shape != hr.shape or y0.shape != d_correction.shape:
        raise ValueError('y0/hr/D must share a shape, got %s %s %s'
                         % (tuple(y0.shape), tuple(hr.shape),
                            tuple(d_correction.shape)))
    b, _c, h, w = y0.shape
    N = ((hr - y0) * d_correction).sum(dim=1, keepdim=True)     # [B,1,H,W]
    Z = (d_correction ** 2).sum(dim=1, keepdim=True)
    gy, gx = as_grid_pair(grid_size)
    # hot path: read the cached (read-only) map instead of rebuilding it
    idx, nby, nbx = _block_index_map_cached(h, w, gy, gx)
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


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
