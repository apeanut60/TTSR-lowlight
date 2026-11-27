import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def validate_reference_indices(indices, n_ref):
    """Range check for indices that did not come from this module's top-k.

    One read of the min and max. Not used on the per-chunk gather path.
    """
    if indices.dtype != torch.long:
        raise TypeError(
            'reference indices must be torch.long, got %s' % indices.dtype)
    if indices.numel() == 0:
        return
    n_ref = int(n_ref)
    lo = int(indices.detach().min())
    hi = int(indices.detach().max())
    if lo < 0 or hi >= n_ref:
        raise IndexError(
            'reference index outside [0, %d); min=%d max=%d' % (n_ref, lo, hi))


def _weighted_gather_unchecked(value, indices, weights):
    """Mix ``k`` value patches. ``value`` is ``[B, D, Nr]``.

    No data-dependent scalar read. Caller guarantees ``indices`` are in range:
    internal top-k / max indices are, and external indices must pass
    ``validate_reference_indices`` first. Shape and dtype checks use Python
    metadata only.
    """
    if value.dim() != 3 or indices.dim() != 3 or weights.dim() != 3:
        raise ValueError(
            'gather expects value [B,D,Nr], indices/weights [B,k,N]')
    if indices.dtype != torch.long:
        raise TypeError(
            'gather indices must be torch.long, got %s' % indices.dtype)
    if indices.shape != weights.shape:
        raise ValueError(
            'indices %s and weights %s must match' % (
                tuple(indices.shape), tuple(weights.shape)))
    b, d, _nr = value.shape
    if int(indices.shape[0]) != int(b):
        raise ValueError(
            'gather batch %d != value batch %d' % (int(indices.shape[0]), int(b)))
    if value.device != indices.device or value.device != weights.device:
        raise ValueError('gather value, indices and weights must share a device')
    k = int(indices.shape[1])
    n = int(indices.shape[2])
    if n == 0:
        return value.new_zeros(b, d, 0)
    flat = indices.reshape(b, k * n)
    gathered = torch.gather(
        value, 2, flat.unsqueeze(1).expand(b, d, k * n))
    gathered = gathered.view(b, d, k, n)
    return (gathered * weights.unsqueeze(1)).sum(dim=2)


def _fold_with_overlap(patches, output_size, kernel, padding, stride):
    """Fold, then divide by a 1-channel geometric overlap count.

    The count is the number of query patches that cover each pixel. It is
    not the candidate count and not a sum of candidate weights. A 1-channel
    ones-fold matches one channel of ``fold(ones_like(unfold))``.
    """
    kk = int(kernel) * int(kernel)
    if int(patches.shape[1]) % kk != 0:
        raise RuntimeError(
            'unfold channels %d not divisible by kernel area %d' % (
                int(patches.shape[1]), kk))
    folded = F.fold(
        patches, output_size=output_size, kernel_size=kernel,
        padding=padding, stride=stride)
    ones = patches.new_ones(1, kk, patches.shape[2])
    norm = F.fold(
        ones, output_size=output_size, kernel_size=kernel,
        padding=padding, stride=stride)
    return folded / norm.clamp(min=1)


class SearchTransfer(nn.Module):
    """Texture search and transfer.

    ``hard_top1`` is the original path: one ``torch.max`` over reference patches.
    ``topk_soft`` mixes the top-k reference patches with a softmax over cosine
    similarity. ``k_eff == 1`` uses ``torch.max`` and weight 1, so it matches
    ``hard_top1`` including ties. For ``k > 1``, ``torch.topk`` chooses among
    ties by its own rule; that order is not claimed to be identical across
    devices.

    ``S`` remains the maximum raw cosine. Candidate weights are not a confidence.
    The three value scales share one lv3 index set and one weight vector.
    Identity transfer still reuses unfold / gather / fold with matching coordinates.
    """

    def __init__(self, matching_mode='hard_top1', match_topk=1,
                 match_temperature=0.1, match_query_chunk_size=128,
                 match_transfer_recompute=True):
        super(SearchTransfer, self).__init__()
        mode = str(matching_mode).strip().lower()
        if mode not in ('hard_top1', 'topk_soft'):
            raise ValueError(
                'matching_mode must be hard_top1 or topk_soft, got %r' % (
                    matching_mode,))
        self.matching_mode = mode
        self.match_topk = int(match_topk)
        self.match_temperature = float(match_temperature)
        self.match_query_chunk_size = int(match_query_chunk_size)
        self.match_transfer_recompute = bool(match_transfer_recompute)
        # Off in production. When on, the shared top-k index is range-checked
        # once, outside the chunk loop and outside activation checkpoint.
        self.debug_check_indices = False
        if self.matching_mode == 'topk_soft':
            self._validate_topk_config()

    def _validate_topk_config(self):
        if self.match_topk < 1:
            raise ValueError(
                'match_topk must be a positive integer, got %r' % (
                    self.match_topk,))
        temp = self.match_temperature
        if not math.isfinite(temp) or temp <= 0:
            raise ValueError(
                'match_temperature must be a finite positive number, got %r' % (
                    temp,))
        if self.match_query_chunk_size < 1:
            raise ValueError(
                'match_query_chunk_size must be >= 1, got %r' % (
                    self.match_query_chunk_size,))

    def bis(self, input, dim, index):
        # batch index select
        views = [input.size(0)] + [
            1 if i != dim else -1 for i in range(1, len(input.size()))
        ]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    @staticmethod
    def identity_match_arg(H_out, W_out, Hr, Wr, batch, device, dtype=torch.long):
        """lv3 Identity indices: Query (y,x) → Key (y,x) on the same grid.

        Requires identical lv3 spatial sizes (paired Low/Ref after preprocessing).
        Raises with a clear diagnostic otherwise — do not silently remap.
        """
        if H_out != Hr or W_out != Wr:
            raise ValueError(
                'Identity matching requires identical Query/Key lv3 shapes; '
                'got Query (%d,%d) vs Key (%d,%d). Ref must share the same '
                'preprocessed full-image / tile coordinates as Low.' % (
                    H_out, W_out, Hr, Wr))
        hw = H_out * W_out
        # Indices are always in [0, hw) — safe at borders for gather.
        return torch.arange(hw, device=device, dtype=dtype).view(1, hw).expand(
            batch, -1).contiguous()

    def transfer_from_arg(self, match_arg, ref_lv1, ref_lv2, ref_lv3,
                          H_out, W_out):
        """Gather/fold Value patches using precomputed lv3 match indices.

        Same unfold kernels, fold sizes, and per-pixel normalization as the
        global SearchTransfer path. ``match_arg`` shape: [N, H_out*W_out].
        """
        n = match_arg.size(0)
        hw = H_out * W_out
        if match_arg.dim() != 2 or match_arg.size(0) != n or match_arg.size(1) != hw:
            raise ValueError(
                'match_arg must be [N, H*W]=[%d, %d], got %s' % (
                    n, hw, tuple(match_arg.shape)))
        max_idx = int(match_arg.max().item()) if match_arg.numel() else -1
        # Key lv3 spatial size inferred from Value lv3 (1:1 with Key grid)
        Hr, Wr = ref_lv3.size()[-2:]
        if max_idx >= Hr * Wr:
            raise IndexError(
                'match_arg index %d out of range for Key/Value lv3 (%d×%d=%d)' % (
                    max_idx, Hr, Wr, Hr * Wr))

        ref_lv3_unfold = F.unfold(
            ref_lv3, kernel_size=(3, 3), padding=1, stride=1)
        ref_lv2_unfold = F.unfold(
            ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
        ref_lv1_unfold = F.unfold(
            ref_lv1, kernel_size=(12, 12), padding=4, stride=4)

        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, match_arg)
        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, match_arg)
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, match_arg)

        H2, W2 = H_out * 2, W_out * 2
        H1, W1 = H_out * 4, W_out * 4

        T_lv3 = F.fold(T_lv3_unfold, output_size=(H_out, W_out),
                       kernel_size=(3, 3), padding=1, stride=1)
        norm_lv3 = F.fold(torch.ones_like(T_lv3_unfold),
                          output_size=(H_out, W_out),
                          kernel_size=(3, 3), padding=1, stride=1)
        T_lv3 = T_lv3 / norm_lv3.clamp(min=1)

        T_lv2 = F.fold(T_lv2_unfold, output_size=(H2, W2),
                       kernel_size=(6, 6), padding=2, stride=2)
        norm_lv2 = F.fold(torch.ones_like(T_lv2_unfold),
                          output_size=(H2, W2),
                          kernel_size=(6, 6), padding=2, stride=2)
        T_lv2 = T_lv2 / norm_lv2.clamp(min=1)

        T_lv1 = F.fold(T_lv1_unfold, output_size=(H1, W1),
                       kernel_size=(12, 12), padding=4, stride=4)
        norm_lv1 = F.fold(torch.ones_like(T_lv1_unfold),
                          output_size=(H1, W1),
                          kernel_size=(12, 12), padding=4, stride=4)
        T_lv1 = T_lv1 / norm_lv1.clamp(min=1)

        return T_lv3, T_lv2, T_lv1

    def transfer_identity(self, ref_lv1, ref_lv2, ref_lv3, H_out, W_out,
                          Hr=None, Wr=None):
        """Identity texture transfer from Reference Value LTE features."""
        if Hr is None or Wr is None:
            Hr, Wr = ref_lv3.size()[-2:]
        n = ref_lv3.size(0)
        match_arg = self.identity_match_arg(
            H_out, W_out, Hr, Wr, n, ref_lv3.device, dtype=torch.long)
        return self.transfer_from_arg(
            match_arg, ref_lv1, ref_lv2, ref_lv3, H_out, W_out)

    def forward(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3,
                return_diag=False, return_match_candidates=False):
        if self.matching_mode == 'hard_top1':
            return self._forward_hard_top1(
                lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3,
                return_diag=return_diag)
        return self._forward_topk_soft(
            lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3,
            return_diag=return_diag,
            return_match_candidates=return_match_candidates)

    def _forward_hard_top1(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3,
                           return_diag=False):
        """
        Args:
            lrsr_lv3: [N, C, H, W] features of low-light image (VGG lv3) — Query
            refsr_lv3: [N, C, Hr, Wr] features of reference image (VGG lv3) — Key
            ref_lv1/2/3: reference Value features
            return_diag: if True, also return a diagnostics dict (does not change
                the primary return tuple used by MainNet).
        Returns:
            S, T_lv3, T_lv2, T_lv1  (and diag if return_diag)
        """
        H_out, W_out = lrsr_lv3.size()[-2:]
        Hr, Wr = refsr_lv3.size()[-2:]

        # ═══ Search (unchanged) ═══
        lrsr_lv3_unfold = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)

        refsr_lv3_unfold = F.normalize(
            refsr_lv3_unfold, dim=2)  # [N, Hr*Wr, C*k*k]
        lrsr_lv3_unfold = F.normalize(
            lrsr_lv3_unfold, dim=1)    # [N, C*k*k, H*W]

        R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold)  # [N, Hr*Wr, H*W]
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1)  # [N, H*W]

        diag = None
        if return_diag:
            diag = self._match_diagnostics(
                R_lv3_star, R_lv3_star_arg, R_lv3,
                H_out, W_out, Hr, Wr, lrsr_lv3, refsr_lv3)
            # Extra fields for offline diagnostics (v1r8); not used in train/eval.
            diag['match_arg'] = R_lv3_star_arg.detach().cpu()  # [N, H*W]
            diag['S_raw'] = R_lv3_star.detach().cpu()          # [N, H*W]
            diag['lv3_hw'] = (int(H_out), int(W_out), int(Hr), int(Wr))
            # Free dense similarity matrix before fold/transfer.
            del R_lv3

        # ═══ Transfer (adapted for 1:1 same resolution) ═══
        T_lv3, T_lv2, T_lv1 = self.transfer_from_arg(
            R_lv3_star_arg, ref_lv1, ref_lv2, ref_lv3, H_out, W_out)

        S = R_lv3_star.view(R_lv3_star.size(0), 1, H_out, W_out)

        if return_diag:
            return S, T_lv3, T_lv2, T_lv1, diag
        return S, T_lv3, T_lv2, T_lv1

    def _check_topk_grids(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3):
        if lrsr_lv3.dim() != 4 or refsr_lv3.dim() != 4:
            raise ValueError('Query and Key must be NCHW')
        if int(lrsr_lv3.shape[0]) != int(refsr_lv3.shape[0]):
            raise ValueError(
                'Query/Key batch mismatch %s vs %s' % (
                    tuple(lrsr_lv3.shape), tuple(refsr_lv3.shape)))
        if int(lrsr_lv3.shape[1]) != int(refsr_lv3.shape[1]):
            raise ValueError(
                'Query/Key channel mismatch %s vs %s' % (
                    tuple(lrsr_lv3.shape), tuple(refsr_lv3.shape)))
        hr, wr = int(refsr_lv3.shape[-2]), int(refsr_lv3.shape[-1])
        if tuple(ref_lv3.shape[-2:]) != (hr, wr):
            raise ValueError(
                'ref lv3 spatial %s != Key %s; refusing to truncate or reorder'
                % (tuple(ref_lv3.shape[-2:]), (hr, wr)))
        if tuple(ref_lv2.shape[-2:]) != (hr * 2, wr * 2):
            raise ValueError(
                'ref lv2 spatial %s != 2x Key %s' % (
                    tuple(ref_lv2.shape[-2:]), (hr, wr)))
        if tuple(ref_lv1.shape[-2:]) != (hr * 4, wr * 4):
            raise ValueError(
                'ref lv1 spatial %s != 4x Key %s' % (
                    tuple(ref_lv1.shape[-2:]), (hr, wr)))
        b = int(lrsr_lv3.shape[0])
        for name, ref in (('lv1', ref_lv1), ('lv2', ref_lv2), ('lv3', ref_lv3)):
            if int(ref.shape[0]) != b:
                raise ValueError(
                    'ref %s batch %d != Query batch %d' % (
                        name, int(ref.shape[0]), b))
        if hr < 1 or wr < 1 or int(lrsr_lv3.shape[-2]) < 1:
            raise ValueError('empty Query/Key grid')

    def _chunked_topk(self, q_norm, k_norm, k_eff):
        """Top-k over reference positions, one query chunk at a time.

        Does not concatenate the full ``[B, Nr, Nq]`` score matrix.
        ``k_eff == 1`` uses ``torch.max`` so ties match hard top-1.
        """
        nq = int(q_norm.shape[-1])
        chunk = int(self.match_query_chunk_size)
        scores_l, idx_l, s_l, arg_l = [], [], [], []
        for start in range(0, nq, chunk):
            q = q_norm[:, :, start:start + chunk]
            scores = torch.bmm(k_norm, q)  # [B, Nr, n]
            n = int(scores.shape[-1])
            if k_eff == 1:
                val, arg = torch.max(scores, dim=1)
                scores_l.append(val.unsqueeze(1))
                idx_l.append(arg.unsqueeze(1))
            else:
                val, arg = torch.topk(scores, k_eff, dim=1)
                scores_l.append(val)
                idx_l.append(arg)
            smax, sarg = torch.max(scores, dim=1)
            s_l.append(smax)
            arg_l.append(sarg)
            del scores
        top_scores = torch.cat(scores_l, dim=2)
        top_idx = torch.cat(idx_l, dim=2)
        s_raw = torch.cat(s_l, dim=1)
        top1_arg = torch.cat(arg_l, dim=1)
        return top_scores, top_idx, s_raw, top1_arg

    def _mix_scale(self, ref, indices, weights, kernel, padding, stride,
                   out_hw, n_ref):
        unfolded = F.unfold(
            ref, kernel_size=kernel, padding=padding, stride=stride)
        if int(unfolded.shape[-1]) != int(n_ref):
            raise ValueError(
                'value unfold has %d patches, Key has %d; refusing to '
                'truncate or reorder reference features' % (
                    int(unfolded.shape[-1]), int(n_ref)))
        # Key count is the lv3 grid; every scale must expose that many patches.
        nq = int(indices.shape[-1])
        parts = []
        chunk = int(self.match_query_chunk_size)
        for start in range(0, nq, chunk):
            idx = indices[:, :, start:start + chunk]
            w = weights[:, :, start:start + chunk]
            needs_grad = torch.is_grad_enabled() and (
                unfolded.requires_grad or w.requires_grad)
            if needs_grad and self.match_transfer_recompute:
                # Bind this chunk's tensors as arguments. Do not close over
                # the loop index, or backward would replay the last chunk.
                # The unfolded map is an input of this recompute, so it can
                # stay live until backward. Chunks share that one storage.
                part = checkpoint(_weighted_gather_unchecked, unfolded, idx, w)
            else:
                part = _weighted_gather_unchecked(unfolded, idx, w)
            parts.append(part)
        mixed = torch.cat(parts, dim=2)
        return _fold_with_overlap(
            mixed, out_hw, kernel, padding, stride)

    def _forward_topk_soft(
            self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3,
            return_diag=False, return_match_candidates=False):
        """Top-k soft transfer. ``S`` stays the max raw cosine.

        For ``k > 1``, ``torch.topk`` breaks ties by its own rule. That order
        is not claimed to match ``torch.max`` or to be identical across devices.
        ``k_eff = 1`` uses ``torch.max`` and weight 1, matching hard top-1.
        """
        self._validate_topk_config()
        self._check_topk_grids(
            lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)
        h_out, w_out = lrsr_lv3.shape[-2:]
        hr, wr = refsr_lv3.shape[-2:]
        q_unf = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
        k_unf = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)
        k_unf = k_unf.permute(0, 2, 1)
        if int(q_unf.shape[-1]) != int(h_out) * int(w_out):
            raise ValueError(
                'Query unfold columns %d != H*W %d' % (
                    int(q_unf.shape[-1]), int(h_out) * int(w_out)))
        if int(k_unf.shape[1]) != int(hr) * int(wr):
            raise ValueError(
                'Key unfold rows %d != Hr*Wr %d' % (
                    int(k_unf.shape[1]), int(hr) * int(wr)))
        q_norm = F.normalize(q_unf, dim=1)
        k_norm = F.normalize(k_unf, dim=2)
        nr = int(k_norm.shape[1])
        k_eff = min(int(self.match_topk), nr)
        top_scores, top_idx, s_raw, top1_arg = self._chunked_topk(
            q_norm, k_norm, k_eff)
        shifted = top_scores - top_scores.max(dim=1, keepdim=True).values
        weights = torch.softmax(shifted / self.match_temperature, dim=1)
        if self.debug_check_indices:
            validate_reference_indices(top_idx, nr)
        # Scales run in order so the k-expanded candidate tensor of one scale
        # is folded before the next scale is gathered. That does not mean the
        # three source unfolds are gone: with grad + checkpoint each unfolded
        # value is saved as a recompute input until backward. Chunks of one
        # scale reference that single storage; the reference count is not a
        # second copy. What is avoided is the full [B, D, k, Nq] candidate.
        t_lv3 = self._mix_scale(
            ref_lv3, top_idx, weights, 3, 1, 1,
            (int(h_out), int(w_out)), nr)
        t_lv2 = self._mix_scale(
            ref_lv2, top_idx, weights, 6, 2, 2,
            (int(h_out) * 2, int(w_out) * 2), nr)
        t_lv1 = self._mix_scale(
            ref_lv1, top_idx, weights, 12, 4, 4,
            (int(h_out) * 4, int(w_out) * 4), nr)
        s = s_raw.view(s_raw.shape[0], 1, int(h_out), int(w_out))
        if not return_diag:
            return s, t_lv3, t_lv2, t_lv1
        w_det = weights.detach()
        ent = -(w_det.clamp_min(1e-12).log() * w_det).sum(dim=1)
        diag = {
            'matching_mode': 'topk_soft',
            'k_configured': int(self.match_topk),
            'k_eff': int(k_eff),
            'temperature': float(self.match_temperature),
            'top1_raw_cosine_mean': float(s_raw.detach().mean()),
            'max_candidate_weight_mean': float(w_det.max(dim=1).values.mean()),
            'candidate_entropy_mean': float(ent.mean()),
            'T_shapes': {
                'lv3': tuple(int(x) for x in t_lv3.shape),
                'lv2': tuple(int(x) for x in t_lv2.shape),
                'lv1': tuple(int(x) for x in t_lv1.shape),
            },
            'lv3_hw': (int(h_out), int(w_out), int(hr), int(wr)),
            # Top-1 index from torch.max (same rule as hard top-1). Candidate
            # weights are not a calibrated confidence, and S is still the max
            # cosine, not a softmax weight.
            'match_arg': top1_arg.detach().cpu(),
            'S_raw': s_raw.detach().cpu(),
        }
        if return_match_candidates:
            diag['topk_scores'] = top_scores.detach()
            diag['topk_indices'] = top_idx.detach()
            diag['topk_weights'] = w_det
        return s, t_lv3, t_lv2, t_lv1, diag

    @staticmethod
    def _match_diagnostics(R_star, R_arg, R_full, H, W, Hr, Wr,
                           q_feat, k_feat):
        """Auxiliary matching stats. Identity only if HxW == Hr x Wr."""
        n = R_star.shape[0]
        hw = H * W
        # max cosine
        max_sim = R_star.detach()
        qs = [0.1, 0.5, 0.9]
        quantiles = torch.quantile(
            max_sim.float().view(-1),
            torch.tensor(qs, device=max_sim.device)).detach().cpu().tolist()

        # feature norms
        q_norm = q_feat.detach().flatten(2).norm(dim=1).mean().item()
        k_norm = k_feat.detach().flatten(2).norm(dim=1).mean().item()

        out = {
            'max_cos_mean': float(max_sim.mean().item()),
            'max_cos_std': float(max_sim.std(unbiased=False).item()),
            'max_cos_q10': float(quantiles[0]),
            'max_cos_q50': float(quantiles[1]),
            'max_cos_q90': float(quantiles[2]),
            'query_feat_norm_mean': q_norm,
            'key_feat_norm_mean': k_norm,
            'identity_valid': False,
            'identity_cos_mean': None,
            'match_l2_mean': None,
            'frac_dist_le_1': None,
            'frac_dist_le_2': None,
            'frac_dist_le_4': None,
            'frac_dist_le_8': None,
        }

        if H == Hr and W == Wr:
            # Identity index for query position i is i (same grid).
            idx = torch.arange(hw, device=R_arg.device).view(1, hw).expand(n, -1)
            # Gather identity cosine: R_full[b, i, i]
            # R_full: [N, Hr*Wr, H*W] with dim1=key, dim2=query
            id_cos = R_full.detach().gather(1, idx.unsqueeze(1)).squeeze(1)
            # Spatial distance between best key and identity
            best = R_arg.detach()
            by = torch.div(best, W, rounding_mode='floor')
            bx = best % W
            iy = torch.div(idx, W, rounding_mode='floor')
            ix = idx % W
            dist = torch.sqrt((by.float() - iy.float()).pow(2) +
                              (bx.float() - ix.float()).pow(2))
            out.update({
                'identity_valid': True,
                'identity_cos_mean': float(id_cos.mean().item()),
                'match_l2_mean': float(dist.mean().item()),
                'frac_dist_le_1': float((dist <= 1).float().mean().item()),
                'frac_dist_le_2': float((dist <= 2).float().mean().item()),
                'frac_dist_le_4': float((dist <= 4).float().mean().item()),
                'frac_dist_le_8': float((dist <= 8).float().mean().item()),
            })
        return out
