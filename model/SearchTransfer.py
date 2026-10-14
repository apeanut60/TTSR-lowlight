import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SearchTransfer(nn.Module):
    """Texture search and transfer module.
    Adapted for 1:1 low-light enhancement:
    - Search: patch matching at lv3 scale (unchanged)
    - Transfer: unfold with stride matching VGG scale ratio,
      fold back to each level's native resolution.
      lv3 → H/4×W/4, lv2 → H/2×W/2, lv1 → H×W
      MainNetEnhance interpolates all to unified feature size.

    Optional eval-only Identity transfer reuses the same unfold/gather/fold
    path with spatially corresponding indices (not RGB interpolation).
    """

    def __init__(self):
        super(SearchTransfer, self).__init__()

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
