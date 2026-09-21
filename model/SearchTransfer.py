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
    """

    def __init__(self):
        super(SearchTransfer, self).__init__()

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [
            1 if i != dim else -1 for i in range(1, len(input.size()))
        ]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3):
        """
        Args:
            lrsr_lv3: [N, C, H, W] features of low-light image (VGG lv3)
            refsr_lv3: [N, C, Hr, Wr] features of reference image (VGG lv3)
            ref_lv1: [N, 64, H1, W1] VGG shallow features
            ref_lv2: [N, 128, H2, W2] VGG mid features
            ref_lv3: [N, 256, H3, W3] VGG deep features
        Returns:
            S: [N, 1, H, W] soft attention map (lv3 resolution)
            T_lv3: [N, 256, H, W] transferred texture at lv3 resolution
            T_lv2: [N, 128, 2H, 2W] transferred texture at lv2 resolution
            T_lv1: [N, 64, 4H, 4W] transferred texture at lv1 resolution
        """
        H_out, W_out = lrsr_lv3.size()[-2:]

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

        # ═══ Transfer (adapted for 1:1 same resolution) ═══
        # VGG feature scales: lv3 @ H/4, lv2 @ H/2, lv1 @ H
        # unfold with matching stride so each level produces the same #patches as lv3
        ref_lv3_unfold = F.unfold(
            ref_lv3, kernel_size=(3, 3), padding=1, stride=1)
        # lv2 is 2x lv3 resolution → stride=2
        ref_lv2_unfold = F.unfold(
            ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
        # lv1 is 4x lv3 resolution → stride=4
        ref_lv1_unfold = F.unfold(
            ref_lv1, kernel_size=(12, 12), padding=4, stride=4)

        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_star_arg)
        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv3_star_arg)
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv3_star_arg)

        # Fold back to NATIVE resolution of each VGG level
        # lv1 @ H×W, lv2 @ H/2×W/2, lv3 @ H/4×W/4
        # MainNetEnhance will interpolate all to unified size via bicubic
        # Use per-pixel normalization mask (fold of all-ones) instead of /k²
        # because stride>1 creates uneven pixel coverage (2.2× variation)
        H2, W2 = H_out * 2, W_out * 2
        H1, W1 = H_out * 4, W_out * 4

        T_lv3 = F.fold(T_lv3_unfold, output_size=(H_out, W_out),
                       kernel_size=(3, 3), padding=1, stride=1)
        norm_lv3 = F.fold(torch.ones_like(T_lv3_unfold), output_size=(H_out, W_out),
                          kernel_size=(3, 3), padding=1, stride=1)
        T_lv3 = T_lv3 / norm_lv3.clamp(min=1)

        T_lv2 = F.fold(T_lv2_unfold, output_size=(H2, W2),
                       kernel_size=(6, 6), padding=2, stride=2)
        norm_lv2 = F.fold(torch.ones_like(T_lv2_unfold), output_size=(H2, W2),
                          kernel_size=(6, 6), padding=2, stride=2)
        T_lv2 = T_lv2 / norm_lv2.clamp(min=1)

        T_lv1 = F.fold(T_lv1_unfold, output_size=(H1, W1),
                       kernel_size=(12, 12), padding=4, stride=4)
        norm_lv1 = F.fold(torch.ones_like(T_lv1_unfold), output_size=(H1, W1),
                          kernel_size=(12, 12), padding=4, stride=4)
        T_lv1 = T_lv1 / norm_lv1.clamp(min=1)

        S = R_lv3_star.view(R_lv3_star.size(0), 1, H_out, W_out)

        return S, T_lv3, T_lv2, T_lv1