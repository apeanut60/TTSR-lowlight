"""
TTSREnhance — Reference-based Low-Light Enhancement Model.
Adapted from TTSR (Texture Transformer Super-Resolution).
Key differences from TTSR:
- Uses MainNetEnhance (1:1 resolution) instead of MainNet (4x SR)
- lrsr/refsr are raw images (no bicubic upsampling needed)
- Supports optional pre-trained weight loading from TTSR
"""

from model import MainNetEnhance, LTE, SearchTransfer

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefCorrectionHead(nn.Module):
    """Always-on reference correction head (global + local).

    The global branch is a 1x1 convolution that learns a per-image global
    brightness/color correction. The local branch is a small 3x3 residual
    corrector for local artifacts. Both outputs are initialized to zero, so
    the checkpoint starts with ``corrected_ref == ref`` exactly.
    """

    def __init__(self, n_feats=16):
        super(RefCorrectionHead, self).__init__()
        self.global_conv = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
        self.local_conv1 = nn.Conv2d(3, n_feats, kernel_size=3, stride=1, padding=1)
        self.local_conv2 = nn.Conv2d(n_feats, 3, kernel_size=3, stride=1, padding=1)

        # Zero initialization keeps old checkpoints exactly unchanged.
        nn.init.zeros_(self.global_conv.weight)
        nn.init.zeros_(self.global_conv.bias)
        nn.init.kaiming_normal_(self.local_conv1.weight, mode='fan_out',
                                nonlinearity='relu')
        nn.init.zeros_(self.local_conv1.bias)
        nn.init.zeros_(self.local_conv2.weight)
        nn.init.zeros_(self.local_conv2.bias)

    def forward(self, x):
        identity = x
        global_delta = self.global_conv(x)
        local = F.relu(self.local_conv1(x))
        local_delta = self.local_conv2(local)
        return torch.clamp(identity + global_delta + local_delta, -1.0, 1.0)


class TTSREnhance(nn.Module):
    def __init__(self, args):
        super(TTSREnhance, self).__init__()
        self.args = args
        self.num_res_blocks = list(
            map(int, args.num_res_blocks.split('+')))
        self.MainNet = MainNetEnhance.MainNetEnhance(
            num_res_blocks=self.num_res_blocks,
            n_feats=args.n_feats,
            res_scale=args.res_scale,
            ref_illum_pool=getattr(args, 'ref_illum_pool', 8))
        freeze_lte = getattr(args, 'freeze_lte', False)
        self.LTE = LTE.LTE(requires_grad=not freeze_lte)
        self.LTE_copy = LTE.LTE(
            requires_grad=False)  # used in transferal perceptual loss
        self.SearchTransfer = SearchTransfer.SearchTransfer()
        self.RefCorrection = None
        if getattr(args, 'ref_correction', True):
            self.RefCorrection = RefCorrectionHead(
                n_feats=getattr(args, 'ref_correction_feats', 16))

    def apply_illum_head(self, x, low):
        """Delegate to MainNet's global illumination head (post-stitch use)."""
        return self.MainNet.apply_illum_head(x, low)

    def apply_ref_illum(self, x, low, ref):
        """Delegate to MainNet's reference illumination transfer (post-stitch)."""
        return self.MainNet.apply_ref_illum(x, low, ref)

    def _dummy_ref_outputs(self, lr):
        """Zero S/T placeholders with the shapes the trainer expects.

        Only consumed by the transferal perceptual loss, which a no-reference
        run does not use. VGG levels downsample by 2 per pooling step.
        """
        n = lr.size(0)
        h1, w1 = lr.size()[-2:]
        h2, w2 = h1 // 2, w1 // 2
        h3, w3 = h2 // 2, w2 // 2

        def z(c, h, w):
            return lr.new_zeros(n, c, h, w)

        return z(1, h3, w3), z(256, h3, w3), z(128, h2, w2), z(64, h1, w1)

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None,
                return_ref=False, apply_illum=True, apply_ref_illum=None,
                use_reference=None):
        if (type(sr) != type(None)):
            # Used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        if apply_ref_illum is None:
            apply_ref_illum = not getattr(self.args, 'no_ref_illum', False)

        if use_reference is None:
            use_reference = not getattr(self.args, 'no_reference', False)

        if not use_reference:
            # No-reference baseline: every reference-dependent path is off,
            # including the reference illumination transfer.
            sr = self.MainNet(lr, apply_illum=apply_illum, ref=None,
                              apply_ref_illum=False, use_reference=False)
            S, T_lv3, T_lv2, T_lv1 = self._dummy_ref_outputs(lr)
            if return_ref:
                return sr, S, T_lv3, T_lv2, T_lv1, None
            return sr, S, T_lv3, T_lv2, T_lv1

        # Extract VGG features for search and texture transfer
        # lrsr/refsr are raw images at the same resolution (no bicubic needed)
        if self.RefCorrection is not None:
            ref_input = self.RefCorrection(ref)
            refsr_input = self.RefCorrection(refsr)
        else:
            ref_input = ref.detach()
            refsr_input = refsr.detach()

        _, _, lrsr_lv3 = self.LTE((lrsr.detach() + 1.) / 2.)
        _, _, refsr_lv3 = self.LTE((refsr_input + 1.) / 2.)

        ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref_input + 1.) / 2.)

        S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(
            lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3,
            oracle_matching=getattr(self.args, 'oracle_matching', 'off'))

        # Control experiment: RefIllumTransfer keeps its capacity and training
        # schedule but receives a constant image instead of the reference, so
        # its gain can be split into "extra correction capacity" vs
        # "reference content". `ref` in MainNet is consumed ONLY by ref_illum.
        illum_ref = ref_input
        if getattr(self.args, 'ref_illum_const_ref', False):
            illum_ref = torch.zeros_like(ref_input)

        # MainNetEnhance at 1:1 resolution
        sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1,
                          apply_illum=apply_illum, ref=illum_ref,
                          apply_ref_illum=apply_ref_illum,
                          use_reference=True)

        if return_ref:
            return sr, S, T_lv3, T_lv2, T_lv1, ref_input
        return sr, S, T_lv3, T_lv2, T_lv1


def load_pretrained_weights(model, pretrain_path, device='cuda'):
    """
    Load pre-trained TTSR weights into TTSREnhance model.
    Only loads matching layers (LTE, SFE shared convs).
    """
    pretrained = torch.load(pretrain_path, map_location=device)
    model_state = model.state_dict()

    matched = 0
    for k, v in pretrained.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            matched += 1

    model.load_state_dict(model_state)
    print(f"[Pretrain] Loaded {matched}/{len(pretrained)} matching layers"
          f" from {pretrain_path}")
    return model
