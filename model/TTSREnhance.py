"""
TTSREnhance — Reference-based Low-Light Enhancement Model.
Adapted from TTSR (Texture Transformer Super-Resolution).
Key differences from TTSR:
- Uses MainNetEnhance (1:1 resolution) instead of MainNet (4x SR)
- lrsr/refsr are raw images (no bicubic upsampling needed)
- Supports optional pre-trained weight loading from TTSR
"""

from model import MainNetEnhance, LTE, SearchTransfer
from model.query_illum_align import align_query_illumination

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
        # Last forward diagnostics (not part of checkpoint)
        self.last_query_illum_info = None
        self.last_match_diag = None
        self.last_texture_fusion_info = None

    def apply_illum_head(self, x, low):
        """Delegate to MainNet's global illumination head (post-stitch use)."""
        return self.MainNet.apply_illum_head(x, low)

    def apply_ref_illum(self, x, low, ref):
        """Delegate to MainNet's reference illumination transfer (post-stitch)."""
        return self.MainNet.apply_ref_illum(x, low, ref)

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None,
                return_ref=False, apply_illum=True, apply_ref_illum=True,
                return_diag=False, query_gamma_override=None,
                query_hist_lut_override=None,
                gate_mode=None, gate_shift=None, gate_smooth_kernel=None,
                collect_gate_stats=False):
        if (type(sr) != type(None)):
            # Used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        # Extract VGG features for search and texture transfer
        # lrsr/refsr are raw images at the same resolution (no bicubic needed)
        if self.RefCorrection is not None:
            ref_input = self.RefCorrection(ref)
            refsr_input = self.RefCorrection(refsr)
        else:
            ref_input = ref.detach()
            refsr_input = refsr.detach()

        # Optional: brightness-align Low Query toward Nano Ref before LTE.
        # MainNet still receives the original `lr`. Value path uses `ref_input`.
        use_qia = getattr(self.args, 'query_illum_align', False)
        alpha = float(getattr(self.args, 'query_illum_alpha', 0.0))
        beta = float(getattr(self.args, 'query_hist_beta', 0.0))
        gmin = float(getattr(self.args, 'query_gamma_min', 0.3))
        gmax = float(getattr(self.args, 'query_gamma_max', 1.0))
        hbins = int(getattr(self.args, 'query_hist_bins', 256))
        use_fusion = bool(getattr(self.args, 'texture_fusion', False))
        fusion_lambda = float(getattr(self.args, 'texture_fusion_lambda', 0.0))
        fusion_s_source = str(
            getattr(self.args, 'texture_fusion_s_source', 'a0')).lower()
        self.last_query_illum_info = None
        self.last_match_diag = None
        self.last_texture_fusion_info = None

        def _aligned_query():
            if use_qia and (alpha > 0.0 or beta > 0.0):
                q, qinfo = align_query_illumination(
                    lrsr.detach(), ref.detach(),
                    alpha=alpha,
                    gamma_min=gmin,
                    gamma_max=gmax,
                    gamma_override=query_gamma_override,
                    beta=beta,
                    hist_bins=hbins,
                    lut_override=query_hist_lut_override,
                )
                return q, qinfo
            q = lrsr.detach()
            qinfo = None
            if use_qia:
                _, qinfo = align_query_illumination(
                    lrsr.detach(), ref.detach(),
                    alpha=0.0, beta=0.0,
                    gamma_min=gmin,
                    gamma_max=gmax,
                    gamma_override=query_gamma_override,
                    hist_bins=hbins,
                )
            return q, qinfo

        # Value + Key from Reference (never GT). Shared by all paths.
        _, _, refsr_lv3 = self.LTE((refsr_input + 1.) / 2.)
        ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref_input + 1.) / 2.)

        if use_fusion:
            # Eval-only Identity↔B2 texture mix. Default ``texture_fusion=False``
            # keeps the original single SearchTransfer path bit-compatible.
            if fusion_s_source not in ('a0', 'search', 'b2'):
                raise ValueError(
                    'texture_fusion_s_source must be a0|search|b2, got %r' %
                    fusion_s_source)
            if not (0.0 <= fusion_lambda <= 1.0):
                raise ValueError(
                    'texture_fusion_lambda must be in [0,1], got %s' %
                    fusion_lambda)

            # A0 Query → S_A0 (and T_A0 only if needed for diagnostics)
            _, _, q_a0_lv3 = self.LTE((lrsr.detach() + 1.) / 2.)
            st_a0 = self.SearchTransfer(
                q_a0_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3,
                return_diag=return_diag)
            if return_diag:
                S_a0, T_a0_3, T_a0_2, T_a0_1, diag_a0 = st_a0
            else:
                S_a0, T_a0_3, T_a0_2, T_a0_1 = st_a0
                diag_a0 = None

            # B2 (or current align settings) Query → T_B2 / S_B2
            low_query, qinfo = _aligned_query()
            self.last_query_illum_info = qinfo
            _, _, q_b2_lv3 = self.LTE((low_query + 1.) / 2.)
            st_b2 = self.SearchTransfer(
                q_b2_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3,
                return_diag=return_diag)
            if return_diag:
                S_b2, T_b2_3, T_b2_2, T_b2_1, diag_b2 = st_b2
            else:
                S_b2, T_b2_3, T_b2_2, T_b2_1 = st_b2
                diag_b2 = None

            H_out, W_out = q_a0_lv3.size()[-2:]
            Hr, Wr = refsr_lv3.size()[-2:]
            T_id_3, T_id_2, T_id_1 = self.SearchTransfer.transfer_identity(
                ref_lv1, ref_lv2, ref_lv3, H_out, W_out, Hr=Hr, Wr=Wr)

            lam = fusion_lambda
            T_lv3 = (1.0 - lam) * T_b2_3 + lam * T_id_3
            T_lv2 = (1.0 - lam) * T_b2_2 + lam * T_id_2
            T_lv1 = (1.0 - lam) * T_b2_1 + lam * T_id_1

            if fusion_s_source == 'a0':
                S = S_a0
            elif fusion_s_source == 'b2':
                S = S_b2
            else:
                S = S_b2  # 'search' ≡ current aligned search soft-attention

            self.last_match_diag = diag_b2 if return_diag else None
            self.last_texture_fusion_info = {
                'lambda': lam,
                's_source': fusion_s_source,
                'T_b2': (T_b2_3, T_b2_2, T_b2_1),
                'T_id': (T_id_3, T_id_2, T_id_1),
                'T_mix': (T_lv3, T_lv2, T_lv1),
                'S_a0': S_a0,
                'S_b2': S_b2,
            }
            if return_diag:
                self.last_match_diag = {
                    'a0': diag_a0,
                    'b2': diag_b2,
                    'fusion_lambda': lam,
                    's_source': fusion_s_source,
                }
        else:
            low_query, qinfo = _aligned_query()
            self.last_query_illum_info = qinfo
            _, _, lrsr_lv3 = self.LTE((low_query + 1.) / 2.)

            st_out = self.SearchTransfer(
                lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3,
                return_diag=return_diag)
            if return_diag:
                S, T_lv3, T_lv2, T_lv1, diag = st_out
                self.last_match_diag = diag
            else:
                S, T_lv3, T_lv2, T_lv1 = st_out

        # Eval-only gate counterfactuals (v1r10). Defaults leave G unchanged.
        if gate_mode is None:
            gate_mode = getattr(self.args, 'gate_mode', None)
        if gate_shift is None:
            gate_shift = getattr(self.args, 'gate_shift', None)
        if isinstance(gate_shift, str):
            gate_shift = gate_shift.strip()
            if gate_shift == '':
                gate_shift = None
            else:
                parts = [int(p) for p in gate_shift.replace(' ', '').split(',')]
                if len(parts) != 2:
                    raise ValueError('gate_shift must be "dy,dx", got %r' %
                                     gate_shift)
                gate_shift = (parts[0], parts[1])
        if gate_smooth_kernel is None:
            gate_smooth_kernel = int(
                getattr(self.args, 'gate_smooth_kernel', 31))
        if not collect_gate_stats:
            collect_gate_stats = bool(
                getattr(self.args, 'collect_gate_stats', False))

        # MainNetEnhance at 1:1 resolution (original Low, not low_query)
        sr = self.MainNet(
            lr, S, T_lv3, T_lv2, T_lv1,
            apply_illum=apply_illum, ref=ref_input,
            apply_ref_illum=apply_ref_illum,
            gate_mode=gate_mode, gate_shift=gate_shift,
            gate_smooth_kernel=gate_smooth_kernel,
            collect_gate_stats=collect_gate_stats)

        if return_ref:
            out = (sr, S, T_lv3, T_lv2, T_lv1, ref_input)
        else:
            out = (sr, S, T_lv3, T_lv2, T_lv1)
        if return_diag:
            return out + (self.last_match_diag, self.last_query_illum_info)
        return out


def load_pretrained_weights(model, pretrain_path, device='cuda',
                            require_core=True):
    """
    Load pre-trained weights into TTSREnhance.

    Reports matched / skipped / missing counts. When ``require_core`` is True
    (default), raises if any MainNet / LTE / RefCorrection / SearchTransfer
    tensor present in the checkpoint is skipped due to shape mismatch, or if
    core prefixes are entirely absent from the checkpoint.
    """
    pretrained = torch.load(pretrain_path, map_location=device)
    if any(k.startswith('module.') for k in pretrained):
        pretrained = {k.replace('module.', '', 1): v for k, v in pretrained.items()}

    model_state = model.state_dict()
    matched = 0
    skipped_shape = []
    ckpt_only = []
    for k, v in pretrained.items():
        if k not in model_state:
            # LTE_copy etc. often absent from saved ckpts — ignore
            if '_copy' in k:
                continue
            ckpt_only.append(k)
            continue
        if model_state[k].shape == v.shape:
            model_state[k] = v
            matched += 1
        else:
            skipped_shape.append(
                (k, tuple(v.shape), tuple(model_state[k].shape)))

    model.load_state_dict(model_state)

    # Expected loadable keys in the live model (exclude LTE_copy buffers)
    loadable = [k for k in model_state if '_copy' not in k]
    missing_in_ckpt = [k for k in loadable if k not in pretrained]
    core_prefixes = ('MainNet.', 'LTE.', 'RefCorrection.', 'SearchTransfer.')
    missing_core = [k for k in missing_in_ckpt
                    if k.startswith(core_prefixes)]
    shape_core = [t for t in skipped_shape if t[0].startswith(core_prefixes)]

    msg = (f"[Pretrain] Loaded {matched}/{len(pretrained)} ckpt tensors "
           f"into model ({len(loadable)} loadable keys); "
           f"missing_in_ckpt={len(missing_in_ckpt)} "
           f"(core={len(missing_core)}); "
           f"shape_mismatch={len(skipped_shape)} "
           f"from {pretrain_path}")
    print(msg)

    if require_core and (missing_core or shape_core):
        detail = []
        if missing_core[:8]:
            detail.append('missing core: ' + ', '.join(missing_core[:8]))
        if shape_core[:4]:
            detail.append('shape: ' + str(shape_core[:4]))
        raise RuntimeError(
            'Strict core weight load failed. ' + '; '.join(detail))
    return model
