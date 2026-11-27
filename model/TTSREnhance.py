"""
TTSREnhance — Reference-based Low-Light Enhancement Model.
Adapted from TTSR (Texture Transformer Super-Resolution).
Key differences from TTSR:
- Uses MainNetEnhance (1:1 resolution) instead of MainNet (4x SR)
- lrsr/refsr are raw images (no bicubic upsampling needed)
- Supports optional pre-trained weight loading from TTSR
"""

from model import MainNetEnhance, LTE, SearchTransfer
from model.CSFIUNet import CSFIUNet
from model.query_illum_align import align_query_illumination
from model.restoration_config import (
    reject_legacy_igab_state, restoration_config_from_args,
)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_to_multiple(x, multiple=4):
    """Reflect-pad H/W up to the next multiple (ASRT-style)."""
    if x is None:
        return None, (0, 0)
    h, w = x.shape[-2:]
    th = int(math.ceil(h / float(multiple)) * multiple)
    tw = int(math.ceil(w / float(multiple)) * multiple)
    ph, pw = th - h, tw - w
    if ph == 0 and pw == 0:
        return x, (h, w)
    mode = 'reflect' if ph < h and pw < w else 'replicate'
    return F.pad(x, (0, pw, 0, ph), mode=mode), (h, w)


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
        backbone = str(getattr(args, 'backbone', 'csfi_unet')).strip().lower()
        self.backbone_name = backbone
        if backbone in ('csfi_unet', 'csfi', 'asrt'):
            # Neutral names; deprecated igab_* aliases warn / conflict-check.
            # Rejects any model_variant other than the one this code builds.
            rcfg = restoration_config_from_args(args)
            self.model_variant = rcfg['model_variant']
            self.restoration_config = rcfg
            self.MainNet = CSFIUNet(
                num_res_blocks=args.num_res_blocks,
                n_feats=args.n_feats,
                res_scale=args.res_scale,
                restoration_block=rcfg['restoration_block'],
                restoration_blocks=rcfg['restoration_blocks'],
                restoration_heads=rcfg['restoration_heads'],
                restoration_ffn_expansion=rcfg['restoration_ffn_expansion'],
                enable_lightup=bool(getattr(args, 'enable_lightup', True)),
                use_global_illum=bool(getattr(args, 'use_global_illum', False)),
                use_ref_illum=bool(getattr(args, 'use_ref_illum', False)),
                ref_illum_pool=getattr(args, 'ref_illum_pool', 8),
                gain_estimator=rcfg['gain_estimator'],
                gain_channels=tuple(
                    int(x) for x in str(rcfg['gain_channels']).split(',')),
                gain_res_blocks=tuple(
                    int(x) for x in str(rcfg['gain_res_blocks']).split(',')),
                gain_attn_heads=rcfg['gain_attn_heads'],
                gain_attn_depth=rcfg['gain_attn_depth'],
                gain_attn_ffn_expansion=rcfg['gain_attn_ffn_expansion'],
                gain_max_tokens=rcfg['gain_max_tokens'])
            match_kwargs = dict(
                matching_mode=rcfg['matching_mode'],
                match_topk=rcfg['match_topk'],
                match_temperature=rcfg['match_temperature'],
                match_query_chunk_size=rcfg['match_query_chunk_size'],
                match_transfer_recompute=rcfg['match_transfer_recompute'])
        elif backbone in ('mainnet', 'mainnet_enhance', 'same_res'):
            self.model_variant = None
            self.restoration_config = None
            self.MainNet = MainNetEnhance.MainNetEnhance(
                num_res_blocks=self.num_res_blocks,
                n_feats=args.n_feats,
                res_scale=args.res_scale,
                ref_illum_pool=getattr(args, 'ref_illum_pool', 8))
            match_kwargs = dict(matching_mode='hard_top1')
        else:
            raise ValueError(
                'Unknown backbone %r (expected csfi_unet|mainnet)' % backbone)
        freeze_lte = getattr(args, 'freeze_lte', False)
        self.LTE = LTE.LTE(requires_grad=not freeze_lte)
        self.LTE_copy = LTE.LTE(
            requires_grad=False)  # used in transferal perceptual loss
        self.SearchTransfer = SearchTransfer.SearchTransfer(**match_kwargs)
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

    def get_effective_reference(self, ref):
        """Reference seen by RefIllumTransfer.

        Matches ``forward``: RefCorrection once when enabled, otherwise the
        raw reference. Never uses GT. ``None`` stays ``None``.
        """
        if ref is None:
            return None
        if self.RefCorrection is None:
            return ref.detach()
        return self.RefCorrection(ref)

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None,
                return_ref=False, apply_illum=True, apply_ref_illum=True,
                return_diag=False, query_gamma_override=None,
                query_hist_lut_override=None,
                gate_mode=None, gate_shift=None, gate_smooth_kernel=None,
                collect_gate_stats=False,
                disable_T_lv3=None, disable_T_lv2=None, disable_T_lv1=None,
                stage_mode_lv3=None, stage_mode_lv2=None, stage_mode_lv1=None,
                collect_residual_decomp=False, stem=None):
        if (type(sr) != type(None)):
            # Used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        # CSFI-U-Net needs H/W divisible by 4 so LTE/SearchTransfer grids match
        # the encoder downs. Pad before LTE; crop after MainNet.
        # When ``stem`` is provided (tiled full-image illum), inputs are already
        # on a /4 canvas and must not be padded again.
        use_csfi = isinstance(self.MainNet, CSFIUNet)
        orig_hw = None
        if use_csfi and lr is not None and stem is None:
            lr, orig_hw = _pad_to_multiple(lr, 4)
            lrsr, _ = _pad_to_multiple(lrsr, 4)
            if ref is not None:
                ref, _ = _pad_to_multiple(ref, 4)
            if refsr is not None:
                refsr, _ = _pad_to_multiple(refsr, 4)

        # Explicit no-reference mode: both ref and refsr must be None.
        # Half-specified reference inputs are rejected.
        if (ref is None) ^ (refsr is None):
            raise ValueError(
                'ref and refsr must both be set or both be None; got ref=%s refsr=%s'
                % (type(ref).__name__, type(refsr).__name__))
        no_ref = ref is None and refsr is None

        self.last_query_illum_info = None
        self.last_match_diag = None
        self.last_texture_fusion_info = None

        if no_ref:
            # Skip RefCorrection / reference LTE / SearchTransfer / inject.
            S = T_lv3 = T_lv2 = T_lv1 = None
            ref_input = None
            main_kwargs = dict(
                apply_illum=apply_illum, ref=None,
                apply_ref_illum=False,
                gate_mode=gate_mode, gate_shift=gate_shift,
                gate_smooth_kernel=gate_smooth_kernel,
                collect_gate_stats=collect_gate_stats,
                disable_T_lv3=True, disable_T_lv2=True, disable_T_lv1=True,
                stage_mode_lv3='branch_bypass',
                stage_mode_lv2='branch_bypass',
                stage_mode_lv1='branch_bypass',
                collect_residual_decomp=collect_residual_decomp,
                use_reference=False)
            if use_csfi and stem is not None:
                main_kwargs['stem'] = stem
            sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1, **main_kwargs)
            if use_csfi and orig_hw is not None:
                sr = sr[..., :orig_hw[0], :orig_hw[1]]
            if return_ref:
                out = (sr, S, T_lv3, T_lv2, T_lv1, None)
            else:
                out = (sr, S, T_lv3, T_lv2, T_lv1)
            if return_diag:
                return out + (None, None)
            return out

        # Extract VGG features for search and texture transfer
        # lrsr/refsr are raw images at the same resolution (no bicubic needed)
        # Do NOT wrap LTE in torch.no_grad() — RefCorrection must receive grads
        # through the frozen LTE graph when it is trainable.
        if self.RefCorrection is not None:
            ref_input = self.RefCorrection(ref)
            refsr_input = self.RefCorrection(refsr)
        else:
            ref_input = ref
            refsr_input = refsr

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

        def _flag(val, name):
            if val is not None:
                return bool(val)
            return bool(getattr(self.args, name, False))

        def _mode(val, name):
            if val is not None:
                return val
            return getattr(self.args, name, None)

        disable_T_lv3 = _flag(disable_T_lv3, 'disable_T_lv3')
        disable_T_lv2 = _flag(disable_T_lv2, 'disable_T_lv2')
        disable_T_lv1 = _flag(disable_T_lv1, 'disable_T_lv1')
        stage_mode_lv3 = _mode(stage_mode_lv3, 'stage_mode_lv3')
        stage_mode_lv2 = _mode(stage_mode_lv2, 'stage_mode_lv2')
        stage_mode_lv1 = _mode(stage_mode_lv1, 'stage_mode_lv1')
        if not collect_residual_decomp:
            collect_residual_decomp = bool(
                getattr(self.args, 'collect_residual_decomp', False))

        # Backbone at 1:1 resolution (original Low, not low_query)
        main_kwargs = dict(
            apply_illum=apply_illum, ref=ref_input,
            apply_ref_illum=apply_ref_illum,
            gate_mode=gate_mode, gate_shift=gate_shift,
            gate_smooth_kernel=gate_smooth_kernel,
            collect_gate_stats=collect_gate_stats,
            disable_T_lv3=disable_T_lv3,
            disable_T_lv2=disable_T_lv2,
            disable_T_lv1=disable_T_lv1,
            stage_mode_lv3=stage_mode_lv3,
            stage_mode_lv2=stage_mode_lv2,
            stage_mode_lv1=stage_mode_lv1,
            collect_residual_decomp=collect_residual_decomp)
        if use_csfi and stem is not None:
            main_kwargs['stem'] = stem
        sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1, **main_kwargs)

        if use_csfi and orig_hw is not None:
            sr = sr[..., :orig_hw[0], :orig_hw[1]]

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
    if isinstance(getattr(model, 'MainNet', None), CSFIUNet):
        # Never shape-filter an IGAB-variant checkpoint into the LFCA backbone.
        reject_legacy_igab_state(pretrained.keys(), pretrain_path)

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
