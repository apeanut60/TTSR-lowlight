"""Retinexformer backbone + single-point (H/4 bottleneck) texture adapter.

This is an **additional** backbone for TTSR-lowlight. The default model is
still ``MainNetEnhance``; this one is selected with ``--enhance_backbone
retinexformer``.

Interface contract — must match ``MainNetEnhance`` so that ``TTSREnhance`` can
call either one without branching:

    forward(x, S, T_lv3, T_lv2, T_lv1, apply_illum, ref, apply_ref_illum,
            use_reference)
    apply_illum_head(x, low)
    apply_ref_illum(x, low, ref)

Submodule names relied upon by ``trainer.py`` (optimizer param groups):

    ref_illum.*           -> own group, driven by --lr_rate_refillum
    global_illum_head.*   -> own group, driven by --lr_rate_illum (optional)

Keeping ``ref_illum`` at exactly this dotted path is load-bearing: if the name
changes, the parameter falls into the generic mainnet group and silently
inherits ``--lr_rate`` instead of ``--lr_rate_refillum``.

Numerical ranges: the outside world (dataloader, LTE, RefIllumTransfer) uses
[-1, 1]; Retinexformer works in [0, 1]. The conversion happens here only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MainNetEnhance import GlobalIllumHead, RefIllumTransfer
from model import retinexformer_arch as R


class RefTextureAdapter(nn.Module):
    """Fuse T_lv3 (256 ch @ H/4) into the bottleneck feature (dim*2^level ch @ H/4).

    ``A3 = Conv1x1(dim+256 -> dim) -> GELU -> Conv3x3(dim -> dim, pad=1)``

    The last layer is zero-initialised, so enabling the adapter is an exact
    no-op at step 0. That is a **one-step** delay, not a permanent block: conv2
    still receives gradient on the first step, after which conv1 starts
    receiving it too. (Measured on the analogous RefIllumTransfer: conv1.grad
    is exactly 0 at step 0 and non-zero after a single optimizer step.) Do not
    add an external learnable gate that also starts at exactly zero — two
    stacked zero blocks would really stall the branch.
    """

    def __init__(self, dim=160, t_ch=256):
        super().__init__()
        self.conv1 = nn.Conv2d(dim + t_ch, dim, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, f3, t_lv3):
        h = F.gelu(self.conv1(torch.cat((f3, t_lv3), dim=1)))
        return self.conv2(h)


class RetinexDenoiser(R.Denoiser):
    """Upstream ``Denoiser`` with one explicit insertion after the bottleneck.

    The body duplicates ``Denoiser.forward`` because upstream exposes no hook
    between the bottleneck and the first decoder upsample. Done explicitly on
    purpose (no global state, no forward hooks) so the data flow stays
    auditable. Keep in sync with ``retinexformer_arch.Denoiser.forward``.
    """

    def __init__(self, in_dim=3, out_dim=3, dim=40, level=2,
                 num_blocks=(1, 2, 2), t_ch=256):
        super().__init__(in_dim=in_dim, out_dim=out_dim, dim=dim, level=level,
                         num_blocks=list(num_blocks))
        self.level = level
        self.ref_adapter = RefTextureAdapter(dim=dim * (2 ** level), t_ch=t_ch)

    def forward(self, x, illu_fea, S=None, T_lv3=None, use_texture=True):
        # --- Embedding ---
        fea = self.embedding(x)

        # --- Encoder ---
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # --- Bottleneck ---
        fea = self.bottleneck(fea, illu_fea)

        # --- Texture injection (new) ---
        if use_texture:
            if T_lv3 is None:
                raise ValueError('use_texture=True but T_lv3 is None')
            if fea.shape[-2:] != T_lv3.shape[-2:]:
                raise ValueError(
                    'bottleneck grid %s does not match T_lv3 grid %s — the input '
                    'must be padded to a multiple of %d before running both LTE '
                    'and the backbone'
                    % (tuple(fea.shape[-2:]), tuple(T_lv3.shape[-2:]),
                       2 ** self.level))
            # Magnitude gate, inherited from the old model. S is a cosine
            # similarity, so this is ~0.6-0.73 in practice and nearly constant;
            # it is a scale, not a reliability probability.
            g = torch.sigmoid(S) if S is not None else 1.0
            fea = fea + g * self.ref_adapter(fea, T_lv3)

        # --- Decoder ---
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = LeWinBlcok(fea, illu_fea)

        # --- Mapping (upstream: out = mapping(fea) + x, where x is x_c) ---
        out = self.mapping(fea) + x

        return out


class RetinexRefMainNet(nn.Module):
    """Retinexformer backbone wrapped in the TTSR-lowlight MainNet contract."""

    def __init__(self, n_feat=40, num_blocks=(1, 2, 2), level=2, t_ch=256,
                 ref_illum_pool=8, use_global_illum=False):
        super().__init__()
        self.n_feat = n_feat
        self.level = level
        self.estimator = R.Illumination_Estimator(n_feat)
        self.denoiser = RetinexDenoiser(in_dim=3, out_dim=3, dim=n_feat,
                                        level=level, num_blocks=num_blocks,
                                        t_ch=t_ch)
        self.ref_illum = RefIllumTransfer(pool=ref_illum_pool)
        # Constructed only when asked for, so a disabled head contributes no
        # parameters to the optimizer at all.
        self.global_illum_head = GlobalIllumHead() if use_global_illum else None

    # --- whole-image post-processing hooks (used by both eval paths) ---

    def apply_ref_illum(self, x, low, ref):
        """Add the reference-driven low-frequency illumination residual."""
        if ref is None or self.ref_illum is None:
            return x
        return self.ref_illum(x, low, ref)

    def apply_illum_head(self, x, low):
        """Per-image global illumination correction (no-op when disabled).

        Kept identical to MainNetEnhance.apply_illum_head, including the
        clamp that keeps the base strictly positive before pow().
        """
        if self.global_illum_head is None:
            return x
        gain, bias, gamma = self.global_illum_head(low)
        x = torch.clamp((x + 1.) / 2., 1e-2, 1.)
        x = torch.pow(x, gamma)
        x = x * 2. - 1.
        x = torch.clamp(x * gain + bias, -1, 1)
        return x

    def forward(self, x, S=None, T_lv3=None, T_lv2=None, T_lv1=None,
                apply_illum=True, ref=None, apply_ref_illum=True,
                use_reference=True, inject_lv2=True, inject_lv1=True):
        # T_lv2 / T_lv1 are accepted for interface compatibility and unused in
        # this version (texture is injected at the H/4 bottleneck only), so
        # inject_lv2 / inject_lv1 are accepted and ignored on purpose.
        low = x

        x01 = (x + 1.0) * 0.5                       # [-1,1] -> [0,1]
        illu_fea, illu_map = self.estimator(x01)
        # Upstream: input_img = img * illu_map + img  (== x01 * (1 + illu_map))
        x_c = x01 * illu_map + x01

        y01 = self.denoiser(x_c, illu_fea, S=S, T_lv3=T_lv3,
                            use_texture=use_reference)
        sr = y01 * 2.0 - 1.0                        # [0,1] -> [-1,1]

        if apply_ref_illum and ref is not None:
            sr = self.apply_ref_illum(sr, low, ref)
        if apply_illum and self.global_illum_head is not None:
            sr = self.apply_illum_head(sr, low)
        return sr
