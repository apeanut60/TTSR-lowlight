"""V3-B.4.1 base-conditioned H/2 feature residual. Sees F_dec_h2 + [F0,T,F0-T]."""

import torch
import torch.nn as nn

from model.V3BFeatureBridge import BASE_H2_CH

REF_CH = 96  # F0+T+(F0-T)
IN_CH = BASE_H2_CH + REF_CH  # 176


class BaseConditionedFeatureResidual(nn.Module):
    """u=[F_dec_h2,F0,T,F0-T] 176ch → ΔF_h2 80ch. Last conv zero-init."""

    def __init__(self, in_ch=IN_CH, hid=64, out_ch=BASE_H2_CH):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.base_feature_ch = BASE_H2_CH
        self.ref_input_ch = REF_CH
        self.net = nn.Sequential(
            nn.Conv2d(self.in_ch, hid, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hid, hid, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hid, self.out_ch, 3, 1, 1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, F_dec, F0, T):
        if int(F_dec.shape[1]) != BASE_H2_CH:
            raise SystemExit('F_dec channels %d != %d'
                             % (int(F_dec.shape[1]), BASE_H2_CH))
        if F0.shape != T.shape or int(F0.shape[1]) != 32:
            raise SystemExit('expected F0/T [B,32,H/2,W/2], got %s %s'
                             % (tuple(F0.shape), tuple(T.shape)))
        if F_dec.shape[-2:] != F0.shape[-2:]:
            raise SystemExit('F_dec spatial %s != F0 %s'
                             % (tuple(F_dec.shape[-2:]), tuple(F0.shape[-2:])))
        u = torch.cat([F_dec, F0, T, F0 - T], dim=1)
        if int(u.shape[1]) != self.in_ch:
            raise SystemExit('concat ch %d != %d' % (int(u.shape[1]), self.in_ch))
        return self.net(u)


def _crop_f(f, y0, x0, th, tw):
    ys, xs = y0 // 2, x0 // 2
    ye, xe = (y0 + th) // 2, (x0 + tw) // 2
    return f[:, :, ys:ye, xs:xe]


def delta_fn_conditioned(adapter, F0, T, cond_mode='normal', F_cond_full=None):
    """Build tiled delta_fn.

    cond_mode:
      normal   — condition on true tile F_h2
      zero     — condition on zeros; still add ΔF onto true F_h2
      shuffled — condition on F_cond_full cropped to tile; add onto true F_h2
    """
    from model.V3BFeatureBridge import crop_ft

    if cond_mode not in ('normal', 'zero', 'shuffled'):
        raise SystemExit('bad cond_mode %r' % cond_mode)
    if cond_mode == 'shuffled' and F_cond_full is None:
        raise SystemExit('shuffled cond needs F_cond_full')

    def delta_fn(y0, x0, th, tw, f_h2):
        f0c, tc = crop_ft(F0, T, y0, x0, th, tw)
        if f0c.shape[-2:] != f_h2.shape[-2:]:
            raise SystemExit('F0 crop %s != F_h2 %s (tile y0=%d x0=%d th=%d tw=%d)'
                             % (tuple(f0c.shape), tuple(f_h2.shape), y0, x0, th, tw))
        if cond_mode == 'normal':
            f_cond = f_h2
        elif cond_mode == 'zero':
            f_cond = torch.zeros_like(f_h2)
        else:
            f_cond = _crop_f(F_cond_full, y0, x0, th, tw)
            if f_cond.shape != f_h2.shape:
                raise SystemExit('F_cond crop %s != F_h2 %s'
                                 % (tuple(f_cond.shape), tuple(f_h2.shape)))
        return adapter(f_cond, f0c, tc)

    return delta_fn
