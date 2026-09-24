"""V2 local reference refinement head.

Operates on the *output* of a frozen enhancer. Two inputs:

    Y0     the frozen enhancer's output (the thing we are refining)
    R_eff  the second path: Y0 itself (``self`` arm) or a reference image

and returns ``sr = Y0 + g * delta_rgb``. Only ``C_out`` is zero-initialised, so
at step 0 the module is an exact identity for ANY finite input -- that keeps the
``self`` and ``nano`` arms comparable from the first update (their only design
difference is the second input, not their initialisation).

The gate is *not* zero-initialised: a second exactly-zero multiplier on top of a
zero-initialised readout would stall the whole branch (the mistake the old
RefIllumTransfer docstring reasoned about incorrectly).

All images stay in the repository's [-1, 1] convention; the encoder is fed
``(x + 1) / 2`` and the residual is predicted in [-1, 1] units.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedEncoder(nn.Module):
    """E.0 / E.1 / E.2 — built once, applied to both input paths."""

    def __init__(self, ch=32):
        super().__init__()
        self.conv0 = nn.Conv2d(3, ch, 3, 1, 1, bias=True)
        self.conv1 = nn.Conv2d(ch, ch, 3, 2, 1, bias=True)   # H -> H/2
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.conv0(x))
        x = self.act(self.conv1(x))
        return self.act(self.conv2(x))


def _neighbour_indices(h, w, radius):
    """[h*w, (2r+1)^2] flat indices of the neighbourhood of every position.

    Built from explicit 2-D coordinates, so a flatten()-adjacent index can
    never wrap across a row (the classic bug when reusing a linear offset).
    """
    dev = torch.device('cpu')
    iy = torch.arange(h, device=dev).view(h, 1).expand(h, w).reshape(-1)
    ix = torch.arange(w, device=dev).view(1, w).expand(h, w).reshape(-1)
    off = torch.arange(-radius, radius + 1, device=dev)
    cy = iy[:, None] + off[None, :]                  # [h*w, k]
    cx = ix[:, None] + off[None, :]
    vy = (cy >= 0) & (cy < h)
    vx = (cx >= 0) & (cx < w)
    valid = vy[:, :, None] & vx[:, None, :]          # [h*w, k, k]
    cyc = cy.clamp(0, h - 1)
    cxc = cx.clamp(0, w - 1)
    flat = (cyc[:, :, None] * w + cxc[:, None, :]).reshape(h * w, -1)
    return flat, valid.reshape(h * w, -1)


class LocalSoftMatch(nn.Module):
    """Single-head cosine matching inside a (2r+1)^2 window."""

    def __init__(self, ch=32, radius=4, tau=0.1, eps=1e-6,
                 chunk=None):
        super().__init__()
        self.q = nn.Conv2d(ch, ch, 1, bias=False)
        self.k = nn.Conv2d(ch, ch, 1, bias=False)
        self.v = nn.Conv2d(ch, ch, 1, bias=False)
        self.radius = radius
        self.tau = tau
        self.eps = eps
        self.chunk = chunk          # None -> dense; int -> query-chunked
        with torch.no_grad():
            eye = torch.eye(ch).view(ch, ch, 1, 1)
            self.q.weight.copy_(eye)
            self.k.weight.copy_(eye)
            self.v.weight.copy_(eye)
        self._cache = {}

    def _idx(self, h, w, device):
        key = (h, w)
        if key not in self._cache:
            self._cache[key] = _neighbour_indices(h, w, self.radius)
        flat, valid = self._cache[key]
        return flat.to(device), valid.to(device)

    @staticmethod
    def _gather_cand(x, flat):
        """x: [B,C,n], flat: [n,j] -> [B,C,j,n] (candidate-major)."""
        return x[:, :, flat].permute(0, 1, 3, 2)

    def forward(self, f0, fr):
        B, C, h, w = f0.shape
        q = F.normalize(self.q(f0), dim=1, eps=self.eps)
        k = F.normalize(self.k(fr), dim=1, eps=self.eps)
        v = self.v(fr)
        flat, valid = self._idx(h, w, f0.device)
        n = h * w
        kf = k.reshape(B, C, n)
        vf = v.reshape(B, C, n)
        qf = q.reshape(B, C, n)
        valid_t = valid.t()                              # [81, n]

        if self.chunk is None or n <= self.chunk:
            kr = self._gather_cand(kf, flat)             # [B,C,81,n]
            vr = self._gather_cand(vf, flat)
            logit = torch.einsum('bcn,bcjn->bjn', qf, kr) / self.tau
            logit = logit.masked_fill(~valid_t[None], float('-inf'))
            p = torch.softmax(logit, dim=1)              # over the 81 candidates
            t = torch.einsum('bjn,bcjn->bcn', p, vr).reshape(B, C, h, w)
        else:
            t = f0.new_empty(B, C, n)
            for s in range(0, n, self.chunk):
                e = min(s + self.chunk, n)
                fl = flat[s:e]                            # [c,81]
                qc = qf[:, :, s:e]
                kr = self._gather_cand(kf, fl)             # [B,C,81,c]
                vr = self._gather_cand(vf, fl)
                logit = torch.einsum('bcm,bcjm->bjm', qc, kr) / self.tau
                logit = logit.masked_fill(~valid_t[:, s:e][None], float('-inf'))
                p = torch.softmax(logit, dim=1)
                t[:, :, s:e] = torch.einsum('bjm,bcjm->bcm', p, vr)
            t = t.reshape(B, C, h, w)
        return t


class Refiner(nn.Module):
    """The whole V2 module: shared encoder + local match + gate + RGB head."""

    def __init__(self, ch=32, radius=4, tau=0.1, chunk=None):
        super().__init__()
        self.encoder = SharedEncoder(ch)
        self.match = LocalSoftMatch(ch, radius, tau, chunk=chunk)
        self.gate = nn.Sequential(
            nn.Conv2d(3 * ch, ch, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(ch, 1, 1, bias=True),
            nn.Sigmoid())
        self.c_hidden = nn.Sequential(
            nn.Conv2d(3 * ch, ch, 3, 1, 1, bias=True),
            nn.GELU())
        self.c_out = nn.Conv2d(ch, 3, 3, 1, 1, bias=True)
        # Only the readout is zeroed (see the module docstring).
        nn.init.zeros_(self.c_out.weight)
        nn.init.zeros_(self.c_out.bias)

    def forward(self, y0, r_eff):
        """y0, r_eff: [B,3,H,W] in [-1,1]. Returns (sr, aux)."""
        f0 = self.encoder((y0 + 1.) * 0.5)
        fr = self.encoder((r_eff + 1.) * 0.5)
        t = self.match(f0, fr)
        u = torch.cat((f0, t, f0 - t), dim=1)
        g_small = self.gate(u)
        delta_small = self.c_hidden(u)
        delta = self.c_out(F.interpolate(
            delta_small, size=y0.shape[-2:], mode='bilinear',
            align_corners=False))
        g = F.interpolate(g_small, size=y0.shape[-2:], mode='bilinear',
                          align_corners=False)
        sr = y0 + g * delta
        aux = dict(gate=g, delta=delta, t=t, f0=f0)
        return sr, aux


def count_params(module):
    return sum(p.numel() for p in module.parameters())
