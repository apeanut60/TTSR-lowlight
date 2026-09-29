"""Clean-room literature-inspired reference evidence modules.

Inspired by:
- Task Decoupled Framework (CVPR 2022): max-correlation confidence.
- ReBaIR (ICCV-W 2025): confidence / offset metadata as downstream features.

This file does NOT copy source code from those repositories. It adapts the ideas
to this project's existing LocalSoftMatch API.

Expected input:
    f0, fr: [B, C, H, W]

Outputs:
    t: aligned/transferred reference feature [B,C,H,W]
    evidence: dict of [B,1,H,W] maps:
        sim_max       max cosine similarity
        pmax          max softmax probability
        entropy       normalized entropy in [0,1]
        margin        top1-top2 cosine margin
        dx, dy        expected local displacement (feature-cell units)
        disp_var      displacement variance
"""

import math
import torch
import torch.nn.functional as F

from model.LocalRefine import LocalSoftMatch


class LocalSoftMatchWithEvidence(LocalSoftMatch):
    """Drop-in LocalSoftMatch variant that also exposes matching evidence.

    It preserves the original q/k/v weights and matching rule. The only new
    behavior is returning diagnostics that were already implicit in the local
    correspondence distribution.
    """

    def _offset_table(self, device, dtype):
        r = int(self.radius)
        off = torch.arange(-r, r + 1, device=device, dtype=dtype)
        dy, dx = torch.meshgrid(off, off, indexing="ij")
        return dx.reshape(-1), dy.reshape(-1)

    def _summarize(self, sim, p):
        """sim,p: [B,J,N] -> dict of [B,1,H,W]-ready flattened maps."""
        # Max similarity: Task-Decoupled-style direct correspondence confidence.
        sim_max = sim.max(dim=1).values

        # Probability confidence and ambiguity.
        top2 = torch.topk(sim, k=min(2, sim.shape[1]), dim=1).values
        margin = top2[:, 0]
        if top2.shape[1] > 1:
            margin = top2[:, 0] - top2[:, 1]

        pmax = p.max(dim=1).values
        entropy = -(p.clamp_min(self.eps) * p.clamp_min(self.eps).log()).sum(dim=1)
        entropy = entropy / math.log(float(p.shape[1]))

        # ReBaIR/SSEN-inspired geometry metadata from the local match distribution.
        dx_tab, dy_tab = self._offset_table(sim.device, sim.dtype)
        dx = (p * dx_tab.view(1, -1, 1)).sum(dim=1)
        dy = (p * dy_tab.view(1, -1, 1)).sum(dim=1)
        disp_var = (
            p * (
                (dx_tab.view(1, -1, 1) - dx[:, None]) ** 2
                + (dy_tab.view(1, -1, 1) - dy[:, None]) ** 2
            )
        ).sum(dim=1)

        return dict(
            sim_max=sim_max,
            pmax=pmax,
            entropy=entropy,
            margin=margin,
            dx=dx,
            dy=dy,
            disp_var=disp_var,
        )

    def forward(self, f0, fr, return_evidence=True):
        B, C, h, w = f0.shape
        q = F.normalize(self.q(f0), dim=1, eps=self.eps)
        k = F.normalize(self.k(fr), dim=1, eps=self.eps)
        v = self.v(fr)

        flat, valid = self._idx(h, w, f0.device)
        n = h * w
        kf = k.reshape(B, C, n)
        vf = v.reshape(B, C, n)
        qf = q.reshape(B, C, n)
        valid_t = valid.t()

        # D0 audit should be deterministic and easy to inspect. For very large
        # images, use the inherited chunk value; evidence is accumulated exactly.
        if self.chunk is None or n <= self.chunk:
            kr = self._gather_cand(kf, flat)
            vr = self._gather_cand(vf, flat)

            # Keep raw cosine separate from temperature-scaled logits.
            sim = torch.einsum("bcn,bcjn->bjn", qf, kr)
            logits = sim / self.tau
            logits = logits.masked_fill(~valid_t[None], float("-inf"))
            p = torch.softmax(logits, dim=1)
            t = torch.einsum("bjn,bcjn->bcn", p, vr).reshape(B, C, h, w)

            if not return_evidence:
                return t

            ev = self._summarize(sim.masked_fill(~valid_t[None], -1.0), p)
            ev = {k: x.reshape(B, 1, h, w) for k, x in ev.items()}
            return t, ev

        # Query-chunked version.
        t = f0.new_empty(B, C, n)
        chunks = {k: [] for k in (
            "sim_max", "pmax", "entropy", "margin", "dx", "dy", "disp_var"
        )}
        for s in range(0, n, self.chunk):
            e = min(s + self.chunk, n)
            fl = flat[s:e]
            qc = qf[:, :, s:e]
            kr = self._gather_cand(kf, fl)
            vr = self._gather_cand(vf, fl)

            sim = torch.einsum("bcm,bcjm->bjm", qc, kr)
            vm = valid_t[:, s:e]
            logits = (sim / self.tau).masked_fill(~vm[None], float("-inf"))
            p = torch.softmax(logits, dim=1)
            t[:, :, s:e] = torch.einsum("bjm,bcjm->bcm", p, vr)

            if return_evidence:
                ev = self._summarize(sim.masked_fill(~vm[None], -1.0), p)
                for name in chunks:
                    chunks[name].append(ev[name])

        t = t.reshape(B, C, h, w)
        if not return_evidence:
            return t

        ev = {
            name: torch.cat(parts, dim=-1).reshape(B, 1, h, w)
            for name, parts in chunks.items()
        }
        return t, ev


def proposal_evidence(gate, delta, d_correction, f0, t):
    """Lightweight Proposal-side evidence maps.

    Returns full-res and feature-res evidence separately so the caller can use
    the project's exact G64 aggregation rather than resizing them ad hoc.
    """
    full = dict(
        gate_v2=gate,
        abs_D=d_correction.abs().mean(dim=1, keepdim=True),
        abs_delta=delta.abs().mean(dim=1, keepdim=True),
    )
    feature = dict(
        f0_minus_t=(f0 - t).abs().mean(dim=1, keepdim=True),
    )
    return dict(full=full, feature=feature)
