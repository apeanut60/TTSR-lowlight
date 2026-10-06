"""Chunked hard correspondence: P=argmax cos, C=max cos. Index/flow detached; C has grad."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MATCH_CHUNK = 1024
CORRELATION_TYPE = 'chunked_global_cosine'
SEARCH_RANGE = 'global'


@dataclass
class MatchResult:
    index: Tensor       # [B,H,W] flattened source index (detached)
    confidence: Tensor  # [B,1,H,W] gathered cosine (live)
    flow: Tensor        # [B,2,H,W] (dx, dy) detached
    sim_max: Tensor     # same as confidence
    margin: Tensor      # [B,1,H,W] sim_max - second
    entropy: Tensor     # [B,1,H,W] softmax entropy over source, detached


class ChunkedHardMatcher(nn.Module):
    """P_i = argmax_j cos(q_i, k_j). Optional local radius with OOB mask."""

    def __init__(self, chunk_size: int = MATCH_CHUNK, eps: float = 1e-6,
                 radius=None):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.eps = float(eps)
        self.radius = None if radius is None else int(radius)

    def forward(self, target_feat: Tensor, ref_feat: Tensor) -> MatchResult:
        if target_feat.shape != ref_feat.shape:
            raise ValueError('shape mismatch: %s vs %s'
                             % (tuple(target_feat.shape), tuple(ref_feat.shape)))
        b, c, h, w = target_feat.shape
        n = h * w
        q = F.normalize(target_feat.flatten(2).transpose(1, 2), dim=-1, eps=self.eps)
        k = F.normalize(ref_feat.flatten(2).transpose(1, 2), dim=-1, eps=self.eps)
        kt = k.transpose(1, 2)

        best_idx = torch.empty(b, n, dtype=torch.long, device=target_feat.device)
        best_sim = torch.empty(b, n, dtype=target_feat.dtype, device=target_feat.device)
        second_sim = torch.empty(b, n, dtype=target_feat.dtype, device=target_feat.device)
        entropy = torch.empty(b, n, dtype=target_feat.dtype, device=target_feat.device)

        ty = torch.arange(h, device=target_feat.device).view(h, 1).expand(h, w).reshape(n)
        tx = torch.arange(w, device=target_feat.device).view(1, w).expand(h, w).reshape(n)
        sy_all = torch.div(torch.arange(n, device=target_feat.device), w,
                           rounding_mode='floor')
        sx_all = torch.arange(n, device=target_feat.device) % w

        win_mask = None
        if self.radius is not None:
            r = self.radius
            dy = (sy_all[None, :] - ty[:, None]).abs()
            dx = (sx_all[None, :] - tx[:, None]).abs()
            win_mask = (dx <= r) & (dy <= r)  # [Nq, Ns]

        with torch.no_grad():
            for s in range(0, n, self.chunk_size):
                e = min(s + self.chunk_size, n)
                sim = torch.bmm(q[:, s:e].detach(), kt.detach())
                if win_mask is not None:
                    m = win_mask[s:e][None].expand(b, -1, -1)
                    sim = sim.masked_fill(~m, float('-inf'))
                topv, topi = sim.topk(2, dim=-1)
                best_sim[:, s:e] = topv[:, :, 0]
                second_sim[:, s:e] = topv[:, :, 1]
                best_idx[:, s:e] = topi[:, :, 0]
                p = torch.softmax(sim, dim=-1)
                entropy[:, s:e] = -(p * (p.clamp_min(1e-8).log())).sum(dim=-1)

        gathered = torch.gather(k, 1, best_idx.unsqueeze(-1).expand(-1, -1, c))
        conf = (q * gathered).sum(dim=-1)

        sy = torch.div(best_idx, w, rounding_mode='floor')
        sx = best_idx.remainder(w)
        flow_x = (sx.float() - tx.float()[None]).to(target_feat.dtype)
        flow_y = (sy.float() - ty.float()[None]).to(target_feat.dtype)
        flow = torch.stack((flow_x, flow_y), dim=1).reshape(b, 2, h, w).detach()

        conf_map = conf.reshape(b, 1, h, w)
        return MatchResult(
            index=best_idx.reshape(b, h, w).detach(),
            confidence=conf_map,
            flow=flow,
            sim_max=conf_map.detach(),
            margin=(best_sim - second_sim).reshape(b, 1, h, w),
            entropy=entropy.reshape(b, 1, h, w),
        )
