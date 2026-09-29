"""Evidence-aware verifier head for V3-A.5D1.

Literature idea:
ReBaIR treats matching confidence / offsets as metadata features instead of
assuming confidence itself is the final restoration gate.

Design for this project:
- current common verifier feature is pooled to exact G64 first;
- evidence is also aggregated to exact G64 by the caller;
- concatenate evidence only at the head input;
- initialize all new evidence-channel weights to zero, so step-0 is identical
  to the current verifier.

This isolates "does explicit evidence help?" from receptive-field changes.
"""

import torch
import torch.nn as nn


class EvidenceAwareHead(nn.Module):
    """Clone the current 1x1 -> 3x3 -> 1x1 head with extra evidence channels."""

    def __init__(self, base_head0, base_head1, base_head2, act, evidence_ch):
        super().__init__()
        self.evidence_ch = int(evidence_ch)
        if self.evidence_ch <= 0:
            raise ValueError("evidence_ch must be positive")

        common_ch = int(base_head0.in_channels)
        hidden = int(base_head0.out_channels)

        self.act = act
        self.head0 = nn.Conv2d(
            common_ch + self.evidence_ch,
            hidden,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=(base_head0.bias is not None),
        )
        self.head1 = nn.Conv2d(
            base_head1.in_channels,
            base_head1.out_channels,
            kernel_size=base_head1.kernel_size,
            stride=base_head1.stride,
            padding=base_head1.padding,
            bias=(base_head1.bias is not None),
        )
        self.head2 = nn.Conv2d(
            base_head2.in_channels,
            base_head2.out_channels,
            kernel_size=base_head2.kernel_size,
            stride=base_head2.stride,
            padding=base_head2.padding,
            bias=(base_head2.bias is not None),
        )

        with torch.no_grad():
            # Copy the current verifier exactly.
            self.head0.weight[:, :common_ch].copy_(base_head0.weight)
            self.head0.weight[:, common_ch:].zero_()
            if base_head0.bias is not None:
                self.head0.bias.copy_(base_head0.bias)

            self.head1.load_state_dict(base_head1.state_dict(), strict=True)
            self.head2.load_state_dict(base_head2.state_dict(), strict=True)

    def forward(self, common_g64, evidence_g64):
        if evidence_g64.shape[1] != self.evidence_ch:
            raise ValueError(
                "expected %d evidence channels, got %d"
                % (self.evidence_ch, evidence_g64.shape[1])
            )
        if common_g64.shape[-2:] != evidence_g64.shape[-2:]:
            raise ValueError(
                "common/evidence spatial mismatch: %s vs %s"
                % (common_g64.shape[-2:], evidence_g64.shape[-2:])
            )

        h = torch.cat([common_g64, evidence_g64], dim=1)
        h = self.act(self.head0(h))
        h = self.act(self.head1(h))
        return torch.sigmoid(self.head2(h))


def evidence_weight_max_abs(head):
    """Diagnostic: evidence weights are exactly zero at initialization."""
    n = int(head.evidence_ch)
    return float(head.head0.weight[:, -n:].detach().abs().max())
