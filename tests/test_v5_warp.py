"""Warp identity and integer-flow gather."""

import unittest

import torch

from model.V5FlowAlign import warp_by_flow


class TestV5Warp(unittest.TestCase):
    def test_zero_flow_identity_interior(self):
        x = torch.randn(1, 4, 12, 16)
        flow = torch.zeros(1, 2, 12, 16)
        y = warp_by_flow(x, flow, mode='bilinear')
        self.assertLessEqual(float((x[:, :, 1:-1, 1:-1] - y[:, :, 1:-1, 1:-1]).abs().max()),
                             1e-5)

    def test_integer_shift_sign(self):
        x = torch.zeros(1, 1, 8, 8)
        x[0, 0, 3, 4] = 1.0
        flow = torch.zeros(1, 2, 8, 8)
        flow[:, 0] = 2  # dx
        flow[:, 1] = -1  # dy
        y = warp_by_flow(x, flow, mode='nearest')
        # output (y,x) samples source (x+2, y-1) → peak at (3-(-1), 4-2)=(4,2)? 
        # sample (x+dx,y+dy)=(4+2,3-1)=(6,2) from output (3,4) wait.
        # Output pixel (oy, ox)=(3,4) samples source (4+2, 3-1)=(6,2) — that's 0.
        # Peak of 1 is at source (3,4). We want output (oy,ox) s.t. (ox+dx, oy+dy)=(4,3)
        # ox+2=4 ⇒ ox=2; oy-1=3 ⇒ oy=4. y[0,0,4,2] ≈ 1
        self.assertGreater(float(y[0, 0, 4, 2]), 0.99)
