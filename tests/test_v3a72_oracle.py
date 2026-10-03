"""T2: binary block oracle on energy-valid cells is the Base/R1 binary choice."""

import unittest

import torch

from v3a5_runtime import (block_energy, block_mean, energy_mask, expand_gate,
                          prepare_geometry, target_geometry)
from v3a7_runtime import block_utility
from v3a72_runtime import binary_block_oracle_q, q_to_grid


class TestV3A72Oracle(unittest.TestCase):
    def test_valid_blocks_match_better_of_base_r1(self):
        torch.manual_seed(0)
        geom = prepare_geometry(target_geometry(40, 60, 'g64'), 'cpu')
        y0 = torch.randn(1, 3, 40, 60)
        D = torch.randn(1, 3, 40, 60) * 0.2
        H = y0 + 0.4 * D
        U = block_utility(y0, H, D, geom)
        mask_t = energy_mask(block_energy(D, geom), 1e-8)
        q = binary_block_oracle_q(U.reshape(-1).numpy(), mask_t.reshape(-1).numpy())
        qg = q_to_grid(q, nby=geom['shape'][0], nbx=geom['shape'][1], device='cpu')
        y_or = y0 + expand_gate(qg, geom) * D
        y_b, y_r = y0, y0 + D
        mse = lambda a: block_mean((a - H).pow(2).mean(dim=1, keepdim=True), geom)
        m_or = mse(y_or).reshape(-1)
        m_b = mse(y_b).reshape(-1)
        m_r = mse(y_r).reshape(-1)
        valid = mask_t.reshape(-1) > 0.5
        better = torch.minimum(m_b, m_r)
        self.assertTrue(torch.all(m_or[valid] <= better[valid] + 1e-5))
        # invalid stays at Base
        invalid = ~valid
        if invalid.any():
            self.assertTrue(torch.allclose(m_or[invalid], m_b[invalid], atol=1e-5))


if __name__ == '__main__':
    unittest.main()
