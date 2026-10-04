"""V3-B.1 reference adaptation modules. Not used in B0 training.

B1a: identity-preserving (zero-init Δγ/Δβ, T_adapt==T at step 0).
B1b: MASA-faithful SAM semantics (step 0 is AdaIN, NOT identity).
Clean-room rewrite from the MASA paper/official SAM *behavior*, not source copy.
"""

import torch
import torch.nn as nn
from model.V3BResidualFusion import V3B0ResidualFusion, count_params

ARMS = ('B1a_identity_adapt', 'B1b_masa_adapt')


class V3B1Model(nn.Module):
    """Frozen-matcher T → T_adapt → same B0 residual head."""

    def __init__(self, arm='B1a_identity_adapt'):
        super().__init__()
        if arm not in ARMS:
            raise SystemExit('unknown B1 arm %r' % arm)
        self.arm = arm
        self.adapt = (V3B1aIdentityAdapt() if arm == 'B1a_identity_adapt'
                      else V3B1bMasaAdapt())
        self.head = V3B0ResidualFusion()

    def forward(self, F0, T, out_hw, return_aux=False):
        t_a, aux = self.adapt.adapt_with_aux(F0, T)
        delta = self.head(F0, t_a, out_hw)
        if return_aux:
            return delta, aux
        return delta


class V3B1aIdentityAdapt(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.norm = nn.InstanceNorm2d(ch, affine=False)
        self.style = nn.Sequential(
            nn.Conv2d(3 * ch, ch, 3, 1, 1, bias=True),
            nn.GELU(),
        )
        self.d_gamma = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)
        self.d_beta = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)
        nn.init.zeros_(self.d_gamma.weight)
        nn.init.zeros_(self.d_gamma.bias)
        nn.init.zeros_(self.d_beta.weight)
        nn.init.zeros_(self.d_beta.bias)

    def adapt_with_aux(self, F0, T):
        t_n = self.norm(T)
        style = self.style(torch.cat([F0, T, F0 - T], dim=1))
        dg, db = self.d_gamma(style), self.d_beta(style)
        t_a = T + dg * t_n + db
        aux = dict(delta_gamma=dg, delta_beta=db, t_adapt=t_a,
                   delta_t=(t_a - T).detach())
        return t_a, aux

    def forward(self, F0, T):
        return self.adapt_with_aux(F0, T)[0]


class V3B1bMasaAdapt(nn.Module):
    """MASA SAM-like: T_adapt = IN(T) * (std(F0)+Δγ) + (mean(F0)+Δβ).

    Zero-init Δγ/Δβ ⇒ step0 AdaIN(T → F0 stats), not identity.
    """

    def __init__(self, ch=32):
        super().__init__()
        self.norm = nn.InstanceNorm2d(ch, affine=False)
        self.shared = nn.Sequential(
            nn.Conv2d(2 * ch, ch, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.d_gamma = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)
        self.d_beta = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)
        nn.init.zeros_(self.d_gamma.weight)
        nn.init.zeros_(self.d_gamma.bias)
        nn.init.zeros_(self.d_beta.weight)
        nn.init.zeros_(self.d_beta.bias)

    def adapt_with_aux(self, F0, T):
        t_n = self.norm(T)
        style = self.shared(torch.cat([F0, T], dim=1))
        mu0 = F0.mean(dim=(2, 3), keepdim=True)
        std0 = F0.std(dim=(2, 3), keepdim=True)
        dg, db = self.d_gamma(style), self.d_beta(style)
        t_a = t_n * (std0 + dg) + (mu0 + db)
        aux = dict(delta_gamma=dg, delta_beta=db, t_adapt=t_a,
                   delta_t=(t_a - T).detach())
        return t_a, aux

    def forward(self, F0, T):
        return self.adapt_with_aux(F0, T)[0]
