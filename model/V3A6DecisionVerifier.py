"""V3-A.6 verifier: MultiScale A1 architecture under decision-aligned arms.

Both A0_qstar and A1_decision_mse share the identical network
(V3A5D2Verifier A1_multiscale). Arms differ only by training objective.
"""

import torch

from model.V3A5D2Verifier import V3A5D2Verifier, build_d2_shared_init
from v3a5_runtime import snapshot_
from v3a6_runtime import ARMS, BOTTLENECK


def build_v3a6_model(arm='A1_decision_mse', bottleneck=BOTTLENECK):
    if arm not in ARMS:
        raise SystemExit('unknown V3-A.6 arm %r (expected %r)' % (arm, ARMS))
    # Always MultiScale; arm name only selects loss in train/eval.
    return V3A5D2Verifier('A1_multiscale', bottleneck=bottleneck)


def build_v3a6_shared_init(seed=42, bottleneck=BOTTLENECK):
    """Identical MultiScale weights for both loss arms; residual zero at step0."""
    d2 = build_d2_shared_init(seed=seed, bottleneck=bottleneck)
    snap = d2['A1_multiscale']
    # deep clone so arms cannot alias
    a0 = {k: v.detach().clone() for k, v in snap.items()}
    a1 = {k: v.detach().clone() for k, v in snap.items()}
    return {'A0_qstar': a0, 'A1_decision_mse': a1}


def load_arm(arm, init_blob_or_path, device='cpu'):
    model = build_v3a6_model(arm).to(device)
    if isinstance(init_blob_or_path, str):
        blob = torch.load(init_blob_or_path, map_location='cpu')
        if isinstance(blob, dict) and arm in blob:
            sd = blob[arm]
        elif isinstance(blob, dict) and 'model' in blob:
            sd = blob['model']
        else:
            sd = blob
    else:
        sd = init_blob_or_path[arm] if arm in init_blob_or_path else init_blob_or_path
    model.load_state_dict(sd, strict=True)
    return model
