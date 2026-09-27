#!/usr/bin/env python
"""V3-A.4 §23 acceptance checklist, runnable before any training.

Every box is checked against the real artifacts and the real model, so a
failure here is a hard stop rather than something discovered 3000 steps later.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CLI = sys.argv[1:]
sys.argv = [sys.argv[0]]

from local_refine_runtime import sha256                           # noqa: E402
from model.V3A4Verifier import ACTION_CH, V3A4Refiner, build_shared_init  # noqa: E402
from v3a4_runtime import (action_features_norm, action_features_raw,  # noqa: E402
                          assert_action_connectivity, load_r1_proposal_strict,
                          masked_mae)
from v3a_runtime import contrast_compress, exposure_gain          # noqa: E402

SRC = '/root/data/experiments/v3a1_lolv2real'
ROOT = '/root/data/experiments/v3a4_lolv2real'
R1_CK = 'R1_v2stable_naive_s42/checkpoint_03000.pt'


def main():
    boxes = []

    def check(name, fn):
        try:
            msg = fn()
            boxes.append((True, name, msg))
        except Exception as e:                                    # noqa: BLE001
            boxes.append((False, name, '%s: %s' % (type(e).__name__, e)))

    lock = json.load(open(os.path.join(ROOT, 'artifact_lock.json'), encoding='utf-8'))
    split = json.load(open(os.path.join(ROOT, 'splits', 'split.json'), encoding='utf-8'))
    rms = json.load(open(os.path.join(ROOT, 'action_stats',
                                      'action_norm.json')))['rms_D']

    m = V3A4Refiner('none')

    def load_strict():
        load_r1_proposal_strict(m, os.path.join(SRC, R1_CK))
        assert float(m.proposal.c_out.weight.abs().max()) > 0
        return 'proposal loaded, c_out max %.3e' % float(
            m.proposal.c_out.weight.abs().max())
    check('R1 proposal strict load, c_out non-zero', load_strict)

    def endpoints():
        m.eval()
        g = torch.Generator().manual_seed(0)
        y0 = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
        low = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
        ref = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
        with torch.no_grad():
            o0, _ = m(y0, ref, low=low, force_qv=0.0)
            o1, _ = m(y0, ref, low=low, force_qv=1.0)
            sr, _ = m.proposal(y0, ref)
        assert torch.equal(o0, y0), 'q=0 != Base'
        assert torch.equal(o1, sr), 'q=1 != proposal'
        assert not torch.equal(o1, y0), 'endpoint check is vacuous'
        return 'q=0 == Base, q=1 == proposal (and != Base)'
    check('q=0 exact Base / q=1 exact R1', endpoints)

    def frozen_mode():
        mm = V3A4Refiner('none')
        load_r1_proposal_strict(mm, os.path.join(SRC, R1_CK))
        for p in mm.proposal.parameters():
            p.requires_grad_(False)
        mm.proposal.eval()
        mm.train()
        assert mm.proposal.training is False, 'm.train() unfroze the proposal'
        before = {k: v.clone() for k, v in mm.proposal.state_dict().items()}
        opt = torch.optim.Adam(mm.verifier.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(1)
        y0 = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
        low = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
        ref = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
        mm.train()
        out, _ = mm(y0, ref, low=low)
        opt.zero_grad(set_to_none=True)
        out.abs().mean().backward()
        opt.step()
        drift = max(float((mm.proposal.state_dict()[k] - before[k]).abs().max())
                    for k in before)
        assert drift == 0.0, 'drift %.3e' % drift
        return 'eval mode held through m.train(); drift = 0'
    check('proposal eval + requires_grad False + drift 0', frozen_mode)

    def mismatch_isolation():
        tr = json.load(open(os.path.join(ROOT, 'mappings',
                                         'mismatch_train_575.json')))
        dv = json.load(open(os.path.join(ROOT, 'mappings',
                                         'mismatch_dev_64.json')))
        assert set(tr) == set(split['train'])
        assert set(dv) == set(split['dev'])
        assert not (set(tr.values()) & set(split['dev']))
        assert not (set(dv.values()) & set(split['train']))
        return 'train/dev donors fully isolated'
    check('mismatch donor isolation', mismatch_isolation)

    def lock_binding():
        for k, p in (('proposal_sha256', os.path.join(SRC, R1_CK)),
                     ('mismatch_train_sha256',
                      os.path.join(ROOT, 'mappings', 'mismatch_train_575.json')),
                     ('mismatch_dev_sha256',
                      os.path.join(ROOT, 'mappings', 'mismatch_dev_64.json')),
                     ('action_norm_sha256',
                      os.path.join(ROOT, 'action_stats', 'action_norm.json')),
                     ('energy_stats_sha256',
                      os.path.join(ROOT, 'action_stats', 'energy.json')),
                     ('split_sha256', os.path.join(ROOT, 'splits', 'split.json'))):
            assert sha256(p) == lock[k], '%s drifted' % k
        return 'all %d SHAs match' % 6
    check('artifact lock SHAs', lock_binding)

    def exposure_semantics():
        x = torch.tensor([-1.0, 0.0, 1.0])
        new = (exposure_gain(x, 0.5) + 1) / 2
        assert torch.allclose(new, torch.tensor([0.0, 0.25, 0.5]), atol=1e-6)
        old = (contrast_compress(x, 0.5) + 1) / 2
        assert torch.allclose(old, torch.tensor([0.25, 0.5, 0.75]), atol=1e-6)
        return 'exposure_gain black->black; contrast_compress black->0.25'
    check('exposure semantics', exposure_semantics)

    def shared_init():
        init = build_shared_init(seed=42, action_rms=rms)
        s = {k: v for k, v in init.items()}
        for key in s['none']:
            if key == 'verifier.head0.weight':
                assert torch.equal(s['none'][key], s['raw'][key][:, :160])
                assert torch.equal(s['norm'][key], s['raw'][key])
            else:
                assert torch.equal(s['none'][key], s['raw'][key])
                assert torch.equal(s['norm'][key], s['raw'][key])
        assert float(s['raw']['verifier.head0.weight'][:, -ACTION_CH:].abs().max()) == 0
        g = torch.Generator().manual_seed(2)
        y0 = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
        low = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
        ref = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
        outs = []
        for mode in ('none', 'raw', 'norm'):
            mm = V3A4Refiner(mode, action_rms=rms).eval()
            mm.load_state_dict(init[mode])
            with torch.no_grad():
                outs.append(mm(y0, ref, low=low)[0])
        assert torch.allclose(outs[0], outs[1], atol=1e-6)
        assert torch.allclose(outs[0], outs[2], atol=1e-6)
        return 'common weights bit-equal; step0 C0==C1==C2'
    check('shared init + step0 equality', shared_init)

    def connectivity():
        msgs = []
        for mode in ('raw', 'norm'):
            mm = V3A4Refiner(mode, action_rms=rms)
            fwd, bwd = assert_action_connectivity(mm, ACTION_CH)
            assert fwd > 0, '%s forward dead' % mode
            assert bwd > 0, '%s backward dead' % mode
            msgs.append('%s fwd %.3e bwd %.3e' % (mode, fwd, bwd))
        return '; '.join(msgs)
    check('action forward + backward connectivity', connectivity)

    def action_scale():
        """Measured on a REAL D from the frozen proposal.

        rms is defined on D4 (post area-pooling), so a synthetic pre-pool tensor
        will not reproduce the statistic -- the check has to use real action.
        """
        from dataset.lolv2real_v3a import TrainSet, pairs_from_manifest
        from option import parser as option_parser
        ns = option_parser.parse_args([])
        ns.dataset_dir = '/root/data/datasets/lol-v2-real'
        ns.v3a_ref_variant = 'nanobanana_ref_v2'
        pairs = pairs_from_manifest(os.path.join(SRC, 'manifests',
                                                 'refiner_train.csv'))[:4]
        ds = TrainSet(ns, crop_size=128, pairs=pairs,
                      y0_cache=os.path.join(SRC, 'cache_y0_lolbase',
                                            'refiner_train'), split='Train')
        mm = V3A4Refiner('norm', action_rms=rms).eval()
        load_r1_proposal_strict(mm, os.path.join(SRC, R1_CK))
        e_raw, e_norm = [], []
        with torch.no_grad():
            for i in range(4):
                b = ds[i]
                _sr, aux = mm.proposal(b['Y0'][None], b['Ref'][None])
                d4, e4 = action_features_raw(aux['gate'] * aux['delta'])
                _d4n, e4n = action_features_norm(aux['gate'] * aux['delta'], rms)
                e_raw.append(float(e4.mean()))
                e_norm.append(float(e4n.mean()))
        assert np.mean(e_norm) > 3 * np.mean(e_raw), \
            'normalization did not lift the scale'
        assert 0.1 < np.mean(e_norm) < 10.0, 'E4_norm still far from O(1): %.4f' \
            % np.mean(e_norm)
        return ('real D: E4 raw %.3e -> norm %.3e (x%.0f), D4 rms norm-to-unit'
                % (np.mean(e_raw), np.mean(e_norm),
                   np.mean(e_norm) / np.mean(e_raw)))
    check('D4_norm/E4_norm are O(1)', action_scale)

    def stats_from_train_only():
        assert lock['split_sha256'] == sha256(os.path.join(ROOT, 'splits', 'split.json'))
        an = json.load(open(os.path.join(ROOT, 'action_stats', 'action_norm.json')))
        assert an['n'] == 575, an['n']
        assert an['split_sha256'] == lock['split_sha256']
        return 'n=575, split_sha bound'
    check('normalization stats from train575 only', stats_from_train_only)

    ok = sum(1 for b in boxes if b[0])
    print('══ V3-A.4 acceptance (§23) ══')
    for good, name, msg in boxes:
        print('  [%s] %-46s %s' % ('x' if good else ' ', name, msg))
    print('\n  %d/%d passed' % (ok, len(boxes)))
    return 0 if ok == len(boxes) else 1


if __name__ == '__main__':
    sys.exit(main())
