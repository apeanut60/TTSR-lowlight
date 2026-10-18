"""Old Y0 caches must not be relabeled as a new checkpoint."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from local_refine_runtime import (  # noqa: E402
    assert_cache_compatible, begin_cache, cache_manifest_sha, write_metadata,
)


def _expect(sha, ids, version='v1'):
    return dict(
        base_checkpoint_sha256=sha,
        manifest_sha256=cache_manifest_sha([{'sample_id': i} for i in ids]),
        p2_config={'texture_fusion_lambda': 0.25, 'stage_mode_lv2': 'branch_bypass'},
        preprocess_version=version,
        dtype='float32',
        range='[-1,1]',
        sample_ids=list(ids),
    )


def _write(cache_dir, expect):
    os.makedirs(cache_dir, exist_ok=True)
    for sid in expect['sample_ids']:
        np.save(os.path.join(cache_dir, sid + '.npy'), np.zeros((3, 4, 4), np.float32))
    write_metadata(cache_dir, dict(complete=True, layout='CHW', **expect))


def test_s1_cache_rejected_for_s1c():
    cache = tempfile.mkdtemp()
    ids = ['r000', 'r001']
    old = _expect('sha-s1-1400', ids)
    _write(cache, old)
    new = _expect('sha-s1c-400', ids)
    before = json.load(open(os.path.join(cache, 'metadata.json')))
    try:
        assert_cache_compatible(cache, new)
        raised = False
    except SystemExit as exc:
        raised = True
        assert 'sha' in str(exc).lower() or 'does not match' in str(exc)
    assert raised
    after = json.load(open(os.path.join(cache, 'metadata.json')))
    assert after['base_checkpoint_sha256'] == before['base_checkpoint_sha256'] == 'sha-s1-1400'
    assert os.path.isfile(os.path.join(cache, 'r000.npy'))
    shutil.rmtree(cache)


def test_match_and_rebuild():
    cache = tempfile.mkdtemp()
    expect = _expect('sha-a', ['a'])
    assert begin_cache(cache, expect, rebuild=False) == 'empty'
    _write(cache, expect)
    assert assert_cache_compatible(cache, expect) == 'match'
    assert begin_cache(cache, expect, rebuild=True) == 'rebuild'
    assert not os.path.isfile(os.path.join(cache, 'a.npy'))
    assert not os.path.isfile(os.path.join(cache, 'metadata.json'))
    shutil.rmtree(cache)
