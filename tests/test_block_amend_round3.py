"""Round-3 engineering tests: RNG device/schema, manifest paths, eval identity.

CPU plus, when CUDA is visible, one real GPU resume. The LTE is a conv stub
(no VGG). This does not validate training quality.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import stat
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from tests.test_block_amend_round2 import (  # noqa: E402
    _patch_lte, _synth_tree, _tiny_args,
)


def _raises(exc, fn, contains=None):
    try:
        fn()
    except exc as e:
        if contains is not None and contains not in str(e):
            raise AssertionError('%r not in %s' % (contains, e))
        return str(e)
    raise AssertionError('expected %s' % exc.__name__)


def _gens():
    return torch.Generator().manual_seed(11), torch.Generator().manual_seed(29)


def _no_ndarray(obj, path='pack'):
    if isinstance(obj, np.ndarray):
        raise AssertionError('ndarray at %s' % path)
    if isinstance(obj, dict):
        for k, v in obj.items():
            _no_ndarray(v, '%s.%s' % (path, k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _no_ndarray(v, '%s[%d]' % (path, i))


# ---------------------------------------------------------------- RNG
def test_rng_roundtrip_and_dtype():
    from block_amend_ckpt import capture_rng_state, restore_rng_state
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    g_train, g_val = _gens()
    saved = capture_rng_state(g_train, g_val)
    assert saved['format'] == 'block_amend_rng_v1'
    assert saved['torch_cpu'].dtype == torch.uint8
    assert saved['torch_cpu'].device.type == 'cpu' and saved['torch_cpu'].is_contiguous()
    for name in ('train_generator', 'val_generator'):
        t = saved[name]
        assert t.dtype == torch.uint8 and t.device.type == 'cpu' and t.dim() == 1
    exp_py = random.random()
    exp_np = float(np.random.rand())
    exp_th = torch.rand(4).clone()
    exp_g = torch.rand(4, generator=g_train).clone()
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    g_train.manual_seed(3)
    g_val.manual_seed(4)
    restore_rng_state(saved, g_train, g_val)
    assert random.random() == exp_py
    assert abs(float(np.random.rand()) - exp_np) < 1e-12
    assert torch.equal(torch.rand(4), exp_th)
    assert torch.equal(torch.rand(4, generator=g_train), exp_g)
    # Wrong dtype is rejected, not cast into a plausible state.
    bad = dict(saved)
    bad['torch_cpu'] = saved['torch_cpu'].float()
    _raises(RuntimeError, lambda: restore_rng_state(bad, *_gens()), 'uint8')
    # A uint8 tensor that a CUDA map_location moved off CPU is brought back.
    if torch.cuda.is_available():
        moved = dict(saved)
        moved['torch_cpu'] = saved['torch_cpu'].cuda()
        assert moved['torch_cpu'].device.type == 'cuda'
        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        gt, gv = _gens()
        restore_rng_state(moved, gt, gv)
        assert random.random() == exp_py
    # Missing loader state is an error, not a reseed.
    missing = dict(saved)
    missing['train_generator'] = None
    _raises(RuntimeError, lambda: restore_rng_state(missing, *_gens()), 'generator')
    # Device-count mismatch is not trimmed.
    if saved['torch_cuda'] is not None:
        mismatch = dict(saved)
        mismatch['torch_cuda'] = list(saved['torch_cuda']) + list(saved['torch_cuda'])
        _raises(RuntimeError, lambda: restore_rng_state(mismatch, *_gens()),
                'device-count')
    print('rng_roundtrip ok')


def test_pack_is_weights_only_friendly_and_schema3_is_refused():
    from block_amend_ckpt import (
        CHECKPOINT_SCHEMA_VERSION, _LOAD_SUPPORTS_WEIGHTS_ONLY,
        atomic_torch_save, capture_rng_state, convert_legacy_rng,
        load_local_checkpoint, migrate_schema3_checkpoint, require_resume_fields,
        restore_rng_state,
    )
    assert CHECKPOINT_SCHEMA_VERSION == 4
    g1, g2 = _gens()
    rng = capture_rng_state(g1, g2)
    _no_ndarray(rng)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'schema4.pt')
    atomic_torch_save({'rng': rng, 'n': 1}, path)
    loaded = load_local_checkpoint(path)
    _no_ndarray(loaded)
    assert loaded['rng']['torch_cpu'].device.type == 'cpu'
    if torch.cuda.is_available():
        # The old map_location=cuda load is what put ByteTensors on the GPU.
        hostile = torch.load(path, map_location='cuda')
        assert hostile['rng']['torch_cpu'].device.type == 'cuda'
    if _LOAD_SUPPORTS_WEIGHTS_ONLY:
        again = torch.load(path, map_location='cpu', weights_only=True)
        assert again['n'] == 1
        print('weights_only=True load executed on this torch')
    else:
        print('weights_only NOT available in torch %s; safe-load branch was '
              'not executed. Compatibility load was used for a local file.'
              % torch.__version__)

    # Policy: when the argument exists, we pass True once and do not retry.
    import block_amend_ckpt as ckpt
    calls = []

    def fake_load(file, map_location=None, weights_only=None, **kwargs):
        calls.append(weights_only)
        if weights_only is None:
            raise AssertionError('weights_only was not explicit')
        if weights_only is True:
            raise RuntimeError('unpickle refused')
        return {'ok': True}

    saved_flag = ckpt._LOAD_SUPPORTS_WEIGHTS_ONLY
    saved_load = ckpt.torch.load
    try:
        ckpt._LOAD_SUPPORTS_WEIGHTS_ONLY = True
        ckpt.torch.load = fake_load
        _raises(RuntimeError, lambda: ckpt.load_local_checkpoint(path), 'unpickle')
        assert calls == [True], calls
        calls.clear()
        ckpt.load_local_checkpoint(path, trust_legacy=True)
        assert calls == [False], calls
    finally:
        ckpt._LOAD_SUPPORTS_WEIGHTS_ONLY = saved_flag
        ckpt.torch.load = saved_load

    # Schema 3 is refused. Migration is explicit and writes a new file.
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    gt, gv = _gens()
    legacy_rng = {
        'python_random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None),
        'train_generator': gt.get_state(),
        'val_generator': gv.get_state(),
    }
    exp_py = random.random()
    converted = convert_legacy_rng(legacy_rng)
    _no_ndarray(converted)
    random.seed(0)
    restore_rng_state(converted, gt, gv)
    assert random.random() == exp_py

    src = os.path.join(tmp, 'schema3.pt')
    # Minimal pack that passes the model-identity check and nothing more.
    from model.CSFIUNet import CSFIUNet
    net = CSFIUNet(num_res_blocks='1+1+1+1', n_feats=16)
    raw3 = {
        'checkpoint_schema_version': 3,
        'model_variant': 'csfi_lfca_v1',
        'model_state_dict': {'MainNet.' + k: v.cpu()
                             for k, v in net.state_dict().items()},
        'optimizer_state_dict': {'state': {}, 'param_groups': []},
        'scheduler_state_dict': {'last_epoch': 1},
        'epoch': 1, 'global_step': 2,
        'best_val_psnr': 1.0, 'best_epoch': 1,
        'config': {'model_variant': 'csfi_lfca_v1', 'restoration_block': 'lfca',
                   'restoration_blocks': '1,1', 'restoration_heads': 1,
                   'restoration_ffn_expansion': 4, 'lowpass_kernel': 'binomial',
                   'lowpass_kernel_size': 5},
        'split_identity': {'train_hash': 't', 'val_hash': 'v',
                           'ref_subdir': 'nanobanana_ref_v2',
                           'manifest_hash_version': 'v2',
                           'reference_source': 'generated_nano'},
        'rng': legacy_rng,
        'metric_protocol': 'rgb_float_clipped_v1',
    }
    torch.save(raw3, src)
    src_bytes = open(src, 'rb').read()
    _raises(RuntimeError, lambda: require_resume_fields(load_local_checkpoint(src), src),
            'Schema 3')
    dst = os.path.join(tmp, 'schema4_migrated.pt')
    migrate_schema3_checkpoint(src, dst)
    assert open(src, 'rb').read() == src_bytes
    _raises(FileExistsError, lambda: migrate_schema3_checkpoint(src, dst))
    migrated = load_local_checkpoint(dst)
    assert migrated['checkpoint_schema_version'] == 4
    assert migrated['split_identity']['train_hash'] == 't'  # not rewritten
    assert migrated['rng']['format'] == 'block_amend_rng_v1'
    _no_ndarray(migrated['rng'])
    print('schema_and_loader ok')


# ---------------------------------------------------------------- paths
def test_manifest_paths_and_migration():
    from dataset.block_amend_lolv2 import (
        collect_triplets, manifest_identity_hash, materialize_items,
        save_split_manifests, split_train_val, validate_split_manifests,
        write_coverage_manifests,
    )
    root = _synth_tree(n_train=8, h=32, w=32)
    ok, missing = collect_triplets(root, 'Train', ref_subdir='nanobanana_ref_v2')
    assert missing == []
    # Hash path and read path are the same relative path.
    for it in ok:
        for key in ('low', 'normal', 'ref'):
            assert os.path.normpath(it[key]) == os.path.normpath(
                os.path.join(it['dataset_dir'], *it[key + '_rel'].split('/')))
        assert os.path.normpath(it['ref']) != os.path.normpath(it['normal'])
    h_a = manifest_identity_hash(ok, root, 'train', 'nanobanana_ref_v2')
    h_b = manifest_identity_hash(ok, root, 'train', 'nanobanana_ref_OTHER')
    assert h_a != h_b

    # Absolute ref changed, relative path left as it was: refuse both.
    broken = [dict(x) for x in ok]
    broken[0]['ref'] = broken[0]['ref'] + '.edited'
    _raises(RuntimeError, lambda: materialize_items(broken, root), 'disagree')

    train, val = split_train_val(ok, val_ratio=0.25, split_seed=0)
    meta = save_split_manifests(train, val, os.path.join(root, 'splits'),
                                dataset_dir=root, ref_subdir='nanobanana_ref_v2',
                                batch_size=2)
    manifest_path = os.path.join(root, 'splits', 'split_train.json')
    original = open(manifest_path, 'rb').read()

    new_root = tempfile.mkdtemp()
    # Copy the dataset tree, not the manifest directory we just wrote inside it.
    for split in ('Train', 'Test'):
        shutil.copytree(os.path.join(root, split), os.path.join(new_root, split))
    with open(manifest_path) as f:
        stored = json.load(f)['items']
    moved = materialize_items(stored, new_root)
    assert open(manifest_path, 'rb').read() == original  # source manifest untouched
    for it in moved:
        assert it['low'].startswith(os.path.normpath(new_root) + os.sep)
        assert it['normal'].startswith(os.path.normpath(new_root) + os.sep)
        assert it['ref'].startswith(os.path.normpath(new_root) + os.sep)
        assert os.path.isfile(it['low']) and os.path.isfile(it['ref'])
    assert manifest_identity_hash(moved, new_root, 'train', 'nanobanana_ref_v2') == meta['train_hash']

    # Duplicate id, overlap, missing file, cross-split path reuse.
    _raises(RuntimeError, lambda: validate_split_manifests(
        train + [dict(train[0])], val, root, 'nanobanana_ref_v2', 2), 'duplicate')
    _raises(RuntimeError, lambda: validate_split_manifests(
        train, val + [dict(train[0])], root, 'nanobanana_ref_v2', 2), 'overlap')
    vanished = [dict(x) for x in val]
    vanished[0]['low'] = vanished[0]['low'] + '.missing'
    vanished[0]['low_rel'] = vanished[0]['low_rel'] + '.missing'
    _raises((FileNotFoundError, RuntimeError), lambda: validate_split_manifests(
        train, vanished, root, 'nanobanana_ref_v2', 2))
    reused = [dict(x) for x in val]
    reused[0]['ref'] = train[0]['ref']
    reused[0]['ref_rel'] = train[0]['ref_rel']
    _raises(RuntimeError, lambda: validate_split_manifests(
        train, reused, root, 'nanobanana_ref_v2', 2), 'reused')

    # Partial coverage drops only a missing ref. A missing GT still fails.
    ref_gone = ok[0]['ref']
    os.remove(ref_gone)
    out = os.path.join(tempfile.mkdtemp(), 'cov')
    report = write_coverage_manifests(
        root, out, ref_subdir='nanobanana_ref_v2', allow_partial=True)
    assert report['exclude_missing_ref'] == 1
    assert report['include_count'] == len(ok) - 1
    included = json.load(open(report['include_manifest']))['items']
    assert all(it['id'] != ok[0]['id'] for it in included)
    normal_gone = ok[1]['normal']
    os.remove(normal_gone)
    _raises(RuntimeError, lambda: write_coverage_manifests(
        root, os.path.join(tempfile.mkdtemp(), 'cov2'),
        ref_subdir='nanobanana_ref_v2', allow_partial=True), 'GT')
    print('manifest_paths ok')


def test_shell_partial_flag():
    script = os.path.join(ROOT, 'train', 'train_block_amend_csfi_lolv2.sh')
    subprocess.run(['bash', '-n', script], check=True)
    stub_dir = tempfile.mkdtemp()
    out = os.path.join(stub_dir, 'argv.json')
    stub = os.path.join(stub_dir, 'python_stub')
    with open(stub, 'w') as f:
        f.write('#!/usr/bin/env python3\nimport json, os, sys\n'
                'open(os.environ["STUB_OUT"], "w").write(json.dumps(sys.argv))\n')
    os.chmod(stub, os.stat(stub).st_mode | stat.S_IEXEC)
    save = os.path.join(stub_dir, 'run')

    def run(value):
        env = os.environ.copy()
        env.update(PYTHON=stub, STUB_OUT=out, ALLOW_PARTIAL_COVERAGE=value,
                   SAVE_DIR=save, DATA='/tmp/unused-data')
        return subprocess.run(['bash', script], env=env, cwd=ROOT,
                              capture_output=True, text=True)

    ok = run('1')
    assert ok.returncode == 0, ok.stderr
    argv = json.load(open(out))
    assert '--allow_partial_coverage' in argv
    assert 'explicit partial coverage' in ok.stdout
    assert 'model_variant=csfi_lfca_v1' in ok.stdout
    assert 'data=/tmp/unused-data' in ok.stdout
    assert 'save_dir=%s' % save in ok.stdout
    strict = run('0')
    assert strict.returncode == 0, strict.stderr
    argv = json.load(open(out))
    assert '--allow_partial_coverage' not in argv
    assert 'strict coverage' in strict.stdout
    bad = run('yes')
    assert bad.returncode != 0
    assert 'ALLOW_PARTIAL_COVERAGE' in bad.stderr
    print('shell_partial_flag ok')


# ---------------------------------------------------------------- eval / resume
def _train_once(save, root, cpu=True):
    import block_amend_main as bam
    _patch_lte()
    saved = dict(bam.BASELINE_FIXED)
    bam.BASELINE_FIXED.update(
        num_res_blocks='1+1+1+1', n_feats=16, ref_correction_feats=4,
        tile_size=32, tile_overlap=8)
    try:
        cmd = ['--mode', 'train', '--dataset_dir', root, '--save_dir', save,
               '--batch_size', '2', '--train_crop_size', '32', '--num_epochs', '1',
               '--val_ratio', '0.25', '--seed', '5', '--print_every', '100',
               '--num_workers', '0']
        if cpu:
            cmd.append('--cpu')
        bam.main(cmd)
    finally:
        bam.BASELINE_FIXED.clear()
        bam.BASELINE_FIXED.update(saved)


def test_eval_identity_and_bare_weights():
    import block_amend_main as bam
    from block_amend_ckpt import load_local_checkpoint
    root = _synth_tree(n_train=8, h=40, w=40)
    save = os.path.join(tempfile.mkdtemp(), 'run')
    _train_once(save, root, cpu=True)
    last = os.path.join(save, 'checkpoint_last.pt')
    pack = load_local_checkpoint(last)
    calls = []
    real = bam.tiled_forward

    def boom(*a, **k):
        calls.append(1)
        raise AssertionError('forward ran')

    def ev(extra):
        calls.clear()
        bam.tiled_forward = boom
        try:
            bam.main(['--mode', 'eval', '--cpu', '--num_workers', '0',
                      '--model_path', last, '--save_dir',
                      os.path.join(save, 'ev_%d' % len(extra))] + extra)
        finally:
            bam.tiled_forward = real

    # A different legal val manifest fails before any forward.
    other = os.path.join(save, 'other_split')
    shutil.copytree(os.path.join(save, 'splits'), other)
    with open(os.path.join(other, 'split_val.json')) as f:
        val = json.load(f)
    extra_ref = os.path.join(root, 'Train', 'nanobanana_ref_v2', 'low_extra.png')
    shutil.copy(val['items'][0]['ref'], extra_ref)
    val['items'][0]['ref'] = extra_ref
    val['items'][0]['ref_rel'] = 'Train/nanobanana_ref_v2/low_extra.png'
    with open(os.path.join(other, 'split_val.json'), 'w') as f:
        json.dump(val, f)
    msg = _raises(RuntimeError, lambda: ev(['--eval_split', 'val', '--split_dir', other]),
                  'split identity')
    assert calls == [], calls
    assert 'val_hash' in msg

    # The untouched internal val still evaluates, and records the pack.
    bam.tiled_forward = real
    ev_dir = os.path.join(save, 'ev_ok')
    bam.main(['--mode', 'eval', '--cpu', '--num_workers', '0', '--eval_split', 'val',
              '--model_path', last, '--save_dir', ev_dir])
    summary = json.load(open(os.path.join(ev_dir, 'eval_val.json')))
    assert summary['split_identity']['val_hash'] == pack['split_identity']['val_hash']
    assert summary['model_variant'] == 'csfi_lfca_v1'
    assert summary['checkpoint_path'] == last
    assert summary['checkpoint_epoch'] == pack['epoch']
    assert summary['metric_protocol'] == 'rgb_float_clipped_v1'

    # Official Test is not asked to match the internal val.
    test_dir = os.path.join(save, 'ev_test')
    bam.main(['--mode', 'eval', '--cpu', '--num_workers', '0', '--eval_split', 'test',
              '--model_path', last, '--save_dir', test_dir])
    test_summary = json.load(open(os.path.join(test_dir, 'eval_official_test.json')))
    assert test_summary['split_identity']['role'] == 'official_test'
    assert test_summary['split_identity']['test_hash'] != pack['split_identity']['val_hash']
    assert test_summary['split_identity']['ref_subdir'] == 'nanobanana_ref_v3'

    # Bare weights fail before a model is built. IGAB weights still say so.
    built = []
    real_build = bam.build_model
    bam.build_model = lambda *a, **k: built.append(1)
    try:
        flat = os.path.join(save, 'flat.pt')
        torch.save(pack['model_state_dict'], flat)
        _raises(SystemExit, lambda: bam.main(
            ['--mode', 'eval', '--cpu', '--model_path', flat,
             '--save_dir', os.path.join(save, 'ev_flat')]), 'bare state_dict')
        assert built == []
        igab = os.path.join(save, 'igab.pt')
        torch.save({'MainNet.coarse_igab.blocks.0.0.to_q.weight': torch.zeros(1)}, igab)
        _raises(RuntimeError, lambda: bam.main(
            ['--mode', 'eval', '--cpu', '--model_path', igab,
             '--save_dir', os.path.join(save, 'ev_igab')]), 'csfi_igab_v0')
        assert built == []
    finally:
        bam.build_model = real_build

    # Schema 3 is refused by the resume entry, source file unchanged.
    schema3 = os.path.join(save, 'old.pt')
    old = dict(pack)
    old['checkpoint_schema_version'] = 3
    old['rng'] = {'python_random': random.getstate(), 'numpy': np.random.get_state()}
    torch.save(old, schema3)
    before = open(schema3, 'rb').read()
    _raises(RuntimeError, lambda: bam.main(
        ['--mode', 'resume', '--cpu', '--num_workers', '0', '--num_epochs', '2',
         '--resume_checkpoint', schema3, '--save_dir', save]), 'Schema 3')
    assert open(schema3, 'rb').read() == before
    print('eval_identity_and_bare_weights ok')


def test_gpu_resume_smoke():
    if not torch.cuda.is_available():
        print('GPU resume smoke NOT RUN: torch.cuda.is_available() is False. '
              'No mock was used in its place.')
        return
    import block_amend_main as bam
    from block_amend_ckpt import load_local_checkpoint
    from block_amend_main import build_optimizer
    root = _synth_tree(n_train=6, h=40, w=40)
    save = os.path.join(tempfile.mkdtemp(), 'gpu_run')
    _train_once(save, root, cpu=False)
    last = os.path.join(save, 'checkpoint_last.pt')
    pack = load_local_checkpoint(last)
    assert pack['rng']['torch_cpu'].device.type == 'cpu'
    assert pack['rng']['torch_cpu'].dtype == torch.uint8
    assert pack['rng']['torch_cuda'] and pack['rng']['torch_cuda'][0].device.type == 'cpu'
    # One more optimizer step on a rebuilt GPU model: momentum follows the
    # parameter onto CUDA, Adam's step counter stays on CPU.
    _patch_lte()
    from model.TTSREnhance import TTSREnhance
    model = TTSREnhance(_tiny_args()).cuda()
    bam_load = __import__('model.weight_init', fromlist=['load_strict_state']).load_strict_state
    bam_load(model, last, mode='resume', preloaded=pack)
    opt, _ = build_optimizer(model, _tiny_args())
    opt.load_state_dict(pack['optimizer_state_dict'])
    g1, g2 = _gens()
    from block_amend_ckpt import restore_rng_state
    restore_rng_state(pack['rng'], g1, g2)
    x = torch.rand(1, 3, 32, 32, device='cuda')
    ref = torch.rand(1, 3, 32, 32, device='cuda')
    opt.zero_grad(set_to_none=True)
    sr = model(lr=x, lrsr=x, ref=ref, refsr=ref)[0]
    sr.mean().backward()
    opt.step()
    assert torch.isfinite(sr).all()
    saw_cuda = saw_cpu_step = False
    for p in model.parameters():
        slot = opt.state.get(p)
        if not slot:
            continue
        if 'exp_avg' in slot:
            assert slot['exp_avg'].device.type == 'cuda'
            saw_cuda = True
        if 'step' in slot and torch.is_tensor(slot['step']):
            assert slot['step'].device.type == 'cpu'
            saw_cpu_step = True
    assert saw_cuda and saw_cpu_step
    # And the real resume entry completes another epoch on this GPU.
    _train_resume = dict(cpu=False)
    bam.main(['--mode', 'resume', '--num_workers', '0', '--num_epochs', '2',
              '--resume_checkpoint', last, '--save_dir', save])
    pack2 = load_local_checkpoint(last)
    assert pack2['epoch'] == 2 and pack2['global_step'] > pack['global_step']
    assert pack2['rng']['torch_cpu'].device.type == 'cpu'
    assert _train_resume['cpu'] is False
    print('gpu_resume_smoke ok (real CUDA, one resumed epoch)')


if __name__ == '__main__':
    print('Python', sys.version.split()[0], 'PyTorch', torch.__version__,
          'cuda', torch.cuda.is_available())
    test_rng_roundtrip_and_dtype()
    test_pack_is_weights_only_friendly_and_schema3_is_refused()
    test_manifest_paths_and_migration()
    test_shell_partial_flag()
    test_eval_identity_and_bare_weights()
    test_gpu_resume_smoke()
    print('all block-amend round3 tests passed')
