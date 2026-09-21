from torch.utils.data import DataLoader
from importlib import import_module
import argparse


def get_dataloader(args):
    ### import module
    m = import_module('dataset.' + args.dataset.lower())

    if (args.dataset == 'CUFED'):
        data_train = getattr(m, 'TrainSet')(args)
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_test = {}
        for i in range(5):
            data_test = getattr(m, 'TestSet')(args=_test_args(args), ref_level=str(i+1))
            dataloader_test[str(i+1)] = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
        dataloader = {'train': dataloader_train, 'test': dataloader_test}

    elif (args.dataset in ['LOL', 'data1', 'data2', 'data1_nanobanana',
                           'lolv2_nanobanana', 'lolv2real', 'lolv2syn', 'SICE',
                           'lolv2_nanobanana_mixed',
                           'mixed_data1_lolv2_nanobanana',
                           'mixed_lolv2_data1',
                           'mixed_data1_nanobanana_lolv2']):
        data_train = getattr(m, 'TrainSet')(args)
        if len(data_train) > 0:
            dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        else:
            dataloader_train = None
        data_test = getattr(m, 'TestSet')(args=_test_args(args), ref_level='1')
        dataloader_test = {'1': DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)}
        dataloader = {'train': dataloader_train, 'test': dataloader_test}

    else:
        raise SystemExit('Error: no such type of dataset!')

    return dataloader


def _test_args(args):
    """Return args for the test loader, honouring --eval_ref_degrade.

    Training uses ``ref_degrade``; when ``eval_ref_degrade`` is set explicitly
    the test loaders use that value instead, so validation can run against a
    clean reference while training still sees degraded references.
    """
    eval_ref_degrade = getattr(args, 'eval_ref_degrade', None)
    if eval_ref_degrade is None:
        return args
    test_args = argparse.Namespace(**vars(args))
    test_args.ref_degrade = eval_ref_degrade
    return test_args


def add_data1_eval(args, dataloader):
    """Inject clean nanobanana data1 eval loaders into the dataloader."""
    if (dataloader is None or not getattr(args, 'eval_data1', False)):
        return dataloader
    return add_data1_nanobanana_eval(args, dataloader)


def _build_test_loader(args, dataset_name, dataset_dir, camera='all', ref_dir=''):
    test_args = argparse.Namespace(**vars(args))
    test_args.dataset = dataset_name
    test_args.dataset_dir = dataset_dir
    test_args.data1_camera = camera
    test_args.ref_dir = ref_dir

    m = import_module('dataset.' + dataset_name)
    data_test = getattr(m, 'TestSet')(args=test_args, ref_level='1')
    return DataLoader(data_test, batch_size=1, shuffle=False,
                      num_workers=getattr(args, 'num_workers', 0))


def add_mixed_eval(args, dataloader):
    """Inject clean-nanobanana data1 + lolv2real + lolv2syn eval loaders."""
    if dataloader is None:
        return dataloader

    add_data1_eval(args, dataloader)
    add_lolv2_nanobanana_eval(args, dataloader)
    return dataloader


def add_data1_nanobanana_eval(args, dataloader):
    """Inject clean-nanobanana data1 eval loaders for Huawei and Nikon."""
    if dataloader is None:
        return dataloader

    if 'extra_test' not in dataloader:
        dataloader['extra_test'] = {}

    for camera, arg_name, key in (
        ('Huawei', 'data1_nanobanana_eval_huawei_ref_dir',
         'data1_nanobanana_huawei'),
        ('Nikon', 'data1_nanobanana_eval_nikon_ref_dir',
         'data1_nanobanana_nikon'),
    ):
        ref_dir = getattr(args, arg_name, '')
        if not ref_dir:
            continue

        data1_args = argparse.Namespace(**vars(args))
        data1_args.dataset = 'data1'
        data1_args.dataset_dir = getattr(args, 'data1_eval_dir',
                                         '/root/data/datasets/data1')
        data1_args.data1_camera = camera
        data1_args.ref_dir = ref_dir
        data1_args.ref_degrade = False

        m = import_module('dataset.data1')
        data_test = getattr(m, 'TestSet')(args=data1_args, ref_level='1')
        loader = DataLoader(data_test, batch_size=1, shuffle=False,
                            num_workers=getattr(args, 'num_workers', 0))
        dataloader['extra_test'][key] = loader

    return dataloader


def add_mixed_data1_nanobanana_eval(args, dataloader):
    """Inject clean data1 nanobanana + clean lolv2 nanobanana eval loaders."""
    if dataloader is None:
        return dataloader

    add_data1_eval(args, dataloader)
    add_lolv2_nanobanana_eval(args, dataloader)
    return dataloader


def add_lolv2_nanobanana_eval(args, dataloader):
    """Inject clean-nanobanana lolv2real/syn eval loaders."""
    if dataloader is None:
        return dataloader

    if 'extra_test' not in dataloader:
        dataloader['extra_test'] = {}

    for key, arg_name, default_dir in (
        ('lolv2real_nanobanana', 'lolv2_nanobanana_eval_real_ref_dir',
         '/root/data/datasets/lol-v2-real/Test/nanobanana_ref_v3'),
        ('lolv2syn_nanobanana', 'lolv2_nanobanana_eval_syn_ref_dir',
         '/root/data/datasets/lol-v2-synthetic/Test/nanobanana_ref_v3'),
    ):
        ref_dir = getattr(args, arg_name, default_dir)
        if not ref_dir:
            continue
        dataset_name = 'lolv2real' if key.startswith('lolv2real') else 'lolv2syn'
        test_args = argparse.Namespace(**vars(args))
        test_args.dataset = dataset_name
        test_args.dataset_dir = (
            getattr(args, 'mixed_lolv2_real_dir', '/root/data/datasets/lol-v2-real')
            if dataset_name == 'lolv2real'
            else getattr(args, 'mixed_lolv2_syn_dir', '/root/data/datasets/lol-v2-synthetic')
        )
        test_args.ref_dir = ref_dir
        test_args.ref_degrade = False
        m = import_module('dataset.' + dataset_name)
        data_test = getattr(m, 'TestSet')(args=test_args, ref_level='1')
        dataloader['extra_test'][key] = DataLoader(
            data_test, batch_size=1, shuffle=False,
            num_workers=getattr(args, 'num_workers', 0))

    return dataloader


def add_lolv2_nanobanana_eval_subset(args, dataloader, subset):
    """Inject only the requested lolv2 nanobanana eval loader."""
    if dataloader is None or subset not in ('real', 'syn'):
        return dataloader

    if 'extra_test' not in dataloader:
        dataloader['extra_test'] = {}

    if subset == 'real':
        key, arg_name, default_dir, dataset_name = (
            'lolv2real_nanobanana', 'lolv2_nanobanana_eval_real_ref_dir',
            '/root/data/datasets/lol-v2-real/Test/nanobanana_ref_v3', 'lolv2real')
        dataset_dir = getattr(args, 'mixed_lolv2_real_dir',
                              '/root/data/datasets/lol-v2-real')
    else:
        key, arg_name, default_dir, dataset_name = (
            'lolv2syn_nanobanana', 'lolv2_nanobanana_eval_syn_ref_dir',
            '/root/data/datasets/lol-v2-synthetic/Test/nanobanana_ref_v3', 'lolv2syn')
        dataset_dir = getattr(args, 'mixed_lolv2_syn_dir',
                              '/root/data/datasets/lol-v2-synthetic')

    ref_dir = getattr(args, arg_name, default_dir)
    if not ref_dir:
        return dataloader

    test_args = argparse.Namespace(**vars(args))
    test_args.dataset = dataset_name
    test_args.dataset_dir = dataset_dir
    test_args.ref_dir = ref_dir
    test_args.ref_degrade = False
    m = import_module('dataset.' + dataset_name)
    data_test = getattr(m, 'TestSet')(args=test_args, ref_level='1')
    dataloader['extra_test'][key] = DataLoader(
        data_test, batch_size=1, shuffle=False,
        num_workers=getattr(args, 'num_workers', 0))
    return dataloader


def add_lol_nanobanana_eval(args, dataloader):
    """Inject a LOLv1 nanobanana-ref eval loader (eval15 + nanobanana_ref)."""
    if dataloader is None:
        return dataloader

    ref_dir = getattr(args, 'lol_nanobanana_eval_ref_dir',
                      '/root/data/datasets/LOLdataset/eval15/nanobanana_ref')
    if not ref_dir:
        return dataloader

    if 'extra_test' not in dataloader:
        dataloader['extra_test'] = {}

    test_args = argparse.Namespace(**vars(args))
    test_args.dataset = 'LOL'
    test_args.dataset_dir = getattr(args, 'dataset_dir',
                                    '/root/data/datasets/LOLdataset')
    test_args.ref_dir = ref_dir
    test_args.ref_degrade = False
    m = import_module('dataset.lol')
    data_test = getattr(m, 'TestSet')(args=test_args, ref_level='1')
    dataloader['extra_test']['lol_nanobanana'] = DataLoader(
        data_test, batch_size=1, shuffle=False,
        num_workers=getattr(args, 'num_workers', 0))
    return dataloader


def add_lolv2real_gt_eval(args, dataloader):
    """Inject a lolv2real GT-degraded eval loader into the dataloader."""
    if dataloader is None:
        return dataloader

    if 'extra_test' not in dataloader:
        dataloader['extra_test'] = {}

    test_args = argparse.Namespace(**vars(args))
    test_args.dataset = 'lolv2real'
    test_args.dataset_dir = getattr(
        args, 'mixed_lolv2_real_dir', '/root/data/datasets/lol-v2-real')
    test_args.ref_dir = ''
    test_args.ref_degrade = True
    m = import_module('dataset.lolv2real')
    data_test = getattr(m, 'TestSet')(args=test_args, ref_level='1')
    dataloader['extra_test']['lolv2real_gt'] = DataLoader(
        data_test, batch_size=1, shuffle=False,
        num_workers=getattr(args, 'num_workers', 0))
    return dataloader


def add_mixed_data1_lolv2_nanobanana_eval(args, dataloader):
    """Inject clean data1 nanobanana and clean lolv2 nanobanana eval loaders."""
    if dataloader is None:
        return dataloader

    add_data1_eval(args, dataloader)
    add_lolv2_nanobanana_eval(args, dataloader)
    return dataloader
