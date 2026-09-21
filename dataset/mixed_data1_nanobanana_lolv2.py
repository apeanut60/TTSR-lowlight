"""Mixed training dataset with per-source reference behaviour.

Sources:
    data1_gt         Ref=HR, ref_degrade=True
    data1_nanobanana Ref=nanobanana_ref, ref_degrade=False
    lolv2real        Ref=GT, ref_degrade=True
    lolv2syn         Ref=GT, ref_degrade=True

Weights are read from args.mixed_weights_4 = 'data1_gt,data1_nanobanana,real,syn'.
"""

import argparse
import random

from torch.utils.data import Dataset

from dataset import data1, data1_nanobanana, lolv2real, lolv2syn


def _sub_args(args, dataset_dir, ref_degrade, camera='all'):
    sub = argparse.Namespace(**vars(args))
    sub.dataset_dir = dataset_dir
    sub.ref_degrade = ref_degrade
    if camera is not None:
        sub.data1_camera = camera
    return sub


class TrainSet(Dataset):
    def __init__(self, args):
        weights = [float(x) for x in args.mixed_weights_4.split(',')]
        if len(weights) != 4 or sum(weights) <= 0:
            raise ValueError('mixed_weights_4 must contain 4 positive numbers')
        self.weights = weights

        self.data1_set = data1.TrainSet(
            _sub_args(args, args.mixed_data1_dir, True, camera='all'))
        self.nanobanana_set = data1_nanobanana.TrainSet(
            _sub_args(args, args.mixed_data1_nanobanana_dir, False, camera='all'))
        self.real_set = lolv2real.TrainSet(
            _sub_args(args, args.mixed_lolv2_real_dir, True, camera=None))
        self.syn_set = lolv2syn.TrainSet(
            _sub_args(args, args.mixed_lolv2_syn_dir, True, camera=None))

        self.sets = [self.data1_set, self.nanobanana_set,
                     self.real_set, self.syn_set]
        self.length = sum(len(ds) for ds in self.sets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dataset = random.choices(self.sets, weights=self.weights, k=1)[0]
        return dataset[random.randrange(len(dataset))]


# Reuse data1.TestSet for the primary validation loader. Additional data1 /
# nanobanana / lolv2 eval loaders are injected by dataloader helpers.
from dataset.data1 import TestSet
