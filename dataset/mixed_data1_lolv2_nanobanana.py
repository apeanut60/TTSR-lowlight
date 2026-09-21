"""Mixed dataset preserving data1 and lolv2 nanobanana references.

Sources:
    data1_gt              Ref=HR, ref_degrade=True
    data1_nanobanana      Ref=data1 nanobanana_ref, ref_degrade=False
    lolv2real_nanobanana  Ref=lolv2real nanobanana_ref, ref_degrade=False
    lolv2syn_nanobanana   Ref=lolv2syn nanobanana_ref, ref_degrade=False

Weights are read from args.mixed_data1_lolv2_nanobanana_weights.
"""

import argparse
import random

from torch.utils.data import Dataset

from dataset import data1, data1_nanobanana, lolv2_nanobanana


def _sub_args(args, dataset_dir, subset=None, ref_degrade=None):
    sub = argparse.Namespace(**vars(args))
    sub.dataset_dir = dataset_dir
    if subset is not None:
        sub.lolv2_nanobanana_subset = subset
    if ref_degrade is not None:
        sub.ref_degrade = ref_degrade
    return sub


class TrainSet(Dataset):
    def __init__(self, args):
        weights = [float(x) for x in args.mixed_data1_lolv2_nanobanana_weights.split(',')]
        if len(weights) != 4 or sum(weights) <= 0:
            raise ValueError('mixed_data1_lolv2_nanobanana_weights must contain 4 positive numbers')
        self.weights = weights

        self.data1_set = data1.TrainSet(
            _sub_args(args, args.mixed_data1_dir, ref_degrade=True))
        degrade_nanobanana = bool(getattr(args, 'ref_degrade_nanobanana', False))
        self.data1_nanobanana_set = data1_nanobanana.TrainSet(
            _sub_args(args, args.mixed_data1_nanobanana_dir,
                      ref_degrade=degrade_nanobanana))
        self.real_nanobanana_set = lolv2_nanobanana.TrainSet(
            _sub_args(args, args.mixed_lolv2_real_dir, subset='real',
                      ref_degrade=degrade_nanobanana))
        self.syn_nanobanana_set = lolv2_nanobanana.TrainSet(
            _sub_args(args, args.mixed_lolv2_syn_dir, subset='syn',
                      ref_degrade=degrade_nanobanana))

        self.sets = [
            self.data1_set,
            self.data1_nanobanana_set,
            self.real_nanobanana_set,
            self.syn_nanobanana_set,
        ]
        self.length = sum(len(ds) for ds in self.sets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dataset = random.choices(self.sets, weights=self.weights, k=1)[0]
        return dataset[random.randrange(len(dataset))]


# Primary loader uses data1.TestSet when dataset_dir points to data1.
from dataset.data1 import TestSet
