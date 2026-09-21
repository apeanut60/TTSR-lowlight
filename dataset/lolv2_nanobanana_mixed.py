"""Mixed lolv2 nanobanana dataset.

Samples lolv2real-nanobanana and lolv2syn-nanobanana together in one dataset.
"""

import argparse
import random

from torch.utils.data import Dataset

from dataset import data1, lolv2_nanobanana


def _sub_args(args, dataset_dir, subset):
    sub = argparse.Namespace(**vars(args))
    sub.dataset_dir = dataset_dir
    sub.lolv2_nanobanana_subset = subset
    sub.ref_degrade = False
    return sub


class TrainSet(Dataset):
    def __init__(self, args):
        weights = [float(x) for x in args.lolv2_nanobanana_mixed_weights.split(',')]
        if len(weights) != 2 or sum(weights) <= 0:
            raise ValueError('lolv2_nanobanana_mixed_weights must contain 2 positive numbers')
        self.weights = weights

        self.real_set = lolv2_nanobanana.TrainSet(
            _sub_args(args, args.mixed_lolv2_real_dir, 'real'))
        self.syn_set = lolv2_nanobanana.TrainSet(
            _sub_args(args, args.mixed_lolv2_syn_dir, 'syn'))
        self.sets = [self.real_set, self.syn_set]
        self.length = sum(len(ds) for ds in self.sets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dataset = random.choices(self.sets, weights=self.weights, k=1)[0]
        return dataset[random.randrange(len(dataset))]


# Primary eval loader is not the main monitoring target; extra loaders are
# injected for data1 GT and lolv2real/syn nanobanana. Reuse data1.TestSet for a
# safe primary loader when dataset_dir points to data1.
TestSet = data1.TestSet
