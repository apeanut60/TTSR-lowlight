"""Mixed training dataset: data1 + lolv2real + lolv2syn.

Sampling is controlled by args.mixed_weights = 'data1,real,syn'.
"""
import argparse
import random

from torch.utils.data import Dataset

from dataset import data1, lolv2real, lolv2syn


def _sub_args(args, dataset_dir):
    sub = argparse.Namespace(**vars(args))
    sub.dataset_dir = dataset_dir
    return sub


class TrainSet(Dataset):
    def __init__(self, args):
        weights = [float(x) for x in args.mixed_weights.split(',')]
        if len(weights) != 3 or sum(weights) <= 0:
            raise ValueError('mixed_weights must contain 3 positive numbers')
        self.weights = weights

        self.data1_set = data1.TrainSet(_sub_args(args, args.mixed_data1_dir))
        self.real_set = lolv2real.TrainSet(_sub_args(args, args.mixed_lolv2_real_dir))
        self.syn_set = lolv2syn.TrainSet(_sub_args(args, args.mixed_lolv2_syn_dir))
        self.sets = [self.data1_set, self.real_set, self.syn_set]
        self.length = sum(len(ds) for ds in self.sets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dataset = random.choices(self.sets, weights=self.weights, k=1)[0]
        return dataset[random.randrange(len(dataset))]


TestSet = data1.TestSet
