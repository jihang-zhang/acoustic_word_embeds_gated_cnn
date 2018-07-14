from os import path
import logging
import numpy as np
import torch
import random

import torch.utils.data as tud
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def load_swbd_labelled(rng, data_dir, min_count=1):
    """
    Load the Switchboard data with their labels.
    Only tokens that occur at least `min_count` times in the training set
    is considered.
    """

    def get_data_and_labels(set):

        npz_fn = path.join(data_dir, "swbd." + set + ".npz")
        logger.info("Reading: " + npz_fn)

        # Load data and shuffle
        npz = np.load(npz_fn)
        utts = sorted(npz.keys())
        rng.shuffle(utts)
        x = [npz[i] for i in utts]

        # Get labels for each utterance
        labels = swbd_utts_to_labels(utts)

        return x, labels

    train_x, train_labels = get_data_and_labels("train")
    dev_x, dev_labels = get_data_and_labels("dev")
    test_x, test_labels = get_data_and_labels("test")

    logger.info("Finding types with at least " + str(min_count) + " tokens")

    # Determine the types with the minimum count
    type_counts = {}
    for label in train_labels:
        if not label in type_counts:
            type_counts[label] = 0
        type_counts[label] += 1
    min_types = set()
    i_type = 0
    word_to_i_map = {}
    for label in type_counts:
        if type_counts[label] >= min_count:
            min_types.add(label)
            word_to_i_map[label] = i_type
            i_type += 1

    # Filter the sets
    def filter_set(x, labels):
        filtered_x = []
        filtered_i_labels = []
        for cur_x, label in zip(x, labels):
            if label in word_to_i_map:
                filtered_x.append(cur_x)
                filtered_i_labels.append(word_to_i_map[label])
        return filtered_x, filtered_i_labels
    train_x, train_labels = filter_set(train_x, train_labels)
    dev_x, dev_labels = filter_set(dev_x, dev_labels)
    test_x, test_labels = filter_set(test_x, test_labels)

    return [(train_x, train_labels), (dev_x, dev_labels), (test_x, test_labels)], word_to_i_map

# Utility Functions
def swbd_utt_to_label(utt):
    return "_".join(utt.split("_")[:-2])

def swbd_utts_to_labels(utts):
    labels = []
    for utt in utts:
        labels.append(swbd_utt_to_label(utt))
    return labels


class BatchRandomSampler(tud.sampler.Sampler):
    """
    Batches the data consecutively and randomly samples
    by batch without replacement.
    """

    def __init__(self, data_source, batch_size):
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i, i + batch_size) for i in range(0, it_end, batch_size)]
        self.data_source = data_source

    def __iter__(self):
        random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)


def make_loader(dataset, batch_size):
    sampler = BatchRandomSampler(dataset, batch_size)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# Siamese DataLoader
