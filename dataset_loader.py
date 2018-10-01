from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import os.path as osp
import lmdb
import io

import torch
from torch.utils.data import Dataset


class JsonDataset(Dataset):
    """Auto Car Json Dataset"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset.data
        if self.dataset.phase == 'train':  # if self.dataset.phase != 'test':
            labels = self.dataset.labels

        return (torch.Tensor(data[index]), torch.Tensor(labels[index])) if self.dataset.phase == 'train' \
            else torch.Tensor(data[index])  # if self.dataset.phase != 'test' \
