from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
# import torch.utils.data.dataset as Dataset
from sklearn.preprocessing import scale
import numpy as np
import json
from numpy import *


class AutoJson(object):
    """
    AutoCar JsonDataset

    Dataset statistics:
    # tracks: 17
    """

    def __init__(self, root='./data', phase='train', verbose=True, **kwargs):
        super(AutoJson, self).__init__()
        self.dataset_dir = osp.join(root, phase + '_data_pre', phase + '_data.json')
        self._check_before_run()

        self.data, self.labels, self.messages = self._process_dir(self.dataset_dir)
        num_total_data = self.data.shape[0]
        num_total_label = len(self.labels)
        assert num_total_data == num_total_label, 'The data are not aligned to the labels'

        if verbose:
            print("=> AutoCar JsonFile loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  Track   |   numbers")
            print("  ------------------------------")
            for _ in self.messages:
                print("  {}   |   {}".format(_[0], _[1]))
                print("  ------------------------------")

        self.num_total_data = num_total_data
        self.num_total_label = num_total_label

    def __len__(self):
        return self.num_total_data

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def _process_dir(self, dir_path):
        data = []
        labels = []
        data_final = []
        labels_final = []
        messages_final = []
        messages = {}
        m_index = 0
        flag = ''
        num = 0
        with open(dir_path, 'r') as s_file:
            for lines in s_file.readlines():
                load_dict = json.loads(lines)
                data.append(load_dict['data'])
                labels.append(load_dict['label'])
                name = load_dict['filename'].split('_')
                track = name[0]
                if (flag == '') & (track != flag):
                    flag = track
                elif (flag != '') & (track != flag):
                    messages[flag] = int(num) + 1
                    flag = track
                    num = name[1]
                else:
                    num = name[1]
            messages[flag] = int(num) + 1

        messages = sorted(messages.items(), key=lambda e: int(e[0].split()[1]), reverse=False)
        for i, _ in enumerate(messages):
            messages_final.append(list(messages[i]))
        messages_value = []
        for i, _ in enumerate(messages):
            messages_value.append(messages[i][1])
        message_index = np.cumsum(messages_value)

        # Outlier Detection
        data = mat(data)
        mask = np.ones((data.shape[0],))
        for i in range(6):
            data_mean = np.mean(data[:, i])
            data_std = np.std(data[:, i], ddof=1)
            for j, _ in enumerate(data[:, i]):
                if abs(_-data_mean) > 3*data_std:
                    mask[j] = 0  # 91173/93766
        data = np.matrix.tolist(data)
        for i, _ in enumerate(mask):
            if _:
                data_final.append(data[i])
                labels_final.append(labels[i])
            else:
                for index, value in enumerate(message_index):
                    if i <= value:
                        m_index = index
                    else:
                        m_index = index
                        break
                messages_final[m_index][1] -= 1
        data_final = mat(data_final)
        data_final[:, 0:6] = scale(data_final[:, 0:6])
        return data_final, labels_final, messages_final
