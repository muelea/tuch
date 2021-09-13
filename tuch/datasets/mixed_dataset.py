# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import numpy as np

from .base_dataset import BaseDataset

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, options, set, **kwargs):

        if set == 'train':
            self.dataset_list = options.ds_names
            self.partition = options.ds_composition
        elif set == 'val':
            self.dataset_list = ['mtp']
            self.partition = [1.0]
        # if dsc data is used, exted dataset_list and partition, so that it uses all three subsets.
        if 'dsc' in self.dataset_list:
            dsc_idx = np.where(np.array(self.dataset_list) == 'dsc')[0][0]
            dsc_share = self.partition[dsc_idx]
            self.dataset_list = [x for x in self.dataset_list if x != 'dsc']
            self.dataset_list += ['dsc_lspet', 'dsc_lsp', 'dsc_df']
            self.partition = [x for i, x in enumerate(self.partition) if i != dsc_idx] 
            self.partition += [dsc_share] * 3
        if 'dsc_eft' in self.dataset_list:
            dsc_idx = np.where(np.array(self.dataset_list) == 'dsc_eft')[0][0]
            dsc_share = self.partition[dsc_idx]
            self.dataset_list = [x for x in self.dataset_list if x != 'dsc_eft']
            self.dataset_list += ['dsc_lspet_eft', 'dsc_lsp_eft', 'dsc_df_eft']
            self.partition = [x for i, x in enumerate(self.partition) if i != dsc_idx] 
            self.partition += [dsc_share] * 3

        self.dataset_dict = dict(zip(self.dataset_list, np.arange(len(self.dataset_list))))
        self.datasets = [BaseDataset(options, ds, set=set, **kwargs) for ds in self.dataset_list]
        self.length = max([len(ds) for ds in self.datasets])
        self.total_length = sum([len(ds) for ds in self.datasets])

        if set == 'train':
            # compute share of in-the-wild datasets without contact annotation or pseudo ground truth SMPL-X
            itw_datasets = ['mpii', 'coco', 'mpii_eft', 'coco_eft']
            itw_index = [i for i, v in enumerate(self.dataset_list) if v in itw_datasets]
            if len(itw_index) > 0:
                length_itw = [len(self.datasets[i]) for i in itw_index]
                length_itw_sum = sum(length_itw)
                for i, j in enumerate(itw_index):
                    self.partition[j] = self.partition[j] * length_itw[i] / length_itw_sum

            # compute share of in-the-wild datasets with contact annotation
            itw_dc_datasets = ['dsc_df', 'dsc_lspet', 'dsc_lsp', 
                               'dsc_df_eft', 'dsc_lspet_eft', 'dsc_lsp_eft']
            itw_dc_index = [i for i, v in enumerate(self.dataset_list) if v in itw_dc_datasets]
            if len(itw_dc_index) > 0:
                length_itw_dc = [len(self.datasets[i]) for i in itw_dc_index]
                length_itw_dc_sum = sum(length_itw_dc)
                for i, j in enumerate(itw_dc_index):
                    self.partition[j] = self.partition[j] * length_itw_dc[i] / length_itw_dc_sum

        self.partition = np.array(self.partition).cumsum()

        print(f'Loading {set} data:')
        for idx in range(len(self.partition)):
            x = self.dataset_list[idx]
            prev = 0 if idx == 0 else self.partition[idx-1]
            y = (self.partition[idx] - prev) * 100
            print(f'  --- {x} share per batch: {y:.02f}% ')

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(len(self.partition)):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
