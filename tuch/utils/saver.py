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
#
# Author: Nikos Kolotouros, nkolot@seas.upenn.edu
# https://github.com/nkolot/SPIN/blob/master/utils/saver.py

from __future__ import division
import os
import os.path as osp
import datetime

import torch

def print_loss_dict(losses):
    printstr = '||| Loss: {0:.5f} ||'.format(losses['loss'].item())
    for key,val in losses.items():
        if key != 'loss':
            printstr += '| {0:.5f} |'.format(val.item())
    print(printstr+'||')

def freq_to_step(freq, num_steps_total):
    if freq == 0.0:
        step = 1
    else:
        step = int(freq * num_steps_total)
    return step

class CheckpointSaver():
    """Class that handles saving and loading checkpoints during training."""
    def __init__(self, save_dir, save_steps=1000):
        self.save_dir = osp.abspath(save_dir)
        self.save_steps = save_steps
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.get_latest_checkpoint()
        return

    def exists_checkpoint(self, checkpoint_file=None):
        """Check if a checkpoint exists in the current directory."""
        if checkpoint_file is None:
            return False if self.latest_checkpoint is None else True
        else:
            return osp.isfile(checkpoint_file)

    def save_checkpoint(self, models, optimizers, epoch, batch_idx, batch_size, total_step_count, val_error, dataset_perm):
        """Save checkpoint."""
        timestamp = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        checkpoint_fn = f'{timestamp}_{epoch:05}_{val_error:.02f}.pt'
        checkpoint_filename = osp.abspath(osp.join(self.save_dir, checkpoint_fn))
        checkpoint = {}
        for model in models:
            checkpoint[model] = models[model].state_dict()
        for optimizer in optimizers:
            checkpoint[optimizer] = optimizers[optimizer].state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['batch_idx'] = batch_idx
        checkpoint['batch_size'] = batch_size
        checkpoint['dataset_perm'] = dataset_perm
        checkpoint['total_step_count'] = total_step_count
        print(timestamp, 'Epoch:', epoch, 'Iteration:', batch_idx)
        print('Saving checkpoint file [' + checkpoint_filename + ']')
        torch.save(checkpoint, checkpoint_filename)
        return

    def load_checkpoint(self, models, optimizers, checkpoint_file=None):
        """Load a checkpoint."""
        if checkpoint_file is None:
            print('Loading latest checkpoint [' + self.latest_checkpoint + ']')
            checkpoint_file = self.latest_checkpoint
        checkpoint = torch.load(checkpoint_file)
        for model in models:
            if model in checkpoint:
                models[model].load_state_dict(checkpoint[model])
        for optimizer in optimizers:
            if optimizer in checkpoint:
                optimizers[optimizer].load_state_dict(checkpoint[optimizer])
        return {'epoch': checkpoint['epoch'],
                'batch_idx': checkpoint['batch_idx'],
                'batch_size': checkpoint['batch_size'],
                'dataset_perm': checkpoint['dataset_perm'],
                'total_step_count': checkpoint['total_step_count']}

    def get_latest_checkpoint(self):
        """Get filename of latest checkpoint if it exists."""
        checkpoint_list = [] 
        for dirpath, dirnames, filenames in os.walk(self.save_dir):
            for filename in filenames:
                if filename.endswith('.pt'):
                    checkpoint_list.append(osp.abspath(osp.join(dirpath, filename)))
        checkpoint_list = sorted(checkpoint_list)
        self.latest_checkpoint =  None if (len(checkpoint_list) == 0) else checkpoint_list[-1]
        return
