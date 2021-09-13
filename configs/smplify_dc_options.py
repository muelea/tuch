import os
import json
import argparse
import numpy as np
from collections import namedtuple

class SMPLifyDCOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--num_workers', type=int, default=8, help='Number of processes used for data loading')

        io = self.parser.add_argument_group('io')
        io.add_argument('--log_dir', default='logs', help='Directory to store logs')
        
        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--batch_size', type=int, default=1, help='Batch size')
        train.add_argument('--img_res', type=int, default=224, help='Rescale bounding boxes to size [img_res, img_res] before feeding them in the network')

        # training data
        train.add_argument('--ds_names', nargs='+', type=str, default=['dsc_df', 'dsc_lspet', 'dsc_lsp'], help='Names of datasets to use in training.')

        # data augmentation
        train.add_argument('--rot_factor', type=float, default=30, help='Random rotation in the range [-rot_factor, rot_factor]')
        train.add_argument('--noise_factor', type=float, default=0.4, help='Randomly multiply pixel values with factor in the range [1-noise_factor, 1+noise_factor]')
        train.add_argument('--scale_factor', type=float, default=0.25, help='Rescale bounding boxes by a factor of [1-scale_factor,1+scale_factor]')
        train.add_argument('--ignore_3d', default=False, action='store_true', help='Ignore GT 3D data (for unpaired experiments')

        # optimization params
        train.add_argument('--num_smplify_iters', default=10, type=int, help='Number of SMPLify iterations')
        train.add_argument('--use_contact_in_the_loop', default=True, type=lambda x: x.lower() in ['true', '1'], help='wether to use additional contact loss in the smplify loop or not')
        train.add_argument('--contact_in_the_loop_loss_weight', default=2000, type=float, help='wether to use additional contact loss weight in the smplify loop or not')

        return

    def parse_args(self):
        """Parse input arguments."""

        self.args = self.parser.parse_args()

        self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
            
        with open(os.path.join(self.args.log_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)

        return self.args