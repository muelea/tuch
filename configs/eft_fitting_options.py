import os
import json
import argparse
import numpy as np
from collections import namedtuple

class FittingOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')
        req.add_argument('--dsname', required=True, help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--num_workers', type=int, default=8, help='Number of processes used for data loading')
        gen.add_argument('--sidx', type=int, default=0, help='start index in data set for cluster jobs')
        gen.add_argument('--cbs', type=int, default=None, help='Number of samples per cluster job')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false')
        gen.set_defaults(pin_memory=True)

        io = self.parser.add_argument_group('io')
        io.add_argument('--pretrained_checkpoint', default=None,
                help='Load a pretrained checkpoint at the beginning training')

        train = self.parser.add_argument_group('Training Options')
        train.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
        train.add_argument('--batch_size', type=int, default=1, help='Batch size')
        train.add_argument('--img_res', type=int, default=224,
        help='Rescale bounding boxes to size [img_res, img_res] before feeding \
                        them in the network')
        train.add_argument('--max_steps', default=50, type=int, help='Weight of SMPL betas loss')
        train.add_argument('--beta_loss_weight', default=1.0, type=float, help='Weight of SMPL betas loss')
        train.add_argument('--keypoint_loss_weight', default=1.0, type=float, help='Weight of 2D and 3D keypoint loss')
        train.add_argument('--contact_loss_weight', default=10.0, type=float, help='weight for contact loss in regressor')

        return

    def parse_args(self):
        """Parse input arguments."""
        self.args = self.parser.parse_args()
        return self.args
