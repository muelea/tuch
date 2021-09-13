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


"""
Before you can process the data, you need to download and prepare it.
Please follow the instructions in lib/datasets/preprocess/README.md
"""
import argparse
from configs import config as cfg
from tuch.datasets.preprocess.pw3d import pw3d_extract
from tuch.datasets.preprocess.mpi_inf_3dhp import mpi_inf_3dhp_extract
from tuch.datasets.preprocess.dsc import dsc_extract
from tuch.datasets.preprocess.mtp import mtp_extract

def main(args):
    # define path to store extra files
    out_path = cfg.DBS_PATH

    if args.train_files_tuch:
        # DSC datasets
        dsc_extract(subset='df')
        dsc_extract(subset='lspet')
        dsc_extract(subset='lsp')

        # MTP dataset
        mtp_extract(mode='train')

    if args.val_files_tuch:
        # MTP dataset validation files
        mtp_extract(mode='val')

    if args.test_files_tuch:

        # MPI-INF-3DHP dataset preprocessing (test set)
        mpi_inf_3dhp_extract(cfg.MPI_INF_3DHP_ROOT, out_path, 'test')

        # 3DPW dataset preprocessing (test set)
        pw3d_extract(cfg.PW3D_ROOT, out_path, set='test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # extract datasets
    parser.add_argument('--train_files_tuch', default=False, action='store_true',
                    help='Extract files needed for training')
    parser.add_argument('--val_files_tuch', default=False, action='store_true',
                    help='Extract files needed for validation during training')
    parser.add_argument('--test_files_tuch', default=False, action='store_true',
                     help='Extract files needed for evaluation')
    args = parser.parse_args()
    main(args)
