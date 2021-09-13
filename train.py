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


import os
# hack to get the correct gpu device id on cluster
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import torch

from configs.train_options import TrainOptions
from tuch.train.trainer import Trainer
from tuch.train.train_module import TUCH
from configs import config
from tuch.models.hmr import hmr
from tuch.smplify.smplifydc import SMPLifyDC
from data.essentials import constants
from tuch.datasets.mixed_dataset import MixedDataset
from tuch.train.loss import RegressorLoss
from tuch.models.smpl import SMPL


def main(options):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset: ToDo: load here
    datasets = (MixedDataset(options, set='train'),
                MixedDataset(options, set='val'))

    # Load SPIN
    modelspin = hmr(config.SMPL_MEAN_PARAMS).to(device)
    spincheckpoint = torch.load(config.SPIN_MODEL_CHECKPOINT)
    modelspin.load_state_dict(spincheckpoint['model'], strict=False)
    modelspin.eval()

    # load regressor
    model = hmr(smpl_mean_params=config.SMPL_MEAN_PARAMS,
                 pretrained=True,
    ).to(device)

    # load body model
    smpl = SMPL(config.SMPL_MODEL_DIR,
              batch_size=options.batch_size,
              create_transl=False
    ).to(device)
    num_verts = smpl.get_num_verts()
    face_tensor = torch.tensor(smpl.faces.astype(np.int64),
                     dtype=torch.long, device=device) \
                     .unsqueeze_(0) \
                     .repeat([options.batch_size,1,1])
    geodistssmpl = torch.tensor(np.load(config.GEODESICS_SMPL),
                      device=device)

    # load optimization routine
    smplify = SMPLifyDC(step_size=1e-2,
                  batch_size=options.batch_size,
                  num_iters=options.num_smplify_iters,
                  focal_length=constants.FOCAL_LENGTH,
                  geodistssmpl=geodistssmpl,
                  geothres=config.geothres,
                  euclthres=config.euclthres,
    )

    # loss function
    loss = RegressorLoss(options=options,
               device=device,
               num_verts=num_verts,
               faces=face_tensor,
               geodistssmpl=geodistssmpl,
               geothres=config.geothres,
               face_tensor=face_tensor,
    )

    # single training step iteration
    # combines regressor and loop
    cospin = TUCH(options=options,
               device=device,
               datasets=datasets,
               bodymodel=smpl,
               spin_model=modelspin,
               regressor=model,
               optimization=smplify,
               criterion=loss,
               geodistssmpl=geodistssmpl,
    )

    # create optimizer, use filter in case layers were fixed
    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=options.lr, weight_decay=0
    )

    # start training
    Trainer(options=options,
        train_module=cospin,
        optimizer=optimizer,
        device=device,
    ).fit()


if __name__ == '__main__':
    options = TrainOptions().parse_args()
    main(options)
