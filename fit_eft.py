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
#os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0]

import numpy as np
import torch

from configs.eft_fitting_options import FittingOptions
from configs import config
from tuch.models.hmr import hmr
from tuch.models.smpl import SMPL
from tuch.eft.fitter import EFTFitter
from tuch.eft.loss import EFTLoss
from tuch.eft.dataset import EFTDataset

def main(options):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create dataset
    dsname = options.dsname
    dataset = EFTDataset(dsname)

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

    # loss function
    loss = EFTLoss(options=options,
               device=device,
               num_verts=num_verts,
               smpl=smpl,
               faces=face_tensor,
               geodistssmpl=geodistssmpl,
               geothres=config.geothres,
               face_tensor=face_tensor,
               keypoint_weight=options.keypoint_loss_weight,
               shape_weight=options.beta_loss_weight,
               contact_weight=options.contact_loss_weight,
    )

    # create optimizer, use filter in case layers were fixed
    optimizer = torch.optim.Adam(lr=options.lr,
                   params=model.parameters(),
                   weight_decay=0
    )

    # start training
    EFTFitter(options=options,
        optimizer=optimizer,
        device=device,
        model=model,
        loss=loss,
        smpl=smpl,
        dsname=dsname,
        dataset=dataset,
    ).fit()


if __name__ == '__main__':
    options = FittingOptions().parse_args()
    main(options)
