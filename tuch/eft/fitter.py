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

from __future__ import division
import os.path as osp
import torch
import numpy as np
import copy
import cv2
import os
import joblib

from configs import config
from data.essentials import constants
from torchgeometry import rotation_matrix_to_angle_axis
from tqdm import tqdm
from tuch.utils.renderer import Renderer

class EFTFitter(object):
    """Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """
    def __init__(
             self,
             options,
             dsname,
             dataset,
             model,
             loss,
             smpl,
             optimizer,
             device,
        ):
        self.device = device
        self.options = options
        self.dsname = dsname
        self.dataset = dataset

        dsidx = self.options.sidx
        cbs = len(dataset) if self.options.cbs is None else self.options.cbs
        self.process_idx = np.arange(dsidx * cbs, dsidx * cbs + cbs)
        self.process_idx = np.array([x for x in self.process_idx if x < len(self.dataset)])

        # prepare dict to save fits and create output dir
        orig_ds_path = config.DATASET_FILES['train'][dsname]
        self.output = joblib.load(orig_ds_path)
        size_ds = len(self.output['imgname'])
        self.output['betas'] = np.zeros((size_ds, 10))
        self.output['pose'] = np.zeros((size_ds, 72))
        if self.options.cbs is None:
            self.outputfn = orig_ds_path.replace(dsname, dsname + '_eft')
        else:
            self.outputfn = config.DATASET_FILES['train'][dsname] \
                            .replace('train.npz', '{}_train_{}.pt' \
                            .format(self.options.name, dsidx))
            self.outputfn = self.outputfn.replace('data/dbs/',
                    'out/temp/{}/{}/'.format(self.options.name, self.dsname))
            os.makedirs(osp.dirname(self.outputfn), exist_ok=True)
        
        print(f'Processing {orig_ds_path}. Save results to {self.outputfn}')

        # class for finetuning a single example
        self.J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
        self.joint_mapper_h36m = constants.H36M_TO_J14

        # current model and optimizers
        self.optimizer = optimizer
        self.optimizer_backup = copy.deepcopy(self.optimizer.state_dict())

        self.smpl = smpl
        self.model = model
        self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)
        self.model_backup = copy.deepcopy(self.model.state_dict())

        # load loss
        self.loss = loss

        # Create renderer
        self.renderer = Renderer(
                            contactlist={'dsc': self.loss.cdict},
                            focal_length=constants.FOCAL_LENGTH,
                            img_res=self.options.img_res,
                            faces=self.smpl.faces)


    def load_pretrained(self, checkpoint_file=None):
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['model'], strict=False)
            print('Pretrained checkpoint loaded')

    def fit(self):
        """Iterate  over all images."""
        # Iterate over all batches in an epoch
        #for step, batch in enumerate(tqdm(self.data_loader)):
        for step in tqdm(self.process_idx):
            batch = self.dataset[step]
            self.step = step

            # move input to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) \
                                else v for k,v in batch.items()}
            batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) \
                                else v for k,v in batch.items()}


            images = batch['img']
            # reset model weights to inital values
            self.model.load_state_dict(self.model_backup)
            self.optimizer.load_state_dict(self.optimizer_backup)
            self.model.train()

            for eft_step in range(self.options.max_steps):
                # model prediction
                rotmat, betas, camera = self.model(images)

                # smpl forward step
                body = self.smpl(betas=betas, body_pose=rotmat[:,1:],
                                 global_orient=rotmat[:,0].unsqueeze(1),
                                 pose2rot=False)

                # save initial prediction for visualisation
                if eft_step == 0:
                    body_init = body
                    camera_init = camera.detach().clone()

                # make the training step: regressor - loop
                loss, loss_dict = self.loss(body, camera, batch)

                # backprop regressor
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if loss.item() < 200 and eft_step > 20:
                    break

            # update output file
            self.output['betas'][step] = betas.detach().cpu().numpy().squeeze()
            rotmat_hom = torch.cat([rotmat.detach().view(-1, 3, 3).detach(),
                torch.tensor([0,0,1], dtype=torch.float32, device=self.device) \
                .view(1, 3, 1).expand(self.options.batch_size * 24, -1, -1)], dim=-1)
            pose = rotation_matrix_to_angle_axis(rotmat_hom) \
                       .contiguous().view(self.options.batch_size, -1)
            pose[torch.isnan(pose)] = 0.0
            self.output['pose'][step] = pose.cpu().numpy().squeeze()

            # visualize result and save output file
            #self.save_results(batch, body, camera,
            #                  body_init, camera_init)

        print('dump output file to', self.outputfn)
        joblib.dump(self.output, self.outputfn)

    def save_results(self, input_batch, body, camera,
                     body_init=None, camera_init=None):
        images = input_batch['img']
        contact = input_batch['contact']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device) \
                                                                .reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device) \
                                                                .reshape(1,3,1,1)

        # render eft fit
        camera_t = torch.stack([camera[:,1], camera[:,2],
                            2*self.loss.focal_length/(self.options.img_res * \
                            camera[:,0] +1e-9)],dim=-1)
        images_eft = self.renderer.visualize_eft(body.vertices.detach(),
                camera_t.detach(), images, contact)
        images_eft = images_eft.cpu().numpy().transpose(1, 2, 0) * 255
        images_out = np.clip(images_eft, 0, 255).astype(np.uint8)

        # render inital fit
        if body_init is not None:
            camera_t_init = torch.stack([camera_init[:,1], camera_init[:,2],
                            2*self.loss.focal_length/(self.options.img_res * \
                            camera_init[:,0] +1e-9)],dim=-1)
            images_init = self.renderer.visualize_eft(body_init.vertices.detach(),
                            camera_t_init.detach(), images, contact)
            images_init = images_init.cpu().numpy().transpose(1, 2, 0) * 255
            images_init = np.clip(images_init, 0, 255).astype(np.uint8)

            images_out = np.hstack((images_init, images_eft))

        # write images to outdir
        os.makedirs('out/temp/{}/'.format(self.options.name), exist_ok=True)
        outpath = osp.join('out/temp/{}/'.format(self.options.name), f'{self.step:05d}_eft.jpg')
        cv2.imwrite(outpath,
            cv2.cvtColor(images_out, cv2.COLOR_BGR2RGB)
        )
        print('image saves to' , outpath)
