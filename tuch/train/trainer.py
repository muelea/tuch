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
import time

import torch
import numpy as np

from configs import config
from data.essentials import constants
from torch.utils.data import DataLoader
from torchgeometry import rotation_matrix_to_angle_axis
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from tuch.utils.saver import CheckpointSaver, print_loss_dict, freq_to_step
from tuch.utils.data_loader import CheckpointDataLoader
from tuch.utils.renderer import Renderer

class Trainer(object):
    """Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """
    def __init__(
             self,
             options,
             train_module,
             optimizer,
             device,
        ):
        self.device = device

        self.options = options
        self.endtime = time.time() + self.options.time_to_run

        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.summary_writer = SummaryWriter(self.options.summary_dir)

        self.optimizer = optimizer

        # class for forward step (regressor and loop)
        self.J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
        self.joint_mapper_h36m = constants.H36M_TO_J14
        self.tuch = train_module

        # current model and optimizers
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.models_dict = {'model': self.tuch.model}

        # training parameters
        self.epoch_count = 0
        self.step_count = 0
        self.checkpoint = None

        if self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(self.models_dict,
                self.optimizers_dict, checkpoint_file=self.options.checkpoint)
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']
        # load weights from old model
        elif self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

        # evaluation files
        self.best_performance = float('inf')
        self.evaluation_accumulators = {}
        self.eval_features = ['betas', 'pose', 'vertices']
        for ef in self.eval_features:
            self.evaluation_accumulators[f'gt_{ef}'] = []
            self.evaluation_accumulators[f'pred_{ef}'] = []

        # Create renderer
        self.renderer = Renderer(
                            contactlist=self.tuch.contactlists,
                            focal_length=constants.FOCAL_LENGTH,
                            img_res=self.options.img_res,
                            faces=self.tuch.smpl.faces)


    def load_pretrained(self, checkpoint_file=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    self.models_dict[model].load_state_dict(checkpoint[model],
                                                            strict=False)
                    print('Pretrained checkpoint loaded')

    def fit(self):
        """Full training process."""
        # Run training for num_epochs epochs
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs),
                    total=self.options.num_epochs, initial=self.epoch_count):

            self.epoch = epoch
            self.train_one_epoch()
            print('================== EPOCH ', epoch, 'DONE =====================')


    def train_one_epoch(self):
        """Single epoch training step."""
        # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
        train_data_loader = CheckpointDataLoader(self.tuch.train_ds,
                                                 checkpoint=self.checkpoint,
                                                 batch_size=self.options.batch_size,
                                                 num_workers=self.options.num_workers,
                                                 pin_memory=self.options.pin_memory,
                                                 shuffle=self.options.shuffle_train)

        # get validation and summary steps from frequency
        num_steps_total = len(self.tuch.train_ds) // self.options.batch_size
        summary_steps = freq_to_step(self.options.summary_freq, num_steps_total)
        checkpoint_steps = freq_to_step(self.options.val_and_checkpoint_freq, num_steps_total)

        # Iterate over all batches in an epoch
        for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch '+str(self.epoch),
                                     total=num_steps_total,
                                     initial=train_data_loader.checkpoint_batch_idx),
                                     train_data_loader.checkpoint_batch_idx):
            # move input to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) \
                                else v for k,v in batch.items()}

            # make the training step: regressor - loop
            loss, loss_dict, output = self.tuch.forward_train_step(batch)

            # backprop regressor
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print loss to console
            print_loss_dict(loss_dict)

            # log loss values  
            for k, v in loss_dict.items():
                self.summary_writer.add_scalar('train/'+k, v, self.step_count)

            # Tensorboard logging every summary_steps steps
            if self.step_count % summary_steps == 0:
                self.add_train_images(batch, output)

            # validate and save checkpoint
            if self.step_count % checkpoint_steps == 0:
                # validate 
                self.validate()
                val_error = self.validate_final_step()
                # save result
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict,
                    self.epoch, 0, self.options.batch_size, self.step_count, val_error, 
                    train_data_loader.sampler.dataset_perm)
                self.tuch.fits_dict.save()

            self.step_count += 1

    def validate(self):

        self.tuch.model.eval()

        # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
        val_data_loader = DataLoader(self.tuch.val_ds,
                                     batch_size=self.options.batch_size,
                                     num_workers=self.options.num_workers,
                                     shuffle=False,
                                     drop_last=True)

        # empty eval data to append to eval measures
        for ef in self.eval_features:
            self.evaluation_accumulators[f'gt_{ef}'] = []
            self.evaluation_accumulators[f'pred_{ef}'] = []


        for step, batch in enumerate(val_data_loader):
            # move btach to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) \
                                else v for k,v in batch.items()}

            with torch.no_grad():
                # get ground truth
                gt_pose = batch['pose']
                gt_betas = batch['betas']
                gt_out = self.tuch.smpl(betas=gt_betas,
                           body_pose=gt_pose[:,3:],
                           global_orient=gt_pose[:,:3])
                gt_vertices = gt_out.vertices

                # get predicton
                pred_rotmat, pred_betas, pred_camera = self.tuch.model(batch['img'])
                pred_output = self.tuch.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:],
                    global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
                pred_vertices = pred_output.vertices
                pred_rotmat_hom = torch.cat([pred_rotmat.detach().view(-1, 3, 3).detach(),
                    torch.tensor([0,0,1], dtype=torch.float32, device=self.device) \
                    .view(1, 3, 1).expand(self.options.batch_size * 24, -1, -1)], dim=-1)
                pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom)
                pred_pose = pred_pose.contiguous().view(self.options.batch_size, -1)
                pred_pose[torch.isnan(pred_pose)] = 0.0

                # append to eval measures
                for ef in self.eval_features:
                    self.evaluation_accumulators[f'gt_{ef}'].append(eval(f'gt_{ef}'))
                    self.evaluation_accumulators[f'pred_{ef}'].append(eval(f'pred_{ef}'))
                
                # save images of first batch in validations set
                if step == 0:
                    pred_cam_t = torch.stack([pred_camera[:,1], pred_camera[:,2],
                        2*constants.FOCAL_LENGTH/(self.options.img_res * \
                            pred_camera[:,0] +1e-9)],dim=-1)
                    out = {'pred_vertices': pred_vertices, 'pred_cam_t': pred_cam_t}
                    self.add_val_images(batch, out)


    def validate_final_step(self):
        gt_vertices = torch.cat(self.evaluation_accumulators['gt_vertices'])
        pred_vertices = torch.cat(self.evaluation_accumulators['pred_vertices'])
        self.J_regressor_batch = self.J_regressor[None, :] \
                                   .expand(gt_vertices.shape[0], -1, -1) \
                                   .to(self.device)

        # ground truth joints
        gt_keypoints_3d = torch.matmul(self.J_regressor_batch, gt_vertices)
        gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
        gt_keypoints_3d = gt_keypoints_3d[:, self.joint_mapper_h36m, :]
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

        # predicted joints
        pred_keypoints_3d = torch.matmul(self.J_regressor_batch, pred_vertices)
        pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
        pred_keypoints_3d = pred_keypoints_3d[:, self.joint_mapper_h36m, :]
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

        # MPJE
        joint_dists = torch.sqrt(torch.sum((pred_keypoints_3d - gt_keypoints_3d) ** 2, dim=-1))
        joint_errors = joint_dists.mean(dim=-1).cpu().numpy()
        mpjpe = np.mean(joint_errors) * 1000

        # v2v Loss
        v2v_dists = torch.sqrt(torch.sum((pred_vertices - gt_vertices) ** 2, dim=-1))
        v2v_mean_dists = v2v_dists.mean(dim=-1).cpu().numpy()
        v2v = np.mean(v2v_mean_dists) * 1000

        # write results and add to tensorboard
        eval_dict = {
            'mpjpe': mpjpe,
            'v2v': v2v
        }

        for k, v in eval_dict.items():
            self.summary_writer.add_scalar('val/'+k, v, self.step_count)

        return mpjpe

    def add_train_images(self, input_batch, output):
        images = input_batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device) \
                                                                .reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device) \
                                                                .reshape(1,3,1,1)

        pred_vertices = output['pred_vertices']
        opt_vertices = output['opt_vertices']
        spin_vertices = output['spin_vertices']
        pred_cam_t = output['pred_cam_t']
        opt_cam_t = output['opt_cam_t']
        spin_cam_t = output['spin_cam_t']
        smplifyoptiverts = output['smplifyoptiverts']
        gt_contact_l3 = output['gt_contact_l3']
        gt_keypoints = output['gt_keypoints']
        if 'gt_verts_incontact' in output.keys():
            gt_verts_incontact = output['gt_verts_incontact']
        else:
            gt_verts_incontact = None
        has_contact_pc = output['has_contact_pc']
        has_contact = output['has_contact']
        valid_kpts_anno = output['valid_kpts_anno']
        images_black = images.clone()
        images_black[~valid_kpts_anno, :, :, :] = 0

        if smplifyoptiverts is not None:
            images_smplifycontactopti = self.renderer.visu_smplifycontactopti(smplifyoptiverts,
                opt_cam_t, images, gt_contact_l3, gt_vertsincontact_idx=gt_verts_incontact)

        images_pred = self.renderer.visualize_tbm(pred_vertices, pred_cam_t, images_black,
                gt_l3_contact=gt_contact_l3, gt_vertsincontact_idx=gt_verts_incontact,
                has_contact_pc=has_contact_pc, has_contact=has_contact)
        images_opt = self.renderer.visualize_tbm(opt_vertices, opt_cam_t, images,
                gt_l3_contact=gt_contact_l3, gt_vertsincontact_idx=gt_verts_incontact,
                has_contact_pc=has_contact_pc, has_contact=has_contact)
        images_spin = self.renderer.visualize_tbm(spin_vertices, spin_cam_t, images,
                gt_l3_contact=gt_contact_l3, gt_vertsincontact_idx=gt_verts_incontact,
                has_contact_pc=has_contact_pc, has_contact=has_contact)

        # add to tensorboard summary
        self.summary_writer.add_image('pred_shape', images_pred, self.step_count)
        self.summary_writer.add_image('opt_shape', images_opt, self.step_count)
        self.summary_writer.add_image('spin_shape', images_spin, self.step_count)
        if smplifyoptiverts is not None:
            self.summary_writer.add_image('smplify_contact_shape',
                                images_smplifycontactopti, self.step_count)

    def add_val_images(self, input_batch, output):
        images = input_batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device) \
                                                                .reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device) \
                                                                .reshape(1,3,1,1)

        pred_vertices = output['pred_vertices']
        pred_cam_t = output['pred_cam_t']

        images_pred = self.renderer.visualize_tbm(
            pred_vertices, pred_cam_t, images,
            gt_l3_contact=None, gt_vertsincontact_idx=None,
            has_contact_pc=None, has_contact=None)

        # add to tensorboard summary
        self.summary_writer.add_image('val/pred_shape', images_pred, self.step_count)