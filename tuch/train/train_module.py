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
import torch.nn as nn
import numpy as np
from torchgeometry import rotation_matrix_to_angle_axis
import pickle
import os.path as osp

from tuch.utils.geometry import perspective_projection, estimate_translation
from tuch.utils.contact import batch_pairwise_dist
from configs import config
from data.essentials import constants
from .fits_dict import FitsDict


class TUCH():
    def __init__(
            self,
            options,
            device,
            datasets,
            bodymodel,
            spin_model,
            regressor,
            optimization,
            criterion,
            geodistssmpl,
        ):
        self.options = options
        self.device = device
        self.focal_length = constants.FOCAL_LENGTH
        self.train_ds, self.val_ds = datasets

        # Load dictionary of fits
        self.fits_dict = FitsDict(self.options, self.train_ds)

        # regressor and loop
        self.modelspin = spin_model
        self.model = regressor
        self.smplify = optimization

        # load smpl model
        self.smpl = bodymodel
        self.geodistssmpl = geodistssmpl

        # load losses
        self.criterion_cospin = criterion

        ######## load contact data ###########
        classes = pickle.load(open(osp.join(config.DSC_ROOT, 'classes.pkl'), 'rb'))
        csig = pickle.load(open(osp.join(config.DSC_ROOT, 'ContactSigSMPL.pkl'), 'rb'))
        self.contactlists = {'classes': classes, 'csig': csig}

    def contact_from_verts(self, verts,
                                 mode = 'regions'):

        """
        compute region based pairwise contact.
        Speed up this function will speed up training loop!
        """
        batch_size = verts.shape[0]

        if mode == 'regions':
            cdict = self.contactlists
            numregpairs = len(cdict['classes'])
            pred_contact = torch.zeros(batch_size, numregpairs, dtype=torch.float).to(self.device)

            for idx, regpair in enumerate(cdict['classes']):
                verts1_idxs, verts2_idxs = cdict['csig'][regpair[0]], cdict['csig'][regpair[1]]
                bp1_verts = verts[:, verts1_idxs, :]
                bp2_verts = verts[:, verts2_idxs, :]

                dists = batch_pairwise_dist(bp1_verts, bp2_verts, squared=True)
                distsflat = dists.view(dists.shape[0], -1)
                pred_contact[:, idx] = torch.min(distsflat, dim=1)[0]
        return pred_contact

    def get_verts_in_contact(self, verts):

        pred_contact = {}
        mgeo = self.geodistssmpl >= config.geothres
        for bidx in range(verts.shape[0]):
            gt_verts_incontact = batch_pairwise_dist(verts[[bidx]],
                                                    verts[[bidx]],
                                                    squared=True)[0]
            meucl = gt_verts_incontact < (config.euclthres**2)
            cmask = meucl * mgeo # vertices in contact
            gt_verts_incontact[~cmask] = 100000
            cinrow = cmask.sum(1) > 0
            gt_verts_incontact_min = torch.min(gt_verts_incontact, dim=1)[1]
            idxs1 = torch.where(cinrow)[0]
            idxs2 = gt_verts_incontact_min[cinrow]
            pred_contact[bidx] = [idxs1, idxs2]

        return pred_contact

    def forward_train_step(self, input_batch):
        camera_center = torch.zeros(self.options.batch_size, 2, device=self.device)

        # set hmr model to training mode
        self.model.train()

        ##### Get data from the batch #####
        # image and augmentation params
        images = input_batch['img']
        batch_size = images.shape[0]
        indices = input_batch['sample_index']
        is_flipped = input_batch['is_flipped']
        rot_angle = input_batch['rot_angle']
        dataset_name = input_batch['dataset_name']
        contactlist = self.contactlists

        # get batch info
        has_pose_3d = input_batch['has_pose_3d'].bool()
        has_disc_contact = input_batch['has_disc_contact'].bool()
        has_2d_keypoints_gtanno = input_batch['has_gt_kpts'].bool()
        has_smpl = input_batch['has_smpl'].bool()
        has_pgt_smpl = input_batch['has_pgt_smpl'].bool()
        has_smpl_ = has_smpl | has_pgt_smpl

        # get ground truth data (includes mimic'ed poses)
        gt_keypoints_2d = input_batch['keypoints'] # 2D keypoints
        gt_joints = input_batch['pose_3d'] # 3D pose
        gt_pose = input_batch['pose']
        gt_betas = input_batch['betas']
        gt_disc_contact = input_batch['contact_vec']
        gt_out = self.smpl(betas=gt_betas,
                           body_pose=gt_pose[:,3:],
                           global_orient=gt_pose[:,:3])
        gt_model_joints = gt_out.joints
        gt_verts = gt_out.vertices

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * \
                                          (gt_keypoints_2d_orig[:, :, :-1] + 1)
        #gt_verts_incontact = self.get_verts_in_contact(gt_verts)

        ##### get data from the dictionary #####
        # get the current best fits
        opt_pose, opt_betas = self.fits_dict[(dataset_name, indices.cpu(),
                                              rot_angle.cpu(), is_flipped.cpu())]
        opt_pose = opt_pose.to(self.device)
        opt_betas = opt_betas.to(self.device)
        opt_output = self.smpl(betas=opt_betas,
                               body_pose=opt_pose[:,3:],
                               global_orient=opt_pose[:,:3])
        opt_vertices = opt_output.vertices
        opt_joints = opt_output.joints
        opt_contact_l3 = self.contact_from_verts(opt_vertices,
                                mode='regions')


        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(gt_model_joints,
                                        gt_keypoints_2d_orig,
                                        focal_length=self.focal_length,
                                        img_size=self.options.img_res,
                                        has_2d_kp_anno=has_2d_keypoints_gtanno)
        opt_cam_t = estimate_translation(opt_joints,
                                        gt_keypoints_2d_orig,
                                        focal_length=self.focal_length,
                                        img_size=self.options.img_res,
                                        has_2d_kp_anno=has_2d_keypoints_gtanno)
        opt_joint_loss = self.smplify.get_fitting_loss(opt_pose, opt_betas,
            opt_cam_t, 0.5 * self.options.img_res * torch.ones(batch_size, 2,
            device=self.device), gt_keypoints_2d_orig, has_2d_keypoints_gtanno).mean(dim=-1)

        # Get the fits of the original SPIN model for visualization in tensorboard.
        with torch.no_grad():
            # forward pass SPIN model in eval mode
            pred_rotmat_spin, pred_betas_spin, pred_camera_spin = self.modelspin(images)
            pred_output_spin = self.smpl(betas=pred_betas_spin, body_pose=pred_rotmat_spin[:,1:],
                    global_orient=pred_rotmat_spin[:,0].unsqueeze(1), pose2rot=False)
            spin_vertices = pred_output_spin.vertices.clone()
            spin_cam_t = torch.stack([pred_camera_spin[:,1],
                                      pred_camera_spin[:,2],
                                      2*self.focal_length/(self.options.img_res * \
                                                    pred_camera_spin[:,0] +1e-9)],dim=-1)

        ############################################
        ############### Regressor ##################
        ############################################
        # feed images to network with contact
        pred_rotmat, pred_betas, pred_camera = self.model(images)
        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:],
                global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        # Convert predicted rotation matrices to axis-angle
        pred_rotmat_hom = torch.cat([pred_rotmat.detach().view(-1, 3, 3).detach(),
            torch.tensor([0,0,1], dtype=torch.float32, device=self.device) \
            .view(1, 3, 1).expand(batch_size * 24, -1, -1)], dim=-1)
        pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(batch_size, -1)
        pred_pose[torch.isnan(pred_pose)] = 0.0
        pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*self.focal_length/(self.options.img_res * \
                                                pred_camera[:,0] +1e-9)],dim=-1)
        pred_keypoints_2d = perspective_projection(pred_joints,
                    rotation=torch.eye(3, device=self.device) \
                        .unsqueeze(0).expand(batch_size, -1, -1),
                    translation=pred_cam_t,
                    focal_length=self.focal_length,
                    camera_center=camera_center)

        # Normalize keypoints to [-1,1] and denormalize to pixel space
        pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)
        #pred_keypoints_2d_orig = 0.5 * self.options.img_res * (pred_keypoints_2d + 1)

        ############################################
        ############# Optimization #################
        ############################################
        # for discrete labeled poses we optimize the predicted pose in
        # the smplify loop
        smplifyoptiverts = None
        if self.options.run_smplify:
            # Run SMPLify optimization starting from network prediction
            new_opt_vertices, new_opt_joints,\
            new_opt_pose, new_opt_betas,\
            new_opt_cam_t, new_opt_joint_loss, \
            smplifyoptiverts = self.smplify(
                pred_pose.detach(),
                pred_betas.detach(),
                pred_cam_t.detach(),
                0.5 * self.options.img_res * \
                torch.ones(batch_size, 2, device=self.device),
                gt_keypoints_2d_orig,
                use_contact=self.options.use_contact_in_the_loop,
                contactlist=contactlist,
                gt_contact=[gt_disc_contact, None],
                ignore_idxs=has_smpl_,
                has_discrete_contact=has_disc_contact,
                has_gt_keypoints=has_2d_keypoints_gtanno,
                contact_loss_weight=self.options.contact_in_the_loop_loss_weight,
                contact_loss_return='sum',
                segments=self.criterion_cospin.segments
            )

            # Find out if optimized fit is better than original fit update the
            # dictionary for the examples where this is the case
            new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)
            update_jointloss = (new_opt_joint_loss <= opt_joint_loss)
            update = update_jointloss

            # for discrete contact we check if the new opt mesh is closer
            # to desired contact
            new_opt_contact_l3 = self.contact_from_verts(new_opt_vertices,
                                    mode='regions')
            update_contact_l3 = ((gt_disc_contact*new_opt_contact_l3) <= \
                                 (gt_disc_contact*opt_contact_l3)).sum(1) > 0
            if self.options.use_contact_in_the_loop:
                update[has_disc_contact] = (update_contact_l3 * update)[has_disc_contact]

            opt_joint_loss[update] = new_opt_joint_loss[update]
            opt_vertices[update, :] = new_opt_vertices[update, :]
            opt_contact_l3[update, :] = new_opt_contact_l3[update, :]
            opt_joints[update, :] = new_opt_joints[update, :]
            opt_pose[update, :] = new_opt_pose[update, :]
            opt_betas[update, :] = new_opt_betas[update, :]
            opt_cam_t[update, :] = new_opt_cam_t[update, :]

            self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(),
                is_flipped.cpu(), update.cpu())] = (opt_pose.cpu(), opt_betas.cpu())

        # Do not replace betas, because it can kill the contact we just optimized.
        # opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.

        # Replace the optimized parameters with the ground truth parameters, if available
        opt_cam_t[has_smpl_, :] = gt_cam_t[has_smpl_, :]
        opt_joints[has_smpl_, :, :] = gt_model_joints[has_smpl_, :, :]
        opt_pose[has_smpl_, :] = gt_pose[has_smpl_, :]
        opt_betas[has_smpl_, :] = gt_betas[has_smpl_, :]
        opt_vertices[has_smpl_, :] = gt_verts[has_smpl_, :]

        # Assert whether a fit is valid by comparing the joint loss with the threshold
        valid_fit = (opt_joint_loss < self.options.smplify_threshold).to(self.device)
        # Add the examples with GT parameters to the list of valid fits
        valid_fit_pose = has_smpl_ | valid_fit
        valid_fit_shape = has_smpl_ | valid_fit

        ############################################
        ################ Losses ####################
        ############################################
        loss, loss_dict = self.criterion_cospin(
                                          pred_rotmat,
                                          pred_betas,
                                          opt_pose,
                                          opt_betas,
                                          pred_keypoints_2d,
                                          gt_keypoints_2d,
                                          pred_joints,
                                          gt_joints,
                                          has_pose_3d,
                                          pred_vertices,
                                          opt_vertices,
                                          pred_camera,
                                          valid_fit_pose,
                                          valid_fit_shape,
        )
        # Pack output arguments for tensorboard logging
        losses = {'loss': loss.detach()}
        for k, val in loss_dict.items():
            losses[k] = val.detach()

        output = {'pred_vertices': pred_vertices.detach(),
                  'spin_vertices': spin_vertices,
                  'opt_vertices':opt_vertices.detach(),
                  'pred_cam_t': pred_cam_t.detach(),
                  'spin_cam_t': spin_cam_t,
                  'opt_cam_t':opt_cam_t.detach(),
                  'smplifyoptiverts': smplifyoptiverts,
                  'gt_contact_l3': gt_disc_contact,
                  'has_contact_pc': has_disc_contact,
                  'has_contact': has_disc_contact,
                  'valid_kpts_anno': valid_fit | has_smpl_,
                  'gt_keypoints': gt_keypoints_2d_orig}

        return loss, losses, output
