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
import os

from tuch.models.smpl import SMPL
from .losses import camera_fitting_loss, body_fitting_loss, contact_fitting_loss
from configs import config
from data.essentials import constants
from .prior import MaxMixturePrior
import numpy as np

class SMPLifyDC():
    """SMPLify-DC optimization follows the SMPLify routine, but takes
    discrete contact annotations into account."""
    def __init__(self,
                 step_size=1e-2,
                 batch_size=66,
                 num_iters=100,
                 focal_length=5000,
                 geodistssmpl=None,
                 geothres=0.0,
                 euclthres=0.0,
                 device=torch.device('cuda')):

        # Store options
        self.device = device
        self.focal_length = focal_length
        self.step_size = step_size

        # Ignore the the following joints for the fitting process
        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip', 'Right Hip', 'Left Hip']
        self.ign_joints = [constants.JOINT_IDS[i] for i in ign_joints]
        self.num_iters = num_iters
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder=config.PRIOR_FOLDER,
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)

        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=batch_size,
                         create_transl=False).to(self.device)

        self.face_tensor = torch.tensor(self.smpl.faces.astype(np.int64),
                               dtype=torch.long, device=self.device) \
                               .unsqueeze_(0) \
                               .repeat([batch_size,1,1])

        self.geodistssmpl = geodistssmpl
        self.geothres = geothres
        self.geomask = self.geodistssmpl > self.geothres
        self.euclthres = euclthres

    def __call__(self, init_pose, init_betas, init_cam_t,
                    camera_center, keypoints_2d, use_contact=False,
                    contactlist=[], gt_contact=None,
                    ignore_idxs=None, has_discrete_contact=None,
                    has_gt_keypoints=None, contact_loss_weight=1,
                    contact_loss_return='sum', segments=None):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """

        batch_size = init_pose.shape[0]

        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.clone()

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1].clone()

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # fix pose and global orient
        body_pose.requires_grad=False
        camera_translation.requires_grad = True
        if use_contact:
            global_orient.requires_grad=False
            # optimize shape and translation
            betas.requires_grad=True
            camera_opt_params = [betas, camera_translation]
        else:
            global_orient.requires_grad=True
            betas.requires_grad=False
            camera_opt_params = [global_orient, camera_translation]

        camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(self.num_iters):
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas)
            # When you use contact and want to let the 
            # camera loss converge use a higher shape prior weight, e.g. 3.0.
            spw = 1.0 if use_contact else 0.0
            loss = camera_fitting_loss(smpl_output, camera_translation,
                                       init_cam_t, camera_center,
                                       joints_2d, joints_conf,
                                       focal_length=self.focal_length,
                                       shape_prior_weight=spw)

            camera_optimizer.zero_grad()
            loss.backward()
            camera_optimizer.step()

        # version smplify-x+c optimization to bring touching bodyparts into contact
        optiverts = []
        if use_contact:
            opt_pose_smplifyloop1 = body_pose.clone()
            opt_global_orient_smplifyloop1 = global_orient.clone()

            # fix camera and shape
            camera_translation.requires_grad=False
            betas.requires_grad=False
            # optimize pose and global orient
            body_pose.requires_grad=True
            global_orient.requires_grad=True

            contact_opt_params = [body_pose, global_orient]
            body_contact_optimizer = torch.optim.Adam(contact_opt_params, lr=self.step_size)

            # For joints ignored during fitting, set the confidence to 0
            joints_conf[:, self.ign_joints] = 0.0

            for i in range(self.num_iters):
                smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas)
                model_joints = smpl_output.joints
                model_verts = smpl_output.vertices
                optiverts += [model_verts]
                loss = contact_fitting_loss(body_pose, global_orient,
                                         opt_pose_smplifyloop1,
                                         opt_global_orient_smplifyloop1,
                                         betas, model_joints,
                                         self.geomask,
                                         self.euclthres,
                                         camera_translation, camera_center,
                                         joints_2d, joints_conf, self.pose_prior,
                                         cdict=contactlist,
                                         gt_contact=gt_contact,
                                         ignore_idxs=ignore_idxs,
                                         has_discrete_contact=has_discrete_contact,
                                         verts=model_verts,
                                         face_tensor = self.face_tensor,
                                         focal_length=self.focal_length,
                                         contact_loss_weight=contact_loss_weight,
                                         output=contact_loss_return,
                                         segments=segments)

                body_contact_optimizer.zero_grad()
                loss.backward()
                body_contact_optimizer.step()
        else:
            # SPIN imprementation of smplify. Contact not included
            # Step 2: Optimize body joints
            # Optimize only the body pose and global orientation of the body
            body_pose.requires_grad=True
            betas.requires_grad=True
            global_orient.requires_grad=True
            camera_translation.requires_grad = False
            body_opt_params = [body_pose, betas, global_orient]

            # For joints ignored during fitting, set the confidence to 0
            joints_conf[:, self.ign_joints] = 0.0

            body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
            for i in range(self.num_iters):
                smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas)
                model_joints = smpl_output.joints
                model_verts = smpl_output.vertices
                optiverts += [model_verts]
                loss = body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                     joints_2d, joints_conf, self.pose_prior,
                                     focal_length=self.focal_length)
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

        if len(optiverts) == 0:
            optiverts = None

        # Get final loss value and get full skin
        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas,
                                    return_full_pose=True)
            model_joints = smpl_output.joints
            # For openpose joints set confidence to 0, if gt joints are available
            if has_gt_keypoints is not None:
                joints_conf[has_gt_keypoints, :25] = 0
            reprojection_loss = body_fitting_loss(body_pose, betas, model_joints,
                                            camera_translation, camera_center,
                                            joints_2d, joints_conf, self.pose_prior,
                                            focal_length=self.focal_length,
                                            output='reprojection')

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        return vertices, joints, pose, betas, camera_translation, reprojection_loss, optiverts

    def get_fitting_loss(self, pose, betas, cam_t, camera_center, keypoints_2d, has_gt_keypoints=None):
        """Given body and camera parameters, compute reprojection loss value.
        Input:
            pose: SMPL pose parameters
            betas: SMPL beta parameters
            cam_t: Camera translation
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            reprojection_loss: Final joint reprojection loss
        """

        batch_size = pose.shape[0]

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]
        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.
        # For openpose joints set confidence to 0, if gt joints are available
        if has_gt_keypoints is not None:
            joints_conf = joints_conf.clone()
            joints_conf[has_gt_keypoints, :25] = 0

        # Split SMPL pose to body pose and global orientation
        body_pose = pose[:, 3:]
        global_orient = pose[:, :3]

        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            model_joints = smpl_output.joints
            reprojection_loss = body_fitting_loss(body_pose, betas, model_joints, cam_t, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior,
                                                  focal_length=self.focal_length,
                                                  output='reprojection')

        return reprojection_loss
