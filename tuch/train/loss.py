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
import sys
from tuch.utils.geometry import batch_rodrigues
from tuch.utils.contact import batch_pairwise_dist
from tuch.utils.contact import winding_numbers
import pickle
import os.path as osp
from configs import config
from data.essentials.segments.smpl import segm_utils as exn
from tuch.utils.segmentation import BatchBodySegment

def batch_face_normals(triangles):
    # Calculate the edges of the triangles
    # Size: BxFx3
    edge0 = triangles[:, :, 1] - triangles[:, :, 0]
    edge1 = triangles[:, :, 2] - triangles[:, :, 0]
    # Compute the cross product of the edges to find the normal vector of
    # the triangle
    aCrossb = torch.cross(edge0, edge1, dim=2)
    # Normalize the result to get a unit vector
    normals = aCrossb / torch.norm(aCrossb, 2, dim=2, keepdim=True)

    return normals


class RegressorLoss(nn.Module):
    def __init__(
            self,
            options,
            device,
            num_verts,
            faces,
            geodistssmpl,
            geothres=0.2,
            euclthres=0.02,
            face_tensor=None,
            use_hd=True,
    ):
        super(RegressorLoss, self).__init__()
        self.device = device
        self.options = options

        # spin criterion
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)

        # contact criterion
        self.faces = faces
        self.nv = num_verts
        self.geodistssmpl = geodistssmpl
        self.geothres = geothres
        self.geomask = geodistssmpl > geothres
        self.euclthres = euclthres

        self.face_tensor = face_tensor

        self.use_hd = use_hd
        if not self.use_hd:
            self.geodistssmpl = geodistssmpl
            self.geomask = geodistssmpl > geothres
        else:
            path = osp.join(config.HD_MODEL_DIR, 'smpl_neutral_hd_vert_regressor.npy')
            self.Vert_Regressor = np.load(path)
            self.Vert_Regressor = torch.tensor(self.Vert_Regressor, device=device).float()
            path = osp.join(config.HD_MODEL_DIR, 'smpl_neutral_hd_sample_from_mesh_out.pkl')
            with open(path, 'rb') as f:
                self.geovec = pickle.load(f)['faces_vert_is_sampled_from']
            self.geovec = torch.tensor(self.geovec, device=device)
            self.geovec_verts = self.face_tensor[0][self.geovec][:,0]

        # create body segments to filer allowed intersection
        self.segments = BatchBodySegment([x for x in exn.segments.keys()], self.face_tensor[0])


    def forward(
            self,
            # spin loss arguments
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
            valid_fit,
            valid_fit_shape,
    ):

        ##### contact criterions #####
        loss_contact = torch.tensor(0)
        if self.options.contact_loss_weight > 0:
            loss_contact = self.contact_loss(pred_vertices, valid_fit)
        contact_loss = self.options.contact_loss_weight * loss_contact

        ##### SPIN criterions #####
        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = self.smpl_losses(
                                                    pred_rotmat,
                                                    pred_betas,
                                                    opt_pose,
                                                    opt_betas,
                                                    valid_fit,
                                                    valid_fit_shape)

        # Compute 2D keypoint loss
        loss_keypoints = self.keypoint_loss(
                                    pred_keypoints_2d,
                                    gt_keypoints_2d,
                                    self.options.openpose_train_weight,
                                    self.options.gt_train_weight,
                                    valid_fit)

        # Compute 3D keypoint loss
        loss_keypoints_3d = self.keypoint_3d_loss(
                                            pred_joints,
                                            gt_joints,
                                            has_pose_3d)

        # Per-vertex loss for the shape
        loss_shape = self.shape_loss(pred_vertices, opt_vertices, valid_fit)

        # camera loss
        cam_loss = ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean()

        ###### total loss ######
        spin_loss = self.options.shape_loss_weight * loss_shape + \
                     self.options.keypoint_loss_weight * loss_keypoints + \
                     self.options.keypoint_loss_weight * loss_keypoints_3d + \
                     self.options.pose_loss_weight * loss_regr_pose + \
                     self.options.beta_loss_weight * loss_regr_betas + \
                     cam_loss

        total_loss = spin_loss + contact_loss
        loss_dict = {
            'loss_shape': loss_shape,
            'loss_keypoints': loss_keypoints,
            'loss_keypoints_3d': loss_keypoints_3d,
            'loss_regr_pose': loss_regr_pose,
            'loss_regr_betas': loss_regr_betas,
            'loss_cam': cam_loss,
            'loss_contact': loss_contact,
        }

        return total_loss, loss_dict



    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d,
                openpose_weight, gt_weight, valid_fit=None):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d,
                                gt_keypoints_2d[:, :, :-1])).mean(axis=(1,2))
        loss = loss[valid_fit].mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint
        annotations are available. The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + \
                           pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d,
                                                    gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL
        annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape,
                                        gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl_pose, has_smpl_shape):
        # get shape loss
        pred_rotmat_valid = pred_rotmat[has_smpl_pose == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3))\
                            .view(-1, 24, 3, 3)[has_smpl_pose == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid,
                                                 gt_rotmat_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)

        # get shape loss
        pred_betas_valid = pred_betas[has_smpl_shape == 1]
        gt_betas_valid = gt_betas[has_smpl_shape == 1]
        if len(pred_betas_valid) > 0:
            loss_regr_betas = self.criterion_regr(pred_betas_valid,
                                                  gt_betas_valid)
        else:
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)

        return loss_regr_pose, loss_regr_betas

    def contact_loss(self, pred_vertices, valid_fit):
        batch_size = pred_vertices.shape[0]
        #triangles = pred_vertices.detach().view([-1, 3])[self.face_tensor]

        contact_loss = torch.FloatTensor(batch_size).fill_(0.).to(self.device)

        # iterate batchbecause of memory
        for bidx in torch.where(valid_fit)[0]:
            v2v_pull = torch.tensor([0.0], device=self.device)
            v2v_push = torch.tensor([0.0], device=self.device)

            with torch.no_grad():

                # for each vertex find closest vertex that is far enough
                # away in geodesicdistance
                v2v = batch_pairwise_dist(pred_vertices[[bidx], :,:],
                                      pred_vertices[[bidx], :,:],
                                      squared=True)

                # compute interior and exterior vertices
                triangles = (pred_vertices[bidx])[self.face_tensor[0]]
                exterior = winding_numbers(pred_vertices.detach()[[bidx],:,:],
                           triangles[None]).squeeze().le(0.99)
                # filter allowed self intersections
                test_segments = self.segments.batch_has_self_isec(pred_vertices[bidx].unsqueeze(0))
                for segm_name, segm_ext in zip(self.segments.names, test_segments):
                    exterior[self.segments.segmentation[segm_name].segment_vidx[(segm_ext).detach().cpu().numpy() == 0]] = 1

                # find vertices is in contact
                v2v[:, ~self.geomask] = float('inf')
                v2v_min_s, v2v_min_idx = torch.min(v2v, dim=1)

            # instead of the v2v distances in SMPL-X topology, use
            # mesh with higher density and evenly distributed vertices
            if self.use_hd:
                with torch.no_grad():
                    # we do not want to consider inside vertices in crook regions. Remove these from inside verts
                    # find vertices inside or in contact
                    verts_in_contact_idx = torch.where((v2v_min_s[0] < self.euclthres**2) | ~exterior)[0]
                    faces_in_contact_idx = torch.where((verts_in_contact_idx.expand(self.faces[0].flatten().shape[0], -1) == \
                                        self.faces[0].flatten().unsqueeze(-1).expand(-1, verts_in_contact_idx.shape[0])).any(1).reshape(-1, 3).any(1))[0]
                    hd_verts_in_contact_idx = (self.geovec.unsqueeze(-1) == faces_in_contact_idx.expand(self.geovec.shape[0], -1)).any(1)

                # if you find vertices in contact of inside
                if hd_verts_in_contact_idx.sum() > 0:
                    hd_verts_in_contact = torch.einsum('ij,njk->nik', self.Vert_Regressor[hd_verts_in_contact_idx], pred_vertices[[bidx], :, :])

                    with torch.no_grad():
                        hd_v2v = batch_pairwise_dist(hd_verts_in_contact, hd_verts_in_contact, squared=True)
                        hd_geo = self.geomask[self.geovec_verts[hd_verts_in_contact_idx],:][:,self.geovec_verts[hd_verts_in_contact_idx]]
                        hd_v2v[:, ~hd_geo] = float('inf')
                        hd_v2v_min, hd_v2v_min_idx = torch.min(hd_v2v, dim=1)
                        #del exterior, v2v, verts_in_contact_idx, faces_in_contact_idx, hd_v2v, v2v_min_s, hd_geo

                        # add little offset to those vertices for in/ex computation
                        face_normals = 0.001 * batch_face_normals(triangles[None])[0]
                        hd_verts_in_contact_offset = hd_verts_in_contact + face_normals[self.geovec[hd_verts_in_contact_idx], :].unsqueeze(0)
                        exterior = winding_numbers(hd_verts_in_contact_offset.detach(), triangles[None]).squeeze().le(0.99)

                    v2v_min = torch.norm(hd_verts_in_contact[0] - hd_verts_in_contact[0, hd_v2v_min_idx, :], dim=2)[0]
                else:
                    v2v_min = None
            else:
                v2v_min = torch.norm(pred_vertices[bidx] - pred_vertices[bidx, v2v_min_idx, :], dim=2)[0]

            if v2v_min is not None:
                # apply contact loss to vertices in contact
                if exterior.sum() > 0:
                    v2v_pull = 0.005 * torch.tanh(v2v_min[exterior] / 0.005)**2

                # apply contact loss to inside vertices
                if (~exterior).sum() > 0:
                    v2v_push = 1.0 * torch.tanh(v2v_min[~exterior] / 0.04)**2

                # compute contact loss
                contact_loss[bidx] = v2v_pull.sum() + v2v_push.sum()

        return contact_loss[valid_fit].mean()