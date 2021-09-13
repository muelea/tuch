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
import pickle
import os.path as osp

from tuch.utils.geometry import perspective_projection
from tuch.utils.contact import batch_pairwise_dist
from configs import config
from data.essentials.segments.smpl import segm_utils as exn
from tuch.utils.segmentation import BatchBodySegment

from tuch.utils.contact import winding_numbers

class EFTLoss(nn.Module):
    def __init__(
            self,
            options,
            device,
            smpl,
            num_verts,
            faces,
            geodistssmpl,
            geothres,
            face_tensor=None,
            use_hd=True,
            keypoint_weight=1.0,
            shape_weight=1.0,
            contact_weight=1.0,
    ):
        super(EFTLoss, self).__init__()
        self.device = device
        self.options = options
        self.focal_length = 5000
        self.camera_center = torch.tensor([0,0]).to('cuda')

        # eft criterion
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)

        # eft weights
        self.keypoints_weight = keypoint_weight
        self.shape_weight = shape_weight
        self.contact_weight = contact_weight

        # contact utils
        self.face_tensor = face_tensor
        self.geodistssmpl = geodistssmpl
        self.geothres = geothres
        self.geomask = self.geodistssmpl > self.geothres
        classes = pickle.load(open(osp.join(config.DSC_ROOT, 'classes.pkl'), 'rb'))
        csig = pickle.load(open(osp.join(config.DSC_ROOT, 'ContactSigSMPL.pkl'), 'rb'))
        self.cdict = {'classes': classes, 'csig': csig}

        # mesh segments for intersection tests
        self.segments = BatchBodySegment([x for x in exn.segments.keys()], self.face_tensor)

    def forward(self, body, camera, batch):
        batch_size = camera.shape[0]
        gt_keypoints, gt_contact = batch['keypoints'], batch['contact']

        model_joints = body.joints
        rotation = torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        camera_t = torch.stack([camera[:,1], camera[:,2],
                                2*self.focal_length/(self.options.img_res * \
                                camera[:,0] +1e-9)],dim=-1)

        pred_keypoints_2d = perspective_projection(model_joints,
                                                   rotation,
                                                   camera_t,
                                                   self.focal_length,
                                                   self.camera_center)
        # Normalize keypoints to [-1,1] and denormalize to pixel space
        pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)
        pred_keypoints_2d_orig = 0.5 * self.options.img_res * (pred_keypoints_2d + 1)

        # keypoints loss
        gt_keypoints_2d_orig = gt_keypoints.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * \
                                          (gt_keypoints_2d_orig[:, :, :-1] + 1)
        loss_keypoints = self.keypoint_loss(pred_keypoints_2d_orig,
                                            gt_keypoints_2d_orig) * \
                         self.keypoints_weight

        # shape loss
        loss_shape = torch.mean(body.betas ** 2) * self.shape_weight

        # contact loss
        loss_contact = torch.tensor(0.0, device=self.device)
        if self.contact_weight > 0:
            loss_contact = self.contact_loss(gt_contact, body.vertices) * self.contact_weight

        loss = 60 * (loss_keypoints + loss_shape + loss_contact)

        loss_dict = {
            'loss_shape': loss_shape,
            'loss_keypoints': loss_keypoints,
            'loss_contact': loss_contact,
        }

        print(loss.item(), loss_shape.item(), loss_keypoints.item(), loss_contact.item())

        return loss, loss_dict

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d):
        """ Compute 2D reprojection loss on the keypoints [openpose | ground truth]).
        The loss is weighted by the confidence.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d,
                                gt_keypoints_2d[:, :, :-1])).mean(axis=(1,2))
        return loss.mean()

    def contact_loss(self, gt_contact, verts):
        """ Compute contact loss for DSC data. """

        batch_size = self.options.batch_size

        # Pull contact areas together if discrete label available
        r2r_loss = torch.zeros(batch_size).to(self.device)

        # Contact loss pulling areas that are close together
        contact_loss = torch.zeros(batch_size).to(self.device)

        for bidx in range(verts.shape[0]):
            # squared pairwise distnace between vertices
            pred_verts_dists = batch_pairwise_dist(verts[[bidx],:,:],
                                                   verts[[bidx],:,:],
                                                   squared=True)
            with torch.no_grad():
                # find intersecting vertices
                triangles = (verts[bidx])[self.face_tensor[0]]
                exterior = winding_numbers(verts[[bidx],:,:], triangles[None]).squeeze().le(0.99)
                # filter allowed self intersections
                test_segments = self.segments.batch_has_self_isec(verts)
                for segm_name, segm_ext in zip(self.segments.names, test_segments):
                    exterior[self.segments.segmentation[segm_name].segment_vidx[(segm_ext).detach().cpu().numpy() == 0]] = 1

                # find closest vertex in contact
                pred_verts_dists[:, ~self.geomask] = float('inf')
                pred_verts_incontact_argmin = torch.argmin(pred_verts_dists, axis=1)[0]

            # general contact term to pull verticse that are close together
            pred_verts_incontact_min = torch.norm(verts[bidx] - verts[bidx, pred_verts_incontact_argmin,:], dim=1)
            v2vinside = torch.tensor([0.0], device=self.device)
            v2voutside = torch.tensor([0.0], device=self.device)
            if (exterior).sum() > 0:
                v2voutside = 0.005 * torch.tanh(pred_verts_incontact_min[exterior] / 0.005)**2
            if (~exterior).sum() > 0:
                v2vinside = 1.0 * torch.tanh(pred_verts_incontact_min[~exterior] / 0.04)**2
            contact_loss[bidx] = v2vinside.mean() + v2voutside.mean()

            # region to region loss for discrete annotated contact
            mindists = 0
            if (gt_contact[bidx] ==1).sum() > 0:
                in_contact_idxs = torch.where(gt_contact[bidx] == 1)[0]
                for in_contact_idx in in_contact_idxs:
                    regpair = self.cdict['classes'][in_contact_idx]
                    verts1_idxs, verts2_idxs = self.cdict['csig'][regpair[0]], self.cdict['csig'][regpair[1]]
                    dists = pred_verts_dists[:, verts1_idxs, :][:, :, verts2_idxs]
                    mindists += torch.min(dists)
                r2r_loss[bidx] = mindists

        total_loss = 100 * (contact_loss + 0.5 * r2r_loss) 

        return total_loss.sum()
