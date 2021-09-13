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

from tuch.utils.geometry import perspective_projection
from tuch.utils.contact import batch_pairwise_dist
from configs import config
import sys
from tuch.utils.contact import winding_numbers

def gmof(x, sigma):
    """
    Geman-McClure error function
    https://github.com/nkolot/SPIN/blob/master/smplify/losses.py
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

def contact_fitting_loss(body_pose, global_orient, body_pose_loop1, opt_global_orient_smplifyloop1,
                      betas, model_joints, geomask, euclthres,
                      camera_t, camera_center,
                      joints_2d, joints_conf, pose_prior,
                      cdict, gt_contact,
                      ignore_idxs, 
                      has_discrete_contact,
                      verts, face_tensor=None,
                      #vertex_normals = None, vertricor=None,
                      device='cuda',
                      focal_length=5000, sigma=100, pose_prior_weight=1.0,
                      shape_prior_weight=1.0, angle_prior_weight=1.0,
                      contact_loss_weight=1000, output='sum',
                      segments=None):
    """
    Loss function for body fitting with contact.
    """

    gt_contact_l3 = gt_contact[0]
    batch_size = body_pose.shape[0]

    # Weighted robust reprojection error
    rotation = torch.eye(3, device=body_pose.device).unsqueeze(0) \
                                        .expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, 
        rotation, camera_t, focal_length, camera_center)
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)

    # Pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)

    # Contact loss pulling areas that are close together
    contact_loss = torch.zeros(batch_size).to(device)

    # Pull contact areas together if discrete label available
    r2r_loss = torch.zeros(batch_size).to(device)

    # optimize all poses with contact information
    opti_bidxs = torch.where(~ignore_idxs)[0]
    for bidx in opti_bidxs:
        # squared pairwise distnace between vertices
        pred_verts_dists = batch_pairwise_dist(verts[[bidx],:,:],
                                               verts[[bidx],:,:],
                                               squared=True)
        with torch.no_grad():
            # find intersecting vertices
            triangles = (verts[bidx])[face_tensor[0]]
            exterior = winding_numbers(verts[[bidx],:,:], triangles[None]).squeeze().le(0.99)

            # filter allowed self intersections
            if segments is not None and (~exterior).sum() > 0:
                test_segments = segments.batch_has_self_isec(verts[[bidx],:,:])
                for segm_name, segm_ext in zip(segments.names, test_segments):
                    exterior[segments.segmentation[segm_name] \
                    .segment_vidx[(segm_ext).detach().cpu().numpy() == 0]] = 1

            # find closest vertex in contact
            pred_verts_dists[:, ~geomask] = float('inf')
            pred_verts_incontact_argmin = torch.argmin(pred_verts_dists, axis=1)[0]

        # general contact term to pull vertices that are close together
        v2vinside = torch.tensor(0.0, device=verts.device)
        v2voutside = torch.tensor(0.0, device=verts.device)
        pred_verts_incontact_min = torch.norm(verts[bidx] - verts[bidx, pred_verts_incontact_argmin,:], dim=1)
        in_contact = pred_verts_incontact_min < euclthres
        if (~exterior).sum() > 0:
            v2vinside = 1.0 * torch.tanh(pred_verts_incontact_min[~exterior] / 0.04)**2
        ext_and_in_contact = exterior & in_contact
        if ext_and_in_contact.sum() > 0:
            v2voutside = 0.005 * torch.tanh(pred_verts_incontact_min[ext_and_in_contact] / 0.005)**2
        contact_loss[bidx] = v2vinside.sum() + v2voutside.sum()

        # region to region loss for discrete annotated contact
        mindists = 0
        if has_discrete_contact[bidx]:
            gt_contact_pose = gt_contact_l3[bidx]
            in_contact_idxs = torch.where(gt_contact_pose == 1)[0]
            for in_contact_idx in in_contact_idxs:
                regpair = cdict['classes'][in_contact_idx]
                verts1_idxs, verts2_idxs = cdict['csig'][regpair[0]], cdict['csig'][regpair[1]]
                dists = pred_verts_dists[:, verts1_idxs, :][:, :, verts2_idxs]
                mindists += torch.min(dists)
            r2r_loss[bidx] = mindists


    total_loss = reprojection_loss.sum(dim=-1) + 10 * contact_loss \
        + pose_prior_loss + contact_loss_weight * r2r_loss

    return total_loss.sum()

def camera_fitting_loss(smpl_output, camera_t, camera_t_est, camera_center, joints_2d, joints_conf,
                        focal_length=5000, depth_loss_weight=100, sigma=100, shape_prior_weight=0.0):
    """
    Loss function for camera and betas optimization.
    """

    # Project model joints
    model_joints = smpl_output.joints
    betas = smpl_output.betas
    batch_size = model_joints.shape[0]

    rotation = torch.eye(3, device=model_joints.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)

    # Weighted robust reprojection error
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight ** 2) * (camera_t[:, 2] - camera_t_est[:, 2]) ** 2

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = reprojection_loss.sum(dim=-1) + depth_loss + shape_prior_loss

    return total_loss.sum()


def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    Code from SPIN - kept for comparison of SMPLify and SMPLify-DC
    https://github.com/nkolot/SPIN/blob/master/smplify/losses.py
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return torch.exp(pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2

def body_fitting_loss(body_pose, betas, model_joints, camera_t, camera_center,
                      joints_2d, joints_conf, pose_prior,
                      focal_length=5000, sigma=100, pose_prior_weight=4.78,
                      shape_prior_weight=5, angle_prior_weight=15.2,
                      output='sum'):
    """
    Loss function for body fitting. 
    Code from SPIN - kept for comparison of SMPLify and SMPLify-DC
    https://github.com/nkolot/SPIN/blob/master/smplify/losses.py
    """

    batch_size = body_pose.shape[0]
    rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)

    # Weighted robust reprojection error
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)

    # Pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)

    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = reprojection_loss.sum(dim=-1) + pose_prior_loss + angle_prior_loss + shape_prior_loss

    if output == 'sum':
        return total_loss.sum()
    elif output == 'reprojection':
        return reprojection_loss