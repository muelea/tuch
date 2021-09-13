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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torchgeometry as tgm

def compute_quaternions_norm(quat1, quat2):
    quatnorms = torch.cat((torch.norm(quat1-quat2, dim=1).unsqueeze(0),
            torch.norm(quat1+quat2, dim=1).unsqueeze(0)))
    quatnorm = torch.min(quatnorms, 0)[0]
    return quatnorm

def compute_quaternions_innerproduct(quat1, quat2):
    return 1 - abs(quat1*quat2)

def compute_MPJAE_rot(rotmat1, rotmat2):
    Rs = torch.matmul(rotmat1, rotmat2.transpose(1,2))
    phis = torch.zeros(Rs.shape[0])
    for idx, R in enumerate(Rs):
        phi = torch.acos((torch.trace(R) - 1)*0.5)
        phis[idx] = phi
    return phis

def compute_MPJAE_quat(quat1, quat2):
    phis = torch.zeros(quat1.shape[0])
    for idx, (q1, q2) in enumerate(zip(quat1, quat2)):
        phi = 2*torch.acos(torch.dot(q1, q2)) #torch.acos(2*torch.dot(q1, q2)**2 - 1)
        phis[idx] = phi
    return phis

def joint_angle_error(P1, P2, reduction='mean', measure='quat_norm'):
    """Compute joint angle error for pose (N x num_joints x 3). Pose in
    axis angle representation. Following approaches discussed in
    Metrics for 3D Rotations: Comparison and Analysis
    Du Q. Huynh"""

    # Norm of the Difference of Quaternions
    re = torch.zeros(P1.shape[0], device=P1.device)
    for idx, (pose1, pose2) in enumerate(zip(P1, P2)):
        quat1 = tgm.angle_axis_to_quaternion(pose1)
        quat2 = tgm.angle_axis_to_quaternion(pose2)
        rotmat1 = tgm.angle_axis_to_rotation_matrix(pose1.squeeze())
        rotmat2 = tgm.angle_axis_to_rotation_matrix(pose2.squeeze())
        if measure == 'quat_norm':
            norm = compute_quaternions_norm(quat1, quat2)
        elif measure == 'degree':
            # http://www.boris-belousov.net/2016/12/01/quat-dist/
            normdegrot = compute_MPJAE_rot(rotmat1[:, :3, :3], rotmat2[:, :3, :3])
            normdegquat = compute_MPJAE_quat(quat1, quat2)
            norm = normdegrot

        re[idx] = norm.sum()

    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    else:
        re = re

    return re
