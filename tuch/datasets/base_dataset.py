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

import torch
import os.path as osp
import time
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
import pickle
import joblib
from os.path import join

from configs import config
from data.essentials import constants
from tuch.utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa

class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(
             self,
             options,
             dataset,
             use_augmentation=True,
             set='train',
        ):
        super(BaseDataset, self).__init__()

        self.dataset_name = dataset
        self.is_train = True if set=='train' else False
        self.set = set
        self.options = options
        self.img_dir = config.IMAGE_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN,
                                        std=constants.IMG_NORM_STD)

        # load data for either train or validation set
        self.data = joblib.load(config.DATASET_FILES[set][dataset])
        self.length = len(self.data['imgname'])
        self.len = len(self.data['imgname'])

        # If False, do not do augmentation
        self.use_augmentation = use_augmentation

        # define data shape, when data not part of dataset and must be created
        self.classes = pickle.load(open(osp.join(config.DSC_ROOT, 'classes.pkl'), 'rb'))
        self.csig = pickle.load(open(osp.join(config.DSC_ROOT, 'ContactSigSMPL.pkl'), 'rb'))
        self.num_classes = len(self.classes)
        self.num_gt_kpts = 24
        self.num_op_kpts = 25

        # list features of each dataset
        if self.dataset_name in ['dsc_df']:
            self.has_disc_contact = np.ones(self.len).astype(np.bool)
            self.has_smpl = np.zeros(self.len)
            self.has_pgt_smpl = np.zeros(self.len).astype(np.bool)
            self.has_pose_3d = np.zeros(self.len)
            self.has_gt_kpts = np.zeros(self.len)

        if self.dataset_name in ['dsc_lspet', 'dsc_lsp']:
            self.has_disc_contact = np.ones(self.len).astype(np.bool)
            self.has_smpl = np.zeros(self.len)
            self.has_pgt_smpl = np.zeros(self.len).astype(np.bool)
            self.has_pose_3d = np.zeros(self.len)
            self.has_gt_kpts = np.ones(self.len)

        if self.dataset_name in ['dsc_df_eft']:
            self.has_disc_contact = np.zeros(self.len).astype(np.bool)
            self.has_smpl = np.zeros(self.len).astype(np.bool)
            self.has_pgt_smpl = np.ones(self.len).astype(np.bool)
            self.has_pose_3d = np.zeros(self.len)
            self.has_gt_kpts = np.zeros(self.len)

        if self.dataset_name in ['dsc_lspet_eft', 'dsc_lsp_eft']:
            self.has_disc_contact = np.zeros(self.len).astype(np.bool)
            self.has_smpl = np.zeros(self.len).astype(np.bool)
            self.has_pgt_smpl = np.ones(self.len).astype(np.bool)
            self.has_pose_3d = np.zeros(self.len)
            self.has_gt_kpts = np.ones(self.len)

        if self.dataset_name in ['mtp']:
            # use has smpl for each image to ignore bad pseudo ground-truth fits
            self.has_disc_contact = np.zeros(self.len).astype(np.bool)
            self.has_smpl =  np.zeros(self.len).astype(np.bool)
            self.has_pgt_smpl = np.ones(self.len)
            self.has_pose_3d = np.zeros(self.len)
            self.has_gt_kpts = np.zeros(self.len)
 
        if self.dataset_name in ['mtp_scans_gt']:
            self.has_disc_contact = np.zeros(self.len).astype(np.bool)
            self.has_smpl = np.ones(self.len)
            self.has_pgt_smpl = np.zeros(self.len)
            self.has_pose_3d = np.zeros(self.len)
            self.has_gt_kpts = np.zeros(self.len)

        if self.dataset_name in ['mpi-inf-3dhp'] and self.is_train:
            self.has_disc_contact = np.zeros(self.len).astype(np.bool)
            self.has_smpl =  self.data['has_smpl']
            self.has_pgt_smpl = np.zeros(self.len)
            self.has_pose_3d = np.ones(self.len)
            self.has_gt_kpts = np.ones(self.len)

        # test sets properties
        if self.dataset_name in ['3dpw'] and not self.is_train:
            self.has_disc_contact = np.zeros(self.len).astype(np.bool)
            self.has_smpl = np.ones(self.len)
            self.has_pgt_smpl = np.zeros(self.len)
            self.has_pose_3d = np.zeros(self.len)
            self.has_gt_kpts = np.zeros(self.len)

        if self.dataset_name in ['mpi-inf-3dhp'] and not self.is_train:
            self.has_disc_contact = np.zeros(self.len).astype(np.bool)
            self.has_smpl =  np.zeros(self.len)
            self.has_pgt_smpl = np.zeros(self.len)
            self.has_pose_3d = np.ones(self.len)
            self.has_gt_kpts = np.ones(self.len)

        if self.options is not None:
            if options.ignore_3d:
                self.has_smpl = np.zeros(self.len)

        # Concatenate ground truth and openpose 2D keypoints
        if 'part' in self.data.keys():
            keypoints_gt = np.array(self.data['part'], dtype=np.float32)
        else:
            keypoints_gt = np.zeros((self.len, self.num_gt_kpts, 3))

        if 'openpose' in self.data.keys():
            keypoints_openpose = np.array(self.data['openpose'], dtype=np.float32)
        else:
            keypoints_openpose = np.zeros((self.len, self.num_op_kpts, 3))

        # concat 25 openpose and 24 gt keypoints
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.get_single_item(index)

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train and self.use_augmentation:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1

            # Each channel is multiplied with a number
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.options.noise_factor, 1+self.options.noise_factor, 3)

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.options.rot_factor,
                    max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.scale_factor,
                    max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0

        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale,
                      [constants.IMG_RES, constants.IMG_RES], rot=rot)
        # flip the image
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale,
                                  [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/constants.IMG_RES - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        elif S.shape[1] == 3:
            S = np.einsum('ij,kj->ki', rot_mat, S)
        else:
            S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1])
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        pose = pose.astype('float32')
        return pose

    def get_single_item(self, index):

        # Bounding boxes params
        scale = np.array(self.data['scale'][index]).copy()
        center = np.array(self.data['center'][index]).copy()

        # Load image and resize directly before cropping, because of speed
        img_path = join(self.img_dir, self.data['imgname'][index])
        orig_img = cv2.imread(img_path)[:,:,::-1].copy().astype(np.float32)
        scale_img = max(448 / orig_img.shape[0], (448 / orig_img.shape[1]))
        new_size_x = int(orig_img.shape[0] * scale_img)
        new_size_y = int(orig_img.shape[1] * scale_img)
        orig_img = cv2.resize(orig_img, (new_size_y, new_size_x))
        orig_shape = np.array(orig_img.shape)[:2].astype(np.float32)
        center *= scale_img
        scale *= scale_img

        # Get augmentation parameters and process image
        flip, pn, rot, sc = self.augm_params()
        s = time.time()
        img = self.rgb_processing(orig_img, center, sc*scale, rot, flip, pn)
        img = torch.from_numpy(img).float()

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()
        keypoints[:,:2] *= scale_img
        keypoints = torch.from_numpy(self.j2d_processing(keypoints,
                               center, sc*scale, rot, flip)).float()

        # Get SMPL parameters, if available
        if self.has_smpl[index] or self.has_pgt_smpl[index]:
            pose = np.array(self.data['pose'][index]).astype(np.float32)
            betas = np.array(self.data['betas'][index]).astype(np.float32)
            if 'gender' in self.data.keys():
                gender = str(self.data['gender'][index])
                gender = 0 if str(gender) == 'm' else 1
            else:
                gender = -1
        else:
            pose = np.zeros(72).astype(np.float32)
            betas = np.zeros(10).astype(np.float32)
            gender = -1

        # get contact vector if object is a person-contact instance
        contactlist = []
        if self.has_disc_contact[index]:
            contact_vec = self.data['contact_vec_mirror_pc'][index] \
                                if flip else self.data['contact_vec_pc'][index]
            contact_vec = np.array(contact_vec).astype(np.float32)
        else:
            contact_vec = np.zeros(self.num_classes).astype(np.float32)

        # Get 3D pose, if available
        if self.has_pose_3d[index]:
            S = self.data['S'][index].copy()
            pose_3d = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
        else:
            pose_3d = torch.zeros(24,4, dtype=torch.float32)

        if self.set != 'test':
            target = {
              'imgname': img_path,
              'img': self.normalize_img(img),
              'keypoints': keypoints,
              'pose': torch.from_numpy(self.pose_processing(pose, rot, flip)).float(),
              'betas': torch.from_numpy(betas).float(),
              'contact_vec': torch.from_numpy(contact_vec).float(),
              'pose_3d': pose_3d,
              'has_smpl': self.has_smpl[index],
              'has_pgt_smpl': self.has_pgt_smpl[index],
              'has_disc_contact': self.has_disc_contact[index],
              'has_gt_kpts': self.has_gt_kpts[index],
              'has_pose_3d': self.has_pose_3d[index],
              'scale': float(sc * scale),
              'center': center.astype(np.float32),
              'is_flipped': flip,
              'rot_angle': np.float32(rot),
              'gender': gender,
              'sample_index': index,
              'dataset_name': self.dataset_name,
              'orig_shape': orig_shape
            }


        # add ground truth smplx params for testing
        else:
            target = {
                'imgname': img_path,
                'orig_shape': orig_shape.astype(np.int),
                'gender': gender,
                'img': self.normalize_img(img),
                'pose': torch.from_numpy(self.pose_processing(pose, rot, flip)).float(),
                'betas': torch.from_numpy(betas).float(),
                'scale': float(sc * scale),
                'center': center.astype(np.float32),
                'pose_3d': pose_3d,
            }
            if 'smplx_pose' in self.data.keys():
                target['smplx_pose'] = torch.from_numpy(self.data['smplx_pose'][index]).float()
                target['smplx_betas'] = torch.from_numpy(self.data['smplx_shape'][index]).float()
                target['smplx_left_hand_pose'] = torch.from_numpy(self.data['smplx_left_hand_pose'][index]).float()
                target['smplx_right_hand_pose'] = torch.from_numpy(self.data['smplx_right_hand_pose'][index]).float()
                target['smplx_global_orient'] = torch.from_numpy(self.data['smplx_global_orient'][index]).float()
            if 'maskname' in self.data.keys():
                target['maskname'] = self.data['maskname'][index]
            if 'partname' in self.data.keys():
                target['partname'] = self.data['partname'][index]

        return target
