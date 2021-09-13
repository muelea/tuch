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

from torch.utils.data import Dataset
from tuch.utils.imutils import crop, transform
import torch
import joblib
import cv2
from data.essentials import constants
from configs import config
from torchvision.transforms import Normalize
import numpy as np
import os.path as osp

class EFTDataset(Dataset):
    def __init__(self, dsname):
        self.dsname = dsname
        self.data = joblib.load(config.DATASET_FILES['train'][dsname])

        self.length = len(self.data['imgname'])
        self.imgs = self.data['imgname']
        self.scale = self.data['scale']
        self.center = self.data['center']
        self.img_dir = config.IMAGE_FOLDERS[dsname]
        if self.img_dir not in self.imgs[0]:
            self.imgs = [osp.join(self.img_dir, x) for x in self.imgs]
        self.num_gt_kpts = 24
        self.num_op_kpts = 25
        if 'contact_vec' in self.data.keys():
            self.dsc = self.data['contact_vec']
        elif 'contact_vec_pc' in self.data.keys():
            self.dsc = self.data['contact_vec_pc']

        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN,
                                        std=constants.IMG_NORM_STD)

        # Concatenate ground truth and openpose 2D keypoints
        if 'part' in self.data.keys():
            keypoints_gt = np.array(self.data['part'], dtype=np.float32)
        else:
            keypoints_gt = np.zeros((self.length, self.num_gt_kpts, 3))

        if 'openpose' in self.data.keys():
            keypoints_openpose = np.array(self.data['openpose'], dtype=np.float32)
        else:
            keypoints_openpose = np.zeros((self.length, self.num_op_kpts, 3))

        # concat 25 openpose and 24 gt keypoints
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

    def __len__(self):
        return self.length

    def j2d_processing(self, kp, center, scale, r):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale,
                                  [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/constants.IMG_RES - 1.
        kp = kp.astype('float32')
        return kp

    def __getitem__(self, idx):
        scale = np.array(self.data['scale'][idx]).copy()
        center = np.array(self.data['center'][idx]).copy()

        # process images
        raw_img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        img = crop(raw_img, center, scale,
                      [constants.IMG_RES, constants.IMG_RES], rot=0)
        img = np.transpose(img.astype('float32'),(2,0,1))/255.0
        img = torch.from_numpy(img).float()

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[idx].copy()
        keypoints = torch.from_numpy(self.j2d_processing(keypoints,
                               center, scale, 0)).float()

        # get contact
        contact = self.dsc[idx]

        target = {
             'img': self.normalize_img(img),
             'keypoints': keypoints,
             'contact': torch.from_numpy(contact).float(),
        }

        return target

