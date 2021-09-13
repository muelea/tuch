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

import os
import cv2
import glob
import json
import numpy as np
import scipy.io as sio
import os.path as osp
import json
from tqdm import tqdm
import pickle
import joblib
import scipy.io as sio
from data.essentials import constants
from configs import config


def get_contact_gt(data, classes):
    """
        Read contact labels and convert it into a binary vector.
    """
    contactvec = np.zeros(len(classes))
    contactvecflip = np.zeros(len(classes))

    for annot in data:
        bp1, bp2 = annot.split('_')
        pair = sorted([bp1, bp2])
        vecidx = np.where(np.all(classes == pair, axis=1))[0]
        if len(vecidx) > 0:
            contactvec[vecidx] = 1

        # contact for the flipped original images
        bp1f = mirror_label(bp1)
        bp2f = mirror_label(bp2)
        pairf = sorted([bp1f, bp2f])
        vecidxf = np.where(np.all(classes == pairf, axis=1))[0]
        if len(vecidxf) > 0:
            contactvecflip[vecidxf] = 1

    return contactvec, contactvecflip

def mirror_label(bp):
    """
        Mirror the label of the self-contact annotations.
    """
    if 'left' in bp:
        bpf = bp.replace('left', 'right')
    elif 'right' in bp:
        bpf = bp.replace('right', 'left')
    else:
         bpf = bp

    return bpf


def bbox_from_openpose(keypoints, rescale=1.2, detection_thresh=0.2):
    """
        Get center and scale for bounding box from openpose detections.
    """

    # check if upper body - four major joints are visible
    op_major_joints = ['OP RAnkle', 'OP LAnkle', 'OP RHip', \
          'OP LHip', 'OP RShoulder', 'OP LShoulder', 'OP RKnee', 'OP LKnee']
    op_joints_ind = [constants.JOINT_IDS[joint] for joint in op_major_joints]
    if sum(keypoints[op_joints_ind, 2] > detection_thresh) < len(op_major_joints):
        center, scale, kpstatus = None, None, False
    else:
        # compute bounding box
        kpstatus = True
        valid = keypoints[:,-1] > detection_thresh
        valid_keypoints = keypoints[valid][:,:-1]
        center = valid_keypoints.mean(axis=0)
        bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
        # adjust bounding box tightness
        scale = bbox_size / 200.0
        scale *= rescale
    return center, scale, kpstatus

def match_op_gt_keypoints(openpose, gt_part, dataset):
    """
        Check if OpenPose and ground truth keypoints are close to each other.
    """
    is_match = False

    # get only the arms/legs joints
    op_to_12 = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7]
    gt_part_vis = gt_part[:,-1] == 1
    op_keyp12 = openpose[op_to_12, :2]
    op_conf12 = openpose[op_to_12, 2:3]
    diff = (op_keyp12 - gt_part[:12, :2])[gt_part_vis[:12]]
    # openpose confidence should be greater 0
    if op_conf12[gt_part_vis[:12]].max() > 0.0:
        dist_conf = np.mean(np.sqrt(np.sum(op_conf12[gt_part_vis[:12]]*diff**2, axis=1)))

        # the exact threshold is not super important but these are the values we used
        if dataset == 'mpii':
            thresh = 30
        elif dataset == 'coco':
            thresh = 10
        elif dataset == 'lsp':
            thresh = 10

        # dataset-specific thresholding based on pixel size of person
        if dist_conf <= thresh:
            is_match = True

    return is_match


def read_df_subset(dsc_dir, df_dir, out_path):
    """
        Preprocess Deep Fashion dataset with self-contact annotations.
    """

    dataset = dict(imgname = [],
                   scale = [],
                   center = [],
                   openpose = [],
                   smplx_pose = [],
                   smplx_betas = [],
                   smplx_global_orient = [],
                   smplx_left_hand_pose = [],
                   smplx_right_hand_pose = [],
                   betas = [],
                   pose = [],
                   contact_vec_mirror_pc = [],
                   contact_vec_pc = [],
                   has_contact_pc = [])

    classes_path = osp.join(dsc_dir, 'tuch_bodypart_pairs.pkl')
    classes = pickle.load(open(classes_path, 'rb'))

    annotation_path = osp.join(dsc_dir, 'df')
    annopaths = glob.glob(osp.join(annotation_path, '*.json'))

    for annopth in tqdm(annopaths):
        with open(annopth, 'r') as f:
            dsc_annotation = json.load(f)

        # dsc annotation
        cvec, cvec_mirror = get_contact_gt(dsc_annotation['contact_annot'], classes)

        # read keypoints for contact annotation
        imgpath = osp.join(df_dir, 'images', dsc_annotation['img'].split('/')[-1])

        # openpose keypoints
        openpose = np.array(dsc_annotation['openpose'])
        if len(openpose) > 0:
            openpose = np.reshape(openpose, [25,3])

            # scale and center and bounding box from openpose
            center, scale, bb_kpstatus = \
                bbox_from_openpose(openpose, detection_thresh=0.2)

            if bb_kpstatus:
                dataset['imgname'].append(imgpath)
                dataset['scale'].append(scale)
                dataset['center'].append(center)
                dataset['openpose'].append(openpose)
                dataset['contact_vec_pc'].append(cvec)
                dataset['contact_vec_mirror_pc'].append(cvec_mirror)

    num_imgs = len(dataset['imgname'])
    print(f'{num_imgs} read from DeepFashion dataset.')
    out_file = osp.join(out_path, 'dsc_df_train.pt')
    joblib.dump(dataset, out_file)


def read_lsp_subset(dsc_dir, img_dir, subset, out_path, scaleFactor=1.2):
    """
        Preprocess LSP / LSPET dataset with self-contact annotations.
    """    
    dataset = dict(imgname = [],
                   scale = [],
                   center = [],
                   part = [],
                   openpose = [],
                   contact_vec_mirror_pc = [],
                   contact_vec_pc = [],
                   has_contact_pc = [])

    # read dsc pairwise classes used in TUCh training
    classes_path = osp.join(dsc_dir, 'tuch_bodypart_pairs.pkl')
    classes = pickle.load(open(classes_path, 'rb'))

    # load ground truth joint matrix
    gt_joints_mat_path = os.path.join(img_dir, 'joints.mat')
    gt_joints_mat = sio.loadmat(gt_joints_mat_path)['joints']

    # read discrete contact annotations
    annotation_path = osp.join(dsc_dir, subset)
    annopaths = glob.glob(osp.join(annotation_path, '*.json'))

    for annopth in tqdm(annopaths):

        has_contact = True 

        with open(annopth, 'r') as f:
            dsc_annotation = json.load(f)
        imgname = dsc_annotation['img'].split('/')[-1]
        imgpath = osp.join(img_dir, 'images', imgname)
        imgname = osp.basename(imgpath)
        IMG = cv2.imread(imgpath)

        # read contact annotation if exists
        cvec, cvec_mirror = get_contact_gt(dsc_annotation['contact_annot'], classes)

        # read ground truth keypoints
        img_jm_idx = int((imgname.split('.')[0]).replace('im', '')) - 1
        if subset == 'lspet':
            gt_keypoints = gt_joints_mat[:, :2, img_jm_idx]
            # non visible keypoints are not necessarily annotated. Just
            # visibility parameter as joints confidence
            gt_keypoints_vis = gt_joints_mat[:, 2, img_jm_idx]
        elif subset == 'lsp':
            gt_keypoints = gt_joints_mat[:2, :, img_jm_idx].T
            # in lsp orig non visible keypoints are also annotated,
            # set visibility to 1
            gt_keypoints_vis = np.ones(14)

        # scale and center and bounding box from gt
        gt_keypoints_visbb = gt_keypoints[gt_keypoints_vis == 1, :]
        bbox = [min(gt_keypoints_visbb[:,0]), min(gt_keypoints_visbb[:,1]),
                max(gt_keypoints_visbb[:,0]), max(gt_keypoints_visbb[:,1])]
        center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
        scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200

        # concat ground-truth and openpose keypoints
        part = np.zeros([24,3])
        part[:14] = np.hstack([gt_keypoints, gt_keypoints_vis.reshape(-1,1)])

        # read openpose keypoints for gt_keypoints
        # openpose keypoints
        openpose = np.array(dsc_annotation['openpose'])
        if len(openpose) > 0:
            openpose = np.reshape(openpose, [25,3])

            # check if keypoint annotation and contact annotation were done for the same person
            # if not use only keypoints, since openpose keypoints with contact is not reliable
            # for lsp data, because of many challenging poses
            is_match = match_op_gt_keypoints(openpose, part, 'lsp')
            if not is_match:
                has_contact = False
                cvec[:], cvec_mirror[:] = 0, 0
                openpose = np.zeros([25,3])

            dataset['imgname'].append(imgpath)
            dataset['scale'].append(scale)
            dataset['center'].append(center)
            dataset['openpose'].append(openpose)
            dataset['part'].append(part)
            dataset['contact_vec_pc'].append(cvec)
            dataset['contact_vec_mirror_pc'].append(cvec_mirror)
            dataset['has_contact_pc'].append(has_contact)

    num_imgs = len(dataset['imgname'])
    print(f'{num_imgs} read from LSP ({subset}) dataset.')

    matches = np.array(dataset['has_contact_pc']).sum()
    print(f'num imgs with contact+keypoints: {matches}')

    out_file = osp.join(out_path, 'dsc_{}_train.pt'.format(subset))
    joblib.dump(dataset, out_file)


def train_data(dsc_dir, subset, out_path):
    if subset == 'df':
        df_dir = config.DF_ROOT
        read_df_subset(dsc_dir, df_dir, out_path)
    elif subset == 'lspet':
        img_dir = config.LSPET_ROOT
        read_lsp_subset(dsc_dir, img_dir, subset, out_path)
    elif subset == 'lsp':
        img_dir = config.LSP_ROOT
        read_lsp_subset(dsc_dir, img_dir, subset, out_path)
    else:
        print(subset, 'not valid subset name')

def dsc_extract(subset):

    dsc_dir = config.DSC_ROOT
    out_path = config.DBS_PATH

    train_data(dsc_dir, subset, out_path)
