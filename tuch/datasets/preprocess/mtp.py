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
import os.path as osp
import json
import pickle
from data.essentials import constants
from configs import config
import torch
import joblib
import torchgeometry as tgm
os.environ["PYOPENGL_PLATFORM"] = "egl"


def get_openpose(jsonfn, imgshape):
    """
        Select the person idx closest to the image center.
    """
    # get only the arms/legs joints
    op_to_12 = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7]

    with open(jsonfn, 'r') as f:
        data = json.load(f)

    h, w = imgshape
    img_center = np.array([w,h])/2
    people = data['people']
    if len(people) == 0:
        keyp25 = np.zeros([25,3])
    else:
        kpselect = np.inf*np.ones(len(people))
        for i, person in enumerate(people):
            op_keyp25 = np.reshape(person['pose_keypoints_2d'], [25,3])
            op_keyp12 = op_keyp25[op_to_12, :2]
            op_conf12 = op_keyp25[op_to_12, 2]
            kpdist = np.linalg.norm(op_keyp12-img_center, axis=1)
            kpconf = np.dot(kpdist, (- op_conf12 + 1))
            kpselect[i] = kpconf
        p_sel = np.argmin(kpselect)
        keyp25 = np.reshape(people[p_sel]['pose_keypoints_2d'], [25,3])

    return keyp25, p_sel


def bbox_from_openpose(openpose_file, person_idx, rescale=1.2, detection_thresh=0.2):
    """
        Get center and scale for bounding box from openpose detections.
    """
    with open(openpose_file, 'r') as f:
        keypoints = json.load(f)['people'][person_idx]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale

    # check if all major keypoints are visible
    kptsvisible = True
    op_major_joints = ['OP RAnkle', 'OP LAnkle', 'OP RHip', \
    'OP LHip', 'OP RShoulder', 'OP LShoulder', 'OP RKnee', 'OP LKnee']
    op_joints_ind = [constants.JOINT_IDS[joint] for joint in op_major_joints]
    if sum(keypoints[op_joints_ind, 2] > 0) < len(op_major_joints):
        kptsvisible = False

    return center, scale, kptsvisible

def read_metadata(ds_dir):

    metadata_path = osp.join(ds_dir, 'subject_meta.json')

    with open(metadata_path, 'r') as f:
        data = json.load(f)

    return data

def fn(path):
    """Remove directory and extension from path, e.g. /aaaa/bbb/cc.ext --> cc """
    return osp.splitext(osp.basename(path))[0]

def read_images(ds_dir, mode):
    """
        Load the image names for train / val subset.
    """
    # load train val  split file 
    with open(osp.join(ds_dir, 'train_val_split.json')) as f:
        mode_ids = json.load(f)[mode]

    img_dir = osp.join(ds_dir, 'images')

    # read images and select the ones in [mode] set
    images = glob.glob(osp.join(img_dir, '**'), recursive=True)
    images = [x for x in images if fn(x) in mode_ids]
    print(f'Total num images in {mode} set {len(images)}')

    return images

def get_params_and_kp_paths(ds_dir, img_path):
    """
        Load paths.
    """

    openpose_dir = osp.join(ds_dir, 'keypoints/openpose')
    smplxparams_dir = osp.join(ds_dir, 'smplify-xmc', 'smplx')
    smplparams_dir = osp.join(ds_dir, 'smplify-xmc', 'smpl')
    img_dir = osp.join(ds_dir, 'images')

    img_fn, _ = osp.splitext(osp.split(img_path)[1])
    relpath = osp.dirname(img_path).replace(img_dir, '').strip('/')

    # find smplx fit and openpose file and model name
    smplx_pth = osp.join(smplxparams_dir, 'params', relpath, img_fn+'.pkl')
    smpl_pth = osp.join(smplparams_dir, 'params',relpath, img_fn+'.pkl')
    op_pth = osp.join(openpose_dir, relpath, img_fn+'.json')

    return smplx_pth, smpl_pth, op_pth

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def train_data(ds_dir, dbs_dir, mode='train'):
    """
        Read MTP data and save it as .pt file.
    """

    assert (mode in ['train', 'val']), 'mode must be train or val'

    dataset = dict(imgname = [],
                   scale = [],
                   center = [],
                   openpose = [],
                   gender = [],
                   smplx_pose = [],
                   smplx_left_hand_pose = [],
                   smplx_right_hand_pose = [],
                   smplx_global_orient = [],
                   smplx_betas=[],
                   smplx_camera_rot=[],
                   betas = [],
                   pose = [])

    img_dir = osp.join(ds_dir, 'images')
    
    metadict = read_metadata(ds_dir)
    images = read_images(ds_dir, mode)

    for idx, img_path in enumerate(images):
        print('------', idx, img_path)

        img_fn, _ = osp.splitext(osp.split(img_path)[1])
        smplx_path, smpl_path, op_path = get_params_and_kp_paths(ds_dir, img_path)

        # load mimicked image and openpose
        img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
        openpose, person_idx = get_openpose(op_path, imgshape=img.shape[:2])
        center, scale, _ = bbox_from_openpose(op_path, person_idx)

        # get gender (female 1, male 0)
        gender = 0 if metadict[img_fn]["SubjectGender"] == 'male' else 1

        # get ground truth smplx params
        smplx_params = pickle.load(open(smplx_path, 'rb'))
        global_orient = smplx_params['global_orient']
        body_pose = smplx_params['body_pose']
        left_hand_pose = smplx_params['left_hand_pose']
        right_hand_pose = smplx_params['right_hand_pose']
        betas = smplx_params['betas']

        # convert smplx to smpl parameters in SPIN format (pose: 72, betas 10)
        smpl_params =  pickle.load(open(smpl_path, 'rb'))
        smpl_betas = np.array(smpl_params['betas'])
        smpl_pose = np.array(smpl_params['pose'])

        # TUCH uses a unit camera rotation. So we apply the SMPLify-XMC
        # camera rotation to the global orientation.
        go_tensor = torch.from_numpy(smplx_params['global_orient'])
        RG = tgm.angle_axis_to_rotation_matrix(go_tensor).cpu().numpy()
        RC = smplx_params['camera_rotation'][0]
        R = np.matmul(RC, RG[0,:3,:3])
        R_hom = torch.from_numpy(np.hstack((R, np.array([[0],[0],[1]]))))
        R_aa = tgm.rotation_matrix_to_angle_axis(R_hom.unsqueeze(0)).float().numpy()
        smpl_pose[:3] = R_aa[0,:3]

        # save results
        dataset['imgname'].append(img_path.replace(img_dir, '').strip('/'))
        dataset['scale'].append(scale)
        dataset['center'].append(center)
        dataset['openpose'].append(openpose)
        dataset['gender'].append(gender)
        dataset['smplx_pose'].append(body_pose.squeeze())
        dataset['smplx_left_hand_pose'].append(left_hand_pose.squeeze())
        dataset['smplx_right_hand_pose'].append(right_hand_pose.squeeze())
        dataset['smplx_global_orient'].append(global_orient.squeeze())
        dataset['smplx_camera_rot'].append(RC.squeeze())
        dataset['smplx_betas'].append(betas.squeeze())
        dataset['betas'].append(smpl_betas.squeeze())
        dataset['pose'].append(smpl_pose.squeeze())

    # save data
    out_file = osp.join(dbs_dir, 'mtp_{}.pt'.format(mode))
    print(f'Saving {out_file} ..')
    joblib.dump(dataset, out_file)


def mtp_extract(mode='train'):

    train_data(
        ds_dir=config.MTP_ROOT,
        dbs_dir=config.DBS_PATH,
        mode=mode
    )
