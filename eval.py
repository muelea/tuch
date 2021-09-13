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
#
# The code is adapted from SPIN, with modifications to visualize contact 
# https://github.com/nkolot/SPIN/blob/master/eval.py


"""
To get the results on the 3dpw test set use:
```
python eval.py --checkpoint=data/tuch_model_checkpoint.pt --dataset=3dpw
```
For mpi-inf-3dhp use 
```
python eval.py --checkpoint=data/tuch_model_checkpoint.pt --dataset=mpi-inf-3dhp
```
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
import torchgeometry as tgm

from configs import config
from data.essentials import constants
from tuch.models.hmr import hmr
from tuch.models.smpl import SMPL
from tuch.datasets.base_dataset import BaseDataset
from tuch.utils.pose_utils import reconstruction_error

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--dataset', default='mpi-inf-3dhp', choices=['3dpw', 'mpi-inf-3dhp'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')
parser.add_argument('--idx', default=None, help='index when evaluation multiple checkpoints on cluster')


def print_interm_result(mpjpe, recon_err, step, batch_size, cnc_arr=None, euclthres_lower=0.01):
    print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
    print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
    print()
    if cnc_arr is not None:
        contact, no_contact = (cnc_arr[:step * batch_size] < euclthres_lower), (cnc_arr[:step * batch_size] == np.inf)
        unclear = ~(no_contact + contact)
        print(contact.sum() + no_contact.sum() + unclear.sum() == contact.shape)
        print('Eval on subsets: contact {}, no-contact {}, unclear {}'.format(contact.sum(), no_contact.sum(), unclear.sum()))
        print('MPJPE contact: ' + str(1000 * mpjpe[:step * batch_size][contact].mean()))
        print('MPJPE no contact: ' + str(1000 * mpjpe[:step * batch_size][no_contact].mean()))
        print('MPJPE unclear: ' + str(1000 * mpjpe[:step * batch_size][unclear].mean()))
        print('Reconstruction Error contact: ' + str(1000 * recon_err[:step * batch_size][contact].mean()))
        print('Reconstruction Error no contact: ' + str(1000 * recon_err[:step * batch_size][no_contact].mean()))
        print('Reconstruction Error unclear: ' + str(1000 * recon_err[:step * batch_size][unclear].mean()))

def print_final_result(mpjpe, recon_err, cnc_arr=None, euclthres_lower=0.01):
    print('MPJPE: ' + str(1000 * mpjpe.mean()))
    print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
    if cnc_arr is not None:
        contact, no_contact = (cnc_arr < euclthres_lower), (cnc_arr == np.inf)
        unclear = ~(no_contact + contact)
        print('Eval on subsets: contact {}, no-contact {}, unclear {}'.format(contact.sum(), no_contact.sum(), unclear.sum()))
        print('MPJPE contact: ' + str(1000 * mpjpe[contact].mean()))
        print('MPJPE no-contact: ' + str(1000 * mpjpe[no_contact].mean()))
        print('MPJPE unclear: ' + str(1000 * mpjpe[unclear].mean()))
        print('Reconstruction Error contact: ' + str(1000 * recon_err[contact].mean()))
        print('Reconstruction Error no contact: ' + str(1000 * recon_err[no_contact].mean()))
        print('Reconstruction Error unclear: ' + str(1000 * recon_err[unclear].mean()))
        print()

def run_evaluation(model, dataset_name, dataset, result_file,
                   batch_size=32, img_res=224,
                   num_workers=32, shuffle=False, log_freq=50):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer model to the GPU
    model.to(device)

    # Load SMPL model
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR,
                       gender='female',
                       create_transl=False).to(device)

    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()

    save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle=False
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)

    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    cnc_arr = None

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    # Choose appropriate evaluation for each dataset
    if dataset_name == '3dpw':
        csigs_arr = np.load(config.THREEDPW_CIG)
        cnc_arr = csigs_arr.min(1).min(1)

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14
    # Iterate over the entire dataset
    #if False:
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:],
                                    global_orient=gt_pose[:, :3]).vertices
        images = batch['img'].to(device)
        gender = batch['gender'].to(device)
        curr_batch_size = images.shape[0]

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(images)
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

        if save_results:
            rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
            rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
            pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
            smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
            smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_betas.cpu().numpy()
            smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_camera.cpu().numpy()

        # 3D pose evaluation
        J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
        # Get 14 ground truth joints
        if dataset_name == 'mpi-inf-3dhp':
            gt_keypoints_3d = batch['pose_3d'].cuda()
            gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
        # For 3DPW get the 14 common joints from the rendered shape
        else:
            gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices
            gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices
            gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
            gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
            gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
            gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

        # Get 14 predicted joints from the mesh
        pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
        if save_results:
            pred_joints[step * batch_size:step * batch_size + curr_batch_size, :, :]  = pred_keypoints_3d.cpu().numpy()
        pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
        pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

        # Absolute error (MPJPE)
        error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

        # Reconstuction_error
        r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
        recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

        # Print intermediate results during evaluation
        if step % log_freq == log_freq - 1:
            print_interm_result(mpjpe, recon_err, step, batch_size, cnc_arr)

    # Print final results during evaluation
    print('*** Final Results *** \n')
    print_final_result(mpjpe, recon_err, cnc_arr)

    # Save reconstructions to a file for further processing
    if save_results:
        os.makedirs('out', exist_ok=True)
        np.savez('out/{}'.format(result_file), 
            pred_joints=pred_joints, 
            pose=smpl_pose, 
            betas=smpl_betas, 
            camera=smpl_camera,
            mpjpe=mpjpe,
            recon_err=recon_err,
        )

if __name__ == '__main__':
    args = parser.parse_args()

    model = hmr(config.SMPL_MEAN_PARAMS)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Setup evaluation dataset
    dataset = BaseDataset(None, args.dataset, set='test')

    # Run evaluation
    if args.result_file is None:
        result_file = '_'.join(args.checkpoint.split('/')[-4:])
    else:
        result_file = args.result_file

    run_evaluation(model, args.dataset, dataset, result_file,
                   batch_size=args.batch_size,
                   shuffle=args.shuffle,
                   log_freq=args.log_freq)
