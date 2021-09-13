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
from torchvision.transforms import Normalize
import numpy as np
import cv2
import os
import pickle
import argparse
import json
import os.path as osp
import glob
import trimesh

from tuch.models.hmr import hmr
from tuch.models.smpl import SMPL
from tuch.utils.imutils import crop
from tuch.utils.renderer import Renderer
from configs import config
from data.essentials import constants

os.environ['PYOPENGL_PLATFORM'] = 'egl'

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
parser.add_argument('--img', type=str, required=True, help='Path to input image')
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. \
                                                                If not set use input filename.')
parser.add_argument('--outdir', type=str, default='out', help='Output dir for results')
parser.add_argument('--spin_img_dir', type=str, default='data/images_spin_fit', help='read images from this dir when images are stacked with spin fit.')
parser.add_argument('--eft_img_dir', type=str, default='data/images_eft_fit', help='read images from this dir when images are stacked with eft fit.')
parser.add_argument('--stack', type=lambda x: x in ['true', 'True'], default=False, help='stack images with eft best model fit.')

def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    with open(openpose_file, 'r') as f:
        people = json.load(f)['people']
    len(people)
    keypoints = people[0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale

def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale

def process_image(img_file, bbox_file, openpose_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    if bbox_file is None and openpose_file is None:
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        if bbox_file is not None:
            center, scale = bbox_from_json(bbox_file)
        elif openpose_file is not None:
            center, scale = bbox_from_openpose(openpose_file)
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img

if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load pretrained model
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)

    # Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    # Setup renderer for visualization
    renderer = Renderer(focal_length=constants.FOCAL_LENGTH,
                        img_res=constants.IMG_RES,
                        faces=smpl.faces,
                        contactlist=[])

    # check if dir
    IMGS = sorted(glob.glob(osp.join(args.img, '*')) \
            if osp.isdir(args.img) else [args.img])
    OPS = sorted(glob.glob(osp.join(args.openpose, '*.json')) \
            if osp.isdir(args.openpose) else [args.openpose])

    for IMG, OP in zip(IMGS, OPS):
        print('processing ', IMG, OP)

        # Preprocess input image and generate predictions
        try:
            img, norm_img = process_image(IMG, args.bbox, OP, input_res=constants.IMG_RES)
        except:
            print(IMG, OP)
            continue
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
            pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], 
                global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

        # create mesh
        mesh = trimesh.Trimesh(pred_vertices.cpu().numpy()[0], smpl.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1,0,0])
        mesh.apply_transform(rot)

        mesh_r1 = trimesh.Trimesh(pred_vertices.cpu().numpy()[0], smpl.faces)
        mesh_r1.apply_transform(rot)
        rot_r1 = trimesh.transformations.rotation_matrix(
            np.radians(60), [0,1,0])
        mesh_r1.apply_transform(rot_r1)

        mesh_r2 = trimesh.Trimesh(pred_vertices.cpu().numpy()[0], smpl.faces)
        mesh_r2.apply_transform(rot)
        rot_r2 = trimesh.transformations.rotation_matrix(
            np.radians(300), [0,1,0])
        mesh_r2.apply_transform(rot_r2)


        # Calculate camera parameters for rendering
        camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2],
          2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
        camera_translation = camera_translation[0].cpu().numpy()

        pred_vertices = pred_vertices[0].cpu().numpy()
        img = img.permute(1,2,0).cpu().numpy()

        # Render parametric shape
        img_shape = renderer(pred_vertices, camera_translation, img)

        # Render side views
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
        center = pred_vertices.mean(axis=0)
        rot_vertices = np.dot((pred_vertices - center), aroundy) + center

        # Render non-parametric shape
        img_shape_side = renderer(rot_vertices, camera_translation, np.ones_like(img))

        if args.outdir is not None:
            os.makedirs(args.outdir, exist_ok=True)
            outfile = IMG.split('.')[0].split('/')[-1]
            outfile = osp.join(args.outdir, outfile)
        else:
            outfile = IMG.split('.')[0] if args.outfile is None else args.outfile
        # export mesh
        mesh.export(outfile+ '.obj')
        mesh_r1.export(outfile+ '_r60.obj')
        mesh_r2.export(outfile+ '_r300.obj')

        camera_translation_1 = camera_translation.copy()
        camera_translation_1[0] *= -1
        cam_out = {
           'spin_output': pred_camera.cpu().numpy(),
           'cam_transform': camera_translation,
           'cam_transform_1': camera_translation_1
        }
        with open(outfile + '_camera.pkl', 'wb') as f:
            pickle.dump(cam_out, f)
        IMGINCROP = 255 * img
        cv2.imwrite(outfile + '_img_in.png', IMGINCROP[:,:,::-1])

        IMGOUT = np.hstack((255 * img, 255 * img_shape, 255 * img_shape_side))
        cv2.imwrite(outfile + '.png', IMGOUT[:,:,::-1])

        print('Saving results to :', outfile)
        if args.stack:
            spin_fit = cv2.imread(osp.join(args.spin_img_dir, outfile.split('/')[-1] + '.png'))
            eft_fit = cv2.imread(osp.join(args.eft_img_dir, outfile.split('/')[-1] + '.png'))
            IMGOUT_SPINEFTFIT = np.hstack((IMGOUT[:,:,::-1], eft_fit, spin_fit))
            cv2.imwrite(outfile + '.png', IMGOUT_SPINEFTFIT)
        else:
            cv2.imwrite(outfile + '.png', IMGOUT[:,:,::-1])
