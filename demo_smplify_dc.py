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
import os.path as osp
import cv2
# hack to get the correct gpu device id on cluster
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import torch
import pickle
from tqdm import tqdm
from torchgeometry import rotation_matrix_to_angle_axis

from configs.smplify_dc_options import SMPLifyDCOptions
from configs import config
from tuch.models.hmr import hmr
from tuch.smplify.smplifydc import SMPLifyDC
from data.essentials import constants
from tuch.datasets.base_dataset import BaseDataset
from tuch.models.smpl import SMPL
from tuch.utils.renderer import Renderer
from data.essentials.segments.smpl import segm_utils as exn
from tuch.utils.segmentation import BatchBodySegment

def main(options):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = options.batch_size

    dataset = BaseDataset(options, options.ds_names[0], use_augmentation=False)

    # Load SPIN to initialize the optimization
    modelspin = hmr(config.SMPL_MEAN_PARAMS).to(device)
    spincheckpoint = torch.load(config.SPIN_MODEL_CHECKPOINT)
    modelspin.load_state_dict(spincheckpoint['model'], strict=False)
    modelspin.eval()

    # load body model
    smpl = SMPL(config.SMPL_MODEL_DIR,
              batch_size=options.batch_size,
              create_transl=False
    ).to(device)
    face_tensor = torch.tensor(smpl.faces.astype(np.int64),
                     dtype=torch.long, device=device) \
                     .unsqueeze_(0) \
                     .expand(options.batch_size,-1,-1)
    geodistssmpl = torch.tensor(np.load(config.GEODESICS_SMPL),
                      device=device)

    # load optimization routine
    smplify = SMPLifyDC(step_size=1e-2,
                  batch_size=options.batch_size,
                  num_iters=options.num_smplify_iters,
                  focal_length=constants.FOCAL_LENGTH,
                  geodistssmpl=geodistssmpl,
                  geothres=config.geothres,
    )

    # load dsc data
    classes = pickle.load(open(osp.join(config.DSC_ROOT, 'classes.pkl'), 'rb'))
    csig = pickle.load(open(osp.join(config.DSC_ROOT, 'ContactSigSMPL.pkl'), 'rb'))
    contactlist = {'classes': classes, 'csig': csig}
    has_smpl_= torch.zeros((options.batch_size)).to(device).bool()

    # Setup renderer for visualization
    renderer = Renderer(focal_length=constants.FOCAL_LENGTH,
                        img_res=constants.IMG_RES,
                        faces=smpl.faces,
                        contactlist=contactlist)

    # segments 
    segments = BatchBodySegment([x for x in exn.segments.keys()], face_tensor[0])
    
    # Process each image
    for idx in tqdm(range(len(dataset.data['imgname']))):

        batch = dataset[idx]

        # create tensor (this is what the torch data loader normally does)
        batch = {k: torch.tensor(v).unsqueeze(0).to(device) if not isinstance(v, str) \
                    else [v] for k,v in batch.items()}

        # move input to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) \
                    else v for k,v in batch.items()}

        images = batch['img']
        gt_keypoints_2d = batch['keypoints']
        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * options.img_res * \
                                          (gt_keypoints_2d_orig[:, :, :-1] + 1)

        gt_disc_contact = batch['contact_vec']
        has_disc_contact = batch['has_disc_contact'].bool()
        has_2d_keypoints_gtanno= batch['has_gt_kpts'].bool()


        # Get the fits of the original SPIN model. To add camera loss
        with torch.no_grad():
            # forward pass SPIN model in eval mode
            init_rotmat, init_betas, init_camera = modelspin(images)
            output = smpl(betas=init_betas, body_pose=init_rotmat[:,1:],
                    global_orient=init_rotmat[:,0].unsqueeze(1), pose2rot=False)
            init_vertices = output.vertices.clone()
            init_joints = output.joints.clone()
            init_cam_t = torch.stack([init_camera[:,1],
                                      init_camera[:,2],
                                      2*constants.FOCAL_LENGTH/(options.img_res * \
                                                    init_camera[:,0] + 1e-9)],dim=-1)

        # Convert predicted rotation matrices to axis-angle
        init_rotmat_hom = torch.cat([init_rotmat.detach().view(-1, 3, 3).detach(),
            torch.tensor([0,0,1], dtype=torch.float32, device=device) \
            .view(1, 3, 1).expand(batch_size * 24, -1, -1)], dim=-1)
        init_pose = rotation_matrix_to_angle_axis(init_rotmat_hom).contiguous().view(batch_size, -1)
        init_pose[torch.isnan(init_pose)] = 0.0

        new_opt_vertices, new_opt_joints,\
            new_opt_pose, new_opt_betas,\
            new_opt_cam_t, new_opt_joint_loss, \
            smplifyoptiverts = smplify(
                init_pose.detach(),
                init_betas.detach(),
                init_cam_t.detach(),
                0.5 * options.img_res * \
                torch.ones(batch_size, 2, device=device),
                gt_keypoints_2d_orig,
                use_contact=options.use_contact_in_the_loop,
                contactlist=contactlist,
                gt_contact=[gt_disc_contact, None],
                ignore_idxs=has_smpl_,
                has_discrete_contact=has_disc_contact,
                has_gt_keypoints=has_2d_keypoints_gtanno,
                contact_loss_weight=options.contact_in_the_loop_loss_weight,
                contact_loss_return='sum',
                segments=segments)

        # render results to logdir
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device) \
                                                                .reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device) \
                                                                .reshape(1,3,1,1)

        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
        for idx in range(batch_size):
            imgname, imgending = batch['imgname'][idx].split('/')[-1].split('.')
            img = images[idx].permute(1,2,0).cpu().numpy()
            img_out = images[idx].permute(1,2,0).cpu().numpy()
            for data in [(init_vertices[idx], init_cam_t[idx]),
                         (new_opt_vertices[idx],new_opt_cam_t[idx])]:
                verts, cam = data[0].cpu().numpy(), data[1].cpu().numpy()
                img_out_front = renderer(verts, cam, img, 
                    contact=gt_disc_contact[idx])
                center = verts.mean(axis=0)
                rot_vertices = np.dot((verts - center), aroundy) + center
                img_out_rot = renderer(rot_vertices, cam, np.zeros_like(img),
                    contact=gt_disc_contact[idx])
                img_out = np.hstack((img_out, img_out_front, img_out_rot))

            cv2.imwrite(osp.join(options.log_dir, '.'.join([imgname, imgending])),
                        img_out[:,:,::-1]*255)



if __name__ == '__main__':
    options = SMPLifyDCOptions().parse_args()
    main(options)
