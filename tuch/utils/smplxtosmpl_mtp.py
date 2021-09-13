import pickle
import os
import os.path as osp
import sys
import argparse
from tqdm import tqdm
import smplx
import numpy as np
import joblib
import torch
import trimesh
import glob


def SMPLXtoSMPL(folder, sidx=None, cbs=None):
    """
    script to convert SMPL-X pose and shape parameters to SMPL parameters.
    input:
        pt_path: path to .pt (joblib) file, with a dict containing:
                 smplx_pose, smplx_shape, smplx_right_hand_pose, smplx_left_hand_pose, smplx_global_orient parameters
                 gender used if availble
        to_neutral: convert to SMPL neutral model (for SPIN). If Falase converts to gender passed in pt file.
        sidx, cdb: start index and batch size on cluser.
    """

    dataset_files = glob.glob(osp.join(folder, '**', '*.pkl'), recursive=True)
    dataset_files = [x for x in dataset_files if not osp.exists(x.replace('/smplx/', '/smpl/'))]
    dataset_size = len(dataset_files)
    print(f'Processing {dataset_size} files ...')

    # creat smplx model
    model_params = dict(create_body_pose=True,
                        create_betas=True)

    #smplx_neutral_model = smplx.create(gender='neutral', model_type='smplx', model_path='/is/cluster/lmueller2/metadata/models', num_pca_comps=12)
    #smplx_faces = smplx_neutral_model.faces

    smpl_neutral_model = smplx.create(gender='neutral',  model_type='smpl', model_path='/is/cluster/lmueller2/metadata/models', **model_params)
    smpl_faces = smpl_neutral_model.faces


    # load matrix to gather smpl vertices from smplx verts
    smplxtosmpl = pickle.load(open('/is/cluster/lmueller2/metadata/models_utils/smplx_to_smpl.pkl', 'rb'))
    smplxtosmplmat = smplxtosmpl['matrix']
    if sidx is None:
        loopthrough = range(dataset_size)
    else:
        sidx = int(sidx)
        cbs = int(cbs)
        loopthrough = np.arange(sidx*cbs, sidx*cbs+cbs)

    for index in tqdm(loopthrough):
        dataset_path = dataset_files[index]
        dataset = pickle.load(open(dataset_files[index], 'rb')) #, allow_pickle=True)
        smplx_vertices = dataset['vertices']
        smplx_pose = torch.from_numpy(dataset['body_pose'])
        smplx_global_orient = torch.from_numpy(dataset['global_orient'])
        smpl_vertices_from_smplx = np.matmul(smplxtosmplmat, smplx_vertices)
        smpl_vertices_from_smplx = torch.from_numpy(smpl_vertices_from_smplx)

        smpl_model = smpl_neutral_model

        smpl_body = smpl_model(
            body_pose=torch.cat((smplx_pose, torch.zeros(1,6)), axis=1),
            global_orient=smplx_global_orient
        )
        smpl_vertices = smpl_body.vertices.detach()[0].cpu().numpy()

        # get initial translation
        smpl_transl = smpl_vertices_from_smplx.mean(axis=0) - \
                      smpl_vertices.mean(axis=0)
        params = dict(
            transl=smpl_transl,
            body_pose=torch.cat((smplx_pose, torch.zeros(1,6)), axis=1),
            global_orient=smplx_global_orient)
        smpl_model.reset_params(**params)

        # start optimization
        smpl_model.betas.requires_grad = True
        smpl_model.body_pose.requires_grad = True
        smpl_model.transl.requires_grad = True
        optimizer = torch.optim.Adam([
              {'params': [smpl_model.body_pose, smpl_model.betas, smpl_model.transl],
               'lr': 1e-2}
        ])

        print('start optimization ... ')
        max_iterations = 5000
        step = 0
        body = smpl_model()
        verts = body.vertices
        while step < max_iterations:
            optimizer.zero_grad()

            body = smpl_model(
                 get_skin = True,
                 global_orient=smplx_global_orient,
            )
            verts = body.vertices
            loss = torch.norm((smpl_vertices_from_smplx - verts), dim=2)
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            step += 1

        # export mesh for visual comparison
        #verts = verts.detach()[0].cpu().numpy()
        #smpl_mesh = trimesh.Trimesh(verts, smpl_faces)
        #smpl_mesh.export('TESTMESH_{}_SMPL.ply'.format(index))
        #smplx_mesh = trimesh.Trimesh(smplx_vertices, smplx_faces)
        #smplx_mesh.export('TESTMESH_{}_SMPLX.ply'.format(index))

        #overwrite pose and save
        newpose = np.hstack((body.global_orient[0].numpy(), body.body_pose.detach().cpu().numpy()[0])).astype(np.float64)
        betas = body.betas.detach().cpu().numpy()[0].astype(np.float64)
        dataout = {'pose': newpose, 'betas': betas}
        path_out = dataset_path.replace('/smplx/', '/smpl/')
        print(path_out)
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        with open(path_out, 'wb') as f:
            pickle.dump(dataout, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True, help='path to pt file to be processed')
    parser.add_argument('--idx', required=False, default=None, help='process single index of pt file')
    parser.add_argument('--cbs', required=False, default=None, help='batch size for cluster jobs')
    args = parser.parse_args()
    folder = args.folder
    sidx = args.idx
    cbs= args.cbs

    SMPLXtoSMPL(folder, sidx, cbs)
