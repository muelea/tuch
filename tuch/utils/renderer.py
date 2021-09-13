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
# https://github.com/nkolot/SPIN/blob/master/utils/renderer.py

import os
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
import sys

def vertstocol(verts):
    mins = torch.min(verts, dim=1)[0]
    vertscol = verts - mins
    maxs = torch.max(verts, dim=1)[0]
    vertscol = (verts * 255/maxs).int()
    return vertscol


class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, contactlist, focal_length=5000, cam_type='perspective', img_res=224, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res,
                                       point_size=1.0)
        self.cam_type = cam_type
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces
        self.contactlist = contactlist

    def visualize_tbm(self, vertices, camera_translation, images, keypoints=None,
          gt_l3_contact=None, gt_vertsincontact_idx={},
          has_contact_pc=None, has_contact=None):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))
        rend_imgs = []
        if keypoints is not None:
            keypoints = keypoints.cpu().numpy()

        for i in range(vertices.shape[0]):
            if gt_vertsincontact_idx is None:
                coloridx = None
            else:
                coloridx = gt_vertsincontact_idx[i] if has_contact[i] else None
            if gt_l3_contact is not None:
                contact_pc_idx = gt_l3_contact[i] if has_contact_pc[i] else None
            else:
                contact_pc_idx = None

            rend_imgs.append(images[i])
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i],
                    camera_translation[i], images_np[i], contact=contact_pc_idx, dorot2=False, colverts=coloridx), (2,0,1))).float()
            rend_imgs.append(rend_img)

            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i],
                    camera_translation[i], images_np[i], contact=contact_pc_idx, dorot2=True, colverts=coloridx), (2,0,1))).float()
            rend_imgs.append(rend_img)

            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i],
                    camera_translation[i], images_np[i], contact=contact_pc_idx,  dorot2=False, dorot3=True, colverts=coloridx), (2,0,1))).float()
            rend_imgs.append(rend_img)

        rend_imgs = make_grid(rend_imgs, nrow=4)
        return rend_imgs

    def visualize_eft(self, vertices, camera_translation, images, contact=None, keypoints=None):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))
        rend_imgs = []
        if keypoints is not None:
            keypoints = keypoints.cpu().numpy()

        for i in range(vertices.shape[0]):
            rend_imgs.append(images[i])
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i],
                    camera_translation[i], images_np[i], contact=contact[i], dorot2=False, colverts=None), (2,0,1))).float()
            rend_imgs.append(rend_img)

            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i],
                    camera_translation[i], images_np[i], contact=contact[i], dorot2=True, colverts=None), (2,0,1))).float()
            rend_imgs.append(rend_img)

            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i],
                    camera_translation[i], images_np[i], contact=contact[i],  dorot2=False, dorot3=True, colverts=None), (2,0,1))).float()
            rend_imgs.append(rend_img)

        rend_imgs = make_grid(rend_imgs, nrow=4)
        return rend_imgs



    def visu_smplifycontactopti(self, verticeslist, camera_translation, images,
    gt_contact_pc, gt_vertsincontact_idx={}, keypoints=None):

        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))
        rend_imgs = []

        for i in range(images.shape[0]):
            #rend_imgs.append(images[i])
            if keypoints is not None:
                fig = plt.figure(figsize=(2.24, 2.24))
                plot = fig.add_subplot(111)
                implot1 = plot.imshow(images_np[i]) #*255)
                rgba_cols = np.zeros((keypoints.shape[1],4))
                rgba_cols[:25,0] = 1.0 #openpose keypoints red
                rgba_cols[25:,2] = 1.0 #gt keypoints blue
                rgba_cols[31:34,2] = 0.5 # gt right hand is light blue
                rgba_cols[34:37,1] = 1.0 # gt left hand is green
                rgba_cols[34:37,2] = 0.0 # gt left hand is green
                rgba_cols[:,-1] = keypoints[i].detach().cpu().numpy()[:,-1]
                implot2 = plot.scatter(keypoints[i].detach().cpu().numpy()[:25,0], keypoints[i].detach().cpu().numpy()[:25,1],
                             color=rgba_cols[:25,:], marker="o")
                implot3 = plot.scatter(keypoints[i].detach().cpu().numpy()[25:,0], keypoints[i].detach().cpu().numpy()[25:,1],
                             color=rgba_cols[25:,:], marker="o")
                fig.canvas.draw()
                # save to numpy
                imgpkpts = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                imgpkpts = imgpkpts.reshape((np.transpose(images[i], (1,2,0))).numpy().shape) / 255
                imgpkpts = torch.from_numpy(np.transpose(imgpkpts, (2, 0, 1))).float()
                #plt.savefig('outdebug/test.png')
                rend_imgs.append(imgpkpts)
                plt.close()
                #import ipdb;ipdb.set_trace()
            rend_imgs.append(images[i])
            plotoptilist=[0,int(len(verticeslist) * 0.5),len(verticeslist)-1]
            for j, optistepverts in enumerate(verticeslist):
                if j in plotoptilist:
                    if gt_vertsincontact_idx is not None:
                        if i in gt_vertsincontact_idx:
                            coloridx = gt_vertsincontact_idx[i]
                    else:
                        coloridx = None
                    vertices = optistepverts[i].detach().cpu().numpy()
                    rend_img = torch.from_numpy(np.transpose(self.__call__(vertices,
                            camera_translation[i], images_np[i], gt_contact_pc[i], dorot2=False, colverts=coloridx), (2,0,1))).float()
                    rend_imgs.append(rend_img)
                    rend_img = torch.from_numpy(np.transpose(self.__call__(vertices,
                            camera_translation[i], images_np[i], gt_contact_pc[i], dorot2=True, colverts=coloridx), (2,0,1))).float()
                    rend_imgs.append(rend_img)

        if keypoints is not None:
            rend_imgs = make_grid(rend_imgs, nrow=1+1+2*len(plotoptilist))
        else:
            rend_imgs = make_grid(rend_imgs, nrow=1+2*len(plotoptilist))
        return rend_imgs

    def __call__(self, vertices, camera_translation, image, contact=None,
        dorot2=False, dorot3=False, colverts=None):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE') #,
            #baseColorFactor=(1.0, 1.0, 0.9, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces, process=False)
        vertexcol = np.array(mesh.visual.vertex_colors)
        vertexcol[:,:3] = [230,230,230]
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1,0,0])
        mesh.apply_transform(rot)
        if dorot2:
            rot2 = trimesh.transformations.rotation_matrix(
            np.radians(60), [0,1,0])
            mesh.apply_transform(rot2)
        if dorot3:
            rot3 = trimesh.transformations.rotation_matrix(
            np.radians(60), [1,0,0])
            mesh.apply_transform(rot3)


        # color vertices based on regions
        if (contact is not None) or (colverts is not None):
            def vertstocol(verts):
                mins = np.min(verts, axis=0)
                verts = verts - mins
                maxs = np.max(verts, axis=0)
                vertscol = (verts * 255 / maxs).astype(np.int)
                return vertscol

            meshcols = vertstocol(vertices)
            if colverts is not None:
                if len(colverts[0]) < vertices.shape[0]:
                    for c1, c2 in zip(colverts[0], colverts[1]):
                        currcol = ((meshcols[c1] + meshcols[c2]) /2).astype(np.int)
                        vertexcol[c1, :3] = currcol
                        vertexcol[c2, :3] = currcol #meshcols[colverts]
            elif contact is not None:
                for i1, val in enumerate(contact):
                    if val == 1:
                        regpair = self.contactlist['classes'][i1]
                        vr1 = self.contactlist['csig'][regpair[0]]
                        vr2 = self.contactlist['csig'][regpair[1]]
                        vertexcol[vr1, :3] = meshcols[vr1[0]]
                        vertexcol[vr2, :3] = meshcols[vr1[0]]

            mesh.visual.vertex_colors = vertexcol

        tmesh = pyrender.Mesh.from_trimesh(mesh) #, material=material)
        if image is not None:
            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        else:
            scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                               ambient_light=(0.3, 0.3, 0.3))

        scene.add(tmesh, 'mesh')

        if self.cam_type == 'perspective':
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_translation
            camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        # EFT uses a different camera model, so we add this here
        elif self.cam_type == 'weak_perspective':
            camera_pose = np.eye(4)
            camera_pose[:2, 3] = camera_translation[1:]
            camera = pyrender.OrthographicCamera(xmag=camera_translation[0], ymag=camera_translation[0])

        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        #light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)
        for lp in [[1,1,-1], [-1,1,-1],[1,-1,-1],[-1,-1,-1]]:
            light_pose[:3, 3] = mesh.vertices.mean(0) + np.array(lp)
            #out_mesh.vertices.mean(0) + np.array(lp)
            scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        # output_img = (color[:, :, :3] * valid_mask +
        #           (1 - valid_mask) * image)

        if dorot2 or dorot3:
            imagebackg = image * 0 + 1
            output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * imagebackg)
        else:
            output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img
