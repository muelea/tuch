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

import sys
import torch
import trimesh
import torch.nn as nn
import numpy as np
import os.path as osp
from tuch.utils.contact import winding_numbers
from configs import config

from data.essentials.segments.smpl import segm_utils as exn


class BodySegment(nn.Module):
    def __init__(self,
                 name,
                 faces,
                 append_idx=None):
        super(BodySegment, self).__init__()
        self.device = faces.device
        self.name = name
        self.append_idx = faces.max().item() if append_idx is None else append_idx

        # read mesh and find faces of segment
        path = osp.join(config.SEGMENT_DIR, 'smpl_segment_{}.ply'.format(name))
        bandmesh = trimesh.load(path, process=False)
        self.segment_vidx = np.where(np.array(bandmesh.visual.vertex_colors[:,0]) == 255)[0]

        # read boundary information
        self.bands = [x for x in exn.segments[name].keys()]
        self.bands_verts = [x for x in exn.segments[name].values()]
        self.bands_faces = self.create_band_faces().to(self.device)

        # read mesh and find
        faces = faces.squeeze()
        segment_faces_ids = np.where(np.isin(faces.cpu().numpy(), self.segment_vidx).sum(1) == 3)[0]
        segment_faces = faces[segment_faces_ids,:]
        segment_faces = torch.cat((faces[segment_faces_ids,:], self.bands_faces), 0)
        self.register_buffer('segment_faces', segment_faces)

    def create_band_faces(self):
        """
            create the faces that close the segment.
        """
        bands_faces = []
        for idx, k in enumerate(self.bands):
            new_vert_idx = self.append_idx + 1 + idx
            new_faces = [[self.bands_verts[idx][i+1], self.bands_verts[idx][i], new_vert_idx] \
                         for i in range(len(self.bands_verts[idx])-1)]
            bands_faces += new_faces
        return torch.tensor(np.array(bands_faces).astype(np.int64), dtype=torch.long)

    def get_closed_segment(self, vertices):
        """
            create the closed segment mesh from SMPL-X vertices.
        """
        segm_verts = vertices.detach()
        # append vertices to SMPLX vertices, that close the segment and compute faces
        for bv in self.bands_verts:
            close_segment_vertices = torch.mean(vertices.detach()[:, bv,:], 1, keepdim=True)
            segm_verts = torch.cat((segm_verts, close_segment_vertices), 1)
        segm_triangles = segm_verts[:, self.segment_faces]

        return segm_triangles

    def has_self_isect(self, vertices):
        """
            check if segment is self intersecting.
        """
        #segm_verts = vertices.detach()
        # append vertices to SMPLX vertices, that close the segment and compute faces
        #for bv in self.bands_verts:
        #    close_segment_vertices = torch.mean(vertices.detach()[:, bv,:], 1, keepdim=True)
        #    segm_verts = torch.cat((segm_verts, close_segment_vertices), 1)
        #segm_triangles = segm_verts[:, self.segment_faces]

        segm_triangles = self.get_closed_segment(vertices)
        # select all vertices on segment
        segm_verts = vertices.detach()[:,self.segment_vidx,:]

        # do inside outside segmentation
        exterior = winding_numbers(segm_verts, segm_triangles).squeeze().le(0.99)

        return exterior


class BatchBodySegment(nn.Module):
    def __init__(self,
                 names,
                 faces):
        super(BatchBodySegment, self).__init__()
        self.names = names
        self.nv = faces.max().item()
        self.append_idx = [len(b) for a,b in exn.segments.items() for c,d in b.items() \
                           if a in self.names]
        self.append_idx = np.cumsum(np.array([self.nv] + self.append_idx))

        self.segmentation = {}
        for idx, name in enumerate(names):
            self.segmentation[name] = BodySegment(name, faces)

    def batch_has_self_isec(self, vertices):
        """
            check is mesh is intersecting with itself
        """
        exteriors = []
        for k, segm in self.segmentation.items():
            exteriors += [segm.has_self_isect(vertices)]
        return exteriors
