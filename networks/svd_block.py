""" The UNet network, code from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py """
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from networks.layers import DoubleConv, OutConv, Down, Up
# from visualization.plots import Plotting

class SVDBlock(nn.Module):
    def __init__(self, config):
        super(SVDBlock, self).__init__()
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, keypoints, pseudo, weight, valid_idx=None):
        '''
        B-batch; N-number of keypoints
        :param keypoints: Bx3xN
        :param pseudo: Bx3xN
        :param weight: Bx1xN
        :param valid_idx: BxN
        :return: rotation and translation from keypoint to pseudo
        '''

        keypoints_centroid = (keypoints @ weight.transpose(2,1).contiguous()) / torch.sum(weight, dim=2, keepdim=True) # Bx3x1
        keypoints_centered = keypoints - keypoints_centroid # Bx3xN

        pseudo_centroid = (pseudo @ weight.transpose(2,1).contiguous()) / torch.sum(weight, dim=2, keepdim=True)
        pseudo_centered = pseudo - pseudo_centroid # Bx3xN

        w = torch.sum(weight, dim=2, keepdim=True) # Bx1x1
        H = (1.0 / w) *  torch.matmul((keypoints_centered * weight), pseudo_centered.transpose(2,1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(keypoints.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous()) # analytical solution for optimal rotation
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = -R @ (keypoints_centroid - R.transpose(2,1).contiguous() @ pseudo_centroid)
        t = t.squeeze()

        return R, t


""" Code from: https://github.com/WangYueFt/dcp/blob/master/model.py """
class SVD(nn.Module):
    def __init__(self):
        super(SVD, self).__init__()

    def forward(self, src_coords, tgt_coords, weights):
        batch_size, _, n_points = src_coords.size()  # B x 3 x N

        # Compute weighted centroids (elementwise multiplication/division)
        src_centroid = torch.sum(src_coords * weights, dim=2, keepdim=True) / torch.sum(weights, dim=2, keepdim=True)  # B x 3 x 1
        tgt_centroid = torch.sum(tgt_coords * weights, dim=2, keepdim=True) / torch.sum(weights, dim=2, keepdim=True)

        src_centered = src_coords - src_centroid  # B x 3 x N
        tgt_centered = tgt_coords - tgt_centroid

        W = torch.diag_embed(weights.reshape(batch_size, n_points))  # B x N x N
        w = torch.sum(weights, dim=2).unsqueeze(2)                   # B x 1 x 1

        H = (1.0 / w) * torch.bmm(tgt_centered, torch.bmm(W, src_centered.transpose(2, 1).contiguous()))  # B x 3 x 3

        U, S, V = torch.svd(H)

        # det_VUT = torch.det(torch.bmm(V, U.transpose(2, 1).contiguous()))
        det_UV = torch.det(U) * torch.det(V)
        ones = torch.ones(batch_size, 2).type_as(V)
        diag = torch.diag_embed(torch.cat((ones, det_UV.unsqueeze(1)), dim=1))  # B x 3 x 3

        # Compute rotation and translation (T_trg_src)
        # R = torch.bmm(V, torch.bmm(diag, U.transpose(2, 1).contiguous()))
        # t = centroid_trg - torch.bmm(R, centroid_src)
        R_tgt_src = torch.bmm(U, torch.bmm(diag, V.transpose(2, 1).contiguous()))  # B x 3 x 3
        t_tgt_src_insrc = src_centroid - torch.bmm(R_tgt_src.transpose(2, 1).contiguous(), tgt_centroid)  # B x 3 x 1
        t_src_tgt_intgt = -R_tgt_src.bmm(t_tgt_src_insrc)

        # Create translation matrix
        zeros = torch.zeros(batch_size, 1, 3).type_as(V)  # B x 1 x 3
        one = torch.ones(batch_size, 1, 1).type_as(V)  # B x 1 x 1
        trans_cols = torch.cat([t_src_tgt_intgt, one], dim=1)  # B x 4 x 1
        rot_cols = torch.cat([R_tgt_src, zeros], dim=1)  # B x 4 x 3
        T_tgt_src = torch.cat([rot_cols, trans_cols], dim=2)  # B x 4 x 4

        return T_tgt_src, R_tgt_src, t_src_tgt_intgt