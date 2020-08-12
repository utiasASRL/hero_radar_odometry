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

        keypoints_centroid = (keypoints @ weight.transpose(2,1)) / torch.sum(weight, dim=2, keepdim=True) # Bx3x1
        keypoints_centered = keypoints - keypoints_centroid # Bx3xN

        pseudo_centroid = (pseudo @ weight.transpose(2,1)) / torch.sum(weight, dim=2, keepdim=True)
        pseudo_centered = pseudo - pseudo_centroid # Bx3xN

        w = torch.sum(weight, dim=2, keepdim=True) # Bx1x1
        H = (1.0 / w) *  torch.matmul((keypoints_centered * weight), pseudo_centered.transpose(2,1))

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

        t = -R @ (keypoints_centroid - R.transpose(2,1) @ pseudo_centroid)
        t = t.squeeze()
        return R, t