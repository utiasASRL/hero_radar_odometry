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
from utils.utils import zn_desc, sample_desc, get_scores
# from visualization.plots import Plotting

class SoftmaxMatcherBlock(nn.Module):
    def __init__(self, config):
        super(SoftmaxMatcherBlock, self).__init__()
        # TODO take the dense match logic outside this block
        self.config = config
        self.match_type = config["networks"]["match_type"]  # zncc, l2, dp
        self.softmax_temperature = config['networks']['matcher_block']['softmax_temperature']

        # Coordinates need for camera sensor
        self.height = config['dataset']['images']['height']
        self.width = config['dataset']['images']['width']
        v_coord, u_coord = torch.meshgrid([torch.arange(0, self.height),
                                           torch.arange(0, self.width)])

        v_coord = v_coord.reshape(self.height * self.width).float()  # HW
        u_coord = u_coord.reshape(self.height * self.width).float()
        image_coords = torch.stack((u_coord, v_coord), dim=1)        # HW x 2
        self.register_buffer('image_coords', image_coords)

    def forward(self, src_coords, tgt_coords, tgt_weights, src_desc, tgt_desc):
        '''
        Descriptors are assumed to be not normalized
        :param src_coords: Bx3xN
        :param tgt_coords: Bx3xM
        :param tgt_weights: Bx1xHxW
        :param src_desc: BxCxN
        :param tgt_desc: BxCxHxW
        :return: pseudo, pseudo_weights, pseudo_desc (UN-Normalized)
        '''

        batch_size, n_features, _, _ = tgt_desc.size()

        # Normalize the descriptors based on match_type
        if self.match_type == 'zncc':
            src_desc_norm = zn_desc(src_desc) # BxCxN
            tgt_desc_norm = zn_desc(tgt_desc.view(batch_size, n_features, -1))  # BxCx(WxH)
        elif self.match_type == 'dp':
            src_desc_norm = F.normalize(src_desc, dim=1)
            tgt_desc_norm = F.normalize(tgt_desc.view(batch_size, n_features, -1), dim=1)
        elif self.match_type == 'l2':
            # TODO implement Mona's setup
            pass
        else:
            assert False, "Match type does NOT support"

        # match points based on match_type
        if self.match_type == 'zncc':
            match_vals = torch.matmul(src_desc_norm.transpose(2, 1).contiguous(),
                                      tgt_desc_norm) / float(src_desc_norm.size(1))    # BxNx(HxW)
            soft_match_vals = F.softmax(match_vals / self.softmax_temperature, dim=2)  # BxNx(HxW)
        else:
            assert False, "Only support match type zncc now"

        # extract pseudo points and attributes associated with them
        # TODO this is different from Mona's implementation cuz she grid-sampled for pseudo scores and desc
        # TODO in my case, I extract desc and then do norm
        valid_pts = torch.ones(batch_size, 1, n_features).type_as(tgt_desc).int()
        if self.config['dataset']['sensor'] == 'velodyne':
            pseudo_coords = torch.matmul(tgt_coords, soft_match_vals.transpose(2, 1)) # Bx3xN
            pseudo_weights = torch.matmul(tgt_weights.view(batch_size, n_features, -1), soft_match_vals.transpose(2, 1)) # Bx1xN
            pseudo_descs = torch.matmul(tgt_desc.view(batch_size, n_features, -1), soft_match_vals.transpose(2, 1)) # BxCxN

            # Compute 2D keypoints from 3D pseudo coordinates
            pseudo_2D = compute_2D_from_3D(pseudo_coords, self.config)
        else:
            batch_image_coords = self.image_coords.unsqueeze(0).expand(batch_size, self.height * self.width, 2)
            pseudo_2D = torch.matmul(batch_image_coords.transpose(2, 1).contiguous(),
                                         soft_match_vals.transpose(2, 1).contiguous()).transpose(2, 1).contiguous()  # BxNx2
            pseudo_coords, valid_pts = self.stereo_cam.inverse_camera_model(keypoints_2D, geometry_img)  # Bx4xN, Bx1xN
            pseudo_coords = pseudo_coords[:, :3, :]  # Bx3xN

            pseudo_weights = get_scores(tgt_weights, pseudo_2D)
            pseudo_weights[valid_pts == 0] = 0.0
            pseudo_descs = sample_desc(tgt_desc, pseudo_2D)

        # return normalized desc
        if self.match_type == 'zncc':
            pseudo_descs = zn_desc(pseudo_descs)
        elif self.match_type == 'dp':
            pseudo_descs = F.normalize(pseudo_descs, dim=1)
        elif self.match_type == 'l2':
            pass
        else:
            assert False, "Cannot normalize because match type is NOT support"

        return pseudo_coords, pseudo_weights, pseudo_descs, pseudo_2D, valid_pts