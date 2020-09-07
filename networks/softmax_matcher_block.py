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
from utils.stereo_camera_model import StereoCameraModel
from utils.utils import zn_desc, sample_desc, get_scores

class SoftmaxMatcherBlock(nn.Module):
    def __init__(self, config):
        super(SoftmaxMatcherBlock, self).__init__()
        # TODO take the dense match logic outside this block
        self.config = config
        self.match_type = config["networks"]["match_type"]  # zncc, l2, dp
        self.softmax_temperature = config['networks']['matcher_block']['softmax_temperature']

        # stereo camera model
        self.stereo_cam = StereoCameraModel()

    def forward(self, geometry_img, tgt_coords, tgt_2D, tgt_weights, tgt_weights_dense, src_desc_norm, tgt_desc,
                tgt_desc_dense, cam_calib, window_size):
        '''
        Descriptors are assumed to be not normalized
        :param tgt_coords: Bx3xM
        :param tgt_weights: Bx1xHxW
        :param src_desc_norm: BxCxN
        :param tgt_desc: BxCxHxW
        :return: pseudo, pseudo_weights, pseudo_desc (UN-Normalized)
        '''

        batch_size = tgt_desc.size(0)
        n_features = tgt_desc.size(1)

        # Normalize the descriptors based on match_type
        if self.match_type == 'zncc':
            tgt_desc_norm = zn_desc(tgt_desc.view(batch_size, n_features, -1))  # BxCx(WxH)
        elif self.match_type == 'dp':
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
            max_softmax = torch.max(torch.max(soft_match_vals, dim=2)[0], dim=1)[0]
            # print(torch.sum(soft_match_vals, dim=))
            # print("softmax src/trg temp3 max: {}".format(torch.max(softmax_score_src_trg_temp3, dim=2)[0]))
        else:
            assert False, "Only support match type zncc now"

        # print(soft_match_vals.size())
        # print(torch.max(soft_match_vals))

        # extract pseudo points and attributes associated with them
        # TODO this is different from Mona's implementation cuz she grid-sampled for pseudo scores and desc
        # TODO in my case, I extract desc and then do norm
        valid_pts = torch.ones(batch_size, 1, src_desc_norm.size(2)).cuda().int()
        if self.config['dataset']['sensor'] == 'velodyne':
            pseudo_coords = torch.matmul(tgt_coords, soft_match_vals.transpose(2, 1).contiguous()) # Bx3xN
            pseudo_weights = torch.matmul(tgt_weights.view(batch_size, 1, -1), soft_match_vals.transpose(2, 1).contiguous()) # Bx1xN
            pseudo_descs = torch.matmul(tgt_desc.view(batch_size, n_features, -1), soft_match_vals.transpose(2, 1).contiguous()) # BxCxN

            # Compute 2D keypoints from 3D pseudo coordinates
            pseudo_2D = compute_2D_from_3D(pseudo_coords, self.config)
        else:
            pseudo_2D = torch.matmul(tgt_2D.transpose(2, 1).contiguous(), soft_match_vals.transpose(2, 1).contiguous()).transpose(2, 1).contiguous()  # BxNx2

            # Compute 3D points with camera model, points are in cam0 frame, Bx3xN, Bx1xN
            # print("Disparity")
            # print(geometry_img[0])
            pseudo_coords, valid_pts = self.stereo_cam.inverse_camera_model(pseudo_2D, geometry_img, cam_calib,
                                                                            start_ind=1, step=window_size)

            pseudo_weights = get_scores(tgt_weights_dense, pseudo_2D)
            pseudo_descs = sample_desc(tgt_desc_dense, pseudo_2D)

        # return normalized desc
        if self.match_type == 'zncc':
            pseudo_descs = zn_desc(pseudo_descs)
        elif self.match_type == 'dp':
            pseudo_descs = F.normalize(pseudo_descs, dim=1)
        elif self.match_type == 'l2':
            pass
        else:
            assert False, "Cannot normalize because match type is NOT support"

        return pseudo_coords, pseudo_weights, pseudo_descs, pseudo_2D, valid_pts, max_softmax