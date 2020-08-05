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
from utils.utils import zn_desc
# from visualization.plots import Plotting

class SoftmaxMatcherBlock(nn.Module):
    def __init__(self, config):
        super(SoftmaxMatcherBlock, self).__init__()
        # TODO take the dense match logic outside this block
        self.match_type = config["networks"]["match_type"] # zncc, l2, dp
        self.softmax_temperature = 0.02
        self.match_vals = torch.tensor(0)

    def forward(self, src_coords, tgt_coords, tgt_weights, src_desc, tgt_desc):
        '''
        Descriptors are assumed to be not normalized
        :param src_coords: Bx3xN
        :param tgt_coords: Bx3xM
        :param tgt_weights: Bx1xM (M=N for keypoint match; M=HW for dense match)
        :param src_desc: BxCxN
        :param tgt_desc: BxCxM
        :return: pseudo, pseudo_weights, pseudo_desc (UN-Normalized)
        '''

        # Normalize the descriptors based on match_type
        if self.match_type == 'zncc':
            src_desc_norm = zn_desc(src_desc) # BxCxN
            tgt_desc_norm = zn_desc(tgt_desc) # BxCxM
        elif self.match_type == 'dp':
            src_desc_norm = F.normalize(src_desc, dim=1)
            tgt_desc_norm = F.normalize(tgt_desc, dim=1)
        elif self.match_type == 'l2':
            # TODO implement Mona's setup
            pass
        else:
            assert False, "Match type does NOT support"

        # match points based on match_type
        if self.match_type == 'zncc':
            self.match_vals = torch.matmul(src_desc_norm.transpose(2, 1).contiguous(),
                                           tgt_desc_norm) / float(src_desc_norm.size(1)) # B x N x M
            soft_match_vals = F.softmax(self.match_vals / self.softmax_temperature, dim=2) # B x N x M
        elif self.match_type == 'dp':
            self.match_vals = torch.matmul(src_desc_norm.transpose(2, 1).contiguous(), tgt_desc_norm) # B x N x M
            soft_match_vals = F.softmax(self.match_vals / self.softmax_temperature, dim=2) # B x N x M
        else:
            assert False, "Only support match type zncc now"

        # extract pseudo points and attri associated with them
        # TODO this is different from Mona's implementation cuz she grid-sampled for pseudo scores and desc
        # TODO in my case, I extract desc and then do norm
        pseudo_coords = torch.matmul(tgt_coords, soft_match_vals.transpose(2, 1)) # Bx3xN
        pseudo_weights = torch.matmul(tgt_weights, soft_match_vals.transpose(2, 1)) # Bx1xN
        pseudo_descs = torch.matmul(tgt_desc, soft_match_vals.transpose(2, 1)) # BxCxN

        # return normalized desc
        if self.match_type == 'zncc':
            pseudo_descs = zn_desc(pseudo_descs)
        elif self.match_type == 'dp':
            pseudo_descs = F.normalize(pseudo_descs, dim=1)
        elif self.match_type == 'l2':
            pass
        else:
            assert False, "Cannot normalize because match type is NOT support"

        return pseudo_coords, pseudo_weights, pseudo_descs