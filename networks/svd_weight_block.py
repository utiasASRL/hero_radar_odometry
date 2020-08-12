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

class SVDWeightBlock(nn.Module):
    def __init__(self, config):
        super(SVDWeightBlock, self).__init__()
        self.match_type = config["networks"]["match_type"] # zncc, l2, dp

    def forward(self, keypoint_desc, pseudo_desc, keypoint_score, pseudo_score):
        '''
        Assume descriptors are normalized beforehand
        :param keypoint_desc: BxCxN
        :param pseudo_desc: BxCxN
        :param keypoint_score: Bx1xN
        :param pseudo_score: Bx1xN
        :return:
        '''

        # supposedly desc_match_score range between -1 and 1
        desc_match_score = torch.sum(keypoint_desc * pseudo_desc, dim=1, keepdim=True) / float(keypoint_desc.size(1)) # Bx1xN = BxCxN * BxCxN
        weights = 0.5 * (desc_match_score + 1) # range between 0 and 1

        # add the weight score for each individual points
        weights *= keypoint_score * pseudo_score # Bx1xN

        return weights