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

from networks.layers import DoubleConv, OutConv, Down, Up, LeakyDoubleConv, LeakyDown, LeakyUp
# from visualization.plots import Plotting

class UNetFBlock(nn.Module):
    def __init__(self, config):
        super(UNetFBlock, self).__init__()
        # TODO relying on hard-coded config params

        # n_channels
        self.n_channels = 0
        self.input_channel = config["dataset"]["images"]["input_channel"]
        self.n_channels += 3 if 'vertex' in self.input_channel else 0
        self.n_channels += 1 if 'intensity' in self.input_channel else 0
        self.n_channels += 1 if 'range' in self.input_channel else 0

        self.n_weight_score = config["networks"]["unet"]["n_weight_score"] # this specifies num_classes for output
        self.bilinear = config["networks"]["unet"]["bilinear"]
        self.first_feature_dimension = config["networks"]["unet"]["first_feature_dimension"]
        self.depth = config["networks"]["unet"]["depth"]

        if config["networks"]["unet"]["leaky"]:
            self.inc = LeakyDoubleConv(self.n_channels, self.first_feature_dimension)    # 512 x 384 (out size after layer)
            self.inc2 = LeakyDoubleConv(self.first_feature_dimension, self.first_feature_dimension)
        else:
            self.inc = DoubleConv(self.n_channels, self.first_feature_dimension)    # 512 x 384 (out size after layer)
            self.inc2 = DoubleConv(self.first_feature_dimension, self.first_feature_dimension)

        # down
        if config["networks"]["unet"]["leaky"]:
            self.down1 = LeakyDown(self.first_feature_dimension, self.first_feature_dimension * 2)                # 256 x 192
            self.down2 = LeakyDown(self.first_feature_dimension * 2, self.first_feature_dimension * 4)               # 128 x 96
            self.down3 = LeakyDown(self.first_feature_dimension * 4, self.first_feature_dimension * 8)               # 64 x 48
            self.down4 = LeakyDown(self.first_feature_dimension * 8, self.first_feature_dimension * 16)              # 32 x 24
        else:
            self.down1 = Down(self.first_feature_dimension, self.first_feature_dimension * 2)                # 256 x 192
            self.down2 = Down(self.first_feature_dimension * 2, self.first_feature_dimension * 4)               # 128 x 96
            self.down3 = Down(self.first_feature_dimension * 4, self.first_feature_dimension * 8)               # 64 x 48
            self.down4 = Down(self.first_feature_dimension * 8, self.first_feature_dimension * 16)              # 32 x 24

        # up 1
        if config["networks"]["unet"]["leaky"]:
            self.up1_pts = LeakyUp(self.first_feature_dimension * (16 + 8), self.first_feature_dimension * 8, self.bilinear)
            self.up2_pts = LeakyUp(self.first_feature_dimension * (8 + 4), self.first_feature_dimension * 4, self.bilinear)
            self.up3_pts = LeakyUp(self.first_feature_dimension * (4 + 2), self.first_feature_dimension * 2, self.bilinear)
            self.up4_pts = LeakyUp(self.first_feature_dimension * (2 + 1), self.first_feature_dimension * 1, self.bilinear)
        else:
            self.up1_pts = Up(self.first_feature_dimension * (16 + 8), self.first_feature_dimension * 8, self.bilinear)
            self.up2_pts = Up(self.first_feature_dimension * (8 + 4), self.first_feature_dimension * 4, self.bilinear)
            self.up3_pts = Up(self.first_feature_dimension * (4 + 2), self.first_feature_dimension * 2, self.bilinear)
            self.up4_pts = Up(self.first_feature_dimension * (2 + 1), self.first_feature_dimension * 1, self.bilinear)
        self.outc_pts = OutConv(self.first_feature_dimension, 1)

        # up 2
        if config["networks"]["unet"]["leaky"]:
            self.up1_score = LeakyUp(self.first_feature_dimension * (16 + 8), self.first_feature_dimension * 8, self.bilinear)
            self.up2_score = LeakyUp(self.first_feature_dimension * (8 + 4), self.first_feature_dimension * 4, self.bilinear)
            self.up3_score = LeakyUp(self.first_feature_dimension * (4 + 2), self.first_feature_dimension * 2, self.bilinear)
            self.up4_score = LeakyUp(self.first_feature_dimension * (2 + 1), self.first_feature_dimension * 1, self.bilinear)
        else:
            self.up1_score = Up(self.first_feature_dimension * (16 + 8), self.first_feature_dimension * 8, self.bilinear)
            self.up2_score = Up(self.first_feature_dimension * (8 + 4), self.first_feature_dimension * 4, self.bilinear)
            self.up3_score = Up(self.first_feature_dimension * (4 + 2), self.first_feature_dimension * 2, self.bilinear)
            self.up4_score = Up(self.first_feature_dimension * (2 + 1), self.first_feature_dimension * 1, self.bilinear)
        self.outc_score = OutConv(self.first_feature_dimension, self.n_weight_score)

        # up 3
        if config["networks"]["unet"]["leaky"]:
            self.up1_desc = LeakyUp(self.first_feature_dimension * (16 + 8), self.first_feature_dimension * 16, self.bilinear)
            self.up2_desc = LeakyUp(self.first_feature_dimension * (16 + 4), self.first_feature_dimension * 16, self.bilinear)
            self.up3_desc = LeakyUp(self.first_feature_dimension * (16 + 2), self.first_feature_dimension * 16, self.bilinear)
            self.up4_desc = LeakyUp(self.first_feature_dimension * (16 + 1), self.first_feature_dimension * 16, self.bilinear)
        else:
            self.up1_desc = Up(self.first_feature_dimension * (16 + 8), self.first_feature_dimension * 16, self.bilinear)
            self.up2_desc = Up(self.first_feature_dimension * (16 + 4), self.first_feature_dimension * 16, self.bilinear)
            self.up3_desc = Up(self.first_feature_dimension * (16 + 2), self.first_feature_dimension * 16, self.bilinear)
            self.up4_desc = Up(self.first_feature_dimension * (16 + 1), self.first_feature_dimension * 16, self.bilinear)
        self.outc_desc = OutConv(self.first_feature_dimension*16, 256)

    def forward(self, x, v):
        batch_size, _, height, width = x.size()

        x1 = self.inc(x)
        x1 = self.inc2(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4_up_pts = self.up1_pts(x5, x4)
        x3_up_pts = self.up2_pts(x4_up_pts, x3)
        x2_up_pts = self.up3_pts(x3_up_pts, x2)
        x1_up_pts = self.up4_pts(x2_up_pts, x1)
        logits_pts = self.outc_pts(x1_up_pts)

        x4_up_score = self.up1_score(x5, x4)
        x3_up_score = self.up2_score(x4_up_score, x3)
        x2_up_score = self.up3_score(x3_up_score, x2)
        x1_up_score = self.up4_score(x2_up_score, x1)
        score = self.outc_score(x1_up_score)

        # Resize outputs of downsampling layers to the size of the original
        # image. Features are interpolated using bilinear interpolation to
        # get gradients for back-prop. Concatenate along the feature channel
        # to get pixel-wise descriptors of size Bx248xHxW
        # f1 = F.interpolate(x1, size=(height, width), mode='bilinear', align_corners=True)
        # f2 = F.interpolate(x2, size=(height, width), mode='bilinear', align_corners=True)
        # f3 = F.interpolate(x3, size=(height, width), mode='bilinear', align_corners=True)
        # f4 = F.interpolate(x4, size=(height, width), mode='bilinear', align_corners=True)
        # f5 = F.interpolate(x5, size=(height, width), mode='bilinear', align_corners=True)
        #
        # feature_list = [f1, f2, f3, f4, f5]
        # features = torch.cat(feature_list, dim=1)

        x4_up_desc = self.up1_desc(x5, x4)
        x3_up_desc = self.up2_desc(x4_up_desc, x3)
        x2_up_desc = self.up3_desc(x3_up_desc, x2)
        x1_up_desc = self.up4_desc(x2_up_desc, x1)
        features = self.outc_desc(x1_up_desc)

        return logits_pts, score, features