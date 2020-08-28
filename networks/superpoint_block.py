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

from networks.layers import DoubleConv, OutConv, Down, Up, SingleConv
# from visualization.plots import Plotting

class SuperpointBlock(nn.Module):
    def __init__(self, config):
        super(SuperpointBlock, self).__init__()
        # TODO relying on hard-coded config params

        # n_channels
        self.n_channels = 0
        self.input_channel = config["dataset"]["images"]["input_channel"]
        self.n_channels += 3 if 'vertex' in self.input_channel else 0
        self.n_channels += 1 if 'intensity' in self.input_channel else 0
        self.n_channels += 1 if 'range' in self.input_channel else 0

        self.n_weight = config['networks']['unet']['n_weight_score']
        self.weight_shuffle = config['networks']['unet']['super_weight_shuffle']

        # encoder
        self.inc = DoubleConv(self.n_channels, 64)    # 512 x 384 (out size after layer)
        self.inc2 = DoubleConv(64, 64)
        self.down1 = Down(64, 128)                # B x 64 x 256 x 192
        self.down1_2 = DoubleConv(128, 128)                # B x 64 x 256 x 192
        self.down2 = Down(128, 256)               # B x 128 x 128 x 96
        self.down2_2 = DoubleConv(256, 256)               # B x 128 x 128 x 96
        self.down3 = Down(256, 512)              # B x 256 x 64 x 48 (H/8 x W/8)

        # decoders
        self.decode_detector = DoubleConv(512, 512)
        self.out_detector = OutConv(512, 64)

        self.decode_weight = DoubleConv(512, 512)
        if self.weight_shuffle:
            self.out_weight = OutConv(512, 64*self.n_weight)
        else:
            self.out_weight = OutConv(512, self.n_weight)

        self.decode_desc = DoubleConv(512, 512)
        self.out_desc = OutConv(512, 256)

        # self.inc = DoubleConv(self.n_channels, 64)    # 512 x 384 (out size after layer)
        # self.inc2 = DoubleConv(64, 64)
        # self.down1 = Down(64, 128)                # B x 64 x 256 x 192
        # self.down2 = Down(128, 256)               # B x 128 x 128 x 96
        # self.down3 = Down(256, 512)              # B x 256 x 64 x 48 (H/8 x W/8)
        #
        # # decoders
        # self.decode_detector = SingleConv(512, 512)
        # self.out_detector = OutConv(512, 64)
        #
        # self.decode_weight = SingleConv(512, 512)
        # self.out_weight = OutConv(512, 64*self.n_weight)
        #
        # self.decode_desc = SingleConv(512, 512)
        # self.out_desc = OutConv(512, 256)

    def forward(self, x, v):
        batch_size, _, height, width = x.size()

        x = self.inc(x)
        x = self.inc2(x)
        x = self.down1(x)
        x = self.down1_2(x)
        x = self.down2(x)
        x = self.down2_2(x)
        x = self.down3(x)                           # B x 256 x H/8 x W/8

        detector = self.decode_detector(x)                # B x 65 x H/8 x W/8
        detector = self.out_detector(detector)
        detector = F.pixel_shuffle(detector, 8)       # B x 1 x H x W

        desc = self.decode_desc(x)                                            # B x 256 x H/8 x W/8
        desc = self.out_desc(desc)
        desc = F.interpolate(desc, size=(height, width), mode='bicubic', align_corners=True)      # B x 256 x H x W

        weight = self.decode_weight(x)              # B x 64 x H/8 x W/8
        weight = self.out_weight(weight)
        if self.weight_shuffle:
            weight = F.pixel_shuffle(weight, 8)          # B x 1 x H x W
        else:
            weight = F.interpolate(weight, size=(height, width), mode='bicubic', align_corners=True)

        return detector, weight, desc