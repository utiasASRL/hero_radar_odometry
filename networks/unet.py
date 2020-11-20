import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers import DoubleConv, OutConv, Down, Up

class UNet(torch.nn.Module):
    """ The UNet network, code from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py """
    def __init__(self, config):
        super().__init__()

        # n_channels
        n_channels = config['input_channels']
        bilinear = config['networks']['unet']['bilinear']
        first_feature_dimension = config['networks']['unet']['first_feature_dimension']

        # down
        self.inc = DoubleConv(n_channels, first_feature_dimension)
        self.down1 = Down(first_feature_dimension, first_feature_dimension * 2)
        self.down2 = Down(first_feature_dimension * 2, first_feature_dimension * 4)
        self.down3 = Down(first_feature_dimension * 4, first_feature_dimension * 8)
        self.down4 = Down(first_feature_dimension * 8, first_feature_dimension * 16)

        self.up1_pts = Up(first_feature_dimension * (16 + 8), first_feature_dimension * 8, bilinear)
        self.up2_pts = Up(first_feature_dimension * (8 + 4), first_feature_dimension * 4, bilinear)
        self.up3_pts = Up(first_feature_dimension * (4 + 2), first_feature_dimension * 2, bilinear)
        self.up4_pts = Up(first_feature_dimension * (2 + 1), first_feature_dimension * 1, bilinear)
        self.outc_pts = OutConv(first_feature_dimension, 1)

        self.up1_score = Up(first_feature_dimension * (16 + 8), first_feature_dimension * 8, bilinear)
        self.up2_score = Up(first_feature_dimension * (8 + 4), first_feature_dimension * 4, bilinear)
        self.up3_score = Up(first_feature_dimension * (4 + 2), first_feature_dimension * 2, bilinear)
        self.up4_score = Up(first_feature_dimension * (2 + 1), first_feature_dimension * 1, bilinear)
        self.outc_score = OutConv(first_feature_dimension, 1)
        self.sigmoid = torch.nn.Sigmoid()

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, height, width = x.size()

        x1 = self.inc(x)
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
        score = self.sigmoid(score)

        # Resize outputs of downsampling layers to the size of the original
        # image. Features are interpolated using bilinear interpolation to
        # get gradients for back-prop. Concatenate along the feature channel
        # to get pixel-wise descriptors of size Bx248xHxW
        f1 = F.interpolate(x1, size=(height, width), mode='bilinear')
        f2 = F.interpolate(x2, size=(height, width), mode='bilinear')
        f3 = F.interpolate(x3, size=(height, width), mode='bilinear')
        f4 = F.interpolate(x4, size=(height, width), mode='bilinear')
        f5 = F.interpolate(x5, size=(height, width), mode='bilinear')

        feature_list = [f1, f2, f3, f4, f5]
        descriptors = torch.cat(feature_list, dim=1)

        return logits_pts, score, descriptors
