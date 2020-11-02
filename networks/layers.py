import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from PIL import ImageFilter

from utils.lie_algebra import se3_log, se3_inv

class SPPLayer(nn.Module):

    def __init__(self, levels=[1,2,3,5], pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.levels = levels
        self.pool_type = pool_type

    def forward(self, x):
        b, c, h, w = x.size()
        pooling_layers = []

        for num_bins in self.levels:
            sizeX = int(math.ceil(float(h) / num_bins))
            strideX = int(math.floor(float(h) / num_bins))
            sizeY = int(math.ceil(float(w) / num_bins))
            strideY = int(math.floor(float(w) / num_bins))

            if self.pool_type == 'max_pool':
                self.pyr_pool = nn.MaxPool2d(kernel_size=[sizeX, sizeY], stride=[strideX, strideY])
            else:
                self.pyr_pool = nn.AdaptiveAvgPool2d(kernel_size=[sizeX, sizeY], stride=[strideX, strideY])

            pooling_layers.append(self.pyr_pool(x).view(b, -1))

        x = torch.cat(pooling_layers, dim=1)

        return x

""" Parts of the U-Net model 
    Code from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# class RANSAC(nn.Module):
#     def __init__(self, inlier_threshold, error_tolerance, num_iterations):
#         super(RANSAC, self).__init__()
#
#         T_s_v = torch.tensor([[0.000796327,         -1.0,       0.0, 0.119873],
#                               [  -0.330472, -0.000263164, -0.943816,  1.49473],
#                               [   0.943815,  0.000751586, -0.330472, 0.354804] ,
#                               [        0.0,          0.0,       0.0,      1.0]])
#         self.register_buffer('T_s_v', T_s_v)
#
#         cu, cv, f, b = 257.446, 197.718, 387.777, 0.239965
#         self.stereo_cam = StereoCameraModel(cu, cv, f, b)
#
#         self.inlier_threshold = inlier_threshold
#         self.error_tolerance = error_tolerance
#         self.num_iterations = num_iterations
#
#         self.svd = SVD()
#
#     def forward(self, keypoints_3D_src, keypoints_3D_trg, keypoints_2D_trg, valid_pts_src, valid_pts_trg, weights):
#         batch_size, _, n_points = keypoints_3D_src.size() # B x 4 x N
#
#         pts_3D_src = keypoints_3D_src.detach()
#         pts_3D_trg = keypoints_3D_trg.detach()
#         pts_2D_trg = keypoints_2D_trg.detach()
#
#         max_num_inliers = torch.zeros(batch_size).type_as(pts_3D_src)
#         max_fraction_inliers = torch.zeros(batch_size).type_as(pts_3D_src)
#         inliers = torch.zeros(batch_size, n_points).type_as(pts_3D_src)
#
#         i = 0
#         ransac_complete = torch.zeros(batch_size).type_as(pts_3D_src).int()
#
#         while (i < self.num_iterations) and (torch.sum(ransac_complete) < batch_size):
#
#             # Pick a random subset of 6 point pairs (3 sufficient, but some poinst will have weight 0 so pick a
#             # few more than needed to increase propbability of getting points with rank 3).
#             rand_index = torch.randint(0, n_points, size=(batch_size, 6)).type_as(pts_3D_src).long()
#             rand_index = rand_index.unsqueeze(1)
#             rand_pts_3D_src = torch.gather(pts_3D_src, dim=2, index=rand_index.expand(batch_size, 4, 6)) # 1 x 4 x M
#             rand_pts_3D_trg = torch.gather(pts_3D_trg, dim=2, index=rand_index.expand(batch_size, 4, 6)) # 1 x 4 x M
#             rand_weights = torch.gather(weights.detach(), dim=2, index=rand_index)                       # 1 x 1 x M
#
#             # Run SVD
#             try:
#                 T_trg_src = self.svd(rand_pts_3D_src, rand_pts_3D_trg, rand_weights)
#             except RuntimeError as e:
#                 print(e)
#                 print("SVD did not converge, re-doing iteration {}".format(i))
#                 print("src pts: {}".format(rand_pts_3D_src[0, 0:3, :]))
#                 print("trg pts: {}".format(rand_pts_3D_trg[0, 0:3, :]))
#                 print("weights: {}".format(rand_weights[0, 0, :]))
#                 print("rank src pts: {}".format(torch.matrix_rank(rand_pts_3D_src[0, 0:3, :])))
#                 print("rank trg pts: {}".format(torch.matrix_rank(rand_pts_3D_trg[0, 0:3, :])), flush=True)
#                 continue
#
#             # Find number of inliers
#             T_s_v = self.T_s_v.expand(batch_size, 4, 4)
#             T_cam = T_s_v.bmm(T_trg_src).bmm(se3_inv(T_s_v))
#             pts_3D_trg_est = T_cam.bmm(pts_3D_src)
#             pts_2D_trg_est = self.stereo_cam.camera_model(pts_3D_trg_est)[:, 0:2, :].transpose(2,1).contiguous()
#
#             err_pts = torch.norm(pts_2D_trg - pts_2D_trg_est, dim=2) # B x N
#             err_pts_small = err_pts < self.error_tolerance
#             err_pts_small[valid_pts_src[:, 0, :] == 0] = 0
#             err_pts_small[valid_pts_trg[:, 0, :] == 0] = 0
#
#             num_inliers = torch.sum(err_pts_small, dim=1)
#
#             fraction_inliers = num_inliers.float() / n_points
#             enough_inliers = fraction_inliers > self.inlier_threshold
#             ransac_complete = ransac_complete | enough_inliers
#
#             for b in range(batch_size):
#                 if num_inliers[b] > max_num_inliers[b]:
#                     max_num_inliers[b] = num_inliers[b]
#                     max_fraction_inliers[b] = fraction_inliers[b]
#                     inliers[b, :] = err_pts_small[b, :]
#
#             i += 1
#
#         return inliers, max_fraction_inliers