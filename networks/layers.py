import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from PIL import ImageFilter

from utils.lie_algebra import se3_log, se3_inv
# from utils.stereo_camera_model import StereoCameraModel
from utils.utils import normalize_coords, get_norm_descriptors, get_scores, l2_norm
# from visualization.plots import Plotting

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

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)

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

# class DescriptorMatchDebug(nn.Module):
#     def __init__(self, results_dir):
#         super(DescriptorMatchDebug, self).__init__()
#
#         self.to_img = transforms.ToPILImage()
#         self.plotting = Plotting(results_dir)
#
#         height, width = 384, 512
#
#         v_coord, u_coord = torch.meshgrid([torch.arange(0, height),
#                                            torch.arange(0, width)])
#
#         v_coord = v_coord.reshape(height * width).float() # HW
#         u_coord = u_coord.reshape(height * width).float()
#         image_coords = torch.stack((u_coord, v_coord), dim=1) # HW x 2
#         self.register_buffer('image_coords', image_coords)
#
#     def forward(self, feature_map_src, feature_map_trg, count, label):
#
#         batch_size, channels, height, width = feature_map_src.size()
#
#         print("feature src min: {}".format(torch.min(feature_map_src)))
#         print("feature src max: {}".format(torch.max(feature_map_src)))
#         print("feature trg min: {}".format(torch.min(feature_map_trg)))
#         print("feature trg max: {}".format(torch.max(feature_map_trg)))
#
#         '''keypoints = torch.tensor([[height/4.0, width/4.0],
#                                   [height/4.0, (3.0 * width)/4.0],
#                                   [height/2.0, width/2.0],
#                                   [(3.0*height)/4.0, width/4.0],
#                                   [(3.0*height)/4.0, (3.0*width)/4.0]]).type_as(feature_map_src).unsqueeze(0)
#         '''
#
#         keypoints = torch.tensor([[width/4.0, height/4.0],
#                                   [(3.0 * width)/4.0, height/4.0],
#                                   [width/2.0, height/2.0],
#                                   [width/4.0, (3.0*height)/4.0],
#                                   [(3.0*width)/4.0, (3.0*height)/4.0]]).type_as(feature_map_src).unsqueeze(0)
#
#         n_points = keypoints.size(1)
#         keypoints = keypoints.expand(batch_size, n_points, 2)
#
#         # Sample descriptors for the chosen test keypoints in the src image
#         descriptors_norm = get_norm_descriptors(feature_map_src, True, keypoints)  # B x C x N
#
#         # Desciptors for every pixel in source image
#         descriptors_norm_src = get_norm_descriptors(feature_map_src)  # B x C x HW
#
#         # Desciptors for every pixel in target image
#         descriptors_norm_trg = get_norm_descriptors(feature_map_trg)  # B x C x HW
#
#         # Get the zncc matching for the src keypoints wrt. every point in the same src image and the trg image.
#         zncc_src_src = torch.matmul(descriptors_norm.transpose(2, 1).contiguous(), descriptors_norm_src) / float(descriptors_norm.size(1)) # B x N x HW
#         zncc_src_trg = torch.matmul(descriptors_norm.transpose(2, 1).contiguous(), descriptors_norm_trg) / float(descriptors_norm.size(1)) # B x N x HW
#
#         print("zncc src/src min: {}".format(torch.min(zncc_src_src, dim=2)[0]))
#         print("zncc src/src max: {}".format(torch.max(zncc_src_src, dim=2)[0]))
#         print("zncc src/trg min: {}".format(torch.min(zncc_src_trg, dim=2)[0]))
#         print("zncc src/trg max: {}".format(torch.max(zncc_src_trg, dim=2)[0]))
#
#         '''
#         # Weigh scores based on distance
#         v_coord, u_coord = torch.meshgrid([torch.arange(0, height),
#                                            torch.arange(0, width)])
#         v_coord = v_coord.reshape(height * width).float()
#         u_coord = u_coord.reshape(height * width).float()
#         image_coords = torch.stack((v_coord, u_coord), dim=1) # HW x 2
#
#         batch_image_coords = image_coords.expand(batch_size, height * width, 2).type_as(feature_map)
#         pts_trg = batch_image_coords.unsqueeze(1).expand(batch_size, n_points, height * width, 2)
#
#         pts_diff = pts_trg - keypoints.unsqueeze(2).expand(batch_size, n_points, height * width, 2)           # B x N x HW x 2
#         pts_norm = torch.norm(pts_diff, dim=3)
#         max_norm = torch.norm(torch.tensor([383.0, 511.0])).item()
#         dist_weight = 1.0 - (pts_norm/max_norm)
#
#         zncc_dist = zncc * (dist_weight ** 2)
#         '''
#
#         '''
#         # Try other matching score functions
#         l1 = torch.zeros(batch_size, n_points, height * width).type_as(feature_map)   # B x N x HW
#         l2 = torch.zeros(batch_size, n_points, height * width).type_as(feature_map)   # B x N x HW
#         mse = torch.zeros(batch_size, n_points, height * width).type_as(feature_map)   # B x N x HW
#
#         for i in range(n_points):
#             descriptor_keypoint = descriptors[:, :, i].reshape(batch_size, channels, 1) # B x C x 1
#
#             descriptor_diff = descriptors_src - descriptor_keypoint                     # B x C x HW
#             l1[:, i, :] = torch.sum(torch.abs(descriptor_diff), dim=1) / descriptor_diff.size(1)
#
#             descriptor_diff = descriptor_diff.transpose(2,1).contiguous().reshape(batch_size * height * width, channels) # BHW x C
#
#             l2_dot = torch.matmul(descriptor_diff.unsqueeze(1), descriptor_diff.unsqueeze(2)) / descriptor_diff.size(1)
#             l2[:, i, :] = l2_dot.reshape(batch_size, height * width) / descriptor_diff.size(1)
#             mse[:, i, :] = torch.sqrt(l2_dot).reshape(batch_size, height * width) / descriptor_diff.size(1)
#
#         l1_inv = 1.0 / l1
#         l2_inv = 1.0 / l2
#         mse_inv = 1.0 / mse
#
#         print("l2 min: {}".format(torch.min(l2)))
#         print("l2 max: {}".format(torch.max(l2)))
#
#         print("s1 min: {}".format(torch.min(score)))
#         print("s1 max: {}".format(torch.max(score)))
#
#         '''
#
#         softmax_score_src_src_temp1 = torch.softmax(zncc_src_src * 10.0, dim=2)
#         softmax_score_src_src_temp2 = torch.softmax(zncc_src_src * 50.0, dim=2)
#         softmax_score_src_src_temp3 = torch.softmax(zncc_src_src * 100.0, dim=2)
#         softmax_score_src_trg_temp1 = torch.softmax(zncc_src_trg * 10.0, dim=2)
#         softmax_score_src_trg_temp2 = torch.softmax(zncc_src_trg * 50.0, dim=2)
#         softmax_score_src_trg_temp3 = torch.softmax(zncc_src_trg * 100.0, dim=2)
#
#         print("softmax src/src temp1 max: {}".format(torch.max(softmax_score_src_src_temp1, dim=2)[0]))
#         print("softmax src/src temp2 max: {}".format(torch.max(softmax_score_src_src_temp2, dim=2)[0]))
#         print("softmax src/src temp3 max: {}".format(torch.max(softmax_score_src_src_temp3, dim=2)[0]))
#         print("softmax src/trg temp1 max: {}".format(torch.max(softmax_score_src_trg_temp1, dim=2)[0]))
#         print("softmax src/trg temp2 max: {}".format(torch.max(softmax_score_src_trg_temp2, dim=2)[0]))
#         print("softmax src/trg temp3 max: {}".format(torch.max(softmax_score_src_trg_temp3, dim=2)[0]))
#
#         # Compute pseudo point pased on all points in target image
#         batch_image_coords = self.image_coords.unsqueeze(0).expand(batch_size, height * width, 2).type_as(feature_map_src)
#         keypoints_2D_pseudo = torch.matmul(batch_image_coords.transpose(2, 1).contiguous(), softmax_score_src_trg_temp3.transpose(2, 1).contiguous()).transpose(2, 1).contiguous() # B x N x 2
#
#         # Sample descriptors for trg pseudopoints and get mean, std and norm needed for zncc
#         descriptors_pseudo_norm = get_norm_descriptors(feature_map_trg, True, keypoints_2D_pseudo)  # B x C x N
#
#         # Get the zncc between each matched point pair (source keypoint to target pseudo point)
#         desc_src = descriptors_norm.transpose(2,1).contiguous().reshape(batch_size * n_points, channels)       # BN x C
#         desc_pseudo = descriptors_pseudo_norm.transpose(2,1).contiguous().reshape(batch_size * n_points, channels) # BN x C
#         desc_zncc = torch.matmul(desc_src.unsqueeze(1), desc_pseudo.unsqueeze(2)) / desc_pseudo.size(1)
#         desc_zncc = desc_zncc.reshape(batch_size, 1, n_points)
#         print("desc_zncc min: {}".format(torch.min(desc_zncc, dim=2)[0]))
#         print("desc_zncc max: {}".format(torch.max(desc_zncc, dim=2)[0]))
#
#         for i in range(n_points):
#             self.plotting.plot_descriptor_match((-(0.5 * (zncc_src_src + 1))).detach().cpu(), i, self.to_img, count, channels, height, width, label, 'zncc_src_src', keypoints[0, i, :])
#             self.plotting.plot_descriptor_match((-(0.5 * (zncc_src_trg + 1))).detach().cpu(), i, self.to_img, count, channels, height, width, label, 'zncc_src_trg', keypoints[0, i, :])
#             '''self.plotting.plot_descriptor_match((softmax_score_src_src_temp1).detach().cpu(), i, self.to_img, count, channels, height, width, label, 'softmax_score_src_src_temp1', keypoints[0, i, :])
#             self.plotting.plot_descriptor_match((softmax_score_src_src_temp2).detach().cpu(), i, self.to_img, count, channels, height, width, label, 'softmax_score_src_src_temp2', keypoints[0, i, :])
#             self.plotting.plot_descriptor_match((softmax_score_src_src_temp3).detach().cpu(), i, self.to_img, count, channels, height, width, label, 'softmax_score_src_src_temp3', keypoints[0, i, :])
#             self.plotting.plot_descriptor_match((softmax_score_src_trg_temp1).detach().cpu(), i, self.to_img, count, channels, height, width, label, 'softmax_score_src_trg_temp1', keypoints[0, i, :])
#             self.plotting.plot_descriptor_match((softmax_score_src_trg_temp2).detach().cpu(), i, self.to_img, count, channels, height, width, label, 'softmax_score_src_trg_temp2', keypoints[0, i, :])
#             self.plotting.plot_descriptor_match((softmax_score_src_trg_temp3).detach().cpu(), i, self.to_img, count, channels, height, width, label, 'softmax_score_src_trg_temp3', keypoints[0, i, :])'''


class DescriptorMatch(nn.Module):
    def __init__(self, window_h, window_w):
        super(DescriptorMatch, self).__init__()

        self.window_h = window_h
        self.window_w = window_w

        height, width = 384, 512

        v_coord, u_coord = torch.meshgrid([torch.arange(0, height),
                                           torch.arange(0, width)])

        v_coord = v_coord.reshape(height * width).float() # HW
        u_coord = u_coord.reshape(height * width).float()
        image_coords = torch.stack((u_coord, v_coord), dim=1) # HW x 2
        self.register_buffer('image_coords', image_coords)


    def forward(self, keypoints_2D_src, keypoints_2D_trg, keypoints_2D_gt, feature_map_src, feature_map_trg,
                scores_map_src, scores_map_trg, use_weights, match_type, all_trg_points):

        batch_size, channels, height, width = feature_map_src.size()
        n_points = keypoints_2D_src.size(1)   # B x N x 2

        # Sample descriptors for the keypoints detected in the source image and compute mean, std and norm
        descriptors_src_norm = get_norm_descriptors(feature_map_src, True, keypoints_2D_src)  # B x C x N

        # Descriptors for all point or points detected in the target image and compute mean, std and norm
        descriptors_trg_norm = None
        if all_trg_points:
            descriptors_trg_norm = get_norm_descriptors(feature_map_trg)                     # B x C x HW
        else:
            descriptors_trg_norm = get_norm_descriptors(feature_map_trg, True, keypoints_2D_trg) # B x C x N

        # For all source keypoints, compute match value between each source point and points in target image.
        # Apply softmax for each source point along the dimension of the target points
        match_vals, soft_match_vals = None, None
        if match_type == 'zncc':
            match_vals = torch.matmul(descriptors_src_norm.transpose(2, 1).contiguous(), descriptors_trg_norm) / float(descriptors_trg_norm.size(1)) # B x N x HW
            soft_match_vals = F.softmax(match_vals/0.01, dim=2) # B x N x HW
        else:
            match_vals = l2_norm(descriptors_src_norm, descriptors_trg_norm) # B x N x N
            soft_match_vals = F.softmin(match_vals/0.01, dim=2)   # B x N x N

        # TODO: consdider wheter this should be included when using dense mathing with target.
        # Only use points that can consitently be match both ways
        '''matched_points_src = torch.ones(batch_size, 1, n_points).type_as(match_vals)
        src_max = torch.max(match_vals.detach(), dim=2)
        trg_max = torch.max(match_vals.detach(), dim=1)
        for i in range(n_points):
            for j in range(batch_size):
                if src_max[1][j, trg_max[1][j, i]] != i:
                    matched_points_src[j, 0, i] = 0'''

        # Compute pseudo point as weighted sum of point coordinates from target image
        keypoints_2D_pseudo = None
        if all_trg_points:
            batch_image_coords = self.image_coords.unsqueeze(0).expand(batch_size, height * width, 2)
            keypoints_2D_pseudo = torch.matmul(batch_image_coords.transpose(2, 1).contiguous(), soft_match_vals.transpose(2, 1).contiguous()).transpose(2, 1).contiguous() # B x N x 2
        else:
            keypoints_2D_pseudo = torch.matmul(keypoints_2D_trg.transpose(2, 1).contiguous(), soft_match_vals.transpose(2, 1).contiguous()).transpose(2, 1).contiguous()   # B x N x 2

        # Sample descriptors for trg pseudo and ground truth points and get mean, std and norm
        descriptors_pseudo_norm = get_norm_descriptors(feature_map_trg, True, keypoints_2D_pseudo)  # B x C x N
        #descriptors_gt_norm = get_norm_descriptors(feature_map_trg, True, keypoints_2D_gt)      # B x C x N

        # Get the zncc between each matched point pair (source keypoint to target pseudo point and target ground truth point)
        desc_src = descriptors_src_norm.transpose(2,1).contiguous().reshape(batch_size * n_points, channels)       # BN x C
        desc_pseudo = descriptors_pseudo_norm.transpose(2,1).contiguous().reshape(batch_size * n_points, channels) # BN x C
        #desc_gt = descriptors_gt_norm.transpose(2,1).contiguous().reshape(batch_size * n_points, channels)         # BN x C

        match_val_pairs, match_val_pos = None, None
        if match_type == 'l2':
            desc_diff = desc_pseudo - desc_src
            # desc_diff_gt = desc_gt - desc_src

            # e^T * e with dimensions BN x 1 x C * BN x C x 1
            match_val_pairs = torch.matmul(desc_diff.unsqueeze(1), desc_diff.unsqueeze(2)) / desc_diff.size(1) # BN x 1
            match_val_pairs = match_val_pairs.reshape(batch_size, 1, n_points)

            #match_val_pos = torch.matmul(desc_diff_gt.unsqueeze(1), desc_diff_gt.unsqueeze(2)) / desc_diff_gt.size(1) # BN x 1
            #match_val_pos = match_val_pos.reshape(batch_size, 1, n_points)
        else:
            match_val_pairs = torch.matmul(desc_src.unsqueeze(1), desc_pseudo.unsqueeze(2)) / desc_pseudo.size(1)
            match_val_pairs = match_val_pairs.reshape(batch_size, 1, n_points)

            #match_val_pos = torch.matmul(desc_src.unsqueeze(1), desc_gt.unsqueeze(2)) / desc_gt.size(1)
            #match_val_pos = match_val_pos.reshape(batch_size, 1, n_points)

        weights, scores_src, scores_pseudo = None, None, None
        if use_weights:

            #variance = torch.var(match_vals, dim=2).unsqueeze(1).detach() # B x 1 x N
            #print("Max variance: {}".format(torch.max(variance, dim=2)[0]))

            # Compute weight for point matches and make sure they lie in range [0, 1].
            if match_type == 'l2':
                weights = 1.0 - (match_val_pairs / torch.max(match_val_pairs, dim=2, keepdim=True)[0])
            else:
                weights = 0.5 * (match_val_pairs + 1.0)

            # Sample scores for the src and pseudo points
            scores_src = get_scores(scores_map_src, keypoints_2D_src)        # B x 1 x N
            scores_pseudo = get_scores(scores_map_trg, keypoints_2D_pseudo)  # B x 1 x N

            weights *= scores_src * scores_pseudo #* (1.0 / variance)
            #weights[matched_points_src == 0] = 0

        return keypoints_2D_pseudo, weights, scores_src, scores_pseudo, match_vals, match_val_pairs.reshape(batch_size, n_points)

""" Code from: https://github.com/WangYueFt/dcp/blob/master/model.py """
class SVD(nn.Module):
    def __init__(self):
        super(SVD, self).__init__()

        T_s_v = torch.tensor([[0.000796327,         -1.0,       0.0, 0.119873],
                              [  -0.330472, -0.000263164, -0.943816,  1.49473],
                              [   0.943815,  0.000751586, -0.330472, 0.354804] ,
                              [        0.0,          0.0,       0.0,      1.0]])
        self.register_buffer('T_s_v', T_s_v)


    def forward(self, keypoints_3D_src, keypoints_3D_trg, weights):

        batch_size, _, n_points = keypoints_3D_src.size() # B x 4 x N

        # Compute weighted centroids (elementwise multiplication/division)
        centroid_src = torch.sum(keypoints_3D_src[:, 0:3, :] * weights, dim=2, keepdim=True) / torch.sum(weights, dim=2, keepdim=True) # B x 3 x 1
        centroid_trg = torch.sum(keypoints_3D_trg[:, 0:3, :] * weights, dim=2, keepdim=True) / torch.sum(weights, dim=2, keepdim=True)

        src_centered = keypoints_3D_src[:, 0:3, :] - centroid_src # B x 3 x N
        trg_centered = keypoints_3D_trg[:, 0:3, :] - centroid_trg

        W = torch.diag_embed(weights.reshape(batch_size, n_points)) # B x N x N
        w = torch.sum(weights, dim=2).unsqueeze(2)                  # B x 1 x 1

        H = (1.0 / w) * torch.bmm(trg_centered,
                                  torch.bmm(W, src_centered.transpose(2, 1).contiguous())) # B x 3 x 3

        U, S, V = torch.svd(H)

        #det_VUT = torch.det(torch.bmm(V, U.transpose(2, 1).contiguous()))
        det_UV = torch.det(U) * torch.det(V)
        ones = torch.ones(batch_size, 2).type_as(V)
        diag = torch.diag_embed(torch.cat((ones, det_UV.unsqueeze(1)), dim=1)) # B x 3 x 3

        # Compute rotation and translation (T_trg_src)
        #R = torch.bmm(V, torch.bmm(diag, U.transpose(2, 1).contiguous()))
        #t = centroid_trg - torch.bmm(R, centroid_src)
        R_trg_src = torch.bmm(U, torch.bmm(diag, V.transpose(2, 1).contiguous()))                        # B x 3 x 3
        t_trg_src_insrc = centroid_src - torch.bmm(R_trg_src.transpose(2,1).contiguous(), centroid_trg)  # B x 3 x 1
        t_src_trg_intrg = -R_trg_src.bmm(t_trg_src_insrc)

        # Create translation matrix
        zeros = torch.zeros(batch_size, 1, 3).type_as(V)                         # B x 1 x 3
        one = torch.ones(batch_size,1, 1).type_as(V)                             # B x 1 x 1
        trans_cols = torch.cat([t_src_trg_intrg, one], dim=1)    # B x 4 x 1
        rot_cols = torch.cat([R_trg_src, zeros], dim=1)                          # B x 4 x 3
        T_trg_src = torch.cat([rot_cols, trans_cols], dim=2)                     # B x 4 x 4

        # Convert from camera to robot frame
        T_s_v = self.T_s_v.expand(batch_size, 4, 4)
        T_trg_src = se3_inv(T_s_v).bmm(T_trg_src).bmm(T_s_v)

        return T_trg_src

# class RANSACSlow(nn.Module):
#     def __init__(self, inlier_threshold, error_tolerance, num_iterations):
#         super(RANSACSlow, self).__init__()
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
#         # Detach the computations from the graph.
#         pts_3D_src = keypoints_3D_src.detach()
#         pts_3D_trg = keypoints_3D_trg.detach()
#         pts_2D_trg = keypoints_2D_trg.detach()
#
#         max_num_inliers = torch.zeros(batch_size).type_as(pts_3D_src)
#         max_fraction_inliers = torch.zeros(batch_size).type_as(pts_3D_src)
#         inliers = torch.zeros(batch_size, n_points).type_as(pts_3D_src)
#
#         # Only use 3D points with valid diparity.
#         valid_pts = valid_pts_src & valid_pts_trg    # B x 1 x N
#         valid_pts_3D = valid_pts.expand(batch_size, 4, n_points).detach()
#         valid_pts_2D = valid_pts.reshape(batch_size, n_points, 1).expand(batch_size, n_points, 2).detach()
#
#         for b in range(batch_size):
#             '''print("++++++++++++++++++++++++STARTING BATCH {} +++++++++++++++++++++++++++++++++++++++++".format(b))
#             print("min z src before: {}".format(torch.min(pts_3D_src[b, 2, :][valid_pts[b, 0, :]])), flush=True)
#             print("max z src before: {}".format(torch.max(pts_3D_src[b, 2, :][valid_pts[b, 0, :]])), flush=True)
#             print("min z trg before: {}".format(torch.min(pts_3D_trg[b, 2, :][valid_pts[b, 0, :]])), flush=True)
#             print("max z trg before: {}".format(torch.max(pts_3D_trg[b, 2, :][valid_pts[b, 0, :]])), flush=True)
#             '''
#
#             pts_3D_src_batch = pts_3D_src[b, :, :][valid_pts_3D[b, :, :] == 1].reshape(4, -1).unsqueeze(0)
#             pts_3D_trg_batch = pts_3D_trg[b, :, :][valid_pts_3D[b, :, :] == 1].reshape(4, -1).unsqueeze(0)
#             pts_2D_trg_batch = pts_2D_trg[b, :, :][valid_pts_2D[b, :, :] == 1].reshape(-1, 2).unsqueeze(0)
#             weights_batch = weights.detach()[b, :, :][valid_pts.detach()[b, :, :] == 1].reshape(1, -1).unsqueeze(0)
#
#             '''print("min z src after: {}".format(torch.min(pts_3D_src_batch[0, 2, :])), flush=True)
#             print("max z src after: {}".format(torch.max(pts_3D_src_batch[0, 2, :])), flush=True)
#             print("min z trg after: {}".format(torch.min(pts_3D_trg_batch[0, 2, :])), flush=True)
#             print("max z trg after: {}".format(torch.max(pts_3D_trg_batch[0, 2, :])), flush=True)
#             '''
#
#             n_points_valid = pts_3D_src_batch.size(2)
#
#             i = 0
#             ransac_complete = False
#
#             while (i < self.num_iterations) and (not ransac_complete):
#
#                 # Pick a random subset of 3 point pairs
#                 rand_index = torch.randint(0, n_points_valid, size=(1, 5)).type_as(pts_3D_src).long()
#                 rand_index = rand_index.unsqueeze(1)
#                 rand_pts_3D_src = torch.gather(pts_3D_src_batch, dim=2, index=rand_index.expand(1, 4, 5)) # 1 x 4 x M
#                 rand_pts_3D_trg = torch.gather(pts_3D_trg_batch, dim=2, index=rand_index.expand(1, 4, 5)) # 1 x 4 x M
#                 rand_weights = torch.gather(weights_batch, dim=2, index=rand_index)                       # 1 x 1 x M
#
#                 '''print("src pts: {}".format(rand_pts_3D_src[0, 0:3, :]))
#                 print("trg pts: {}".format(rand_pts_3D_trg[0, 0:3, :]))
#                 print("rank src pts: {}".format(torch.matrix_rank(rand_pts_3D_src[0, 0:3, :])))
#                 print("rank trg pts: {}".format(torch.matrix_rank(rand_pts_3D_trg[0, 0:3, :])), flush=True)
#                 '''
#
#                 # Run SVD
#                 try:
#                     T_trg_src = self.svd(rand_pts_3D_src, rand_pts_3D_trg, rand_weights)
#                     #T_trg_src = self.svd(rand_pts_3D_src, rand_pts_3D_trg, torch.ones(1, 1, 5).type_as(pts_3D_src))
#                 except RuntimeError as e:
#                     print(e)
#                     print("SVD did not converge, re-doing iteration {}".format(i))
#                     print("src pts: {}".format(rand_pts_3D_src[0, 0:3, :]))
#                     print("trg pts: {}".format(rand_pts_3D_trg[0, 0:3, :]))
#                     print("weights: {}".format(rand_weights[0, 0, :]))
#                     print("rank src pts: {}".format(torch.matrix_rank(rand_pts_3D_src[0, 0:3, :])))
#                     print("rank trg pts: {}".format(torch.matrix_rank(rand_pts_3D_trg[0, 0:3, :])), flush=True)
#                     continue
#                     #break
#
#                 # Find number of inliers
#                 T_s_v = self.T_s_v.expand(1, 4, 4)
#                 T_cam = T_s_v.bmm(T_trg_src).bmm(se3_inv(T_s_v))
#                 pts_3D_trg_est = T_cam.bmm(pts_3D_src[b, :, :].unsqueeze(0))
#                 pts_2D_trg_est = self.stereo_cam.camera_model(pts_3D_trg_est)[:, 0:2, :].transpose(2,1).contiguous()
#
#                 err_pts = torch.norm(pts_2D_trg[b, :, :].unsqueeze(0) - pts_2D_trg_est, dim=2) # 1 x N
#                 err_pts_small = err_pts < 20.0
#                 err_pts_small[valid_pts[b, :, :] == 0] = 0
#
#                 num_inliers = torch.sum(err_pts_small)
#                 fraction_inliers = float(num_inliers) / n_points_valid
#
#                 if fraction_inliers > self.inlier_threshold:
#                     ransac_complete = True
#
#                 if num_inliers > max_num_inliers[b]:
#                     max_num_inliers[b] = num_inliers
#                     max_fraction_inliers[b] = fraction_inliers
#                     inliers[b, :] = err_pts_small[0, :]
#
#                 i += 1
#
#             #if not ransac_complete:
#             #    print("RANSAC finished with inliers below threshold: {} out of {} for batch ind {}".format(max_num_inliers[b], n_points_valid, b))
#
#         return inliers, max_fraction_inliers


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