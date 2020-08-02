import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from utils.lie_algebra import se3_inv, se3_log, se3_exp
from utils.utils import zn_desc, T_inv
from networks.unet_block import UNetBlock
from networks.softmax_matcher_block import SoftmaxMatcherBlock
from networks.svd_weight_block import SVDWeightBlock
from networks.svd_block import SVDBlock
from networks.keypoint_block import KeypointBlock

class SVDPoseModel(nn.Module):
    def __init__(self, config, window_size, batch_size):
        super(SVDPoseModel, self).__init__()

        # load configs
        self.config = config
        self.window_size = window_size # TODO this is hard-fixed at the moment
        self.match_type = config["networks"]["match_type"] # zncc, l2, dp

        # network arch
        self.unet_block = UNetBlock(self.config)

        self.keypoint_block = KeypointBlock(self.config, window_size, batch_size)

        self.softmax_matcher_block = SoftmaxMatcherBlock(self.config)

        self.svd_weight_block = SVDWeightBlock(self.config)

        self.svd_block = SVDBlock(self.config)

    def forward(self, data):
        '''
        Estimate transform between two frames
        :param data: dictionary containing outputs of getitem function
        :return:
        '''

        # parse data
        geometry_img, images, T_iv = data['geometry'], data['input'], data['T_iv']

        # move to GPU
        geometry_img = geometry_img.cuda()
        images = images.cuda()
        T_iv = T_iv.cuda()

        # Extract features, detector scores and weight scores
        detector_scores, weight_scores, descs = self.unet_block(images)

        # Use detector scores to compute keypoint locations in 3D along with their weight scores and descs
        keypoint_coords, keypoint_descs, keypoint_weights = self.keypoint_block(geometry_img, descs, detector_scores, weight_scores)

        # Match the points in src frame to points in target frame to generate pseudo points
        pseudo_coords, pseudo_weights, pseudo_descs = self.softmax_matcher_block(keypoint_coords[::self.window_size],
                                                                                 keypoint_coords[1::self.window_size],
                                                                                 keypoint_weights[1::self.window_size],
                                                                                 keypoint_descs[::self.window_size],
                                                                                 keypoint_descs[1::self.window_size])

        # Normalize src desc based on match type
        if self.match_type == 'zncc':
            src_descs = zn_desc(keypoint_descs[::self.window_size])
        elif self.match_type == 'dp':
            src_descs = F.normalize(keypoint_descs[::self.window_size], dim=1)
        elif self.match_type == 'l2':
            pass
        else:
            assert False, "Cannot normalize because match type is NOT support"


        # Compute matching pair weights for matching pairs
        svd_weights = self.svd_weight_block(src_descs,
                                            pseudo_descs,
                                            keypoint_weights[::self.window_size],
                                            pseudo_weights)

        # Use SVD to solve for optimal transform
        R_pred, t_pred = self.svd_block(keypoint_coords[::self.window_size],
                                        pseudo_coords,
                                        svd_weights)

        # Compute ground truth transform
        T_i_src = T_iv[::self.window_size]
        T_i_tgt = T_iv[1::self.window_size]
        T_src_tgt = T_inv(T_i_src) @ T_i_tgt
        R_src_tgt = T_src_tgt[:,:3,:3]
        t_src_tgt = T_src_tgt[:,:3, 3]

        loss_dict = {}
        loss = 0

        # SVD loss
        svd_loss = self.SVD_loss(R_src_tgt, R_pred, t_src_tgt, t_pred)
        loss += svd_loss

        loss_dict['SVD_LOSS'] = svd_loss.item()
        loss_dict['LOSS'] = loss.item()

        return loss_dict

    def SVD_loss(self, R, R_pred, t, t_pred, rel_w=10.0):
        '''
        Compute SVD loss
        :param R: ground truth rotation matrix
        :param R_pred: estimated rotation matrix
        :param t: ground truth translation vector
        :param t_pred: estimated translation vector
        :return:
        '''
        batch_size = R.size(0)
        alpha = rel_w
        identity = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        loss_fn = torch.nn.MSELoss()
        R_loss = alpha * loss_fn(R_pred.transpose(2,1) @ R, identity)
        t_loss = 1.0 * loss_fn(t_pred, t)
        #     print(R_loss, t_loss)
        svd_loss = R_loss + t_loss
        return svd_loss

    def print_loss(self, loss):
        message = 'LOSS={:.6f} SL={:.6f}'.format(loss['LOSS'],
                                               loss['SVD_LOSS'])
        print(message)
