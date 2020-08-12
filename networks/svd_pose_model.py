import random
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from utils.lie_algebra import se3_inv, se3_log, se3_exp
from utils.utils import zn_desc, T_inv
from utils.helper_func import compute_2D_from_3D
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
        self.batch_size = batch_size
        self.window_size = window_size # TODO this is hard-fixed at the moment
        self.match_type = config["networks"]["match_type"] # zncc, l2, dp

        # network arch
        self.unet_block = UNetBlock(self.config)

        self.keypoint_block = KeypointBlock(self.config, window_size, batch_size)

        self.softmax_matcher_block = SoftmaxMatcherBlock(self.config)

        self.svd_weight_block = SVDWeightBlock(self.config)

        self.svd_block = SVDBlock(self.config)

        self.sigmoid = torch.nn.Sigmoid()

        # intermediate output
        self.result_path = 'results/' + self.config['session_name'] + '/intermediate_outputs'
        self.save_dict = {}
        self.overwrite_flag = False
        self.save_names = ['T_i_src', 'T_i_tgt', 'R_tgt_src', 't_tgt_src', 'keypoint_coords', 'pseudo_gt_coords',
                           'pseudo_coords', 'geometry_img', 'images', 'T_iv', 'valid_idx',
                           'keypoints_2D', 'keypoints_gt_2D', 'svd_weights', 'detector_scores', 'weight_scores',
                           'pseudo_2D']

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

        # Pass weight scores into Sigmoid func
        weight_scores = self.sigmoid(weight_scores)

        # Use detector scores to compute keypoint locations in 3D along with their weight scores and descs
        keypoints_2D, keypoint_coords, keypoint_descs, keypoint_weights = self.keypoint_block(geometry_img,
                                                                                              descs,
                                                                                              detector_scores,
                                                                                              weight_scores)

        # Match the points in src frame to points in target frame to generate pseudo points
        n_features = descs.size(1)
        pseudo_coords, pseudo_weights, pseudo_descs = self.softmax_matcher_block(keypoint_coords[::self.window_size],
                                                                                 geometry_img[1::self.window_size].view(self.batch_size, 3, -1),
                                                                                 weight_scores[1::self.window_size].view(self.batch_size, 1, -1),
                                                                                 keypoint_descs[::self.window_size],
                                                                                 descs[1::self.window_size].view(self.batch_size, n_features, -1))

        # # UNCOMMENT this line to do keypoint matching instead of dense matching
        # pseudo_coords, pseudo_weights, pseudo_descs = self.softmax_matcher_block(keypoint_coords[::self.window_size],
        #                                                                          keypoint_coords[1::self.window_size],
        #                                                                          keypoint_weights[1::self.window_size],
        #                                                                          keypoint_descs[::self.window_size],
        #                                                                          keypoint_descs[1::self.window_size])

        # Compute 2D keypoints from 3D pseudo coordinates
        pseudo_2D = compute_2D_from_3D(pseudo_coords, self.config)
        # print(pseudo_2D.shape)

        # Normalize src desc based on match type
        if self.match_type == 'zncc':
            src_descs = zn_desc(keypoint_descs[::self.window_size])
        elif self.match_type == 'dp':
            src_descs = F.normalize(keypoint_descs[::self.window_size], dim=1)
        elif self.match_type == 'l2':
            pass
        else:
            assert False, "Cannot normalize because match type is NOT support"


        # Compute ground truth transform
        T_i_src = T_iv[::self.window_size]
        T_i_tgt = T_iv[1::self.window_size]
        T_src_tgt = T_inv(T_i_src) @ T_i_tgt
        R_src_tgt = T_src_tgt[:,:3,:3]
        t_src_tgt = T_src_tgt[:,:3, 3]
        T_tgt_src = T_inv(T_src_tgt)
        R_tgt_src = T_tgt_src[:,:3,:3]
        t_tgt_src = T_tgt_src[:,:3, 3]

        loss_dict = {}
        loss = 0

        ###############
        # Keypoint loss
        ###############
        pseudo_gt_coords = torch.matmul(R_tgt_src, keypoint_coords[::self.window_size]) + t_tgt_src.unsqueeze(-1)
        keypoints_gt_2D = compute_2D_from_3D(pseudo_gt_coords, self.config)

        # remove keypoints that fall on gap pixels
        self.valid_idx = torch.sum(keypoint_coords[::self.window_size] ** 2, dim=1) != 0

        # compute loss term
        # keypoint_loss = self.Keypoint_loss(pseudo_coords,
        #                                    pseudo_gt_coords,
        #                                    inliers=True)
        keypoint_loss = self.Keypoint_2D_loss(pseudo_2D, keypoints_gt_2D)

        ##########
        # SVD loss
        ##########

        # # UNCOMMENT these two lines to verify SVD module is working properly with
        # # ground truth matches
        # pseudo_gt_coords = torch.matmul(R_tgt_src, keypoint_coords[::self.window_size]) + t_tgt_src.unsqueeze(-1)
        # R_pred, t_pred = self.svd_block(keypoint_coords[::self.window_size],
        #                                 pseudo_gt_coords,
        #                                 svd_weights)


        # Compute matching pair weights for matching pairs
        svd_weights = self.svd_weight_block(src_descs,
                                            pseudo_descs,
                                            keypoint_weights[::self.window_size],
                                            pseudo_weights)

        # Use SVD to solve for optimal transform
        R_pred, t_pred = [], []
        for i in range(svd_weights.size(0)):
            R_pred_i, t_pred_i = self.svd_block(keypoint_coords[::self.window_size][i][:, self.valid_idx[i]].unsqueeze(0),
                                                pseudo_coords[i][:, self.valid_idx[i]].unsqueeze(0),
                                                svd_weights[i][:, self.valid_idx[i]].unsqueeze(0))
            R_pred.append(R_pred_i)
            t_pred.append(t_pred_i.unsqueeze(0))
        R_pred = torch.cat(R_pred, dim=0)
        t_pred = torch.cat(t_pred, dim=0)

        svd_loss = self.SVD_loss(R_tgt_src, R_pred, t_tgt_src, t_pred)
        # loss += svd_loss
        loss_dict['SVD_LOSS'] = svd_loss

        # save intermediate outputs
        if len(self.save_names) > 0:
            # print("Saving {}".format(self.save_names))

            self.save_dict['T_i_src'] = T_i_src if 'T_i_src' in self.save_names else None
            self.save_dict['T_i_tgt'] = T_i_tgt if 'T_i_tgt' in self.save_names else None
            self.save_dict['R_tgt_src'] = R_tgt_src if 'R_tgt_src' in self.save_names else None
            self.save_dict['t_tgt_src'] = t_tgt_src if 't_tgt_src' in self.save_names else None
            self.save_dict['keypoint_coords'] = keypoint_coords if 'keypoint_coords' in self.save_names else None
            self.save_dict['pseudo_gt_coords'] = pseudo_gt_coords if 'pseudo_gt_coords' in self.save_names else None
            self.save_dict['pseudo_coords'] = pseudo_coords if 'pseudo_coords' in self.save_names else None
            self.save_dict['geometry_img'] = geometry_img if 'geometry_img' in self.save_names else None
            self.save_dict['images'] = images if 'images' in self.save_names else None
            self.save_dict['T_iv'] = T_iv if 'T_iv' in self.save_names else None
            self.save_dict['valid_idx'] = self.valid_idx if 'valid_idx' in self.save_names else None
            self.save_dict['keypoints_2D'] = keypoints_2D if 'keypoints_2D' in self.save_names else None
            self.save_dict['keypoints_gt_2D'] = keypoints_gt_2D if 'keypoints_gt_2D' in self.save_names else None
            self.save_dict['svd_weights'] = svd_weights if 'svd_weights' in self.save_names else None
            self.save_dict['detector_scores'] = detector_scores if 'detector_scores' in self.save_names else None
            self.save_dict['weight_scores'] = weight_scores if 'weight_scores' in self.save_names else None
            self.save_dict['pseudo_2D'] = pseudo_2D if 'pseudo_2D' in self.save_names else None

        loss += keypoint_loss
        loss_dict['KEY_LOSS'] = keypoint_loss

        loss_dict['LOSS'] = loss

        return loss_dict

    def Keypoint_loss(self, src, target, inliers=True):
        '''
        Compute mean squared loss for keypoint pairs
        :param src: source points Bx3xN
        :param target: target points Bx3xN
        :param inliers:
        :return:
        '''
        e = (src - target) # Bx3xN

        if self.valid_idx is None:
            return torch.mean(torch.sum(e ** 2, dim=1))
        else:
            if inliers:
                inlier_thresh = 1.6 ** 2
                inlier_idx = torch.sum(e ** 2, dim=1) < inlier_thresh
                inlier_idx = self.valid_idx * inlier_idx
            return torch.mean(torch.sum(e ** 2, dim=1)[inlier_idx])

    def Keypoint_2D_loss(self, src, target):
        '''
        Compute mean squared loss for 2D keypoint pairs
        :param src: source points BxNx2
        :param target: target points BxNx2
        :return:
        '''
        loss_fn = torch.nn.MSELoss()
        keypoint_2D_loss = loss_fn(src, target)
        return keypoint_2D_loss


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

    def print_loss(self, loss, epoch, iter):
        message = 'e{:03d}-i{:04d} => LOSS={:.6f} SL={:.6f} KL={:.6f}'.format(epoch,
                                                                              iter,
                                                                              loss['LOSS'].item(),
                                                                              loss['SVD_LOSS'].item(),
                                                                              loss['KEY_LOSS'].item())
        print(message)

    def save_intermediate_outputs(self):
        if len(self.save_dict) > 0 and not self.overwrite_flag:
            if not os.path.exists(self.result_path):
                os.makedirs(self.result_path)
            for key in self.save_dict.keys():
                value = self.save_dict[key]
                if value is not None:
                    np.save('{}/{}.npy'.format(self.result_path, key), value.detach().cpu().numpy())
            self.overwrite_flag = True
        else:
            # print("No intermediate output is saved")
            pass