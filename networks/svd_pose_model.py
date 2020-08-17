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
from networks.svd_block import SVDBlock, SVD
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

        self.svd_block = SVD()

        self.sigmoid = torch.nn.Sigmoid()

        # intermediate output
        #self.result_path = 'results/' + self.config['session_name'] + '/intermediate_outputs'
        self.save_dict = {}
        self.save_names = self.config['record']['save_names']
        #self.overwrite_flag = False

    def forward(self, data, epoch):
        '''
        Estimate transform between two frames
        :param data: dictionary containing outputs of getitem function
        :return:
        '''

        # parse data
        geometry_img, images, T_iv, cam_calib = data['geometry'], data['input'], data['T_iv'], data['cam_calib']

        # move to GPU
        geometry_img = geometry_img.cuda()
        images = images.cuda()
        T_iv = T_iv.cuda()

        # Extract features, detector scores and weight scores
        detector_scores, weight_scores, descs = self.unet_block(images)

        # Pass weight scores into Sigmoid func
        weight_scores = self.sigmoid(weight_scores)

        # Use detector scores to compute keypoint locations in 3D along with their weight scores and descs
        keypoints_2D, keypoint_coords, keypoint_descs, keypoint_weights, keypoint_valid = self.keypoint_block(geometry_img,
                                                                                                     descs,
                                                                                                     detector_scores,
                                                                                                     weight_scores)

        src_coords = keypoint_coords[::self.window_size]
        src_valid = keypoint_valid[::self.window_size]
        src_2D = keypoints_2D[::self.window_size]
        src_weights = keypoint_weights[::self.window_size]

        # Normalize src desc based on match type
        if self.match_type == 'zncc':
            src_descs = zn_desc(keypoint_descs[::self.window_size])
        elif self.match_type == 'dp':
            src_descs = F.normalize(keypoint_descs[::self.window_size], dim=1)
        elif self.match_type == 'l2':
            pass
        else:
            assert False, "Cannot normalize because match type is NOT support"

        # TODO: loop if we have more than two frames as input (for cycle loss)

        # Match the points in src frame to points in target frame to generate pseudo points
        pseudo_coords, pseudo_weights, pseudo_descs, pseudo_2D, pseudo_valid = self.softmax_matcher_block(src_coords,
                                                                                 geometry_img[1::self.window_size].view(self.batch_size, 3, -1),
                                                                                 weight_scores[1::self.window_size].view(self.batch_size, 1, -1),
                                                                                 src_descs,
                                                                                 descs[1::self.window_size])

        # # UNCOMMENT this line to do keypoint matching instead of dense matching
        # pseudo_coords, pseudo_weights, pseudo_descs = self.softmax_matcher_block(keypoint_coords[::self.window_size],
        #                                                                          keypoint_coords[1::self.window_size],
        #                                                                          keypoint_weights[1::self.window_size],
        #                                                                          keypoint_descs[::self.window_size],
        #                                                                          keypoint_descs[1::self.window_size])


        # Compute ground truth transform
        T_i_src = T_iv[::self.window_size]
        T_i_tgt = T_iv[1::self.window_size]
        T_tgt_src = se3_inv(T_i_tgt).bmm(T_i_src)
        R_tgt_src = T_tgt_src[:,:3,:3]
        t_tgt_src = T_tgt_src[:,:3, 3]

        # Compute ground truth coordinates for the pseudo points
        pseudo_gt_coords = R_tgt_src.bmm(src_coords) + t_tgt_src.unsqueeze(-1)
        if self.config['dataset']['sensor'] == 'velodyne':
            pseudo_gt_2D = compute_2D_from_3D(pseudo_gt_coords, self.config)
        else:
            pseudo_gt_2D = self.stereo_cam.camera_model(pseudo_gt_coords)[:, 0:2, :].transpose(2,1)  # BxNx2

        # Outlier rejection based on error between predicted pseudo point and ground truth
        valid_err = torch.ones(keypoint_valid.size()).type_as(keypoint_valid)
        if self.config['networks']['outlier_rejection']['on']:
            if self.config['networks']['outlier_rejection']['type'] == '3D':
                # Find outliers with large keypoint errors in 3D
                err = torch.norm(pseudo_coords - pseudo_gt_coords, dim=1) # B x N
                valid_err = err < self.config['networks']['outlier_rejection']['threshold']
                valid_err = valid_err.unsqueeze(1) # B x 1 x N
            elif self.config['networks']['outlier_rejection']['type'] == '2D':
                # Find outliers with large keypoint errors in 2D
                err = torch.norm(pseudo_2D - pseudo_gt_2D, dim=2)  # B x N
                valid_err = err < self.config['networks']['outlier_rejection']['threshold']
                valid_err = valid_err.unsqueeze(1) # B x 1 x N
            else:
                assert False, "Outlier rejection type must be 2D or 3D"


        ###############
        # Compute loss
        ###############

        loss_dict = {}
        loss = 0

        ###############
        # Keypoint loss
        ###############

        if ('keypoint_2D' in self.config['loss']['types']) or ('keypoint_3D' in self.config['loss']['types']):

            if self.config['dataset']['sensor'] == 'velodyne':
                # remove keypoints that fall on gap pixels
                self.valid_idx = torch.sum(keypoint_coords[::self.window_size] ** 2, dim=1) != 0

                # compute loss term
                if 'keypoint_3D' in self.config['loss']['types']:
                    keypoint_loss = self.Keypoint_loss(pseudo_coords,
                                                   pseudo_gt_coords,
                                                   inliers=True)
                else:
                    keypoint_w = 0.001
                    keypoint_loss = keypoint_w * self.Keypoint_2D_loss(pseudo_2D, pseudo_gt_2D)

            else:
                # compute loss term
                if 'keypoint_3D' in self.config['loss']['types']:
                    valid = src_valid & pseudo_valid & valid_err                 # B x 1 x N
                    valid = valid.expand(self.batch_size, 3, valid_err.size(2))  # B x 3 x N
                    keypoint_loss = self.Keypoint_2D_loss(pseudo_coords[valid], pseudo_coords_gt[valid])
                    keypoint_loss *= self.config['loss']['weights']['keypoint_3D']
                else:
                    valid = src_valid & valid_err                                                # B x 1 x N
                    valid = valid.transpose(2, 1).expand(self.batch_size, valid_err.size(2), 2)  # B x N x 2
                    keypoint_loss = self.Keypoint_2D_loss(pseudo_2D[valid], pseudo_2D_gt[valid])
                    keypoint_loss *= self.config['loss']['weights']['keypoint_2D']

            loss += keypoint_loss
            loss_dict['KEY_LOSS'] = keypoint_loss

        ##########
        # SVD loss
        ##########

        if ('svd' in self.config['loss']['types']) and (epoch > self.config['loss']['start_svd_epoch']):

            # # UNCOMMENT these two lines to verify SVD module is working properly with
            # # ground truth matches
            # pseudo_gt_coords = torch.matmul(R_tgt_src, keypoint_coords[::self.window_size]) + t_tgt_src.unsqueeze(-1)
            # R_pred, t_pred = self.svd_block(keypoint_coords[::self.window_size],
            #                                 pseudo_gt_coords,
            #                                 svd_weights)


            # Compute matching pair weights for matching pairs
            svd_weights = self.svd_weight_block(src_descs,
                                                pseudo_descs,
                                                src_weights,
                                                pseudo_weights)

            # # Use SVD to solve for optimal transform
            # R_pred, t_pred = [], []
            # # if torch.sum(torch.isnan(pseudo_coords)):
            # #     np.save('{}/pseudo_coords.npy'.format(self.result_path), pseudo_coords.detach().cpu().numpy())
            # for i in range(svd_weights.size(0)):
            #     R_pred_i, t_pred_i = self.svd_block(keypoint_coords[::self.window_size][i][:, self.valid_idx[i]].unsqueeze(0),
            #                                         pseudo_coords[i][:, self.valid_idx[i]].unsqueeze(0),
            #                                         svd_weights[i][:, self.valid_idx[i]].unsqueeze(0))
            #     R_pred.append(R_pred_i)
            #     t_pred.append(t_pred_i.unsqueeze(0))
            # R_pred = torch.cat(R_pred, dim=0)
            # t_pred = torch.cat(t_pred, dim=0

            # Use SVD to solve for optimal transform
            valid = src_valid & pseudo_valid & valid_err
            svd_weights[valid == 0] = 0.0
            T_tgt_src_pred, R_tgt_src_pred, t_tgt_src_pred = self.svd_block(src_coords, pseudo_coords, svd_weights)

            svd_loss, R_loss, t_loss = self.SVD_loss(R_tgt_src, R_tgt_src_pred, t_tgt_src, t_tgt_src_pred)
            svd_loss_weight = self.config['loss']['weights']['svd']
            loss += svd_loss_weight * svd_loss
            loss_dict['SVD_LOSS'] = svd_loss_weight * svd_loss
            loss_dict['SVD_R_LOSS'] = svd_loss_weight * R_loss
            loss_dict['SVD_t_LOSS'] = svd_loss_weight * t_loss

        loss_dict['LOSS'] = loss

        # save intermediate outputs
        if (len(self.save_names) > 0) and (epoch > self.config['loss']['start_svd_epoch']):

            self.save_dict['T_tgt_src'] = T_trg_src if 'T_tgt_src' in self.save_names else None
            self.save_dict['T_tgt_src_pred'] = T_trg_src_pred if 'T_tgt_src_pred' in self.save_names else None
            self.save_dict['inliers'] = valid_err if 'inliers' in self.save_names else None
        #     # print("Saving {}".format(self.save_names))
        #
        #     self.save_dict['T_i_src'] = T_i_src if 'T_i_src' in self.save_names else None
        #     self.save_dict['T_i_tgt'] = T_i_tgt if 'T_i_tgt' in self.save_names else None
        #     self.save_dict['R_tgt_src'] = R_tgt_src if 'R_tgt_src' in self.save_names else None
        #     self.save_dict['t_tgt_src'] = t_tgt_src if 't_tgt_src' in self.save_names else None
        #     self.save_dict['R_pred'] = R_pred if 'R_pred' in self.save_names else None
        #     self.save_dict['t_pred'] = t_pred if 't_pred' in self.save_names else None
        #     self.save_dict['keypoint_coords'] = keypoint_coords if 'keypoint_coords' in self.save_names else None
        #     self.save_dict['pseudo_gt_coords'] = pseudo_gt_coords if 'pseudo_gt_coords' in self.save_names else None
        #     self.save_dict['pseudo_coords'] = pseudo_coords if 'pseudo_coords' in self.save_names else None
        #     self.save_dict['geometry_img'] = geometry_img if 'geometry_img' in self.save_names else None
        #     self.save_dict['images'] = images if 'images' in self.save_names else None
        #     self.save_dict['T_iv'] = T_iv if 'T_iv' in self.save_names else None
        #     self.save_dict['valid_idx'] = self.valid_idx if 'valid_idx' in self.save_names else None
        #     self.save_dict['keypoints_2D'] = keypoints_2D if 'keypoints_2D' in self.save_names else None
        #     self.save_dict['keypoints_gt_2D'] = keypoints_gt_2D if 'keypoints_gt_2D' in self.save_names else None
        #     self.save_dict['svd_weights'] = svd_weights if 'svd_weights' in self.save_names else None
        #     self.save_dict['detector_scores'] = detector_scores if 'detector_scores' in self.save_names else None
        #     self.save_dict['weight_scores'] = weight_scores if 'weight_scores' in self.save_names else None
        #     self.save_dict['pseudo_2D'] = pseudo_2D if 'pseudo_2D' in self.save_names else None

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
            inlier_thresh = 4.0 ** 2
            inlier_idx = torch.sum(e ** 2, dim=1) < inlier_thresh
            if inliers:
                inlier_idx = self.valid_idx * inlier_idx
            else:
                inlier_idx = self.valid_idx
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
        return svd_loss, R_loss, t_loss

    # def print_loss(self, loss, epoch, iter):
    #     message = 'e{:03d}-i{:04d} => LOSS={:.6f} KL={:.6f} SL={:.6f} SRL={:.6f} STL={:.6f}'.format(epoch,
    #                                                                           iter,
    #                                                                           loss['LOSS'].item(),
    #                                                                           loss['KEY_LOSS'].item(),
    #                                                                           loss['SVD_LOSS'].item(),
    #                                                                           loss['SVD_R_LOSS'].item(),
    #                                                                           loss['SVD_t_LOSS'].item())
    #     print(message)

    def print_loss(self, loss, epoch, iter):
        message = 'epoch: {} i: {} => '.format(epoch, iter)

        for key in loss:
            message = '{} {}: {:.6f},'.format(message, key, loss[key].item())

        print(message, flush=True)

    def get_pose_error(self):

        T_tgt_src = self.save_dict['T_tgt_src']
        T_tgt_src_pred = self.save_dict['T_tgt_src_pred']

        return se3_log(se3_inv(T_tgt_src).bmm(T_tgt_src_pred))

    def print_inliers(self, epoch, iter):

        print('epoch: {} i: {} => num inliers: {}'.format(epoch, iter,
                                                          torch.sum(self.save_dict['inliers'], dim=2).squeeze()))

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

    def return_save_dict(self):
        return self.save_dict