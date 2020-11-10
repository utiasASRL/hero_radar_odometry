import os

import torch
import torch.nn.functional as F
import numpy as np

from utils.lie_algebra import se3_inv, se3_log, se3_exp, so3_log
from utils.utils import zn_desc, T_inv
from networks.unet_block import UNetBlock
from networks.softmax_matcher_block import SoftmaxMatcherBlock
from networks.svd_weight_block import SVDWeightBlock
from networks.svd_block import SVD
from networks.keypoint_block_radar import KeypointBlock

class SVDPoseModel(torch.nn.Module):
    def __init__(self, config):
        super(SVDPoseModel, self).__init__()
        # load configs
        self.config = config
        self.window_size = config['window_size']
        # network arch
        self.unet_block = UNetBlock(self.config)
        self.keypoint_block = KeypointBlock(self.config)
        self.softmax_matcher_block = SoftmaxMatcherBlock(self.config)
        self.svd_weight_block = SVDWeightBlock(self.config)
        self.svd_block = SVD()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):
        input = data['input']
        T_21 = data['T_21']

        if self.config['gpuid'] != "cpu":
            input.cuda()
            T_21.cuda()

        bsz, _, height, width = input.size()
        self.batch_size = bsz / self.window_size

        detector_scores, weight_scores, descs = self.unet_block(input)
        weight_scores = self.sigmoid(weight_scores)
        # Use detector scores to compute keypoint locations along with their weight scores and descriptors
        keypoint_coords, keypoint_descs, keypoint_weights = self.keypoint_block(descs, detector_scores, weight_scores)
        # Normalize src desc based on match type
        src_descs = zn_desc(keypoint_descs[::self.window_size])
        # TODO: loop if we have more than two frames as input (for cycle loss)

        v_coord, u_coord = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        v_coord = v_coord.reshape(height * width).float()  # HW
        u_coord = u_coord.reshape(height * width).float()
        image_coords = torch.stack((u_coord, v_coord), dim=1)  # HW x 2
        tgt_2D_dense = image_coords.unsqueeze(0).expand(self.batch_size, height * width, 2)
        if self.config['gpuid'] != "cpu":
            tgt_2D_dense.cuda()
        # Match the points in src frame to points in target frame to generate pseudo points
        pseudo_coords, pseudo_weights, pseudo_descs = self.softmax_matcher_block(keypoint_coords[::self.window_size],
                                                                                 keypoint_coords[1::self.window_size],
                                                                                 keypoint_weights[1::self.window_size],
                                                                                 keypoint_descs[::self.window_size],
                                                                                 keypoint_descs[1::self.window_size]
                                                                                 src_descs)
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
            pseudo_gt_2D = self.stereo_cam.camera_model(pseudo_gt_coords, cam_calib, start_ind=1,
                                                        step=self.window_size)[:, 0:2, :].transpose(2,1).contiguous() # BxNx2

        # Outlier rejection based on error between predicted pseudo point and ground truth
        inlier_valid = torch.ones(src_valid.size()).type_as(src_valid)
        if self.config['networks']['outlier_rejection']['on']:
            if self.config['networks']['outlier_rejection']['type'] == '3D':
                # Find outliers with large keypoint errors in 3D
                err = torch.norm(pseudo_coords - pseudo_gt_coords, dim=1) # B x N
                inlier_valid = err < self.config['networks']['outlier_rejection']['threshold']
                inlier_valid = inlier_valid.unsqueeze(1) # B x 1 x N
            elif self.config['networks']['outlier_rejection']['type'] == '2D':
                # Find outliers with large keypoint errors in 2D
                err = torch.norm(pseudo_2D - pseudo_gt_2D, dim=2)  # B x N
                # print("err")
                # print(err)
                inlier_valid = err < self.config['networks']['outlier_rejection']['threshold']
                inlier_valid = inlier_valid.unsqueeze(1) # B x 1 x N
            else:
                assert False, "Outlier rejection type must be 2D or 3D"

        pseudo_gt_valid = (pseudo_gt_2D[:, :, 0] > 0.0) & (pseudo_gt_2D[:, :, 0] < width) & \
                          (pseudo_gt_2D[:, :, 1] > 0.0) & (pseudo_gt_2D[:, :, 1] < height)  # B x N
        pseudo_gt_valid = pseudo_gt_valid.unsqueeze(1)

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
                    valid = src_valid & pseudo_valid & inlier_valid                 # B x 1 x N
                    valid = valid.expand(self.batch_size, 3, inlier_valid.size(2))  # B x 3 x N
                    keypoint_loss = self.Keypoint_2D_loss(pseudo_coords[valid], pseudo_gt_coords[valid])
                    keypoint_loss *= self.config['loss']['weights']['keypoint_3D']
                else:
                    valid = src_valid & inlier_valid                                                # B x 1 x N
                    valid = valid.transpose(2, 1).contiguous().expand(self.batch_size, inlier_valid.size(2), 2)  # B x N x 2
                    keypoint_loss = self.Keypoint_2D_loss(pseudo_2D[valid], pseudo_gt_2D[valid])
                    keypoint_loss *= self.config['loss']['weights']['keypoint_2D']

            loss += keypoint_loss
            loss_dict['KEY_LOSS'] = keypoint_loss

        ##########
        # SVD loss
        ##########

        if ('svd' in self.config['loss']['types']) and (epoch >= self.config['loss']['start_svd_epoch']):

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

            valid = src_valid & pseudo_valid & inlier_valid
            svd_weights[valid == 0] = 0.0

            # # Use SVD to solve for optimal transform
            # R_pred, t_pred = [], []
            # # if torch.sum(torch.isnan(pseudo_coords)):
            # #     np.save('{}/pseudo_coords.npy'.format(self.result_path), pseudo_coords.detach().cpu().numpy())
            # for i in range(svd_weights.size(0)):
            #     R_pred_i, t_pred_i = self.svd_block(src_coords[i].unsqueeze(0),
            #                                         pseudo_coords[i].unsqueeze(0),
            #                                         svd_weights[i].unsqueeze(0))
            #     R_pred.append(R_pred_i)
            #     t_pred.append(t_pred_i.unsqueeze(0))
            # R_pred = torch.cat(R_pred, dim=0)
            # t_pred = torch.cat(t_pred, dim=0)

            # Use SVD to solve for optimal transform
            # valid = src_valid & pseudo_valid & inlier_valid
            # svd_weights[valid == 0] = 0.0
            T_tgt_src_pred, R_tgt_src_pred, t_tgt_src_pred = self.svd_block(src_coords, pseudo_coords, svd_weights)

            # T_tgt_src_test, _, _ = self.svd_block(src_coords, pseudo_gt_coords, svd_weights)
            # print('SVD test diff: {}'.format(T_tgt_src_test.bmm(se3_inv(T_tgt_src))))

            svd_loss, R_loss, t_loss = self.SVD_loss(R_tgt_src, R_tgt_src_pred, t_tgt_src.unsqueeze(-1), t_tgt_src_pred)
            # svd_loss, R_loss, t_loss = self.SVD_loss(R_tgt_src, R_tgt_src_pred, t_tgt_src, t_tgt_src_pred)
            svd_loss_weight = self.config['loss']['weights']['svd']
            loss += svd_loss_weight * svd_loss
            loss_dict['SVD_LOSS'] = svd_loss_weight * svd_loss
            loss_dict['SVD_R_LOSS'] = svd_loss_weight * R_loss
            loss_dict['SVD_t_LOSS'] = svd_loss_weight * t_loss

        loss_dict['LOSS'] = loss

        # save intermediate outputs
        if (len(self.save_names) > 0):

            self.save_dict['inliers'] = inlier_valid if 'inliers' in self.save_names else None
            self.save_dict['src_valid'] = src_valid if 'src_valid' in self.save_names else None
            self.save_dict['pseudo_valid'] = pseudo_valid if 'pseudo_valid' in self.save_names else None
            self.save_dict['pseudo_gt_valid'] = pseudo_gt_valid if 'pseudo_valid' in self.save_names else None
            self.save_dict['T_i_src'] = T_i_src if 'T_i_src' in self.save_names else None
            self.save_dict['T_i_tgt'] = T_i_tgt if 'T_i_tgt' in self.save_names else None

            if epoch >= self.config['loss']['start_svd_epoch']:
                self.save_dict['T_tgt_src'] = T_tgt_src if 'T_tgt_src' in self.save_names else None
                self.save_dict['T_tgt_src_pred'] = T_tgt_src_pred if 'T_tgt_src_pred' in self.save_names else None
                self.save_dict['R_tgt_src'] = R_tgt_src if 'T_tgt_src' in self.save_names else None
                self.save_dict['R_tgt_src_pred'] = R_tgt_src_pred if 'T_tgt_src_pred' in self.save_names else None
                self.save_dict['t_tgt_src'] = t_tgt_src.unsqueeze(-1) if 'T_tgt_src' in self.save_names else None
                self.save_dict['t_tgt_src_pred'] = t_tgt_src_pred if 'T_tgt_src_pred' in self.save_names else None
                self.save_dict['weights'] = svd_weights if 'weights' in self.save_names else None

            self.save_dict['src_coords'] = src_coords if 'src_coords' in self.save_names else None
            self.save_dict['src_2D'] = src_2D if 'src_2D' in self.save_names else None
            self.save_dict['pseudo_2D'] = pseudo_2D if 'pseudo_2D' in self.save_names else None
            self.save_dict['pseudo_gt_2D'] = pseudo_gt_2D if 'pseudo_gt_2D' in self.save_names else None
            self.save_dict['weight_scores_src'] = weight_scores[::self.window_size] if 'weight_scores_src' in self.save_names else None
            self.save_dict['weight_scores_tgt'] = weight_scores[1::self.window_size] if 'weight_scores_tgt' in self.save_names else None
            self.save_dict['max_softmax'] = max_softmax if 'max_softmax' in self.save_names else None

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
        R_loss = alpha * loss_fn(R_pred.transpose(2,1).contiguous() @ R, identity)
        t_loss = 1.0 * loss_fn(t_pred, t)
        #     print(R_loss, t_loss)
        svd_loss = R_loss + t_loss
        return svd_loss, R_loss, t_loss

    def print_loss(self, loss, epoch, iter):
        message = 'epoch: {} i: {} => '.format(epoch, iter)

        for key in loss:
            message = '{} {}: {:.6f},'.format(message, key, loss[key].item())

        print(message + '\n', flush=True)

    def get_pose_error(self):
        T_tgt_src = self.save_dict['T_tgt_src']
        T_tgt_src_pred = self.save_dict['T_tgt_src_pred']
        return se3_log(T_tgt_src) - se3_log(T_tgt_src_pred)

    def get_inliers(self, epoch):
        num_inliers = torch.sum(self.save_dict['inliers'])
        num_nonzero_weights = 0
        if epoch >= self.config['loss']['start_svd_epoch']:
            num_nonzero_weights = torch.sum(self.save_dict['weights'] > 0.0)
        return num_inliers, num_nonzero_weights

    def print_inliers(self, epoch, iter):
        print('epoch: {} i: {}'.format(epoch, iter))
        print('inliers:         {}'.format(torch.sum(self.save_dict['inliers'], dim=2).squeeze()))
        print('src_valid:       {}'.format(torch.sum(self.save_dict['src_valid'], dim=2).squeeze()))
        print('pseudo_valid:    {}'.format(torch.sum(self.save_dict['pseudo_valid'], dim=2).squeeze()))
        print('pseudo_gt_valid: {}'.format(torch.sum(self.save_dict['pseudo_gt_valid'], dim=2).squeeze()))

        if 'max_softmax' in self.save_dict:
            print('max_softmax:     {}'.format(self.save_dict['max_softmax']))

        if epoch >= self.config['loss']['start_svd_epoch']:
            print('weights:         {}'.format(torch.sum(self.save_dict['weights'] > 0.0, dim=2).squeeze()))

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
