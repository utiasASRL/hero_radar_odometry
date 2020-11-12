import os
import torch
import torch.nn.functional as F
import numpy as np

from utils.lie_algebra import se3_log
from networks.unet import UNet
from networks.softmax_matcher import SoftmaxMatcher
from networks.svd import SVD
from networks.keypoint import Keypoint

class SVDPoseModel(torch.nn.Module):
    def __init__(self, config):
        super(SVDPoseModel, self).__init__()

        self.config = config
        self.window_size = config['window_size']
        self.outlier_rejection = config['networks']['outlier_rejection']['on']
        self.outlier_threshold = self.config['networks']['outlier_rejection']['threshold']

        self.unet = UNet(self.config)
        self.keypoint = Keypoint(self.config)
        self.softmax_matcher = SoftmaxMatcher(self.config)
        self.svd = SVD(self.config)

    def forward(self, data):
        input = data['input']
        T_21 = data['T_21']

        if self.config['gpuid'] != "cpu":
            input.cuda()
            T_21.cuda()

        bsz, _, height, width = input.size()
        self.batch_size = bsz / self.window_size

        detector_scores, weight_scores, desc = self.unet(input)

        keypoint_coords, keypoint_scores, keypoint_desc = self.keypoint(detector_scores, weight_scores, desc)

        pseudo_coords, match_weights = self.softmax_matcher(keypoint_scores, keypoint_desc, weight_scores, desc)

        T_tgt_src_pred, R_tgt_src_pred, t_tgt_src_pred = self.svd(keypoint_coords, pseudo_coords, match_weights)

        # Get ground truth transforms
        T_tgt_src = T_21[::self.window_size]
        R_tgt_src = T_tgt_src[:,:3,:3]
        t_tgt_src = T_tgt_src[:,:3, 3]

        loss_dict = {}
        loss = 0

        if ('keypoint_2D' in self.config['loss']['types']):
            # Compute ground truth coordinates for the pseudo points
            pseudo_gt_coords = R_tgt_src.bmm(src_coords) + t_tgt_src.unsqueeze(-1)
            inlier_valid = torch.ones(pseudo_coords.size()).int()
            if self.outlier_rejection:
                err = torch.norm(pseudo_coords - pseudo_gt_coords, dim=1) # B x N
                inlier_valid = err < self.outlier_threshold
                inlier_valid = inlier_valid.unsqueeze(1) # B x 1 x N
            keypoint_w = self.config['loss']['weights']['keypoint_2D']
            keypoint_loss = keypoint_w * self.Keypoint_2D_loss(pseudo_coords, pseudo_gt_coords)
            loss += keypoint_loss
            loss_dict['KEY_LOSS'] = keypoint_loss

        svd_loss, R_loss, t_loss = self.SVD_loss(R_tgt_src, R_tgt_src_pred, t_tgt_src.unsqueeze(-1), t_tgt_src_pred)
        svd_loss_weight = self.config['loss']['weights']['svd']
        loss += svd_loss_weight * svd_loss
        loss_dict['SVD_LOSS'] = svd_loss_weight * svd_loss
        loss_dict['SVD_R_LOSS'] = svd_loss_weight * R_loss
        loss_dict['SVD_t_LOSS'] = svd_loss_weight * t_loss

        loss_dict['LOSS'] = loss

        return loss_dict

    def Keypoint_2D_loss(self, src, target):
        loss_fn = torch.nn.MSELoss()
        keypoint_2D_loss = loss_fn(src, target)
        return keypoint_2D_loss

    def SVD_loss(self, R, R_pred, t, t_pred, rel_w=10.0):
        batch_size = R.size(0)
        alpha = rel_w
        identity = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        loss_fn = torch.nn.MSELoss()
        R_loss = alpha * loss_fn(R_pred.transpose(2,1).contiguous() @ R, identity)
        t_loss = 1.0 * loss_fn(t_pred, t)
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
