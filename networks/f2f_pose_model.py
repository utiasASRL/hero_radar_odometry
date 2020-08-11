import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import zn_desc, T_inv
from networks.unet_block import UNetBlock
from networks.unetf_block import UNetFBlock
from networks.superpoint_block import SuperpointBlock
from networks.softmax_matcher_block import SoftmaxMatcherBlock
from networks.svd_weight_block import SVDWeightBlock
from networks.svd_block import SVDBlock
from networks.keypoint_block import KeypointBlock

class F2FPoseModel(nn.Module):
    def __init__(self, config, window_size, batch_size):
        super(F2FPoseModel, self).__init__()

        # load configs
        self.config = config
        self.window_size = 2 # TODO this is hard-fixed at the moment
        self.match_type = config["networks"]["match_type"] # zncc, l2, dp

        # network arch
        if self.config['networks']['base_net'] == 'unet':
            self.unet_block = UNetBlock(self.config)
        elif self.config['networks']['base_net'] == 'super':
            self.unet_block = SuperpointBlock(self.config)
        elif self.config['networks']['base_net'] == 'unetf':
            self.unet_block = UNetFBlock(self.config)
        else:
            assert False, "Base network should be unet or super"

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
        geometry_img, images, T_iv, return_mask = data['geometry'], data['input'], data['T_iv'], data['return_mask']

        # move to GPU
        geometry_img = geometry_img.cuda()
        images = images.cuda()
        T_iv = T_iv.cuda()
        return_mask = return_mask.cuda()

        # divide range by 100
        # images[1, :, :] = images[1, :, :]/100.0

        # Extract features, detector scores and weight scores
        detector_scores, weight_scores, descs = self.unet_block(images)

        # Use detector scores to compute keypoint locations in 3D along with their weight scores and descs
        keypoint_coords, keypoint_descs, keypoint_weights = self.keypoint_block(geometry_img, return_mask, descs, detector_scores, weight_scores)

        # Match the points in src frame to points in target frame to generate pseudo points
        # first input is src. Computes pseudo with target
        pseudo_coords, pseudo_weights, pseudo_descs = self.softmax_matcher_block(keypoint_coords[::self.window_size],
                                                                                 keypoint_coords[1::self.window_size],
                                                                                 keypoint_weights[1::self.window_size],
                                                                                 keypoint_descs[::self.window_size],
                                                                                 keypoint_descs[1::self.window_size])

        # compute loss
        loss = self.loss(keypoint_coords[::self.window_size], pseudo_coords,
                         keypoint_weights[::self.window_size], T_iv)


        return loss

    def loss(self, src_coords, tgt_coords, weights, T_iv):
        '''
        Compute loss
        :param src_coords: src keypoint coordinates
        :param tgt_coords: tgt keypoint coordinates
        :param weights: weights for match pair
        :param T_iv: groundtruth transform
        :return:
        '''
        loss = 0

        # loop over each batch
        for batch_i in range(src_coords.size(0)):
            b = 1
            # get src points
            points1 = src_coords[batch_i, :, :].transpose(0, 1)

            # check for no returns in src keypoints
            nr_ids = torch.nonzero(torch.sum(points1, dim=1), as_tuple=False).squeeze()
            points1 = points1[nr_ids, :]

            # get tgt points
            points2 = tgt_coords[batch_i, :, nr_ids].transpose(0, 1)

            # get weights
            w = weights[batch_i, :, nr_ids].transpose(0, 1)
            Wmat, d = self.convertWeightMat(w)

            # get gt poses
            src_id = 2*batch_i
            tgt_id = 2*batch_i + 1
            T_21 = self.se3_inv(T_iv[tgt_id, :, :])@T_iv[src_id, :, :]

            # match consistency
            w12 = self.softmax_matcher_block.match_vals[batch_i, nr_ids, :] # N x M
            # w12 = torch.zeros((5, 4))
            _, ind2to1 = torch.max(w12, dim=1)  # N
            _, ind1to2 = torch.max(w12, dim=0)  # M
            mask = torch.eq(ind1to2[ind2to1], torch.arange(ind2to1.__len__(), device=ind1to2.device))
            mask_ind = torch.nonzero(mask, as_tuple=False).squeeze()
            points1 = points1[mask_ind, :]
            points2 = points2[mask_ind, :]
            # w = w[mask_ind, :]
            Wmat = Wmat[mask_ind, :, :]
            d = d[mask_ind, :]

            # error rejection
            points1_in_2 = points1@T_21[:3, :3].T + T_21[:3, 3].unsqueeze(0)
            error = torch.sum((points1_in_2 - points2) ** 2, dim=1)
            ids = torch.nonzero(error < self.config["networks"]["keypoint_loss"]["error_thresh"] ** 2,
                                as_tuple=False).squeeze()

            if ids.nelement() <= 1:
                print("WARNING: ELEMENTS LESS THAN 1")
                continue

            loss += self.weighted_mse_loss(points1_in_2[ids, :],
                                           points2[ids, :],
                                           Wmat[ids, :, :])

            # loss -= torch.mean(3*w[ids, :])
            loss -= torch.mean(torch.sum(d[ids, :], 1))

        return loss

    def weighted_mse_loss(self, data, target, weight):
        # return 3.0*torch.mean(torch.exp(weight) * (data - target) ** 2)
        e = (data - target).unsqueeze(-1)
        return torch.mean(e.transpose(1, 2)@weight@e)

    def print_loss(self, epoch, iter, loss):
        message = '{:d},{:d},{:.6f}'.format(epoch, iter, loss)
        print(message)

    def se3_inv(self, Tf):
        Tinv = torch.zeros_like(Tf)
        Tinv[:3, :3] = Tf[:3, :3].T
        Tinv[:3, 3:] = -Tf[:3, :3].T@Tf[:3, 3:]
        Tinv[3, 3] = 1
        return Tinv

    def forward_encoder_decoder(self, images):
        detector_scores, weight_scores, descs = self.unet_block(images)
        return detector_scores, weight_scores, descs

    def forward_keypoints(self, data):
        # parse data
        geometry_img, images, T_iv, return_mask = data['geometry'], data['input'], data['T_iv'], data['return_mask']

        # move to GPU
        geometry_img = geometry_img.cuda()
        images = images.cuda()
        T_iv = T_iv.cuda()
        return_mask = return_mask.cuda()

        # divide range by 100
        # images[1, :, :] = images[1, :, :]/100.0

        # Extract features, detector scores and weight scores
        detector_scores, weight_scores, descs = self.unet_block(images)

        # Use detector scores to compute keypoint locations in 3D along with their weight scores and descs
        keypoint_coords, keypoint_descs, keypoint_weights = self.keypoint_block(geometry_img, return_mask, descs, detector_scores, weight_scores)

        # Match the points in src frame to points in target frame to generate pseudo points
        # first input is src. Computes pseudo with target
        pseudo_coords, pseudo_weights, pseudo_descs = self.softmax_matcher_block(keypoint_coords[::self.window_size],
                                                                                 keypoint_coords[1::self.window_size],
                                                                                 keypoint_weights[1::self.window_size],
                                                                                 keypoint_descs[::self.window_size],
                                                                                 keypoint_descs[1::self.window_size])

        return keypoint_coords[::self.window_size], pseudo_coords, keypoint_weights[::self.window_size]

    def convertWeightMat(self, w):
        if w.size(1) == 1:
            # scalar weight
            A = torch.zeros(w.size(0), 9, device=w.device)
            A[:, (0, 4, 8)] = torch.exp(w)
            A = A.reshape((-1, 3, 3))

            d = torch.zeros(w.size(0), 3, device=w.device)
            d += w
        elif w.size(1) == 6:
            # 3x3 matrix
            L = torch.zeros(w.size(0), 9, device=w.device)
            L[:, (0, 4, 8)] = 1
            L[:, (3, 6, 7)] = w[:, :3]
            L = L.reshape((-1, 3, 3))

            D = torch.zeros(w.size(0), 9, device=w.device)
            D[:, (0, 4, 8)] = torch.exp(w[:, 3:])
            D = D.reshape((-1, 3, 3))

            A = L@D@L.transpose(1, 2)
            d = w[:, 3:]
        else:
            assert False, "Weight should be dim 1 or 6"

        return A, d
