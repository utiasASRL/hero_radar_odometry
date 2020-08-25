import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.utils import zn_desc, T_inv
from networks.unet_block import UNetBlock
from networks.unetf_block import UNetFBlock
from networks.superpoint_block import SuperpointBlock
from networks.softmax_matcher_block import SoftmaxMatcherBlock
from networks.svd_weight_block import SVDWeightBlock
from networks.svd_block import SVDBlock
from networks.keypoint_block import KeypointBlock

# steam
import cpp_wrappers.cpp_steam.build.steampy_f2f as steampy_f2f

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

        # sobel kernel
        sobelx = torch.tensor([[1., 0., -1.],
                                [2., 0., -2.],
                                [1., 0., -1.]])
        sobely = torch.tensor([[1., 2., 1.],
                                [0., 0., 0.],
                                [-1., -2., -1.]])

        sobelx = sobelx.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        sobely = sobely.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        self.register_buffer('sobelx', sobelx)
        self.register_buffer('sobely', sobely)

        # patch dimesions
        self.patch_height = config['networks']['keypoint_block']['patch_height']
        self.patch_width = config['networks']['keypoint_block']['patch_width']

        # vehicle mask
        if "vehicle" in self.config['networks']['sobel_mask']:
            self.vehicle_mask = self.create_vehicle_mask()

    def forward(self, data):
        '''
        Estimate transform between two frames
        :param data: dictionary containing outputs of getitem function
        :return:
        '''

        # parse data
        geometry_img, images, T_iv, return_mask, canny_edge = data['geometry'], data['input'], data['T_iv'], \
                                                              data['return_mask'], data['canny_edge']

        # move to GPU
        geometry_img = geometry_img.cuda()
        images = images.cuda()
        T_iv = T_iv.cuda()
        return_mask = return_mask.cuda()
        canny_edge = canny_edge.cuda()

        # divide range by 100
        # images[1, :, :] = images[1, :, :]/100.0

        # Extract features, detector scores and weight scores
        detector_scores, weight_scores, descs = self.unet_block(images)

        # Use detector scores to compute keypoint locations in 3D along with their weight scores and descs
        keypoint_coords, keypoint_descs, keypoint_weights, patch_mask = self.keypoint_block(
            geometry_img, return_mask, descs, detector_scores, weight_scores)

        # sobel mask
        if self.config['networks']['sobel_mask']:
            patch_mask = self.sobel_mask(images, patch_mask, canny_edge)

        # Match the points in src frame to points in target frame to generate pseudo points
        # first input is src. Computes pseudo with target
        match_vals = self.softmax_matcher_block(keypoint_coords[::self.window_size],
                                                 keypoint_coords[1::self.window_size],
                                                 keypoint_weights[1::self.window_size],
                                                 keypoint_descs[::self.window_size],
                                                 keypoint_descs[1::self.window_size])

        # compute loss
        if self.config['networks']['loss'] == "gt":
            loss = self.loss(keypoint_coords[::self.window_size], keypoint_coords[1::self.window_size],
                             keypoint_weights[::self.window_size], patch_mask, T_iv)
        elif self.config['networks']['loss'] == "steam":
            loss = self.loss_steam(keypoint_coords[::self.window_size], keypoint_coords[1::self.window_size],
                                   keypoint_weights[::self.window_size], patch_mask)
        else:
            assert False, "Loss must be gt or steam"


        return loss

    def loss(self, src_coords, tgt_coords, weights, patch_mask, T_iv):
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

            # mask
            nr_ids1 = torch.nonzero(patch_mask[batch_i*self.window_size, :, :].squeeze(), as_tuple=False).squeeze()
            nr_ids2 = torch.nonzero(patch_mask[batch_i*self.window_size+1, :, :].squeeze(), as_tuple=False).squeeze()

            # get src points
            points1 = src_coords[batch_i, :, nr_ids1].transpose(0, 1)

            # get tgt points
            w12 = self.softmax_matcher_block.match_vals[batch_i, nr_ids1, :] # N x M
            w12 = w12[:, nr_ids2]
            points2 = F.softmax(w12*self.config['networks']['pseudo_temp'], dim=1)@tgt_coords[batch_i, :, nr_ids2].transpose(0, 1)

            # get weights
            w = weights[batch_i, :, nr_ids1].transpose(0, 1)
            Wmat, d = self.convertWeightMat(w)

            # match consistency
            _, ind2to1 = torch.max(w12, dim=1)  # N
            _, ind1to2 = torch.max(w12, dim=0)  # M
            mask = torch.eq(ind1to2[ind2to1], torch.arange(ind2to1.__len__(), device=ind1to2.device))
            mask_ind = torch.nonzero(mask, as_tuple=False).squeeze()
            points1 = points1[mask_ind, :]
            points2 = points2[mask_ind, :]
            # w = w[mask_ind, :]
            Wmat = Wmat[mask_ind, :, :]
            d = d[mask_ind, :]

            # get gt poses
            src_id = 2*batch_i
            tgt_id = 2*batch_i + 1
            T_21 = self.se3_inv(T_iv[tgt_id, :, :])@T_iv[src_id, :, :]

            if not self.config['networks']['steam_pseudo']:
                points2e = tgt_coords[batch_i, :, nr_ids2].transpose(0, 1)
                points2e = points2e[ind2to1, :]
                points2e = points2e[mask_ind, :]
            else:
                points2e = points2

            # error rejection
            points1_in_2 = points1@T_21[:3, :3].T + T_21[:3, 3].unsqueeze(0)
            if self.config["networks"]["keypoint_loss"]["mah"]:
                error = (points1_in_2 - points2e).unsqueeze(-1)
                mah = error.transpose(1, 2)@Wmat@error
                ids = torch.nonzero(mah.squeeze() < self.config["networks"]["keypoint_loss"]["error_thresh"] ** 2,
                                    as_tuple=False).squeeze()
            else:
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

    def loss_steam(self, src_coords, tgt_coords, weights, patch_mask):
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

            # mask
            # nr_ids = torch.nonzero(torch.sum(points1, dim=1), as_tuple=False).squeeze()
            nr_ids1 = torch.nonzero(patch_mask[batch_i*self.window_size, :, :].squeeze(), as_tuple=False).squeeze()
            nr_ids2 = torch.nonzero(patch_mask[batch_i*self.window_size+1, :, :].squeeze(), as_tuple=False).squeeze()

            # get src points
            points1 = src_coords[batch_i, :, nr_ids1].transpose(0, 1)

            # get tgt points (pseudo)
            w12 = self.softmax_matcher_block.match_vals[batch_i, nr_ids1, :] # N x M
            w12 = w12[:, nr_ids2]
            points2 = F.softmax(w12*self.config['networks']['pseudo_temp'], dim=1)@tgt_coords[batch_i, :, nr_ids2].transpose(0, 1)

            # get weights
            w = weights[batch_i, :, nr_ids1].transpose(0, 1)
            Wmat, d = self.convertWeightMat(w)

            # match consistency
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

            # solve steam problem
            T_21_temp = np.zeros((13, 4, 4), dtype=np.float32)
            pts1_npy = points1.detach().cpu().numpy()
            Wmat_npy = Wmat.detach().cpu().numpy()
            pts2_npy = points2.detach().cpu().numpy()
            if not self.config['networks']['steam_pseudo']:
                points2e = tgt_coords[batch_i, :, nr_ids2].transpose(0, 1)
                points2e = points2e[ind2to1, :]
                points2e = points2e[mask_ind, :]
                pts2_npy = points2e.detach().cpu().numpy()
            else:
                points2e = points2
            steampy_f2f.run_steam_best_match(pts1_npy,
                                             pts2_npy,
                                             Wmat_npy, T_21_temp)
            T_21 = torch.from_numpy(T_21_temp)
            T_21 = T_21.cuda()

            # error rejection
            points1_in_2 = points1@T_21[0, :3, :3].T + T_21[0, :3, 3].unsqueeze(0)
            if self.config["networks"]["keypoint_loss"]["mah"]:
                error = (points1_in_2 - points2e).unsqueeze(-1)
                mah = error.transpose(1, 2)@Wmat@error
                ids = torch.nonzero(mah.squeeze() < self.config["networks"]["keypoint_loss"]["error_thresh"] ** 2,
                                    as_tuple=False).squeeze()
            else:
                error = torch.sum((points1_in_2 - points2) ** 2, dim=1)
                ids = torch.nonzero(error < self.config["networks"]["keypoint_loss"]["error_thresh"] ** 2,
                                    as_tuple=False).squeeze()

            if ids.nelement() <= 1:
                print("WARNING: ELEMENTS LESS THAN 1")
                continue

            for ii in range(12):
                points1_in_2 = points1@T_21[1+ii, :3, :3].T + T_21[1+ii, :3, 3].unsqueeze(0)
                loss += self.weighted_mse_loss(points1_in_2[ids, :],
                                               points2[ids, :],
                                               Wmat[ids, :, :])/12.0

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
        geometry_img, images, T_iv, return_mask, canny_edge = data['geometry'], data['input'], data['T_iv'], \
                                                              data['return_mask'], data['canny_edge']

        # move to GPU
        geometry_img = geometry_img.cuda()
        images = images.cuda()
        # T_iv = T_iv.cuda()
        return_mask = return_mask.cuda()
        canny_edge = canny_edge.cuda()

        # divide range by 100
        # images[1, :, :] = images[1, :, :]/100.0

        # Extract features, detector scores and weight scores
        detector_scores, weight_scores, descs = self.unet_block(images)

        # Use detector scores to compute keypoint locations in 3D along with their weight scores and descs
        keypoint_coords, keypoint_descs, keypoint_weights, patch_mask = self.keypoint_block(geometry_img, return_mask, descs, detector_scores, weight_scores)

        # sobel mask
        if self.config['networks']['sobel_mask']:
            patch_mask = self.sobel_mask(images, patch_mask, canny_edge)

        # Match the points in src frame to points in target frame to generate pseudo points
        # first input is src. Computes pseudo with target
        match_vals = self.softmax_matcher_block(keypoint_coords[::self.window_size],
                                                                 keypoint_coords[1::self.window_size],
                                                                 keypoint_weights[1::self.window_size],
                                                                 keypoint_descs[::self.window_size],
                                                                 keypoint_descs[1::self.window_size])

        return keypoint_coords[::self.window_size], keypoint_coords[1::self.window_size], keypoint_weights[::self.window_size], patch_mask

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

    def sobel(self, images):
        sobel_x = torch.tensor([[1., 0., -1.],
                                [2., 0., -2.],
                                [1., 0., -1.]], device = images.device)
        sobel_y = torch.tensor([[1., 2., 1.],
                                [0., 0., 0.],
                                [-1., -2., -1.]], device = images.device)

        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(images.size(1), images.size(1), 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(images.size(1), images.size(1), 1, 1)

        outx = F.conv2d(images, sobel_x)
        outy = F.conv2d(images, sobel_y)
        outx = F.pad(outx, (1,1,1,1))
        outy = F.pad(outy, (1,1,1,1))

        return outx, outy

    def sobel_mask(self, images, patch_mask, canny_edge):
        # bitwise or between intensity and range sobel masks
        # bitwise and with result and patch_mask

        pixel_mask = torch.zeros_like(images[:, 0:1, :, :])

        # canny (assume its use is always on its own)
        if "canny" in self.config['networks']['sobel_mask']:
            pixel_mask = (pixel_mask + canny_edge) > 0

        # intensity
        if "intensity" in self.config['networks']['sobel_mask']:
            # assume intensity channel is 0
            sobel_out = F.conv2d(images[:, 0:1, :, :], self.sobelx)
            sobel_out = F.pad(sobel_out, (1,1,1,1))
            sobel_out = sobel_out > self.config['networks']['sobel_int_thresh']
            pixel_mask = (pixel_mask + sobel_out) > 0

        # range
        if "range" in self.config['networks']['sobel_mask']:
            # assume range channel is 1
            sobelx_out = F.conv2d(images[:, 1:2, :, :], self.sobelx)
            sobely_out = F.conv2d(images[:, 1:2, :, :], self.sobely)
            sobelx_out = F.pad(sobelx_out, (1,1,1,1))
            sobely_out = F.pad(sobely_out, (1,1,1,1))
            sobel_out = torch.sqrt(sobelx_out ** 2 + sobely_out ** 2)
            sobel_out = sobel_out > self.config['networks']['sobel_ran_thresh']
            pixel_mask = (pixel_mask + sobel_out) > 0

        # vehicle mask
        if "vehicle" in self.config['networks']['sobel_mask']:
            pixel_mask = pixel_mask*self.vehicle_mask

        # unfold
        mask_patches = F.unfold(pixel_mask.float(), kernel_size=(self.patch_height, self.patch_width),
                                stride=(self.patch_height, self.patch_width))  # B x num_patch_elements x num_patches
        mask_patches = torch.sum(mask_patches, dim=1).unsqueeze(1) > 0
        out_mask = patch_mask*mask_patches

        return out_mask

    def create_vehicle_mask(self):
        # vehicle mask
        # TODO: Hardcoded for 64 x 720 images
        vehicle_mask = torch.ones(64, 720)
        vehicle_mask = vehicle_mask.cuda()

        # left/right vertical border
        vborder = 6
        vehicle_mask[:, :vborder+1] = 0
        vehicle_mask[:, -vborder:] = 0

        # bottom-left blob
        vehicle_mask[-23:, :48] = 0
        vehicle_mask[-35:, 24:54] = 0
        vehicle_mask[-15:, 48:82] = 0
        vehicle_mask[-23:, 72:112] = 0

        # bottom-centre blob
        vehicle_mask[-17:, 275:440] = 0

        # bottom-centre-right blob
        vehicle_mask[-27:, 460:533] = 0
        vehicle_mask[-17:, 533:541] = 0

        # bottom-right blob
        vehicle_mask[-20:, 580:655] = 0
        vehicle_mask[-32:, 655:695] = 0
        vehicle_mask[-22:, 695:] = 0

        vehicle_mask = vehicle_mask.unsqueeze(0)
        vehicle_mask = vehicle_mask.unsqueeze(0)

        return vehicle_mask
