import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from networks.unet import UNet
from networks.keypoint import Keypoint, normalize_coords
from networks.softmax_ref_matcher import SoftmaxRefMatcher
import cpp.build.SteamSolver as steamcpp
from utils.utils import convert_to_radar_frame


class SteamPoseModel(torch.nn.Module):
    """
        TODO
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpuid = config['gpuid']
        self.cart_pixel_width = config['cart_pixel_width']
        self.cart_resolution = config['cart_resolution']
        self.unet = UNet(config)
        self.keypoint = Keypoint(config)
        # self.softmax_matcher = SoftmaxMatcher(config)
        self.softmax_matcher = SoftmaxRefMatcher(config)
        self.solver = SteamSolver(config)
        self.patch_size = config['networks']['keypoint_block']['patch_size']
        self.border = config['steam']['border']
        self.min_abs_vel = config['steam']['min_abs_vel']
        self.mah_thresh = config['steam']['mah_thresh']
        self.nms_thresh = config['steam']['nms_thresh']
        self.relu_detector = nn.ReLU()
        self.zero_int_detector = config['steam']['zero_int_detector']
        self.expect_approx_opt = config['steam']['expect_approx_opt']

    def forward(self, batch):
        data = batch['data'].to(self.gpuid)
        mask = batch['mask'].to(self.gpuid)

        detector_scores, weight_scores, desc = self.unet(data)

        if self.zero_int_detector:
            detector_scores = self.relu_detector(detector_scores)
            detector_scores *= mask

        keypoint_coords, keypoint_scores, keypoint_desc = self.keypoint(detector_scores, weight_scores, desc)

        pseudo_coords, match_weights, key_ids = self.softmax_matcher(keypoint_scores, keypoint_desc, weight_scores, desc)
        keypoint_coords = keypoint_coords[key_ids]

        pseudo_coords_xy = convert_to_radar_frame(pseudo_coords, self.cart_pixel_width, self.cart_resolution,
                                                  self.gpuid)
        keypoint_coords_xy = convert_to_radar_frame(keypoint_coords, self.cart_pixel_width, self.cart_resolution,
                                                    self.gpuid)
        pseudo_coords_xy[:, :, 1] *= -1.0
        keypoint_coords_xy[:, :, 1] *= -1.0

        # binary masks
        keypoint_ints_zif = self.zero_intensity_filter(mask[key_ids]) >= self.solver.zero_int_thresh
        keypoint_ints_bf = self.border_filter(keypoint_coords)
        keypoint_ints = keypoint_ints_zif * keypoint_ints_bf

        R_tgt_src_pred, t_tgt_src_pred = self.solver.optimize(keypoint_coords_xy, pseudo_coords_xy, match_weights,
                                                              keypoint_ints)

        return {'R': R_tgt_src_pred, 't': t_tgt_src_pred, 'scores': weight_scores, 'tgt': keypoint_coords_xy,
                'src': pseudo_coords_xy, 'match_weights': match_weights, 'keypoint_ints': keypoint_ints,
                'detector_scores': detector_scores, 'tgt_rc': keypoint_coords, 'src_rc': pseudo_coords, 'key_ids': key_ids}

    def loss(self, keypoint_coords, pseudo_coords, match_weights, keypoint_ints, scores, batch):
        point_loss = 0
        logdet_loss = 0
        mask_loss = 0
        mask = batch['mask'].to(self.gpuid)

        # loop through each batch
        bcount = 0
        for b in range(self.solver.batch_size):
            # check average velocity
            if np.mean(np.sqrt(np.sum(self.solver.vels[b]*self.solver.vels[b], axis=1))) < self.min_abs_vel:
                continue
            bcount += 1
            i = b * (self.solver.window_size-1)    # first index of window

            # loop for each window frame
            for w in range(i, i + self.solver.window_size - 1):
                # filter by zero intensity and/or border
                ids = torch.nonzero(keypoint_ints[w, 0] > 0, as_tuple=False).squeeze(1)

                # points must be list of N x 3
                points1 = src_coords[w, ids].T    # 2 x N
                points2 = tgt_coords[w, ids].T  # 2 x N
                weights = match_weights[w, :, ids]   # 1 x N

                # get R_21 and t_12_in_2
                R_21 = torch.from_numpy(self.solver.poses[b, w-i+1][:2, :2]).to(self.gpuid)
                t_12_in_2 = torch.from_numpy(self.solver.poses[b, w-i+1][:2, 3:4]).to(self.gpuid)
                error = points2 - (R_21 @ points1 + t_12_in_2)
                #error2 = torch.sum(error * error * torch.exp(weights), dim=0)
                error2 = torch.sum(error * error * weights, dim=0)

                # error threshold
                errorT = min(self.mah_thresh ** 2, self.nms_thresh * self.nms_thresh * torch.max(error2))
                if self.mah_thresh > 0:
                    ids = torch.nonzero(error2.squeeze() < errorT, as_tuple=False).squeeze()
                else:
                    ids = torch.arange(error2.squeeze().size(0))

                # squared mah error
                if self.expect_approx_opt == 0:
                    # only mean
                    #point_loss += torch.mean(torch.sum(error[:, ids] * error[:, ids] * torch.exp(weights[:, ids]), dim=0))
                    point_loss += torch.mean(torch.sum(torch.abs(error[:, ids]) * weights[:, ids], dim=0))
                elif self.expect_approx_opt == 1:
                    # sigmapoints
                    Rsp = torch.from_numpy(self.solver.poses_sp[b, w-i, :, :2, :2]).to(self.gpuid).unsqueeze(1)  # s x 1 x 2 x 2
                    tsp = torch.from_numpy(self.solver.poses_sp[b, w-i, :, :2, 3:4]).to(self.gpuid).unsqueeze(1) # s x 1 x 2 x 1

                    points2 = points2[:, ids].T.unsqueeze(0).unsqueeze(-1)  # 1 x n x 2 x 1
                    points1_in_2 = Rsp@(points1[:, ids].T.unsqueeze(0).unsqueeze(-1)) + tsp  # s x n x 2 x 1
                    error = points2 - points1_in_2  # s x n x 2 x 1
                    #temp = torch.sum(error*error*torch.exp(weights[:, ids].unsqueeze(-1).unsqueeze(-1)), dim=0).squeeze(-1)/12.0
                    temp = torch.sum(torch.abs(error)*weights[:, ids].unsqueeze(-1).unsqueeze(-1), dim=0).squeeze(-1)/12.0
                    point_loss += torch.mean(torch.sum(temp, dim=1))
                else:
                    raise NotImplementedError('Steam loss method not implemented!')

                # log det
                #logdet_loss -= 3 * torch.mean(weights[:, ids])
                logdet_loss -= 3 * torch.log(torch.mean(weights[:, ids]))

                # mask loss
                bceloss = torch.nn.BCELoss()
                mask_loss += bceloss(scores[w], mask[w])

        # average over batches
        if bcount > 0:
            point_loss /= bcount
            logdet_loss /= bcount
        total_loss = point_loss + logdet_loss + mask_loss
        dict_loss = {'point_loss': point_loss, 'logdet_loss': logdet_loss, 'mask_loss': mask_loss}
        return total_loss, dict_loss

    def zero_intensity_filter(self, data):
        int_patches = F.unfold(data, kernel_size=(self.patch_size, self.patch_size),
                               stride=(self.patch_size, self.patch_size))  # N x patch_elements x num_patches
        keypoint_int = torch.mean(int_patches, dim=1, keepdim=True)

        return keypoint_int

    def border_filter(self, keypoint_coords):
        # self.cart_pixel_width
        width = self.cart_pixel_width
        border = self.border
        keypoint_int = (keypoint_coords[:, :, 0] >= border) * (keypoint_coords[:, :, 0] <= width - border) \
                       * (keypoint_coords[:, :, 1] >= border) * (keypoint_coords[:, :, 1] <= width - border)
        return keypoint_int.unsqueeze(1)


class SteamSolver():
    """
        TODO
    """

    def __init__(self, config):
        # parameters
        # self.sliding_flag = config['steam']['sliding_flag']   # doesn't do anything yet
        self.batch_size = config['batch_size']
        self.window_size = config['window_size']
        self.gpuid = config['gpuid']

        # state variables
        self.poses = np.tile(
            np.expand_dims(np.expand_dims(np.eye(4, dtype=np.float32), 0), 0),
            (self.batch_size, self.window_size, 1, 1))  # B x W x 4 x 4
        self.vels = np.zeros((self.batch_size, self.window_size, 6), dtype=np.float32)  # B x W x 6
        self.poses_sp = np.tile(
            np.expand_dims(np.expand_dims(np.expand_dims(np.eye(4, dtype=np.float32), 0), 0), 0),
            (self.batch_size, self.window_size - 1, 12, 1, 1))  # B x (W-1) x 12 x 4 x 4

        # steam solver (c++)
        self.solver_cpp = steamcpp.SteamSolver(config['steam']['time_step'], self.window_size)
        self.zero_int_thresh = config['steam']['zero_int_thresh']
        self.weight_thresh_mult = config['steam']['weight_thresh_mult']
        self.sigmapoints_flag = (config['steam']['expect_approx_opt'] == 1)

    def optimize(self, keypoint_coords, pseudo_coords, match_weights, keypoint_ints):
        """optimize for training"""
        # update batch size
        self.batch_size = int(keypoint_coords.size(0) / (self.window_size-1))
        self.poses = np.tile(
            np.expand_dims(np.expand_dims(np.eye(4, dtype=np.float32), 0), 0),
            (self.batch_size, self.window_size, 1, 1))  # B x W x 4 x 4
        self.vels = np.zeros((self.batch_size, self.window_size, 6), dtype=np.float32)  # B x W x 6
        self.poses_sp = np.tile(
            np.expand_dims(np.expand_dims(np.expand_dims(np.eye(4, dtype=np.float32), 0), 0), 0),
            (self.batch_size, self.window_size - 1, 12, 1, 1))  # B x (W-1) x 12 x 4 x 4

        # keypoint_coords B*(W-1) x 400 x 2
        # weights B*(W-1) x 1 x 400
        # pseudo_coords B*(W-1) x 400 x 2
        num_points = keypoint_coords.size(1)
        zeros_vec = np.zeros((num_points, 1), dtype=np.float32)
        identity_weights = np.tile(np.expand_dims(np.eye(3, dtype=np.float32), 0), (num_points, 1, 1))

        R_tgt_src = np.zeros((self.batch_size, self.window_size, 3, 3), dtype=np.float32)
        t_src_tgt_in_tgt = np.zeros((self.batch_size, self.window_size, 3, 1), dtype=np.float32)

        # loop through each batch
        for b in range(self.batch_size):
            i = b * (self.window_size-1)    # first index of window
            points1 = []
            points2 = []
            weights = []
            # loop for each window frame
            for w in range(i, i + self.window_size - 1):
                # filter by zero intensity and/or border
                ids = torch.nonzero(keypoint_ints[w, 0] > 0, as_tuple=False).squeeze(1)
                ids_cpu = ids.cpu()

                # points must be list of N x 3
                points1_temp = pseudo_coords[w, ids].detach().cpu().numpy()
                points2_temp = keypoint_coords[w, ids].detach().cpu().numpy()
                zeros_vec_temp = zeros_vec[ids_cpu]

                # weights must be list of N x 3 x 3
                #torch_weights = torch.exp(match_weights[w, 0, ids])
                torch_weights = match_weights[w, 0, idx]
                weights_temp = torch_weights.view(-1, 1, 1).detach().cpu().numpy() * identity_weights[ids_cpu]

                # weight threshold
                weight_thresh = torch.mean(torch_weights)*self.weight_thresh_mult
                ids = torch.nonzero(torch_weights > weight_thresh, as_tuple=False).squeeze(1)
                ids_cpu = ids.cpu()

                # append
                points1 += [np.concatenate((points1_temp[ids_cpu], zeros_vec_temp[ids_cpu]), 1)]
                points2 += [np.concatenate((points2_temp[ids_cpu], zeros_vec_temp[ids_cpu]), 1)]
                weights += [weights_temp[ids_cpu]]

            # solver
            self.solver_cpp.resetTraj()
            self.solver_cpp.setMeas(points2, points1, weights)
            self.solver_cpp.optimize()

            # get pose output
            self.solver_cpp.getPoses(self.poses[b])
            self.solver_cpp.getVelocities(self.vels[b])

            # sigmapoints output
            if self.sigmapoints_flag:
                self.solver_cpp.getSigmapoints2NP1(self.poses_sp[b])

            # set output
            R_tgt_src[b] = self.poses[b, :, :3, :3]
            t_src_tgt_in_tgt[b] = self.poses[b, :, :3, 3:4]

        return torch.from_numpy(R_tgt_src).to(self.gpuid), torch.from_numpy(t_src_tgt_in_tgt).to(self.gpuid)
