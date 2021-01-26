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

    def loss(self, src_coords, tgt_coords, match_weights, keypoint_ints):
        point_loss = 0
        logdet_loss = 0

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
                zeros_vec = torch.zeros_like(src_coords[w, ids, 0:1])
                points1 = torch.cat((src_coords[w, ids], zeros_vec), dim=1).unsqueeze(-1)    # N x 3 x 1
                points2 = torch.cat((tgt_coords[w, ids], zeros_vec), dim=1).unsqueeze(-1)    # N x 3 x 1
                # weights = match_weights[w, :, ids]   # 1 x N
                weights_mat, weights_d = self.solver.convert_to_weight_matrix(match_weights[w, :, ids].T)

                # get R_21 and t_12_in_2
                R_21 = torch.from_numpy(self.solver.poses[b, w-i+1][:3, :3]).to(self.gpuid).unsqueeze(0)
                t_12_in_2 = torch.from_numpy(self.solver.poses[b, w-i+1][:3, 3:4]).to(self.gpuid).unsqueeze(0)
                error = points2 - (R_21 @ points1 + t_12_in_2)
                mah2_error = error.transpose(1, 2)@weights_mat@error

                # error threshold
                if self.mah_thresh > 0:
                    ids = torch.nonzero(mah2_error.squeeze() < self.mah_thresh ** 2, as_tuple=False).squeeze()
                else:
                    ids = torch.arange(mah2_error.size(0))

                # squared mah error
                if self.expect_approx_opt == 0:
                    # only mean
                    point_loss += torch.mean(error[ids].transpose(1, 2)@weights_mat[ids]@error[ids])
                elif self.expect_approx_opt == 1:
                    # sigmapoints
                    Rsp = torch.from_numpy(self.solver.poses_sp[b, w-i, :, :3, :3]).to(self.gpuid).unsqueeze(1)  # s x 1 x 3 x 3
                    tsp = torch.from_numpy(self.solver.poses_sp[b, w-i, :, :3, 3:4]).to(self.gpuid).unsqueeze(1) # s x 1 x 3 x 1

                    points2 = points2[ids].unsqueeze(0)  # 1 x n x 3 x 1
                    points1_in_2 = Rsp@(points1[ids].unsqueeze(0)) + tsp  # s x n x 3 x 1
                    error = points2 - points1_in_2  # s x n x 2 x 1
                    temp = torch.sum(error.transpose(2, 3)@weights_mat[ids].unsqueeze(0)@error, dim=0)/Rsp.size(0)
                    point_loss += torch.mean(temp)
                else:
                    raise NotImplementedError('Steam loss method not implemented!')

                # log det (ignore 3rd dim since it's a constant)
                logdet_loss -= torch.mean(torch.sum(weights_d[ids, 0:2], dim=1))

        # average over batches
        if bcount > 0:
            point_loss /= bcount
            logdet_loss /= bcount
        total_loss = point_loss + logdet_loss
        dict_loss = {'point_loss': point_loss, 'logdet_loss': logdet_loss}
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

        # z weight value
        # 9.2103 = log(1e4), 1e4 is inverse variance of 1cm std dev
        self.z_weight = 9.2103     # TODO: should this be a config parameter?

        # state variables
        self.poses = np.tile(
            np.expand_dims(np.expand_dims(np.eye(4, dtype=np.float32), 0), 0),
            (self.batch_size, self.window_size, 1, 1))  # B x W x 4 x 4
        self.vels = np.zeros((self.batch_size, self.window_size, 6), dtype=np.float32)  # B x W x 6
        self.poses_sp = np.tile(
            np.expand_dims(np.expand_dims(np.expand_dims(np.eye(4, dtype=np.float32), 0), 0), 0),
            (self.batch_size, self.window_size - 1, 12, 1, 1))  # B x (W-1) x 12 x 4 x 4

        # steam solver (c++)
        self.solver_cpp = steamcpp.SteamSolver(config['steam']['time_step'],
                                               self.window_size, config['steam']['zero_vel_prior'])
        self.zero_int_thresh = config['steam']['zero_int_thresh']
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
                weights_temp, _ = self.convert_to_weight_matrix(match_weights[w, :, ids].T)

                # append
                points1 += [np.concatenate((points1_temp, zeros_vec_temp), 1)]
                points2 += [np.concatenate((points2_temp, zeros_vec_temp), 1)]
                weights += [weights_temp.detach().cpu().numpy()]

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

    def convert_to_weight_matrix(self, w):
        if w.size(1) == 1:
            # scalar weight
            A = torch.zeros(w.size(0), 9, device=w.device)
            A[:, (0, 4)] = torch.exp(w)
            A[:, 8] = torch.exp(torch.tensor(self.z_weight))
            A = A.reshape((-1, 3, 3))

            d = torch.zeros(w.size(0), 3, device=w.device)
            d[:, 0:2] += w
            d[:, 2] += self.z_weight
        elif w.size(1) == 3:
            # 2x2 matrix
            L = torch.zeros(w.size(0), 4, device=w.device)
            L[:, (0, 3)] = 1
            L[:, 2] = w[:, 0]
            L = L.reshape((-1, 2, 2))

            D = torch.zeros(w.size(0), 4, device=w.device)
            D[:, (0, 3)] = torch.exp(w[:, 1:])
            D = D.reshape((-1, 2, 2))

            A2x2 = L@D@L.transpose(1, 2)

            A = torch.zeros(w.size(0), 3, 3, device=w.device)
            A[:, 0:2, 0:2] = A2x2
            A[:, 2, 2] = torch.exp(torch.tensor(self.z_weight))
            d = torch.ones(w.size(0), 3, device=w.device)*self.z_weight
            d[:, 0:2] = w[:, 1:]
        else:
            assert False, "Weight should be dim 1 or 3"

        return A, d