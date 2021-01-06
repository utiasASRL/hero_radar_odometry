import torch
import numpy as np
import torch.nn.functional as F
from networks.unet import UNet
from networks.keypoint import Keypoint, normalize_coords
from networks.softmax_matcher import SoftmaxMatcher
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
        self.softmax_matcher = SoftmaxMatcher(config)
        self.solver = SteamSolver(config)
        self.patch_size = config['networks']['keypoint_block']['patch_size']

    def forward(self, batch):
        data = batch['data'].to(self.gpuid)

        detector_scores, weight_scores, desc = self.unet(data)

        keypoint_coords, keypoint_scores, keypoint_desc = self.keypoint(detector_scores, weight_scores, desc)

        pseudo_coords, match_weights = self.softmax_matcher(keypoint_scores, keypoint_desc, weight_scores, desc)

        pseudo_coords = convert_to_radar_frame(pseudo_coords, self.cart_pixel_width, self.cart_resolution, self.gpuid)
        keypoint_coords = convert_to_radar_frame(keypoint_coords, self.cart_pixel_width, self.cart_resolution, self.gpuid)

        keypoint_ints = self.zero_intensity_filter(data, keypoint_coords)

        R_tgt_src_pred, t_tgt_src_pred = self.solver.optimize(keypoint_coords, pseudo_coords, match_weights,
                                                              keypoint_ints)

        return {'R': R_tgt_src_pred, 't': t_tgt_src_pred, 'scores': weight_scores, 'src': keypoint_coords,
                'tgt': pseudo_coords, 'match_weights': match_weights, 'keypoint_ints': keypoint_ints}

    def loss(self, keypoint_coords, pseudo_coords, match_weights, keypoint_ints):
        point_loss = 0
        logdet_loss = 0

        # loop through each batch
        # TODO: currently only implemented for mean approx and window_size = 2
        bcount = 0
        for b in range(self.solver.batch_size):
            # check velocity
            if np.linalg.norm(self.solver.vels[b, 1]) < 0.03:   # TODO: make this a config parameter
                continue
            bcount += 1
            i = b * self.solver.window_size

            # filter by zero intensity
            ids = torch.nonzero(keypoint_ints[i, 0] > self.solver.zero_int_thresh, as_tuple=False).squeeze(1)

            points1 = keypoint_coords[i, ids].T   # 2 x N
            points2 = pseudo_coords[b, ids].T     # 2 x N
            weights = match_weights[b, :, ids]  # 1 x N
            # get R_21 and t_12_in_2
            R_21 = torch.from_numpy(self.solver.poses[b, 1][:2, :2]).to(self.gpuid)
            t_12_in_2 = torch.from_numpy(self.solver.poses[b, 1][:2, 3:4]).to(self.gpuid)
            # error threshold
            error = points2 - (R_21 @ points1 + t_12_in_2)
            error2 = torch.sum(error * error * torch.exp(weights), dim=0)
            ids = torch.nonzero(error2.squeeze() < 4.0 ** 2, as_tuple=False).squeeze()
            # squared error
            point_loss += torch.mean(torch.sum(error[:, ids] * error[:, ids] * torch.exp(weights[:, ids]), dim=0))
            # log det
            logdet_loss -= 3 * torch.mean(weights)

        # average over batches
        if bcount > 0:
            point_loss /= bcount
            logdet_loss /= bcount
        total_loss = point_loss + logdet_loss
        dict_loss = {'point_loss': point_loss, 'logdet_loss': logdet_loss}
        return total_loss, dict_loss

    def zero_intensity_filter(self, data, keypoint_coords):
        # N, _, height, width = data.size()
        # norm_keypoints2D = normalize_coords(keypoint_coords, width, height).unsqueeze(1)
        # keypoint_int = F.grid_sample(data, norm_keypoints2D, mode='bilinear')
        # keypoint_int = keypoint_int.reshape(N, data.size(1), keypoint_coords.size(1))  # N x 1 x n_patch

        int_patches = F.unfold(data, kernel_size=(self.patch_size, self.patch_size),
                            stride=(self.patch_size, self.patch_size))  # N x patch_elements x num_patches
        keypoint_int = torch.mean(int_patches, dim=1, keepdim=True)

        return keypoint_int


class SteamSolver():
    """
        TODO
    """
    def __init__(self, config):
        # parameters
        self.sliding_flag = config['steam']['sliding_flag']
        self.batch_size = config['batch_size']
        self.window_size = config['window_size']
        self.gpuid = config['gpuid']
        # state variables
        self.poses = np.tile(
            np.expand_dims(np.expand_dims(np.eye(4, dtype=np.float32), 0), 0),
            (self.batch_size, self.window_size, 1, 1))  # B x W x 4 x 4
        self.vels = np.zeros((self.batch_size, self.window_size, 6), dtype=np.float32)  # B x W x 6
        # steam solver (c++)
        self.solver_cpp = steamcpp.SteamSolver(config['steam']['time_step'], self.window_size)
        self.zero_int_thresh = config['steam']['zero_int_thresh']

    def optimize(self, keypoint_coords, pseudo_coords, match_weights, keypoint_ints):
        # update batch size
        self.batch_size = int(keypoint_coords.size(0)/self.window_size)
        self.poses = np.tile(
            np.expand_dims(np.expand_dims(np.eye(4, dtype=np.float32), 0), 0),
            (self.batch_size, self.window_size, 1, 1))  # B x W x 4 x 4
        self.vels = np.zeros((self.batch_size, self.window_size, 6), dtype=np.float32)  # B x W x 6
        # keypoint_coords (BxW) x 400 x 2
        # weights (BxW/2) x 1 x 400
        # pseudo_coords (BxW/2) x 400 x 2
        num_points = keypoint_coords.size(1)
        zeros_vec = np.zeros((num_points, 1), dtype=np.float32)
        identity_weights = np.tile(np.expand_dims(np.eye(3, dtype=np.float32), 0), (num_points, 1, 1))

        R_tgt_src = np.zeros((self.batch_size, 3, 3))
        t_src_tgt_in_tgt = np.zeros((self.batch_size, 3, 1))

        # loop through each batch
        for b in range(self.batch_size):
            # TODO: This implementation only works for window_size = 2
            assert self.window_size == 2, "Currently only implemented for window of 2."
            i = b*self.window_size

            # filter by zero intensity
            ids = torch.nonzero(keypoint_ints[i, 0] > self.zero_int_thresh, as_tuple=False).squeeze(1)
            ids_cpu = ids.cpu()

            # points must be list of N x 3
            points1 = keypoint_coords[i, ids].detach().cpu().numpy()
            points2 = pseudo_coords[b, ids].detach().cpu().numpy()

            # weights must be list of N x 3 x 3
            weights = torch.exp(match_weights[b, 0, ids]).view(-1, 1, 1).detach().cpu().numpy()*identity_weights[ids_cpu]

            # solver
            self.solver_cpp.resetTraj()
            self.solver_cpp.setMeas([np.concatenate((points2, zeros_vec[ids_cpu]), 1)],
                                    [np.concatenate((points1, zeros_vec[ids_cpu]), 1)], [weights])
            self.solver_cpp.optimize()
            # get pose output
            self.solver_cpp.getPoses(self.poses[b])
            self.solver_cpp.getVelocities(self.vels[b])
            # set output
            R_tgt_src[b, :, :] = self.poses[b, 1, :3, :3]
            t_src_tgt_in_tgt[b, :, :] = self.poses[b, 1, :3, 3:4]

        return torch.from_numpy(R_tgt_src).to(self.gpuid), torch.from_numpy(t_src_tgt_in_tgt).to(self.gpuid)
