import torch
import numpy as np
from networks.unet import UNet
from networks.keypoint import Keypoint
from networks.softmax_matcher import SoftmaxMatcher
import cpp.build.SteamSolver as SteamCpp

class SteamPoseModel(torch.nn.Module):
    """
        TODO
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.gpuid = config['gpuid']

        self.unet = UNet(config)
        self.keypoint = Keypoint(config)
        self.softmax_matcher = SoftmaxMatcher(config)
        self.solver = SteamSolver(config)

    def forward(self, batch):
        data = batch['data'].to(self.gpuid)

        detector_scores, weight_scores, desc = self.unet(data)

        keypoint_coords, keypoint_scores, keypoint_desc = self.keypoint(detector_scores, weight_scores, desc)

        pseudo_coords, match_weights = self.softmax_matcher(keypoint_scores, keypoint_desc, weight_scores, desc)

        # R_tgt_src_pred, t_tgt_src_pred = self.svd(keypoint_coords, pseudo_coords, match_weights)
        self.solver.optimize(keypoint_coords, pseudo_coords, match_weights)

        # return {'R': R_tgt_src_pred, 't': t_tgt_src_pred, 'scores': weight_scores, 'src': keypoint_coords,
        #         'tgt': pseudo_coords, 'match_weights': match_weights}

        return


class SteamSolver():
    """
        TODO
    """
    def __init__(self, config):
        # parameters
        self.sliding_flag = config['steam']['sliding_flag']
        self.batch_size = config['batch_size']
        self.window_size = config['window_size']

        # state variables
        self.poses = np.tile(
            np.expand_dims(np.expand_dims(np.eye(4, dtype=np.float32), 0), 0),
            (self.batch_size, self.window_size, 1, 1))  # B x W x 4 x 4
        self.vels = np.zeros((self.batch_size, self.window_size, 6), dtype=np.float32)  # B x W x 6

        # steam solver (c++)
        self.solver_cpp = SteamCpp.SteamSolver(config['steam']['time_step'], self.window_size)

    def optimize(self, keypoint_coords, pseudo_coords, match_weights):
        # keypoint_coords (BxW) x 400 x 2
        # weights (BxW/2) x 1 x 400
        # pseudo_coords (BxW/2) x 400 x 2
        num_points = keypoint_coords.size(1)
        zeros_vec = np.zeros((num_points, 1), dtype=np.float32)
        identity_weights = np.tile(np.expand_dims(np.eye(3, dtype=np.float32), 0), (num_points, 1, 1))

        # loop through each batch
        for b in range(self.batch_size):
            # TODO: This implementation only works for window_size = 2
            assert self.window_size == 2, "Currently only implemented for window of 2."

            # points must be list of N x 3
            id = b*self.window_size
            points1 = keypoint_coords[id].detach().cpu().numpy()
            points2 = pseudo_coords[id].detach().cpu().numpy()
            # np.concatenate((points1, zeros_vec), 1)

            # weights must be list of N x 3 x 3
            weights = torch.exp(match_weights[id, 0]).unsqueeze(-1).unsqueeze(-1).detach().cpu().numpy()*identity_weights

            # solver
            self.solver_cpp.resetTraj()
            self.solver_cpp.setMeas([np.concatenate((points2, zeros_vec), 1)],
                                    [np.concatenate((points1, zeros_vec), 1)], weights)
            self.solver_cpp.optimize()

