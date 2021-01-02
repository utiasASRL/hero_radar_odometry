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

        self.cart_pixel_width = config['cart_pixel_width']
        self.cart_resolution = config['cart_resolution']
        if (self.cart_pixel_width % 2) == 0:
            self.cart_min_range = (self.cart_pixel_width / 2 - 0.5) * self.cart_resolution
        else:
            self.cart_min_range = self.cart_pixel_width // 2 * self.cart_resolution

        self.unet = UNet(config)
        self.keypoint = Keypoint(config)
        self.softmax_matcher = SoftmaxMatcher(config)
        self.solver = SteamSolver(config)

    def forward(self, batch):
        data = batch['data'].to(self.gpuid)

        detector_scores, weight_scores, desc = self.unet(data)

        keypoint_coords, keypoint_scores, keypoint_desc = self.keypoint(detector_scores, weight_scores, desc)

        pseudo_coords, match_weights = self.softmax_matcher(keypoint_scores, keypoint_desc, weight_scores, desc)

        # convert to radar frame points
        pseudo_coords = self.convert_to_radar_frame(pseudo_coords)
        keypoint_coords = self.convert_to_radar_frame(keypoint_coords)

        # steam optimization
        R_tgt_src_pred, t_tgt_src_pred = self.solver.optimize(keypoint_coords, pseudo_coords, match_weights)

        return {'R': R_tgt_src_pred, 't': t_tgt_src_pred, 'scores': weight_scores, 'src': keypoint_coords,
                'tgt': pseudo_coords, 'match_weights': match_weights}

    def loss(self, keypoint_coords, pseudo_coords, match_weights):
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
            i = b*self.solver.window_size
            points1 = keypoint_coords[i].T   # 2 x N
            points2 = pseudo_coords[b].T     # 2 x N
            weights = match_weights[b]  # 1 x N

            # get R_21 and t_12_in_2
            R_21 = torch.from_numpy(self.solver.poses[b, 1][:2, :2]).to(self.gpuid)
            t_12_in_2 = torch.from_numpy(self.solver.poses[b, 1][:2, 3:4]).to(self.gpuid)

            # error threshold
            error = points2 - (R_21@points1 + t_12_in_2)
            error2 = torch.sum(error*error, dim=0)
            ids = torch.nonzero(error2.squeeze() < 2.0 ** 2, as_tuple=False).squeeze()

            # squared error
            point_loss += torch.mean(torch.sum(error[:, ids]*error[:, ids]*torch.exp(weights[:, ids]), dim=0))

            # log det
            logdet_loss -= 2*torch.mean(weights)

        # average over batches
        if bcount > 0:
            point_loss /= bcount
            logdet_loss /= bcount
        total_loss = point_loss + logdet_loss
        dict_loss = {'point_loss': point_loss, 'logdet_loss': logdet_loss}
        return total_loss, dict_loss

    def convert_to_radar_frame(self, pixel_coords):
        """Converts pixel_coords (B x N x 2) from pixel coordinates to metric coordinates in the radar frame."""
        B, N, _ = pixel_coords.size()
        R = torch.tensor([[0, -self.cart_resolution], [self.cart_resolution, 0]]).expand(B, 2, 2).to(self.gpuid)
        t = torch.tensor([[self.cart_min_range], [-self.cart_min_range]]).expand(B, 2, N).to(self.gpuid)
        return (torch.bmm(R, pixel_coords.transpose(2, 1)) + t).transpose(2, 1)

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
        self.solver_cpp = SteamCpp.SteamSolver(config['steam']['time_step'], self.window_size)

    def optimize(self, keypoint_coords, pseudo_coords, match_weights):
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

            # points must be list of N x 3
            i = b*self.window_size
            points1 = keypoint_coords[i].detach().cpu().numpy()
            points2 = pseudo_coords[b].detach().cpu().numpy()

            # weights must be list of N x 3 x 3
            weights = torch.exp(match_weights[b, 0]).unsqueeze(-1).unsqueeze(-1).detach().cpu().numpy()*identity_weights

            # solver
            self.solver_cpp.resetTraj()
            self.solver_cpp.setMeas([np.concatenate((points2, zeros_vec), 1)],
                                    [np.concatenate((points1, zeros_vec), 1)], [weights])
            self.solver_cpp.optimize()

            # get pose output
            self.solver_cpp.getPoses(self.poses[b])
            self.solver_cpp.getVelocities(self.vels[b])

            # set output
            R_tgt_src[b, :, :] = self.poses[b, 1, :3, :3]
            t_src_tgt_in_tgt[b, :, :] = self.poses[b, 1, :3, 3:4]

        return torch.from_numpy(R_tgt_src).to(self.gpuid), torch.from_numpy(t_src_tgt_in_tgt).to(self.gpuid)