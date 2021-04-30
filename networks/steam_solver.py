import numpy as np
import torch
import cpp.build.SteamSolver as steamcpp
from utils.utils import convert_to_weight_matrix

class SteamSolver():
    """
        This class provides a simple to use python wrapper around our boost-python/c++/steam optimization code.
        Given matched keypoint coordiante, target coordinates, and match weights over a sliding window, the
        optimize() method will use STEAM to find the best set of transformation matrices and body velocity vectors
        to fit the given data.
    """

    def __init__(self, config):
        # parameters
        self.sliding_flag = False   # should always be false during training
        self.batch_size = config['batch_size']
        self.window_size = config['window_size']
        self.gpuid = config['gpuid']
        self.log_det_thres_flag = config['steam']['log_det_thres_flag']
        self.log_det_thres_val = config['steam']['log_det_thres_val']
        self.log_det_topk = config['steam']['log_det_topk']
        self.flip_y = config['flip_y']
        self.dataset = config['dataset']
        self.T_aug = []
        # state variables
        self.poses = np.tile(np.expand_dims(np.expand_dims(np.eye(4, dtype=np.float32), 0), 0),
                             (self.batch_size, self.window_size, 1, 1))  # B x W x 4 x 4
        self.vels = np.zeros((self.batch_size, self.window_size, 6), dtype=np.float32)  # B x W x 6
        self.poses_sp = np.tile(np.expand_dims(np.expand_dims(np.expand_dims(np.eye(4, dtype=np.float32), 0), 0), 0),
                                (self.batch_size, self.window_size - 1, 12, 1, 1))  # B x (W-1) x 12 x 4 x 4
        # steam solver (c++)
        self.solver_cpp = steamcpp.SteamSolver(config['steam']['time_step'], self.window_size)
        qc_diag = np.array(config['qc_diag']).reshape(6, 1)
        self.solver_cpp.setQcInv(qc_diag)
        if config['steam']['use_ransac']:
            self.solver_cpp.useRansac()
            self.solver_cpp.setRansacVersion(config['steam']['ransac_version'])
        if config['steam']['use_ctsteam']:
            self.solver_cpp.useCTSteam()
        self.sigmapoints_flag = (config['steam']['expect_approx_opt'] == 1)
        self.T_sv = np.eye(4, dtype=np.float32)
        k = 0
        for i in range(3):
            self.T_sv[i, 3] = config['steam']['ex_translation_vs_in_s'][i]
            for j in range(3):
                self.T_sv[i, j] = config['steam']['ex_rotation_sv'][k]
                k += 1
        self.solver_cpp.setExtrinsicTsv(self.T_sv)

    def optimize(self, keypoint_coords, pseudo_coords, match_weights, keypoint_ints, time_tgt, time_src,
                 t_ref_tgt, t_ref_src):
        """
            keypoint_coords: B*(W-1)x400x2
            pseudo_coords: B*(W-1)x400x2
            match_weights: B*(W-1)xSx400
        """
        self.poses = np.tile(np.expand_dims(np.expand_dims(np.eye(4, dtype=np.float32), 0), 0),
                             (self.batch_size, self.window_size, 1, 1))  # B x W x 4 x 4
        self.vels = np.zeros((self.batch_size, self.window_size, 6), dtype=np.float32)  # B x W x 6
        self.poses_sp = np.tile(np.expand_dims(np.expand_dims(np.expand_dims(np.eye(4, dtype=np.float32), 0), 0), 0),
                                (self.batch_size, self.window_size - 1, 12, 1, 1))  # B x (W-1) x 12 x 4 x 4
        if self.sliding_flag:
            self.solver_cpp.slideTraj()
        else:
            self.solver_cpp.resetTraj()
        num_points = keypoint_coords.size(1)
        zeros_vec = np.zeros((num_points, 1), dtype=np.float32)

        R_tgt_src = np.zeros((self.batch_size, self.window_size, 3, 3), dtype=np.float32)
        t_src_tgt_in_tgt = np.zeros((self.batch_size, self.window_size, 3, 1), dtype=np.float32)
        # loop through each batch
        for b in range(self.batch_size):
            i = b * (self.window_size-1)    # first index of window
            points1 = []
            points2 = []
            times1 = []
            times2 = []
            weights = []
            t_refs = []
            # loop for each window frame
            for w in range(i, i + self.window_size - 1):
                # filter by zero intensity patches
                ids = torch.nonzero(keypoint_ints[w, 0] > 0, as_tuple=False).squeeze(1)
                # points must be list of N x 3
                points1_temp = pseudo_coords[w, ids].detach().cpu().numpy()
                points2_temp = keypoint_coords[w, ids].detach().cpu().numpy()
                zeros_vec_temp = zeros_vec[ids.cpu()]
                # weights must be list of N x 3 x 3
                weights_temp, weights_d = convert_to_weight_matrix(match_weights[w, :, ids].T, w, self.T_aug)
                # threshold on log determinant
                if self.log_det_thres_flag:
                    ids = torch.nonzero(torch.sum(weights_d[:, 0:2], dim=1) > self.log_det_thres_val,
                                        as_tuple=False).squeeze().detach().cpu()
                    if ids.squeeze().nelement() <= self.log_det_topk:
                        print('Warning: Log det threshold output less than specified top k.')
                        _, ids = torch.topk(torch.sum(weights_d[:, 0:2], dim=1), self.log_det_topk, largest=True)
                        ids = ids.squeeze().detach().cpu()
                else:
                    ids = np.arange(weights_temp.size(0)).squeeze()
                # append
                points1 += [np.concatenate((points1_temp[ids], zeros_vec_temp[ids]), 1)]
                points2 += [np.concatenate((points2_temp[ids], zeros_vec_temp[ids]), 1)]
                weights += [weights_temp[ids].detach().cpu().numpy()]
                times1 += [time_src[w].cpu().numpy().squeeze()]
                times2 += [time_tgt[w].cpu().numpy().squeeze()]
                if w == i:
                    t_refs.append(t_ref_src[w, 0].cpu().item())
                t_refs.append(t_ref_tgt[w, 0].cpu().item())
            # solver
            timestamps1, timestamps2 = self.getApproxTimeStamps(points1, points2, times1, times2)
            self.solver_cpp.setMeas(points2, points1, weights, timestamps2, timestamps1, t_refs)
            self.solver_cpp.optimize()
            # get pose output
            self.solver_cpp.getPoses(self.poses[b])
            self.solver_cpp.getVelocities(self.vels[b])
            # sigmapoints output
            if self.sigmapoints_flag:
                self.solver_cpp.getSigmapoints2N(self.poses_sp[b])
            # set output
            R_tgt_src[b] = self.poses[b, :, :3, :3]
            t_src_tgt_in_tgt[b] = self.poses[b, :, :3, 3:4]

        return torch.from_numpy(R_tgt_src).to(self.gpuid), torch.from_numpy(t_src_tgt_in_tgt).to(self.gpuid)

    def getApproxTimeStamps(self, points1, points2, times1, times2):
        azimuth_step = (2 * np.pi) / 400
        # Helper to ensure angle is within [0, 2 * PI)
        def wrapto2pi(phi):
            if phi < 0:
                return phi + 2 * np.pi
            elif phi >= 2 * np.pi:
                return phi - 2 * np.pi
            else:
                return phi
        timestamps1 = []
        timestamps2 = []
        for i in range(len(points1)):
            for j in range(2):
                if j == 0:
                    p = points1[i]
                    times = times1[i]
                else:
                    p = points2[i]
                    times = times2[i]
                point_times = []
                for k in range(p.shape[0]):
                    x = p[k, 0]
                    y = p[k, 1]
                    if self.flip_y:
                        y *= -1
                    phi = np.arctan2(y, x)
                    phi = wrapto2pi(phi)
                    time_idx = phi / azimuth_step
                    t1 = times[int(np.floor(time_idx))]
                    idx2 = int(np.ceil(time_idx))
                    t2 = times[idx2 if idx2 < 400 else 399]
                    # interpolate to get slightly more precise timestamp
                    ratio = time_idx % 1
                    t = int(t1 + ratio * (t2 - t1))
                    point_times.append(t)
                point_times = np.array(point_times)
                if j == 0:
                    timestamps1.append(point_times)
                else:
                    timestamps2.append(point_times)
        return timestamps1, timestamps2
