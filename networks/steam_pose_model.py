import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from networks.unet import UNet
from networks.keypoint import Keypoint
from networks.softmax_ref_matcher import SoftmaxRefMatcher
import cpp.build.SteamSolver as steamcpp
from utils.utils import convert_to_radar_frame

class SteamPoseModel(torch.nn.Module):
    """
        This model performs unsupervised radar odometry using a sliding window optimization with window
        size between 2 (regular frame-to-frame odometry) and 4. A python wrapper around the STEAM library is used
        to optimize for the best set of transformations over the sliding window.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpuid = config['gpuid']
        self.unet = UNet(config)
        self.keypoint = Keypoint(config)
        self.softmax_matcher = SoftmaxRefMatcher(config)
        self.solver = SteamSolver(config)
        self.patch_size = config['networks']['keypoint_block']['patch_size']
        self.mah_thres = config['steam']['mah_thres']
        self.patch_mean_thres = config['steam']['patch_mean_thres']
        self.expect_approx_opt = config['steam']['expect_approx_opt']
        self.topk_backup = config['steam']['topk_backup']
        self.prior_learn_const = config['steam']['prior_learn_const']

        self.q = nn.Parameter(torch.log(torch.tensor([0.37, 0.043, 0.13, 0.0073, 0.0076, 0.002], requires_grad=True)))

    def forward(self, batch):
        data = batch['data'].to(self.gpuid)
        mask = batch['mask'].to(self.gpuid)

        detector_scores, weight_scores, desc = self.unet(data)
        keypoint_coords, keypoint_scores, keypoint_desc = self.keypoint(detector_scores, weight_scores, desc)
        pseudo_coords, match_weights, tgt_ids = self.softmax_matcher(keypoint_scores, keypoint_desc, desc)
        keypoint_coords = keypoint_coords[tgt_ids]

        pseudo_coords_xy = convert_to_radar_frame(pseudo_coords, self.config)
        keypoint_coords_xy = convert_to_radar_frame(keypoint_coords, self.config)
        # rotate back if augmented
        if 'T_aug' in batch:
            T_aug = torch.stack(batch['T_aug'], dim=0).to(self.gpuid)
            keypoint_coords_xy = torch.matmul(keypoint_coords_xy, T_aug[:, :2, :2].transpose(1, 2))
            self.solver.T_aug = batch['T_aug']
        else:
            self.solver.T_aug = []

        pseudo_coords_xy[:, :, 1] *= -1.0
        keypoint_coords_xy[:, :, 1] *= -1.0

        # binary mask to remove keypoints from 'empty' regions of the input radar scan
        keypoint_ints = self.mask_intensity_filter(mask[tgt_ids])

        R_tgt_src_pred, t_tgt_src_pred = self.solver.optimize(keypoint_coords_xy, pseudo_coords_xy, match_weights,
                                                              keypoint_ints, self.q)

        return {'R': R_tgt_src_pred, 't': t_tgt_src_pred, 'scores': weight_scores, 'tgt': keypoint_coords_xy,
                'src': pseudo_coords_xy, 'match_weights': match_weights, 'keypoint_ints': keypoint_ints,
                'detector_scores': detector_scores, 'tgt_rc': keypoint_coords, 'src_rc': pseudo_coords}

    def mask_intensity_filter(self, data):
        int_patches = F.unfold(data, kernel_size=self.patch_size, stride=self.patch_size)
        keypoint_int = torch.mean(int_patches, dim=1, keepdim=True)  # BW x 1 x num_patches
        return keypoint_int >= self.patch_mean_thres

    def loss(self, src_coords, tgt_coords, match_weights, keypoint_ints, scores, batch):
        point_loss = 0
        logdet_loss = 0
        unweighted_point_loss = 0
        prior_loss = 0

        # loop through each batch
        bcount = 0
        for b in range(self.solver.batch_size):
            bcount += 1
            i = b * (self.solver.window_size-1)    # first index of window
            # loop for each window frame
            for w in range(i, i + self.solver.window_size - 1):
                # filter by zero intensity patches
                ids = torch.nonzero(keypoint_ints[w, 0] > 0, as_tuple=False).squeeze(1)
                if ids.size(0) == 0:
                    print('WARNING: filtering by zero intensity patches resulted in zero keypoints!')
                    continue

                # points must be list of N x 3
                zeros_vec = torch.zeros_like(src_coords[w, ids, 0:1])
                points1 = torch.cat((src_coords[w, ids], zeros_vec), dim=1).unsqueeze(-1)    # N x 3 x 1
                points2 = torch.cat((tgt_coords[w, ids], zeros_vec), dim=1).unsqueeze(-1)    # N x 3 x 1
                weights_mat, weights_d = self.solver.convert_to_weight_matrix(match_weights[w, :, ids].T, w)
                ones = torch.ones(weights_mat.shape).to(self.gpuid)

                # get R_21 and t_12_in_2
                R_21 = torch.from_numpy(self.solver.poses[b, w-i+1][:3, :3]).to(self.gpuid).unsqueeze(0)
                t_12_in_2 = torch.from_numpy(self.solver.poses[b, w-i+1][:3, 3:4]).to(self.gpuid).unsqueeze(0)
                error = points2 - (R_21 @ points1 + t_12_in_2)
                mah2_error = error.transpose(1, 2) @ weights_mat @ error

                # error threshold
                errorT = self.mah_thres**2
                if errorT > 0:
                    ids = torch.nonzero(mah2_error.squeeze() < errorT, as_tuple=False).squeeze()
                else:
                    ids = torch.arange(mah2_error.size(0))

                if ids.squeeze().nelement() <= 1:
                    print('Warning: MAH threshold output has 1 or 0 elements.')
                    error2 = error.transpose(1, 2) @ error
                    k = min(len(error2.squeeze()), self.topk_backup)
                    _, ids = torch.topk(error2.squeeze(), k, largest=False)

                # squared mah error
                if self.expect_approx_opt == 0:
                    # only mean
                    point_loss += torch.mean(error[ids].transpose(1, 2) @ weights_mat[ids] @ error[ids])
                    unweighted_point_loss += torch.mean(error[ids].transpose(1, 2) @ ones[ids] @ error[ids])
                elif self.expect_approx_opt == 1:
                    # sigmapoints
                    Rsp = torch.from_numpy(self.solver.poses_sp[b, w-i, :, :3, :3]).to(self.gpuid).unsqueeze(1)  # s x 1 x 3 x 3
                    tsp = torch.from_numpy(self.solver.poses_sp[b, w-i, :, :3, 3:4]).to(self.gpuid).unsqueeze(1) # s x 1 x 3 x 1

                    points2 = points2[ids].unsqueeze(0)  # 1 x n x 3 x 1
                    points1_in_2 = Rsp @ (points1[ids].unsqueeze(0)) + tsp  # s x n x 3 x 1
                    error = points2 - points1_in_2  # s x n x 3 x 1
                    temp = torch.sum(error.transpose(2, 3) @ weights_mat[ids].unsqueeze(0) @ error, dim=0)/Rsp.size(0)
                    unweighted_point_loss += torch.mean(error.transpose(2, 3) @ ones[ids].unsqueeze(0) @ error)
                    point_loss += torch.mean(temp)
                else:
                    raise NotImplementedError('Steam loss method not implemented!')

                # log det (ignore 3rd dim since it's a constant)
                logdet_loss -= torch.mean(torch.sum(weights_d[ids, 0:2], dim=1))

                # motion prior
                Qk_inv = self.construct_Qk_inv()
                outer_prod = np.zeros((12, self.solver.window_size - 1), dtype=np.float32)
                self.solver.solver_cpp.getMotionErrorOuterProd(0, outer_prod)
                outer_prod = torch.from_numpy(outer_prod).to(self.gpuid)

                prior_loss += torch.trace(torch.matmul(Qk_inv, outer_prod)) + 2*(self.q[0] + self.q[1] + self.q[5])

        # average over batches
        if bcount > 0:
            point_loss /= (bcount * (self.solver.window_size - 1))
            logdet_loss /= (bcount * (self.solver.window_size - 1))
            prior_loss /= (bcount * (self.solver.window_size - 1))
        total_loss = point_loss + logdet_loss + self.prior_learn_const*prior_loss
        dict_loss = {'point_loss': point_loss, 'logdet_loss': logdet_loss,
                     'unweighted_point_loss': unweighted_point_loss, 'prior_loss': prior_loss}
        return total_loss, dict_loss

    def construct_Qk_inv(self):
        qc_diag = torch.exp(self.q.detach())
        qc_diag[0] = torch.exp(self.q[0])
        qc_diag[1] = torch.exp(self.q[1])
        qc_diag[5] = torch.exp(self.q[5])
        Qc_inv = torch.diag(1.0/qc_diag)
        return kronecker_product(self.solver.Qdt_inv.to(Qc_inv.device), Qc_inv)

def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2

class SteamSolver():
    """
        TODO
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
        self.T_aug = []
        self.time_step = config['steam']['time_step']
        dti1 = 1.0/self.time_step
        dti2 = dti1*dti1
        dti3 = dti1*dti2
        self.Qdt_inv = torch.tensor([[12.0*dti3, -6.0*dti2], [-6.0*dti2, 4.0*dti1]], dtype=torch.float32)
        # z weight value
        # 9.2103 = log(1e4), 1e4 is inverse variance of 1cm std dev
        self.z_weight = 9.2103
        # state variables
        self.poses = np.tile(np.expand_dims(np.expand_dims(np.eye(4, dtype=np.float32), 0), 0),
                             (self.batch_size, self.window_size, 1, 1))  # B x W x 4 x 4
        self.vels = np.zeros((self.batch_size, self.window_size, 6), dtype=np.float32)  # B x W x 6
        self.poses_sp = np.tile(np.expand_dims(np.expand_dims(np.expand_dims(np.eye(4, dtype=np.float32), 0), 0), 0),
                                (self.batch_size, self.window_size - 1, 12, 1, 1))  # B x (W-1) x 12 x 4 x 4
        # steam solver (c++)
        self.solver_cpp = steamcpp.SteamSolver(self.time_step,
                                               self.window_size, config['steam']['zero_vel_prior'])
        self.sigmapoints_flag = (config['steam']['expect_approx_opt'] == 1)

    def optimize(self, keypoint_coords, pseudo_coords, match_weights, keypoint_ints, q):
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
            weights = []
            # loop for each window frame
            for w in range(i, i + self.window_size - 1):
                # filter by zero intensity patches
                ids = torch.nonzero(keypoint_ints[w, 0] > 0, as_tuple=False).squeeze(1)
                ids_cpu = ids.cpu()
                # points must be list of N x 3
                points1_temp = pseudo_coords[w, ids].detach().cpu().numpy()
                points2_temp = keypoint_coords[w, ids].detach().cpu().numpy()
                zeros_vec_temp = zeros_vec[ids_cpu]
                # weights must be list of N x 3 x 3
                weights_temp, weights_d = self.convert_to_weight_matrix(match_weights[w, :, ids].T, w)
                # threshold on log determinant
                if self.log_det_thres_flag:
                    ids = torch.nonzero(torch.sum(weights_d[:, 0:2], dim=1) > self.log_det_thres_val, as_tuple=False).squeeze().detach().cpu()
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
            # solver
            self.solver_cpp.setQcInv(torch.exp(q.unsqueeze(1)).detach().cpu().numpy())
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

    def convert_to_weight_matrix(self, w, window_id):
        """
            w: n_points x S
            This function converts the S-dimensional weights estimated for each keypoint into
            a 2x2 weight (inverse covariance) matrix for each keypoint.
            If S = 1, Wout = diag(exp(w), exp(w), 1e4)
            If S = 3, use LDL^T to obtain 2x2 covariance, place on top-LH corner. 1e4 bottom-RH corner.
        """
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
            A2x2 = L @ D @ L.transpose(1, 2)

            if self.T_aug:  # if list is not empty
                Rot = self.T_aug[window_id].to(w.device)[:2, :2].unsqueeze(0)
                A2x2 = Rot.transpose(1, 2) @ A2x2 @ Rot

            A = torch.zeros(w.size(0), 3, 3, device=w.device)
            A[:, 0:2, 0:2] = A2x2
            A[:, 2, 2] = torch.exp(torch.tensor(self.z_weight))
            d = torch.ones(w.size(0), 3, device=w.device)*self.z_weight
            d[:, 0:2] = w[:, 1:]
        else:
            assert False, "Weight scores should be dim 1 or 3"

        return A, d
