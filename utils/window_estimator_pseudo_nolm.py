import torch
import torch.nn.functional as F
import time
import numpy as np
import collections
import cpp_wrappers.cpp_steam.build.steampy_lm as steampy_lm

class WindowEstimatorPseudoNoLM:
    """
    Class that implements a windowed batch optimization with steam
    """

    # Initialization
    def __init__(self, window_size, sf_temp, mah_thresh):
        # config
        # TODO: use json config
        self.window_size = window_size
        self.sf_temp = sf_temp
        self.mah_thresh = mah_thresh

        # measurement variables
        self.mframes_deq = collections.deque()
        self.frame_counter = 0

    # Added new frame
    def add_frame(self, coords, descs, weights, T_k0=torch.Tensor()):

        # pop front frame if window size will be exceeded
        if len(self.mframes_deq) == self.window_size:
            # pop left (front)
            self.mframes_deq.popleft()

        # def __init__(self, coords, descs, weights, frame_id):
        self.mframes_deq.append(MeasurementFrame(coords, descs, weights, self.frame_counter))
        self.frame_counter += 1


    def isWindowFull(self):
        return len(self.mframes_deq) == self.window_size

    def optimize(self):

        meas_list = []
        ref_list = []
        weight_list = []
        pose_list = []
        vel_list = []
        # loop through every frame
        for k, frame in enumerate(self.mframes_deq):

            pose_list += [frame.pose]
            vel_list += [frame.vel]
            if k == 0:
                ref_frame = frame
                continue

            w_12 = ref_frame.descs@frame.descs.T
            _, id_12 = torch.max(w_12, dim=1)   # gives indices of 2 (size of 1)
            _, id_21 = torch.max(w_12, dim=0)   # gives indices of 1 (size of 2)
            mask1 = torch.eq(id_21[id_12], torch.arange(id_12.__len__(), device=w_12.device))
            mask1_ind = torch.nonzero(mask1, as_tuple=False).squeeze(1)  # ids of 1 that are successful matches

            w_12_sf = F.softmax(w_12*self.sf_temp, dim=1)
            meas = w_12_sf@frame.coords
            meas_weights, _ = frame.convertWeightMat(w_12_sf@frame.weights)

            meas_list += [meas[mask1_ind, :].detach().cpu().numpy()]
            weight_list += [meas_weights[mask1_ind, :, :].detach().cpu().numpy()]
            ref_list += [ref_frame.coords[mask1_ind, :].detach().cpu().numpy()]

        poses = np.stack(pose_list, axis=0)
        vels = np.stack(vel_list, axis=0)
        # steampy_lm.run_steam_lm(meas_list, match_list, weight_list, poses, vels, lm_coords)
        steampy_lm.run_steam_no_lm(meas_list, ref_list, weight_list, poses, vels, False, np.zeros(0))

        for k, frame in enumerate(self.mframes_deq):
            frame.pose = poses[k, :, :]
            frame.vel = vels[k, :]

    def loss(self, T_k0, use_gt, cov_diag_list):
        pass

    def getFirstPose(self):
        return self.mframes_deq[0].pose, self.mframes_deq[0].frame_id

    def se3_inv(self, Tf):
        Tinv = np.zeros_like(Tf)
        Tinv[:3, :3] = Tf[:3, :3].T
        Tinv[:3, 3:] = -Tf[:3, :3].T@Tf[:3, 3:]
        Tinv[3, 3] = 1
        return Tinv

    def ransac_svd(self, points1, points2, thresh):
        reflect = torch.eye(3, device=points1.device)
        reflect[2, 2] = -1
        T_21 = torch.eye(4, device=points1.device)
        pick_size = 5
        total_points = points1.shape[0]
        best_inlier_size = 0

        for i in range(100):
            # randomly select query
            query = np.random.randint(0, high=total_points, size=pick_size)

            # centered
            mean1 = points1[query, :].mean(dim=0, keepdim=True)
            mean2 = points2[query, :].mean(dim=0, keepdim=True)
            diff1 = points1[query, :] - mean1
            diff2 = points2[query, :] - mean2

            # svd
            H = torch.matmul(diff1.T, diff2)
            u, s, v = torch.svd(H)
            r = torch.matmul(v, u.T)
            r_det = torch.det(r)
            if r_det < 0:
                v = torch.matmul(v, reflect)
                r = torch.matmul(v, u.T)
            R_21 = r
            t_21 = torch.matmul(-r, mean1.T) + mean2.T

            points2tran = points2@R_21 - (R_21.T@t_21).T
            error = points1 - points2tran
            error = torch.sum(error ** 2, 1)
            inliers = torch.where(error < thresh ** 2)
            if inliers[0].size(0) > best_inlier_size:
                best_inlier_size = inliers[0].size(0)
                best_inlier_ids = inliers[0]

            if best_inlier_size > 0.8*total_points:
                break

        # svd with inliers
        query = best_inlier_ids
        mean1 = points1[query, :].mean(dim=0, keepdim=True)
        mean2 = points2[query, :].mean(dim=0, keepdim=True)
        diff1 = points1[query, :] - mean1
        diff2 = points2[query, :] - mean2

        # svd
        H = torch.matmul(diff1.T, diff2)
        u, s, v = torch.svd(H)
        r = torch.matmul(v, u.T)
        r_det = torch.det(r)
        if r_det < 0:
            v = torch.matmul(v, reflect)
            r = torch.matmul(v, u.T)
        R_21 = r
        t_21 = torch.matmul(-r, mean1.T) + mean2.T

        T_21[:3, :3] = R_21
        T_21[:3, 3:] = t_21

        return T_21, query

    def gt_inliers(self, points1, points2, T_21, thresh):
        points1_in_2 = points1@T_21[:3, :3].T + T_21[:3, 3].unsqueeze(0)

        error = points2 - points1_in_2
        error = torch.sum(error ** 2, 1)
        # inliers = torch.where(error < thresh ** 2)
        inliers = torch.nonzero(error < thresh ** 2, as_tuple=False).squeeze(1)

        return inliers

class MeasurementFrame:
    """
    Class that implements a measurement frame
    """

    # Initialization
    def __init__(self, coords, descs, weights, frame_id):
        self.coords = coords
        self.descs = descs
        self.weights = weights

        self.pose = np.eye(4, dtype=np.float32)
        self.vel = np.zeros((6), dtype=np.float32)
        self.frame_id = frame_id

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
