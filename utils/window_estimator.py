import torch
import numpy as np
import collections
import cpp_wrappers.cpp_steam.build.steampy_lm as steampy_lm

class WindowEstimator:
    """
    Class that implements a windowed batch optimization with steam
    """

    # Initialization
    def __init__(self):
        # config
        # TODO: use json config
        self.window_size = 10
        self.min_solve_obs = 5

        # landmark variables
        self.lm_coords = torch.Tensor()
        # self.lm_descs = torch.Tensor()
        self.lm_obs_count = torch.Tensor()

        # measurement variables
        self.mframes_deq = collections.deque()
        self.frame_counter = 0

    # Added new frame
    def add_frame(self, coords, descs, weights):
        # if first frame
        if len(self.lm_obs_count) == 0:
            # initialize landmarks
            self.lm_coords = coords
            # self.lm_descs = descs
            self.lm_obs_count = torch.ones(coords.size(0), device=coords.device, dtype=torch.long)  # order corresponds to row

            # initialize measurements
            self.mframes_deq.append(
                MeasurementFrame(coords, descs, weights,
                                 torch.arange(coords.size(0), dtype=torch.long, device=coords.device), self.frame_counter))
            self.frame_counter += 1
            return

        # pop front frame if window size will be exceeded
        if len(self.mframes_deq) == self.window_size:
            # pop left (front)
            pop_frame = self.mframes_deq.popleft()
            pop_match_id = pop_frame.match_ids[pop_frame.active_ids]

            # update observation count
            self.lm_obs_count[pop_match_id] -= 1

            # drop landmarks with 0 observations
            self.removeZeroLandmarks()


        # match new frame to current landmarks
        prev_frame = self.mframes_deq[-1]
        w_12 = prev_frame.descs@descs.T
        _, id_12 = torch.max(w_12, dim=1)   # gives indices of 2 (size of 1)
        _, id_21 = torch.max(w_12, dim=0)   # gives indices of 1 (size of 2)
        mask2 = torch.eq(id_12[id_21], torch.arange(id_21.__len__(), device=descs.device))
        mask2_ind = torch.nonzero(mask2, as_tuple=False).squeeze(1)  # successful matches
        new_match_ids = -1*torch.ones(mask2.size(0), dtype=torch.long, device=mask2.device)

        # TODO: optional ransac b/w new frame and previous frame for inliers
        T_21_r, inliers = self.ransac_svd(prev_frame.coords[id_21[mask2_ind], :], coords[mask2_ind, :])
        mask2_ind = mask2_ind[inliers]


        # existing landmarks
        existing_lm = torch.nonzero(prev_frame.match_ids[id_21[mask2_ind]] >= 0, as_tuple=False).squeeze(1)
        new_match_ids[mask2_ind[existing_lm]] = prev_frame.match_ids[id_21[mask2_ind[existing_lm]]]
        self.lm_obs_count[prev_frame.match_ids[id_21[mask2_ind[existing_lm]]]] += 1

        # new landmarks
        if existing_lm.size() < mask2_ind.size():
            new_lm = torch.nonzero(prev_frame.match_ids[id_21[mask2_ind]] == -1, as_tuple=False).squeeze(1)
            new_match_ids[mask2_ind[new_lm]] = torch.arange(self.lm_coords.size(0),
                                                            self.lm_coords.size(0) + new_lm.size(0),
                                                            device=new_lm.device)

            # update previous frame
            prev_frame.match_ids[id_21[mask2_ind[new_lm]]] = torch.arange(self.lm_coords.size(0),
                                                             self.lm_coords.size(0) + new_lm.size(0),
                                                             device=new_lm.device)
            prev_frame.active_ids = torch.nonzero(prev_frame.match_ids >= 0, as_tuple=False).squeeze(1)

            # update landmarks
            self.lm_coords = torch.cat((self.lm_coords, coords[mask2_ind[new_lm], :]), dim=0)
            self.lm_obs_count = torch.cat((self.lm_obs_count,
                                           2*torch.ones(new_lm.size(0), dtype=torch.long, device=new_lm.device)),dim=0)

        new_frame = MeasurementFrame(coords, descs, weights, new_match_ids, self.frame_counter)
        self.mframes_deq.append(new_frame)
        self.frame_counter += 1
        new_frame.pose = T_21_r.detach().cpu().numpy()@prev_frame.pose # T_k0

        # drop landmarks with 0 observations
        # self.removeZeroLandmarks()
        geq3 = torch.nonzero(self.lm_obs_count >= 3, as_tuple=False).squeeze()
        geq4 = torch.nonzero(self.lm_obs_count >= 4, as_tuple=False).squeeze()
        geq5 = torch.nonzero(self.lm_obs_count >= 5, as_tuple=False).squeeze()
        print("# total landmarks: ", len(self.lm_obs_count), ", obs >= 3: ", geq3.nelement(), geq4.nelement(), geq5.nelement())

    def removeZeroLandmarks(self):
        # find landmarks with at least 1 observation
        nonzero_ids = torch.nonzero(self.lm_obs_count, as_tuple=False).squeeze(1)
        if nonzero_ids.nelement() == self.lm_obs_count.size(0):    # TODO: check that this condition is sufficient
            return

        # simply remove
        m = self.lm_obs_count.size(0)
        self.lm_coords = self.lm_coords[nonzero_ids]
        # self.lm_descs = self.lm_descs[nonzero_ids]
        self.lm_obs_count = self.lm_obs_count[nonzero_ids]

        # update id mapping
        remap_ids = -1*torch.ones(m, dtype=torch.long, device=self.lm_obs_count.device)
        remap_ids[nonzero_ids] = torch.arange(nonzero_ids.size(0), dtype=torch.long, device=remap_ids.device)
        for frame in self.mframes_deq:
            frame.match_ids[frame.active_ids] = remap_ids[frame.match_ids[frame.active_ids]]
            frame.active_ids = torch.nonzero(frame.match_ids >= 0, as_tuple=False).squeeze(1)

    def isWindowFull(self):
        return len(self.mframes_deq) == self.window_size

    def optimize(self):
        # determine landmarks with minimum observations
        lm_ids = torch.nonzero(self.lm_obs_count >= self.min_solve_obs, as_tuple=False).squeeze(1)
        remap_ids = -1*torch.ones(self.lm_obs_count.size(0), dtype=torch.long, device=self.lm_obs_count.device)
        remap_ids[lm_ids] = torch.arange(lm_ids.size(0), dtype=torch.long, device=remap_ids.device)
        lm_coords = self.lm_coords[lm_ids, :].detach().cpu().numpy()

        meas_list = []
        match_list = []
        weight_list = []
        pose_list = []
        vel_list = []
        # loop through every frame
        for frame in self.mframes_deq:
            match_list += [remap_ids[frame.match_ids[frame.active_ids]].detach().cpu().numpy()]
            meas_list += [frame.coords[frame.active_ids, :].detach().cpu().numpy()]
            weight_list += [frame.wmats[frame.active_ids, :].detach().cpu().numpy()]
            pose_list += [frame.pose]
            # pose_list += [np.arange(16, dtype=np.float).reshape((4, 4))]
            # print(pose_list[-1])
            vel_list += [frame.vel]

        poses = np.stack(pose_list, axis=0)
        vels = np.stack(vel_list, axis=0)
        steampy_lm.run_steam_lm(meas_list, match_list, weight_list, poses, vels, lm_coords)

        self.lm_coords[lm_ids, :] = torch.from_numpy(lm_coords).cuda()
        count = 0
        for frame in self.mframes_deq:
            frame.pose = poses[count, :, :]
            frame.vel = vels[count, :]
            count += 1

    def getFirstPose(self):
        return self.mframes_deq[0].pose, self.mframes_deq[0].frame_id

    def ransac_svd(self, points1, points2):
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
            inliers = torch.where(error < 0.09)
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

class MeasurementFrame:
    """
    Class that implements a measurement frame
    """

    # Initialization
    def __init__(self, coords, descs, weights, match_ids, frame_id):
        self.coords = coords
        self.descs = descs
        # self.weights = weights
        A, d = self.convertWeightMat(weights)
        self.wmats = A
        self.wds = d
        self.match_ids = match_ids   # landmark id (-1 if not active)
        self.active_ids = torch.nonzero(match_ids >= 0, as_tuple=False).squeeze(1)
        self.pose = np.eye(4)
        self.vel = np.zeros((6))
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

# def add_frame(self, coords, descs, weights):
    #     # if first frame
    #     if len(self.lm_descs) == 0:
    #         # initialize landmarks
    #         self.lm_coords = coords
    #         # self.lm_descs = descs
    #         self.lm_obs_count = torch.ones(coords.size(0))  # order corresponds to row
    #
    #         # initialize measurements
    #         self.mframes_deq.append(MeasurementFrame(coords, descs, weights, torch.arange(coords.size(0))))
    #         return
    #
    #     # pop front frame if window size will be exceeded
    #     if len(self.mframes_deq) == self.window_size:
    #         # pop left (front)
    #         pop_frame = self.mframes_deq.popleft()
    #         pop_match_id = pop_frame.match_ids
    #
    #         # update observation count
    #         self.lm_obs_count[pop_match_id] -= 1
    #
    #         # drop landmarks with 0 observations
    #         self.removeZeroLandmarks()
    #
    #     # TODO: optional ransac b/w new frame and previous frame for inliers
    #
    #     # match new frame to current landmarks
    #     w_lf = self.lm_descs@descs.T
    #     _, id_lf = torch.max(w_lf, dim=1)
    #     _, id_fl = torch.max(w_lf, dim=0)
    #     mask_lf = torch.eq(id_lf[id_fl], torch.arange(id_fl.__len__(), device=descs.device))
    #     mask_lf_ind = torch.nonzero(mask_lf, as_tuple=False).squeeze()
    #     new_frame = MeasurementFrame(coords[mask_lf_ind, :], descs[mask_lf_ind, :],
    #                                  weights[mask_lf_ind, :], id_fl)
    #
    #     # TODO: replace landmark descriptors
    #
    #     # are there measurements remaining?
    #     if mask_lf_ind.size(0) < mask_lf.size(0):
    #         # match remaining measurements to previous frame to spawn new landmarks
    #         mask_lf_ind_ = torch.nonzero(mask_lf == 0, as_tuple=False).squeeze()
    #         prev_frame = self.mframes_deq[-1]
    #         w_12 = prev_frame.descs@descs[mask_lf_ind_, :].T
    #         _, id_12 = torch.max(w_12, dim=1)
    #         _, id_21 = torch.max(w_12, dim=0)
    #         mask_12 = torch.eq(id_12[id_21], torch.arange(id_21.__len__(), device=descs.device))
    #         mask_12_ind = torch.nonzero(mask_12, as_tuple=False).squeeze()
    #
    #     # drop landmarks with 0 observations
    #     self.removeZeroLandmarks()