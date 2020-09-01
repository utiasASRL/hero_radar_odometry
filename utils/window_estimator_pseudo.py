import torch
import torch.nn.functional as F
import time
import numpy as np
import collections
import cpp_wrappers.cpp_steam.build.steampy_lm as steampy_lm

class WindowEstimatorPseudo:
    """
    Class that implements a windowed batch optimization with steam
    """

    # Initialization
    def __init__(self, window_size, min_solve_obs, sf_temp, mah_thresh):
        # config
        # TODO: use json config
        self.window_size = window_size
        self.min_solve_obs = min_solve_obs
        self.sf_temp = sf_temp
        self.mah_thresh = mah_thresh

        # landmark variables
        self.lm_coords = torch.Tensor()
        self.lm_obs_count = torch.Tensor()

        # measurement variables
        self.mframes_deq = collections.deque()
        self.frame_counter = 0

    # Added new frame
    def add_frame(self, coords, descs, weights, T_k0=torch.Tensor()):
        # if first frame
        if len(self.mframes_deq) == 0:
            # initialize landmarks
            # self.lm_coords = coords
            # self.lm_descs = descs
            # self.lm_obs_count = torch.ones(coords.size(0), device=coords.device, dtype=torch.long)  # order corresponds to row

            # initialize measurements
            self.mframes_deq.append(
                MeasurementFrame(coords, descs, weights,
                                 -1*torch.ones(coords.size(0), device=coords.device, dtype=torch.long),
                                 self.frame_counter))
            self.frame_counter += 1
            return

        # pop front frame if window size will be exceeded
        if len(self.mframes_deq) == self.window_size:
            # pop left (front)
            pop_frame = self.mframes_deq.popleft()
            pop_match_id = pop_frame.lm_ids

            # update observation count
            self.lm_obs_count[pop_match_id] -= 1

            # drop landmarks with 0 observations
            self.removeZeroLandmarks()

        # TODO: optional ransac b/w new frame and previous frame for inliers

        prev_frame = self.mframes_deq[-1]
        w_12 = prev_frame.descs@descs.T
        mv_12, id_12 = torch.max(w_12, dim=1)   # gives indices of 2 (size of 1)
        mv_21, id_21 = torch.max(w_12, dim=0)   # gives indices of 1 (size of 2)
        mask1 = torch.eq(id_21[id_12], torch.arange(id_12.__len__(), device=descs.device))
        mask1_ind = torch.nonzero(mask1, as_tuple=False).squeeze(1)  # ids of 1 that are successful matches

        w_12_sf = F.softmax(w_12*self.sf_temp, dim=1)
        meas = w_12_sf@coords

        # ransac
        if T_k0.size(0) == 0:
            T_21_r, inliers = self.ransac_svd(prev_frame.coords[mask1_ind, :], meas[mask1_ind, :], self.mah_thresh)
        else:
            T_21_r = T_k0.detach().cpu().numpy()@self.se3_inv(prev_frame.pose.astype(np.float32))
            inliers = self.gt_inliers(prev_frame.coords[mask1_ind, :], meas[mask1_ind, :],
                                      torch.from_numpy(T_21_r).cuda(), self.mah_thresh)

        if inliers.size(0) == 0:
            assert False, "No inliers in frame"
        mask1_ind = mask1_ind[inliers]

        meas = meas[mask1_ind, :]
        meas_weights = w_12_sf@weights
        meas_weights = meas_weights[mask1_ind, :]

        # create new frame
        new_frame = MeasurementFrame(coords, descs, weights,
                                     -1*torch.ones(coords.size(0), device=coords.device, dtype=torch.long),
                                     self.frame_counter)
        if T_k0.size(0) == 0:
            new_frame.pose = T_21_r.detach().cpu().numpy()@prev_frame.pose # T_k0
        else:
            new_frame.pose = T_21_r@prev_frame.pose
        self.frame_counter += 1

        # check if any are of existing landmarks
        existing_lm = torch.nonzero(prev_frame.match_ids[mask1_ind] >= 0, as_tuple=False).squeeze(1)
        if existing_lm.size(0) > 0:
            # update existing landmarks
            lm_ids_to_update = prev_frame.match_ids[mask1_ind[existing_lm]]
            self.lm_obs_count[lm_ids_to_update] += 1

            # add new measurements
            coord_ids_of_new_frame = id_12[mask1_ind[existing_lm]]
            new_frame.addMeas(meas[existing_lm, :], meas_weights[existing_lm, :],
                              coord_ids_of_new_frame, lm_ids_to_update)

        # spawn new landmarks with leftover
        if existing_lm.size(0) < mask1_ind.size(0):
            new_lm = torch.nonzero(prev_frame.match_ids[mask1_ind] == -1, as_tuple=False).squeeze(1)
            new_lm_coords = prev_frame.coords[mask1_ind[new_lm], :]
            new_lm_ids = self.addNewLandmarks(new_lm_coords)

            # update previous frame
            prev_frame.addMeas(prev_frame.coords[mask1_ind[new_lm], :], prev_frame.weights[mask1_ind[new_lm], :],
                               mask1_ind[new_lm], new_lm_ids)

            # update new frame
            coord_ids_of_new_frame = id_12[mask1_ind[new_lm]]
            new_frame.addMeas(meas[new_lm, :], meas_weights[new_lm, :],
                              coord_ids_of_new_frame, new_lm_ids)

        # add frame to deque
        self.mframes_deq.append(new_frame)

        # geq3 = torch.nonzero(self.lm_obs_count >= 3, as_tuple=False).squeeze()
        # geq4 = torch.nonzero(self.lm_obs_count >= 4, as_tuple=False).squeeze()
        # geq5 = torch.nonzero(self.lm_obs_count >= 5, as_tuple=False).squeeze()
        # print("# total landmarks: ", len(self.lm_obs_count), ", obs >= 3: ", geq3.nelement(), geq4.nelement(), geq5.nelement())

    def removeZeroLandmarks(self):
        # find landmarks with at least 1 observation
        nonzero_ids = torch.nonzero(self.lm_obs_count, as_tuple=False).squeeze(1)
        if nonzero_ids.nelement() == self.lm_obs_count.size(0):    # TODO: check that this condition is sufficient
            return

        # simply remove
        m = self.lm_obs_count.size(0)
        self.lm_coords = self.lm_coords[nonzero_ids]
        self.lm_obs_count = self.lm_obs_count[nonzero_ids]

        # update id mapping
        remap_ids = -1*torch.ones(m, dtype=torch.long, device=self.lm_obs_count.device)
        remap_ids[nonzero_ids] = torch.arange(nonzero_ids.size(0), dtype=torch.long, device=remap_ids.device)
        for frame in self.mframes_deq:
            active_ids = torch.nonzero(frame.match_ids >= 0, as_tuple=False).squeeze(1)
            frame.match_ids[active_ids] = remap_ids[frame.match_ids[active_ids]]
            frame.lm_ids = remap_ids[frame.lm_ids]

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
            temp_match_ids = remap_ids[frame.lm_ids]    # temp ids remapped for min obs
            mask_ids = torch.nonzero(temp_match_ids >= 0, as_tuple=False).squeeze(1)
            match_list += [temp_match_ids[mask_ids].detach().cpu().numpy()]
            meas_list += [frame.meas[mask_ids].detach().cpu().numpy()]
            weight_list += [frame.wmats[mask_ids].detach().cpu().numpy()]
            pose_list += [frame.pose]
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

    def loss(self, T_k0, use_gt, cov_diag_list):
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
        l_sp_list = []
        mask_ids_list = []
        # loop through every frame
        for k, frame in enumerate(self.mframes_deq):
            temp_match_ids = remap_ids[frame.lm_ids]    # temp ids remapped for min obs
            mask_ids = torch.nonzero(temp_match_ids >= 0, as_tuple=False).squeeze(1)
            mask_ids_list += [mask_ids]
            match_list += [temp_match_ids[mask_ids].detach().cpu().numpy()]
            meas_list += [frame.meas[mask_ids].detach().cpu().numpy()]
            weight_list += [frame.wmats[mask_ids].detach().cpu().numpy()]
            pose_list += [frame.pose]
            vel_list += [frame.vel]

            if use_gt:
                pose_list[-1] = T_k0[k, :, :].detach().cpu().numpy()

            # empty list object
            if k == 0:
                l_sp_list += [np.zeros((6+1, meas_list[-1].shape[0], 3), dtype=np.float32)]
            else:
                l_sp_list += [np.zeros((18+1, meas_list[-1].shape[0], 3), dtype=np.float32)]

        poses = np.stack(pose_list, axis=0)
        vels = np.stack(vel_list, axis=0)

        cov_diag = np.asarray(cov_diag_list)

        # steampy_lm.run_steam_lm_sp(meas_list, match_list, weight_list, poses, vels, lm_coords, l_sp_list
        #                         use_gt, cov_diag)
        steampy_lm.run_steam_lm_spb(meas_list, match_list, weight_list, poses, vels, lm_coords, l_sp_list,
                                    use_gt, cov_diag)

        loss = 0
        sum_meas = 0
        for k, frame in enumerate(self.mframes_deq):

            meas_k = frame.meas[mask_ids_list[k]]
            l_sp = torch.from_numpy(l_sp_list[k]).cuda()
            wmats_k = frame.wmats[mask_ids_list[k]]
            wds_k = frame.wds[mask_ids_list[k]]

            # error thresh
            # error = (meas_k - l_sp[0, :, :]).unsqueeze(-1)
            # mah = error.transpose(1, 2)@wmats_k@error
            # ids = torch.nonzero(mah.squeeze() < self.mah_thresh ** 2, as_tuple=False).squeeze(1)
            ids = torch.arange(meas_k.size(0))

            # compute loss over sigmapoints

            # batch version
            error = (meas_k[ids, :].unsqueeze(0) - l_sp[1:, ids, :]).unsqueeze(-1)
            mah = error.transpose(2,3)@wmats_k[ids, :, :].unsqueeze(0)@error/(l_sp.size(0)-1)
            loss += torch.sum(mah.squeeze())

            # loop version
            # out2 = torch.zeros(ids.size()).cuda()
            # for l in torch.arange(l_sp.size(0)-1):
            #     error = (meas_k[ids, :] - l_sp[1+l, ids, :]).unsqueeze(-1)
            #     mah = error.transpose(1,2)@wmats_k[ids, :, :]@error/(l_sp.size(0)-1)
            #     out2 += mah.squeeze()

            # weights
            loss -= torch.sum(wds_k[ids, :])
            sum_meas += ids.size(0)
        if sum_meas > 0:
            loss = loss/sum_meas
        return loss

    def getFirstPose(self):
        return self.mframes_deq[0].pose, self.mframes_deq[0].frame_id

    def addNewLandmarks(self, new_lm_coords):
        # check if empty
        if self.lm_obs_count.size(0) == 0:
            self.lm_coords = new_lm_coords
            self.lm_obs_count = 2*torch.ones(new_lm_coords.size(0), device=new_lm_coords.device)
            return torch.arange(new_lm_coords.size(0), device=new_lm_coords.device)
        else:
            # append
            m = self.lm_coords.size(0)
            self.lm_coords = torch.cat((self.lm_coords, new_lm_coords), dim=0)
            self.lm_obs_count = torch.cat((self.lm_obs_count,
                                           2*torch.ones(new_lm_coords.size(0), device=new_lm_coords.device)), dim=0)
            return torch.arange(m, m + new_lm_coords.size(0), device=new_lm_coords.device)

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
    def __init__(self, coords, descs, weights, match_ids, frame_id):
        self.coords = coords
        self.descs = descs
        self.weights = weights
        self.match_ids = match_ids

        self.pose = np.eye(4, dtype=np.float32)
        self.vel = np.zeros((6), dtype=np.float32)
        self.frame_id = frame_id

        self.meas = torch.Tensor()
        self.lm_ids = torch.Tensor()
        self.wmats = torch.Tensor()
        self.wds = torch.Tensor()

        # self.match_ids = match_ids   # landmark id (-1 if not active)
        # self.active_ids = torch.nonzero(match_ids >= 0, as_tuple=False).squeeze(1)
        # self.meas = meas
        # A, d = self.convertWeightMat(meas_weights)
        # self.wmats = A
        # self.wds = d

    def addMeas(self, meas, weights, coord_ids, lm_ids):
        A, d = self.convertWeightMat(weights)
        if self.meas.size(0) == 0:
            # empty
            self.meas = meas
            self.lm_ids = lm_ids
            self.wmats = A
            self.wds = d
        else:
            # concatenate
            self.meas = torch.cat((self.meas, meas), dim=0)
            self.lm_ids = torch.cat((self.lm_ids, lm_ids), dim=0)
            self.wmats = torch.cat((self.wmats, A), dim=0)
            self.wds = torch.cat((self.wds, d), dim=0)
        self.match_ids[coord_ids] = lm_ids

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