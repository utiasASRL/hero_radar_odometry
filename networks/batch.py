import torch
import numpy as np
from utils.utils import convert_to_radar_frame, mask_intensity_filter
import cv2
import cpp.build.BatchSolver as steamcpp
from datasets.features import *
from utils.utils import getApproxTimeStamps

class Batch():
    """Hand-crafted feature extraction (cen2018) and descriptors (orb)."""

    def __init__(self, config):
        self.config = config
        self.orb = cv2.ORB_create()
        self.orb.setPatchSize(21)
        self.orb.setEdgeThreshold(21)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING) #, crossCheck=True)
        self.radar_resolution = config['radar_resolution']
        self.cart_resolution = config['cart_resolution']
        self.cart_pixel_width = config['cart_pixel_width']
        self.solver = steamcpp.BatchSolver()

    def add_frame_pair(self, batch):
        data = batch['data']
        timestamps = batch['timestamps']
        t_ref = batch['t_ref']
        tgt_ids = [1]
        src_ids = [0]
        polar = batch['polar'].numpy().squeeze()
        azimuths = batch['azimuths'].numpy().squeeze()
        ptgt1 = cen2018features(polar[0])
        ptgt2 = cen2018features(polar[1])
        tgt1 = polar_to_cartesian_points(azimuths[0], ptgt1, self.radar_resolution)
        tgt2 = polar_to_cartesian_points(azimuths[1], ptgt2, self.radar_resolution)
        tgt1, tgt1p = convert_to_bev2(tgt1, self.cart_resolution, self.cart_pixel_width, 21)
        tgt2, tgt2p = convert_to_bev2(tgt2, self.cart_resolution, self.cart_pixel_width, 21)
        kp1 = convert_to_keypoints(tgt1p)
        kp2 = convert_to_keypoints(tgt2p)
        img1 = (data[0] * 255).numpy().squeeze().astype(np.uint8)
        img2 = (data[1] * 255).numpy().squeeze().astype(np.uint8)
        kp1_, des1 = self.orb.compute(img1, kp1)
        kp2_, des2 = self.orb.compute(img2, kp2)
        assert(len(kp1_) == len(kp1))

        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        N = len(good_matches)

        p1 = np.zeros((N, 2))
        p2 = np.zeros((N, 2))
        p1p = np.zeros((N, 2))
        p2p = np.zeros((N, 2))
        for i,match in enumerate(good_matches):
            p1[i, :] = tgt1[match.queryIdx, :]
            p1p[i, :] = tgt1p[match.queryIdx, :]
            p2[i, :] = tgt2[match.trainIdx, :]
            p2p[i, :] = tgt2p[match.trainIdx, :]

        p1 = np.expand_dims(p1, axis=0)
        p2 = np.expand_dims(p2, axis=0)
        p1p = np.expand_dims(p1p, axis=0)
        p2p = np.expand_dims(p2p, axis=0)
        pseudo_coords_xy = torch.from_numpy(p1)
        keypoint_coords_xy = torch.from_numpy(p2)

        pseudo_coords = torch.from_numpy(p1p)
        keypoint_coords = torch.from_numpy(p2p)
        time_tgt = timestamps[tgt_ids]
        time_src = timestamps[src_ids]
        # t_ref_tgt = t_ref[tgt_ids]
        # t_ref_src = t_ref[src_ids]
        # keypoint_ints = torch.ones((1, 1, p1.shape[1]))
        # match_weights = torch.ones((1, 1, p1.shape[1]))

        # prepare input for cpp call
        points1 = pseudo_coords_xy[0].detach().cpu().numpy()
        points2 = keypoint_coords_xy[0].detach().cpu().numpy()
        times1 = time_src[0].cpu().numpy().squeeze()
        times2 = time_tgt[0].cpu().numpy().squeeze()
        timestamps1 = getApproxTimeStamps([points1], [times1])[0]
        timestamps2 = getApproxTimeStamps([points2], [times2])[0]

        # add measurements
        self.solver.addFramePair(points2, points1,
            timestamps2, timestamps1, np.min(timestamps1), np.max(timestamps2))

        return