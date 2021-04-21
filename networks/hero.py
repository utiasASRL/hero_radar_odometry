import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from networks.unet import UNet
from networks.keypoint import Keypoint
from networks.softmax_ref_matcher import SoftmaxRefMatcher
from networks.steam_solver import SteamSolver
from utils.utils import convert_to_radar_frame, mask_intensity_filter

import cv2
from datasets.features import *

class HERO(torch.nn.Module):
    """
        This model performs unsupervised radar odometry using a sliding window optimization with a window
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
        self.patch_mean_thres = config['steam']['patch_mean_thres']
        self.orb = cv2.ORB_create()
        self.orb.setPatchSize(21)
        self.orb.setEdgeThreshold(21)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING) #, crossCheck=True)

    def forward(self, batch):
        data = batch['data'].to(self.gpuid)
        mask = batch['mask'].to(self.gpuid)
        timestamps = batch['timestamps'].to(self.gpuid)

        '''detector_scores, weight_scores, desc = self.unet(data)
        keypoint_coords, keypoint_scores, keypoint_desc = self.keypoint(detector_scores, weight_scores, desc)
        pseudo_coords, match_weights, tgt_ids, src_ids = self.softmax_matcher(keypoint_scores, keypoint_desc, desc)
        keypoint_coords = keypoint_coords[tgt_ids]

        pseudo_coords_xy = convert_to_radar_frame(pseudo_coords, self.config)
        keypoint_coords_xy = convert_to_radar_frame(keypoint_coords, self.config)'''

        tgt_ids = [1]
        src_ids = [0]
        polar = batch['polar'].cpu().numpy().squeeze()
        azimuths = batch['azimuths'].cpu().numpy().squeeze()
        ptgt1 = cen2018features(polar[0])
        ptgt2 = cen2018features(polar[1])
        tgt1 = polar_to_cartesian_points(azimuths[0], ptgt1, 0.0432)
        tgt2 = polar_to_cartesian_points(azimuths[1], ptgt2, 0.0432)
        tgt1, tgt1p = convert_to_bev2(tgt1, 0.2592, 640, 21)
        tgt2, tgt2p = convert_to_bev2(tgt2, 0.2592, 640, 21)

        kp1 = convert_to_keypoints(tgt1p)
        kp2 = convert_to_keypoints(tgt2p)
        img1 = (data[0] * 255).cpu().numpy().squeeze().astype(np.uint8)
        img2 = (data[1] * 255).cpu().numpy().squeeze().astype(np.uint8)
        kp1_, des1 = self.orb.compute(img1, kp1)
        kp2_, des2 = self.orb.compute(img2, kp2)
        assert(len(kp1_) == len(kp1))

        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        N = len(good_matches)
        print(N)
        p1 = np.zeros((N, 2))
        p2 = np.zeros((N, 2))
        for i,match in enumerate(good_matches):
            p1[i, :] = tgt1[match.queryIdx, :]
            p2[i, :] = tgt2[match.trainIdx, :]

        p1 = np.expand_dims(p1, axis=0)
        p2 = np.expand_dims(p2, axis=0)
        pseudo_coords_xy = torch.from_numpy(p1)
        keypoint_coords_xy = torch.from_numpy(p2)

        # rotate back if augmented
        if 'T_aug' in batch:
            T_aug = torch.stack(batch['T_aug'], dim=0).to(self.gpuid)
            keypoint_coords_xy = torch.matmul(keypoint_coords_xy, T_aug[:, :2, :2].transpose(1, 2))
            self.solver.T_aug = batch['T_aug']

        if self.config["flip_y"]:
            pseudo_coords_xy[:, :, 1] *= -1.0
            keypoint_coords_xy[:, :, 1] *= -1.0

        # binary mask to remove keypoints from 'empty' regions of the input radar scan
        #keypoint_ints = mask_intensity_filter(mask[tgt_ids], self.patch_size, self.patch_mean_thres)
        time_tgt = timestamps[tgt_ids]
        time_src = timestamps[src_ids]
        keypoint_ints = torch.ones((1, 1, p1.shape[1]))
        match_weights = torch.ones((1, 1, p1.shape[1]))
        #time_tgt = torch.index_select(timestamps, 0, tgt_ids)
        #time_src = torch.index_select(timestamps, 0, src_ids)
        R_tgt_src_pred, t_tgt_src_pred = self.solver.optimize(keypoint_coords_xy, pseudo_coords_xy, match_weights,
                                                              keypoint_ints, time_tgt, time_src)

        weight_scores = None
        detector_scores = None
        keypoint_coords = None
        pseudo_coords = None

        return {'R': R_tgt_src_pred, 't': t_tgt_src_pred, 'scores': weight_scores, 'tgt': keypoint_coords_xy,
                'src': pseudo_coords_xy, 'match_weights': match_weights, 'keypoint_ints': keypoint_ints,
                'detector_scores': detector_scores, 'tgt_rc': keypoint_coords, 'src_rc': pseudo_coords}
