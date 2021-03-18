import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from networks.unet import UNet
from networks.keypoint import Keypoint
from networks.softmax_ref_matcher import SoftmaxRefMatcher
from networks.steam_solver import SteamSolver
from utils.utils import convert_to_radar_frame, mask_intensity_filter

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

    def forward(self, batch):
        data = batch['data'].to(self.gpuid)
        mask = batch['mask'].to(self.gpuid)
        timestamps = batch['timestamps'].to(self.gpuid)

        detector_scores, weight_scores, desc = self.unet(data)
        keypoint_coords, keypoint_scores, keypoint_desc = self.keypoint(detector_scores, weight_scores, desc)
        pseudo_coords, match_weights, tgt_ids, src_ids = self.softmax_matcher(keypoint_scores, keypoint_desc, desc)
        keypoint_coords = keypoint_coords[tgt_ids]

        pseudo_coords_xy = convert_to_radar_frame(pseudo_coords, self.config)
        keypoint_coords_xy = convert_to_radar_frame(keypoint_coords, self.config)
        # rotate back if augmented
        if 'T_aug' in batch:
            T_aug = torch.stack(batch['T_aug'], dim=0).to(self.gpuid)
            keypoint_coords_xy = torch.matmul(keypoint_coords_xy, T_aug[:, :2, :2].transpose(1, 2))
            self.solver.T_aug = batch['T_aug']

        pseudo_coords_xy[:, :, 1] *= -1.0
        keypoint_coords_xy[:, :, 1] *= -1.0

        # binary mask to remove keypoints from 'empty' regions of the input radar scan
        keypoint_ints = mask_intensity_filter(mask[tgt_ids], self.patch_size, self.patch_mean_thres)

        time_tgt = torch.index_select(timestamps, 0, tgt_ids)
        time_src = torch.index_select(timestamps, 0, src_ids)
        R_tgt_src_pred, t_tgt_src_pred = self.solver.optimize(keypoint_coords_xy, pseudo_coords_xy, match_weights,
                                                              keypoint_ints, time_tgt, time_src)

        return {'R': R_tgt_src_pred, 't': t_tgt_src_pred, 'scores': weight_scores, 'tgt': keypoint_coords_xy,
                'src': pseudo_coords_xy, 'match_weights': match_weights, 'keypoint_ints': keypoint_ints,
                'detector_scores': detector_scores, 'tgt_rc': keypoint_coords, 'src_rc': pseudo_coords}
