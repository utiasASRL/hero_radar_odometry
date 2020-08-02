import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from utils.lie_algebra import se3_inv, se3_log, se3_exp
from utils.utils import zn_desc
from networks.UNetBlock import UNetBlock
from networks.SoftmaxMatcherBlock import SoftmaxMatcherBlock
from networks.SVDWeightBlock import SVDWeightBlock
from networks.SVDBlock import SVDBlock

class PoseModel(nn.Module):
    def __init__(self, config):
        super(PoseModel, self).__init__()

        # load configs
        self.config = config
        self.window_size = self.config["dataset"]["window_size"]
        self.match_type = config["networks"]["match_type"] # zncc, l2, dp

        # network arch
        self.feature_model = UNetBlock(self.config)

        self.matcher = SoftmaxMatcherBlock(self.config)

        self.weigh_svd = SVDWeightBlock(self.config)

        self.SVD = SVDBlock(self.config)

    def forward(self, images):
        '''
        Estimate transform between two frames
        :param images: BxCxN
        :return:
        '''

        # Extract features, detector scores and weight scores
        detector_scores, weight_scores, descs = self.feature_model(images)

        # Use detector scores to compute keypoint locations in 3D along with their weight scores and descs
        keypoint_coords, keypoint_descs, keypoint_weights = None, None, None

        # Match the points in src frame to points in target frame to generate pseudo points
        pseudo_coords, pseudo_weights, pseudo_descs = self.matcher(keypoint_coords[::self.window_size],
                                                                   keypoint_coords[1::self.window_size],
                                                                   keypoint_weights[1::self.window_size],
                                                                   keypoint_descs[::self.window_size],
                                                                   keypoint_descs[1::self.window_size])

        # Normalize src desc based on match type
        if self.match_type == 'zncc':
            src_descs = zn_desc(keypoint_descs[::self.window_size])
        elif self.match_type == 'dp':
            src_descs = F.normalize(keypoint_descs[::self.window_size], dim=1)
        elif self.match_type == 'l2':
            pass
        else:
            assert False, "Cannot normalize because match type is NOT support"


        # Compute matching pair weights for matching pairs
        svd_weights = self.weigh_svd(src_descs,
                                     pseudo_descs,
                                     keypoint_weights[::self.window_size],
                                     pseudo_weights)

        # Use SVD to solve for optimal transform
        R_pred, t_pred = self.SVD(keypoint_coords[::self.window_size],
                                  pseudo_coords,
                                  svd_weights)

        return R_pred, t_pred


