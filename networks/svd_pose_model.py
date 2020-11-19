import torch
import torch.nn.functional as F

from networks.unet import UNet
from networks.keypoint import Keypoint
from networks.softmax_matcher import SoftmaxMatcher
from networks.svd import SVD

class SVDPoseModel(torch.nn.Module):
    def __init__(self, config):
        super(SVDPoseModel, self).__init__()

        self.config = config
        self.gpuid = config['gpuid']

        self.unet = UNet(config)
        self.keypoint = Keypoint(config)
        self.softmax_matcher = SoftmaxMatcher(config)
        self.svd = SVD(config)

    def forward(self, batch):
        input = batch['input'].to(self.gpuid)

        detector_scores, weight_scores, desc = self.unet(input)

        keypoint_coords, keypoint_scores, keypoint_desc = self.keypoint(detector_scores, weight_scores, desc)

        pseudo_coords, match_weights = self.softmax_matcher(keypoint_scores, keypoint_desc, weight_scores, desc)

        R_tgt_src_pred, t_tgt_src_pred = self.svd(keypoint_coords, pseudo_coords, match_weights)

        return {'R': R_tgt_src_pred, 't': t_tgt_src_pred, 'scores': weight_scores, 'src': keypoint_coords,
            'tgt': pseudo_coords, 'match_weights': match_weights}
