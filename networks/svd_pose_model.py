import torch
from networks.unet import UNet
from networks.keypoint import Keypoint
from networks.softmax_matcher import SoftmaxMatcher
from networks.svd import SVD

class SVDPoseModel(torch.nn.Module):
    """
        This model computes a 3x3 Rotation matrix and a 3x1 translation vector describing the transformation
        between two radar scans. This transformation can be used for odometry or metric localization.
        It is intended to be an implementation of Under the Radar (Barnes et al., 2020)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpuid = config['gpuid']
        self.unet = UNet(config)
        self.keypoint = Keypoint(config)
        self.softmax_matcher = SoftmaxMatcher(config)
        self.svd = SVD(config)

    def forward(self, batch):
        data = batch['data'].to(self.gpuid)
        mask = batch['mask'].to(self.gpuid)

        detector_scores, weight_scores, desc = self.unet(data, mask)

        keypoint_coords, keypoint_scores, keypoint_desc, kpmask  = self.keypoint(detector_scores, weight_scores, desc)

        pseudo_coords, match_weights, kp_inds = self.softmax_matcher(keypoint_scores, keypoint_desc, weight_scores, desc)
        src_coords = keypoint_coords[kp_inds]

        R_tgt_src_pred, t_tgt_src_pred = self.svd(src_coords, pseudo_coords, match_weights, kpmask)

        return {'R': R_tgt_src_pred, 't': t_tgt_src_pred, 'scores': weight_scores, 'src': src_coords,
                'tgt': pseudo_coords, 'match_weights': match_weights, 'dense_weights': weight_scores}
