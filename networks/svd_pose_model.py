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
        self.window_size = config['window_size']
        self.outlier_rejection = config['networks']['outlier_rejection']['on']
        self.outlier_threshold = config['networks']['outlier_rejection']['threshold']
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

        return R_tgt_src_pred, t_tgt_src_pred

def supervised_loss(R_tgt_src_pred, t_tgt_src_pred, batch, config):
    T_21 = batch['T_21'].to(config['gpuid'])
    # Get ground truth transforms
    T_tgt_src = T_21[::config['window_size']]
    R_tgt_src = T_tgt_src[:,:3,:3]
    t_tgt_src = T_tgt_src[:,:3, 3]
    svd_loss, R_loss, t_loss = SVD_loss(R_tgt_src, R_tgt_src_pred, t_tgt_src.unsqueeze(-1), t_tgt_src_pred, config['gpuid'])
    return svd_loss, R_loss, t_loss

def SVD_loss(R, R_pred, t, t_pred, gpuid='cpu', rel_w=10.0):
    batch_size = R.size(0)
    alpha = rel_w
    identity = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(gpuid)
    loss_fn = torch.nn.MSELoss()
    R_loss = alpha * loss_fn(R_pred.transpose(2,1).contiguous() @ R, identity)
    t_loss = 1.0 * loss_fn(t_pred, t)
    svd_loss = R_loss + t_loss
    return svd_loss, R_loss, t_loss
