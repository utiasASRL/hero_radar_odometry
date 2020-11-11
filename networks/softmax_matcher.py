import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxMatcher(nn.Module):
    def __init__(self, config):
        super(SoftmaxMatcherBlock, self).__init__()
        self.softmax_temp = config["networks"]["matcher_block"]["softmax_temp"]
        self.window_size = config["window_size"]

    def forward(self, keypoint_coords, keypoint_weights, keypoint_desc):
        '''
        Descriptors are assumed to be not normalized
        :param src_coords: Bx2xN
        :param tgt_coords: Bx2xM
        :param src_weights: Bx1xN
        :param tgt_weights: Bx1xM (M=N for keypoint match)
        :param src_desc: BxCxN
        :param tgt_desc: BxCxM
        '''
        batch_size, n_features, _, _ = tgt_desc.size()
        # TODO: loop if window_size is greater than 2 (for cycle loss)
        src_coords = keypoint_coords[::self.window_size]
        tgt_coords = keypoint_coords[1::self.window_size]
        src_weights = keypoint_weights[::self.window_size]
        tgt_weights = keypoint_weights[1::self.window_size]
        src_desc = keypoint_desc[::self.window_size]
        tgt_desc = keypoint_desc[1::self.window_size]

        src_desc_norm = F.normalize(src_desc, dim=1)
        tgt_desc_norm = F.normalize(tgt_desc, dim=1)

        match_vals = torch.matmul(src_desc_norm.transpose(2, 1).contiguous(), tgt_desc_norm) # BxNxM
        soft_match_vals = F.softmax(match_vals / self.softmax_temp, dim=2)  # BxNxM

        pseudo_coords = torch.matmul(tgt_coords, soft_match_vals.transpose(2, 1).contiguous()) # Bx3xN
        pseudo_desc = torch.matmul(tgt_desc.view(batch_size, n_features, -1), soft_match_vals.transpose(2, 1).contiguous()) # BxCxN
        pseudo_desc = F.normalize(pseudo_descs, dim=1)
        pseudo_weights = torch.matmul(tgt_weights.view(batch_size, 1, -1), soft_match_vals.transpose(2, 1).contiguous()) # Bx1xN

        desc_match_score = torch.sum(src_desc * pseudo_desc, dim=1, keepdim=True) / float(src_desc.size(1)) # Bx1xN = BxCxN * BxCxN
        match_weights = 0.5 * (desc_match_score + 1) * src_weights * pseudo_weights

        return pseudo_coords, match_weights
