import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxMatcher(nn.Module):
    def __init__(self, config):
        super(SoftmaxMatcherBlock, self).__init__()
        self.softmax_temp = config["networks"]["matcher_block"]["softmax_temp"]
        self.window_size = config["window_size"]

    def forward(self, keypoint_scores, keypoint_desc, scores_dense, desc_dense):
        '''
        :param keypoint_scores: Bx1xN
        :param keypoint_desc: BxCxN
        :param scores_dense: Bx1xHxW
        :param desc_dense: BxCxHxW
        '''
        # TODO: loop if window_size is greater than 2 (for cycle loss)
        bsz, encoder_dim, _ = keypoint_desc.size()
        batch_size = bsz / self.window_size
        _, _, height, width = desc_dense.size()

        src_desc = keypoint_desc[::self.window_size]
        src_desc = F.normalize(src_desc, dim=1)

        tgt_desc_dense = desc_dense[1::self.window_size] # B x C x H x W
        tgt_desc_dense = F.normalize(tgt_desc_dense.view(batch_size, encoder_dim, -1), dim=1) # B x C x HW

        match_vals = torch.matmul(src_desc.transpose(2, 1).contiguous(), tgt_desc_dense) # B x N x HW
        soft_match_vals = F.softmax(match_vals / self.softmax_temp, dim=2)  # B x N x HW

        v_coord, u_coord = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        v_coord = v_coord.reshape(height * width).float()  # HW
        u_coord = u_coord.reshape(height * width).float()
        coords = torch.stack((u_coord, v_coord), dim=1)  # HW x 2
        tgt_coords_dense = coords.unsqueeze(0).expand(batch_size, height * width, 2) # B x HW x 2
        if config['gpuid'] != 'cpu':
            tgt_coords_dense = tgt_coords_dense.cuda()

        pseudo_coords = torch.matmul(tgt_coords_dense.transpose(2, 1).contiguous(),
            soft_match_vals.transpose(2, 1).contiguous()).transpose(2, 1).contiguous()  # BxNx2

        # GET SCORES for pseudo point locations
        n_points = keypoint_scores.size(2)
        pseudo_norm = normalize_coords(pseudo_coords, batch_size, height, width).unsqueeze(1)   # B x 1 x N x 2
        tgt_scores_dense = scores_dense[1::self.window_size]
        pseudo_scores = F.grid_sample(tgt_scores_dense, pseudo_norm, mode='bilinear')           # B x 1 x 1 x N
        pseduo_scores = pseduo_scores.reshape(batch_size, 1, n_points)                          # B x 1 x N
        # GET DESCRIPTORS for pseduo point locations
        pseudo_desc = F.grid_sample(tgt_desc_dense, pseudo_norm, mode='bilinear')               # B x C x 1 x N
        pseudo_desc = pseudo_desc.reshape(batch_size, channels, keypoints.size(1))              # B x C x N

        desc_match_score = torch.sum(src_desc * pseudo_desc, dim=1, keepdim=True) / float(src_desc.size(1)) # Bx1xN = BxCxN * BxCxN
        src_scores = keypoint_scores[::self.window_size]
        match_weights = 0.5 * (desc_match_score + 1) * src_scores * pseudo_scores

        return pseudo_coords, match_weights
