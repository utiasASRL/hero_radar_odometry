import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_indices

class SoftmaxMatcher(nn.Module):
    """
        Performs soft matching between keypoint descriptors and a dense map of descriptors.
        A temperature-weighted softmax is used which can approximate argmax at low temperatures.
    """
    def __init__(self, config):
        super().__init__()
        self.softmax_temp = config['networks']['matcher_block']['softmax_temp']
        self.window_size = config['window_size']
        self.gpuid = config['gpuid']
        self.score_comp = config['networks']['matcher_block']['score_comp']

    def forward(self, keypoint_scores, keypoint_desc, scores_dense, desc_dense):
        """
            keypoint_scores: BWx1xN
            keypoint_desc: BWxCxN
            scores_dense: BWx1xHxW
            desc_dense: BWxCxHxW
        """
        # TODO: loop if window_size is greater than 2 (for cycle loss)
        BW, encoder_dim, n_points = keypoint_desc.size()
        batch_size = int(BW / self.window_size)
        _, _, height, width = desc_dense.size()
        kp_inds, dense_inds = get_indices(batch_size, self.window_size)

        src_desc = keypoint_desc[kp_inds]  # B x C x N
        src_desc = F.normalize(src_desc, dim=1)
        B = src_desc.size(0)

        tgt_desc_dense = desc_dense[dense_inds]  # B x C x H x W
        tgt_desc_unrolled = F.normalize(tgt_desc_dense.view(B, encoder_dim, -1), dim=1)  # B x C x HW

        match_vals = torch.matmul(src_desc.transpose(2, 1), tgt_desc_unrolled)  # B x N x HW
        soft_match_vals = F.softmax(match_vals / self.softmax_temp, dim=2)  # B x N x HW

        v_coord, u_coord = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        v_coord = v_coord.reshape(height * width).float()  # HW
        u_coord = u_coord.reshape(height * width).float()
        coords = torch.stack((u_coord, v_coord), dim=1)  # HW x 2
        tgt_coords_dense = coords.unsqueeze(0).expand(B, height * width, 2).to(self.gpuid)  # B x HW x 2

        pseudo_coords = torch.matmul(tgt_coords_dense.transpose(2, 1),
                                     soft_match_vals.transpose(2, 1)).transpose(2, 1)  # BxNx2

        # GET SCORES for pseudo point locations
        pseudo_norm = normalize_coords(pseudo_coords, height, width).unsqueeze(1)          # B x 1 x N x 2
        tgt_scores_dense = scores_dense[dense_inds]
        pseudo_scores = F.grid_sample(tgt_scores_dense, pseudo_norm, mode='bilinear')           # B x 1 x 1 x N
        pseudo_scores = pseudo_scores.reshape(B, 1, n_points)                          # B x 1 x N
        # GET DESCRIPTORS for pseudo point locations
        pseudo_desc = F.grid_sample(tgt_desc_dense, pseudo_norm, mode='bilinear')               # B x C x 1 x N
        pseudo_desc = pseudo_desc.reshape(B, encoder_dim, n_points)                    # B x C x N

        desc_match_score = torch.sum(src_desc * pseudo_desc, dim=1, keepdim=True) / float(encoder_dim)  # Bx1xN
        src_scores = keypoint_scores[kp_inds]
        if self.score_comp:
            match_weights = 0.5 * (desc_match_score + 1) * src_scores * pseudo_scores
        else:
            match_weights = src_scores

        return pseudo_coords, match_weights, kp_inds

def normalize_coords(coords_2D, width, height):
    """Normalizes coords_2D (B x N x 2) to be within [-1, 1] """
    batch_size = coords_2D.size(0)
    u_norm = (2 * coords_2D[:, :, 0].reshape(batch_size, -1) / (width - 1)) - 1
    v_norm = (2 * coords_2D[:, :, 1].reshape(batch_size, -1) / (height - 1)) - 1
    return torch.stack([u_norm, v_norm], dim=2)  # B x num_patches x 2
