import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import normalize_coords

class SoftmaxRefMatcher(nn.Module):
    """
        Performs soft matching between keypoint descriptors and a dense map of descriptors.
        A temperature-weighted softmax is used which can approximate argmax at low temperatures.
    """
    def __init__(self, config):
        super().__init__()
        self.softmax_temp = config['networks']['matcher_block']['softmax_temp']
        self.window_size = config['window_size']
        self.B = config['batch_size']
        self.gpuid = config['gpuid']
        self.width = config['cart_pixel_width']
        v_coord, u_coord = torch.meshgrid([torch.arange(0, self.width), torch.arange(0, self.width)])
        v_coord = v_coord.reshape(self.width**2).float()  # HW
        u_coord = u_coord.reshape(self.width**2).float()
        coords = torch.stack((u_coord, v_coord), dim=1)  # HW x 2
        self.src_coords_dense = coords.unsqueeze(0).to(self.gpuid)  # 1 x HW x 2

    def forward(self, keypoint_scores, keypoint_desc, desc_dense, times_img):
        """
            keypoint_scores: BWxSxN
            keypoint_desc: BWxCxN
            desc_dense: BWxCxHxW
        """
        bsz, encoder_dim, n_points = keypoint_desc.size()
        src_desc_dense = desc_dense[::self.window_size]
        src_desc_unrolled = F.normalize(src_desc_dense.view(self.B, encoder_dim, -1), dim=1)  # B x C x HW
        # build pseudo_coords
        pseudo_coords = torch.zeros((self.B * (self.window_size - 1), n_points, 2),
                                    device=self.gpuid)  # B*(window - 1) x N x 2
        tgt_ids = torch.zeros(self.B * (self.window_size - 1), dtype=torch.int64)    # B*(window - 1)
        # loop for each batch
        for i in range(self.B):
            win_ids = torch.arange(i * self.window_size + 1, i * self.window_size + self.window_size)
            tgt_desc = keypoint_desc[win_ids]  # (window - 1) x C x N
            tgt_desc = F.normalize(tgt_desc, dim=1)
            match_vals = torch.matmul(tgt_desc.transpose(2, 1), src_desc_unrolled[i:i+1])  # (window - 1) x N x HW
            soft_match_vals = F.softmax(match_vals / self.softmax_temp, dim=2)  # (window - 1) x N x HW
            pseudo_ids = torch.arange(i * (self.window_size - 1), i * (self.window_size - 1) + self.window_size - 1)
            pseudo_coords[pseudo_ids] = torch.matmul(self.src_coords_dense.transpose(2, 1),
                soft_match_vals.transpose(2, 1)).transpose(2, 1)  # (window - 1) x N x 2
            tgt_ids[pseudo_ids] = win_ids

        # pseudo times
        _, _, height, width = desc_dense.size()
        src_times_dense = times_img[::self.window_size].repeat_interleave(self.window_size-1, dim=0)
        norm_pseudopoints2D = normalize_coords(pseudo_coords, height, width).unsqueeze(1)
        pseudo_times = F.grid_sample(src_times_dense, norm_pseudopoints2D, mode='bilinear')
        pseudo_times = pseudo_times.view(self.B * (self.window_size - 1), 1, pseudo_coords.size(1))  # BW x 1 x n_patch
        return pseudo_coords, keypoint_scores[tgt_ids], tgt_ids, pseudo_times
