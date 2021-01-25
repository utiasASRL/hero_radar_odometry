import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxRefMatcher(nn.Module):
    """
        Performs soft matching between keypoint descriptors and a dense map of descriptors.
        A temperature-weighted softmax is used which can approximate argmax at low temperatures.
    """
    def __init__(self, config):
        super().__init__()
        self.softmax_temp = config['networks']['matcher_block']['softmax_temp']
        self.window_size = config['window_size']
        self.gpuid = config['gpuid']
        # self.score_comp = config['networks']['matcher_block']['score_comp']

    def forward(self, keypoint_scores, keypoint_desc, scores_dense, desc_dense):
        """
            keypoint_scores: Bx1xN
            keypoint_desc: BxCxN
            scores_dense: Bx1xHxW
            desc_dense: BxCxHxW
        """
        # TODO: loop if window_size is greater than 2 (for cycle loss)
        bsz, encoder_dim, n_points = keypoint_desc.size()
        batch_size = int(bsz / self.window_size)
        _, _, height, width = desc_dense.size()

        # src_desc = keypoint_desc[::self.window_size]  # B x C x N
        # src_desc = F.normalize(src_desc, dim=1)
        #
        # tgt_desc_dense = desc_dense[1::self.window_size]  # B x C x H x W
        # tgt_desc_unrolled = F.normalize(tgt_desc_dense.view(batch_size, encoder_dim, -1), dim=1)  # B x C x HW

        # setup 2D grid coord
        v_coord, u_coord = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        v_coord = v_coord.reshape(height * width).float()  # HW
        u_coord = u_coord.reshape(height * width).float()
        coords = torch.stack((u_coord, v_coord), dim=1)  # HW x 2
        src_coords_dense = coords.unsqueeze(0).to(self.gpuid)  # 1 x HW x 2

        src_desc_dense = desc_dense[::self.window_size]
        src_desc_unrolled = F.normalize(src_desc_dense.view(batch_size, encoder_dim, -1), dim=1)  # B x C x HW

        # build pseudo coords
        pseudo_coords = torch.zeros((batch_size*(self.window_size - 1), n_points, 2), device=self.gpuid) # B*(window - 1) x N x 2
        tgt_ids = torch.zeros(batch_size*(self.window_size - 1), dtype=torch.int64)    # B*(window - 1)
        # loop for each batch
        for i in range(src_desc_unrolled.size(0)):
            win_ids = torch.arange(i*self.window_size+1, i*self.window_size+self.window_size)
            tgt_desc = keypoint_desc[win_ids]  # (window - 1) x C x N
            match_vals = torch.matmul(tgt_desc.transpose(2, 1), src_desc_unrolled[i:i+1])  # (window - 1) x N x HW
            soft_match_vals = F.softmax(match_vals / self.softmax_temp, dim=2)  # (window - 1) x N x HW

            pseudo_ids = torch.arange(i*(self.window_size-1), i*(self.window_size-1)+self.window_size-1)
            pseudo_coords[pseudo_ids] = torch.matmul(src_coords_dense.transpose(2, 1),
                soft_match_vals.transpose(2, 1)).transpose(2, 1)  # (window - 1) x N x 2
            tgt_ids[pseudo_ids] = win_ids

        return pseudo_coords, keypoint_scores[tgt_ids], tgt_ids

def normalize_coords(coords_2D, width, height):
    """Normalizes coords_2D (B x N x 2) to be within [-1, 1] """
    batch_size = coords_2D.size(0)
    u_norm = (2 * coords_2D[:, :, 0].reshape(batch_size, -1) / (width - 1)) - 1
    v_norm = (2 * coords_2D[:, :, 1].reshape(batch_size, -1) / (height - 1)) - 1
    return torch.stack([u_norm, v_norm], dim=2)  # B x num_patches x 2
