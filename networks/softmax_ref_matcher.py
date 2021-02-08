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
        self.B = config['batch_size']
        self.P = self.get_num_pairs()
        self.gpuid = config['gpuid']
        self.width = config['cart_pixel_width']
        v_coord, u_coord = torch.meshgrid([torch.arange(0, self.width), torch.arange(0, self.width)])
        v_coord = v_coord.reshape(self.width**2).float()  # HW
        u_coord = u_coord.reshape(self.width**2).float()
        coords = torch.stack((u_coord, v_coord), dim=1)  # HW x 2
        self.src_coords_dense = coords.unsqueeze(0).to(self.gpuid)  # 1 x HW x 2
        self.tgt_ids = torch.zeros(self.B * self.P, dtype=torch.int64, device=self.gpuid)    # B*P
        self.src_ids = torch.zeros(self.B * self.P, dtype=torch.int64, device=self.gpuid)    # B*P
        p = 0
        for b in range(self.B):
            for w in range(self.window_size - 1):
                src_idx = w + b * self.window_size
                win_ids = torch.arange(w + 1, self.window_size, device=self.gpuid) + b * self.window_size
                pseudo_ids = torch.arange(p, p + len(win_ids), device=self.gpuid)
                p += len(win_ids)
                self.tgt_ids[pseudo_ids] = win_ids
                self.src_ids[pseudo_ids] = src_idx


    def forward2(self, keypoint_scores, keypoint_desc, desc_dense):
        """
            keypoint_scores: BWx1xN
            keypoint_desc: BWxCxN
            desc_dense: BWxCxHxW
        """
        bsz, encoder_dim, n_points = keypoint_desc.size()
        src_desc_dense = desc_dense[::self.window_size]
        src_desc_unrolled = F.normalize(src_desc_dense.view(self.B, encoder_dim, -1), dim=1)  # B x C x HW
        # build pseudo_coords
        pseudo_coords = torch.zeros((self.B * (self.window_size - 1), n_points, 2),
                                    device=self.gpuid) # B*(window - 1) x N x 2
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
        return pseudo_coords, keypoint_scores[tgt_ids], tgt_ids

    def forward(self, keypoint_scores, keypoint_desc, desc_dense):
        """
            keypoint_scores: BWxSxN
            keypoint_desc: BWxCxN
            desc_dense: BWxCxHxW
        """
        BW, encoder_dim, n_points = keypoint_desc.size()
        src_desc_unrolled = F.normalize(desc_dense.view(BW, encoder_dim, -1), dim=1)  # B x C x HW

        # build pseudo_coords
        # pseudo_coords = torch.zeros((self.B * self.P, n_points, 2), device=self.gpuid) # B*P x N x 2
        # tgt_scores = torch.zeros(self.B * self.P, keypoint_scores.size(1), n_points, device=self.gpuid)

        tgt_scores = keypoint_scores.index_select(0, tgt_ids)
        tgt_desc = keypoint_desc.index_select(0, tgt_ids)
        tgt_desc = F.normalize(tgt_desc, dim=1)
        src_desc = src_desc_unrolled.index_select(0, src_ids)
        match_vals = torch.matmul(tgt_desc.transpose(2, 1), src_desc)  # * x N x HW
        soft_match_vals = F.softmax(match_vals / self.softmax_temp, dim=2)
        pseudo_coords = torch.matmul(self.src_coords_dense.transpose(2, 1),
            soft_match_vals.transpose(2, 1)).transpose(2, 1)  # * x N x 2

        # p = 0
        # for b in range(self.B):
        #     for w in range(self.window_size - 1):
        #         src_idx = w + b * self.window_size
        #         win_ids = torch.arange(w + 1, self.window_size, device=self.gpuid) + b * self.window_size
        #         tgt_desc = keypoint_desc[win_ids]
        #         tgt_desc = F.normalize(tgt_desc, dim=1)
        #         match_vals = torch.matmul(tgt_desc.transpose(2, 1), src_desc_unrolled[src_idx:src_idx+1])  # * x N x HW
        #         soft_match_vals = F.softmax(match_vals / self.softmax_temp, dim=2)  # * x N x HW
        #         pseudo_ids = torch.arange(p, p + len(win_ids), device=self.gpuid)
        #         p += len(win_ids)
        #         pseudo_coords[pseudo_ids] = torch.matmul(self.src_coords_dense.transpose(2, 1),
        #             soft_match_vals.transpose(2, 1)).transpose(2, 1)  # * x N x 2
        #         tgt_ids[pseudo_ids] = win_ids
        #         src_ids[pseudo_ids] = src_idx
        #         tgt_scores[pseudo_ids] = keypoint_scores[win_ids]
        return pseudo_coords, tgt_scores, tgt_ids, src_ids

    def get_num_pairs(self):
        if self.window_size == 2:
            return 1
        elif self.window_size == 3:
            return 3
        elif self.window_size == 4:
            return 6
        else:
            assert(False),"Unsupported window size: {}".format(self.window_size)
        return self.window_size - 1
