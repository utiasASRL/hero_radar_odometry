import torch
import torch.nn.functional as F
from utils.utils import convert_to_radar_frame, get_indices

class SVD(torch.nn.Module):
    """
        Computes a 3x3 rotation matrix SO(3) and a 3x1 translation vector from pairs of 3D point clouds aligned
        according to known correspondences. The forward() method uses singular value decomposition to do this.
        This implementation is differentiable and follows the derivation from State Estimation for Robotics (Barfoot).
    """
    def __init__(self, config):
        super().__init__()
        self.window_size = config['window_size']
        self.batch_size = config['batch_size']
        self.cart_pixel_width = config['cart_pixel_width']
        self.cart_resolution = config['cart_resolution']
        self.gpuid = config['gpuid']


    def forward(self, src_coords, tgt_coords, weights, convert_from_pixels=True):
        if src_coords.size(0) > tgt_coords.size(0):
            BW = src_coords.size(0)
            batch_size = int(BW / self.window_size)
            kp_inds, _ = get_indices(batch_size, self.window_size)
            src_coords = src_coords[kp_inds]
        assert(src_coords.size() == tgt_coords.size())
        B = src_coords.size(0)  # B x N x 2
        if convert_from_pixels:
            src_coords = convert_to_radar_frame(src_coords, self.cart_pixel_width, self.cart_resolution, self.gpuid)
            tgt_coords = convert_to_radar_frame(tgt_coords, self.cart_pixel_width, self.cart_resolution, self.gpuid)
        if src_coords.size(2) < 3:
            pad = 3 - src_coords.size(2)
            src_coords = F.pad(src_coords, [0, pad, 0, 0])
        if tgt_coords.size(2) < 3:
            pad = 3 - tgt_coords.size(2)
            tgt_coords = F.pad(tgt_coords, [0, pad, 0, 0])
        src_coords = src_coords.transpose(2, 1)  # B x 3 x N
        tgt_coords = tgt_coords.transpose(2, 1)

        # Compute weighted centroids
        w = torch.sum(weights, dim=2, keepdim=True) + 1e-4
        src_centroid = torch.sum(src_coords * weights, dim=2, keepdim=True) / w  # B x 3 x 1
        tgt_centroid = torch.sum(tgt_coords * weights, dim=2, keepdim=True) / w

        src_centered = src_coords - src_centroid  # B x 3 x N
        tgt_centered = tgt_coords - tgt_centroid

        W = torch.bmm(tgt_centered * weights, src_centered.transpose(2, 1)) / w  # B x 3 x 3

        try:
            U, _, V = torch.svd(W)
        except:     # torch.svd sometimes has convergence issues, this has yet to be patched.
            print(W)
            print(tgt_centered)
            print(src_centered)
            print(weights)
            print('Adding turbulence to patch convergence issue')
            U, _, V = torch.svd(W + 1e-4 * W.mean() * torch.rand(1, 3).to(self.gpuid))

        det_UV = torch.det(U) * torch.det(V)
        ones = torch.ones(B, 2).type_as(V)
        S = torch.diag_embed(torch.cat((ones, det_UV.unsqueeze(1)), dim=1))  # B x 3 x 3

        # Compute rotation and translation (T_tgt_src)
        R_tgt_src = torch.bmm(U, torch.bmm(S, V.transpose(2, 1)))  # B x 3 x 3
        t_tgt_src_insrc = src_centroid - torch.bmm(R_tgt_src.transpose(2, 1), tgt_centroid)  # B x 3 x 1
        t_src_tgt_intgt = -R_tgt_src.bmm(t_tgt_src_insrc)

        return R_tgt_src.transpose(2, 1), t_src_tgt_intgt

    # def forward(self, src_coords, tgt_coords, weights, mask, convert_from_pixels=True):
    #     src_inds, _ = get_indices(self.batch_size, self.window_size)
    #     src_mask = mask[src_inds]
    #     if src_coords.size(0) > tgt_coords.size(0):
    #         src_coords = src_coords[src_inds]
    #     assert(src_coords.size() == tgt_coords.size())
    #     B = src_coords.size(0)  # B x N x 2
    #     if convert_from_pixels:
    #         src_coords = convert_to_radar_frame(src_coords, self.cart_pixel_width, self.cart_resolution, self.gpuid)
    #         tgt_coords = convert_to_radar_frame(tgt_coords, self.cart_pixel_width, self.cart_resolution, self.gpuid)
    #     if src_coords.size(2) < 3:
    #         pad = 3 - src_coords.size(2)
    #         src_coords = F.pad(src_coords, [0, pad, 0, 0])
    #     if tgt_coords.size(2) < 3:
    #         pad = 3 - tgt_coords.size(2)
    #         tgt_coords = F.pad(tgt_coords, [0, pad, 0, 0])
    #     src_coords = src_coords.transpose(2, 1)  # B x 3 x N
    #     tgt_coords = tgt_coords.transpose(2, 1)
    #
    #     # Compute weighted centroids
    #     R = torch.zeros(B, 3, 3).to(self.gpuid)
    #     t = torch.zeros(B, 3, 1).to(self.gpuid)
    #
    #     for i in range(B):
    #         smask = src_mask[i]
    #         indices = torch.nonzero(smask).squeeze()
    #         src = src_coords[i]
    #         tgt = tgt_coords[i]
    #         w = weights[i]
    #         src = src[:, indices]
    #         tgt = tgt[:, indices]
    #         w = w[:, indices]
    #         wsum = torch.sum(w) + 1e-15
    #         src_centroid = torch.sum(src * w, dim=1, keepdim=True) / wsum  # 3 x 1
    #         tgt_centroid = torch.sum(tgt * w, dim=1, keepdim=True) / wsum
    #
    #         src_centered = src - src_centroid  # 3 x N
    #         tgt_centered = tgt - tgt_centroid
    #
    #         W = torch.matmul(tgt_centered * w, src_centered.transpose(1, 0)) / wsum  # 3 x 3
    #
    #         U, _, V = torch.svd(W)
    #
    #         det_UV = torch.det(U) * torch.det(V)
    #         ones = torch.ones(1, 2).type_as(V)
    #         S = torch.diag_embed(torch.cat((ones, det_UV.reshape(1, 1)), dim=-1)).squeeze()  # 3 x 3
    #         # Compute rotation and translation (T_tgt_src)
    #         R_tgt_src = torch.matmul(U, torch.matmul(S, V.transpose(1, 0)))  # 3 x 3
    #         t_tgt_src_insrc = src_centroid - torch.matmul(R_tgt_src.transpose(1, 0), tgt_centroid)  # B x 3 x 1
    #         t_src_tgt_intgt = -R_tgt_src.matmul(t_tgt_src_insrc)
    #         R[i] = R_tgt_src.transpose(1, 0)
    #         t[i] = t_src_tgt_intgt
    #
    #     return R, t
