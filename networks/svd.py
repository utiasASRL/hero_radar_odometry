import torch
import torch.nn.functional as F

class SVD(torch.nn.Module):
    def __init__(self, config):
        super(SVD, self).__init__()
        self.window_size = config['window_size']
        self.cart_pixel_width = config['cart_pixel_width']
        self.cart_resolution = config['cart_resolution']
        if (self.cart_pixel_width % 2) == 0:
            self.cart_min_range = (self.cart_pixel_width / 2 - 0.5) * self.cart_resolution
        else:
            self.cart_min_range = self.cart_pixel_width // 2 * self.cart_resolution
        self.gpuid = config['gpuid']

    def forward(self, keypoint_coords, tgt_coords, weights, convert_from_pixels=True):
        src_coords = keypoint_coords[::self.window_size]
        batch_size, n_points, _ = src_coords.size()  # B x N x 2
        if convert_from_pixels:
            src_coords = self.convert_to_radar_frame(src_coords)
            tgt_coords = self.convert_to_radar_frame(tgt_coords)
        if src_coords.size(2) < 3:
            pad = 3 - src_coords.size(2)
            src_coords = F.pad(src_coords, [0, pad, 0, 0])
        if tgt_coords.size(2) < 3:
            pad = 3 - tgt_coords.size(2)
            tgt_coords = F.pad(tgt_coords, [0, pad, 0, 0])
        src_coords = src_coords.transpose(2, 1) # B x 3 x N
        tgt_coords = tgt_coords.transpose(2, 1)

        # Compute weighted centroids (elementwise multiplication/division)
        w = torch.sum(weights, dim=2, keepdim=True)
        src_centroid = torch.sum(src_coords * weights, dim=2, keepdim=True) / w  # B x 3 x 1
        tgt_centroid = torch.sum(tgt_coords * weights, dim=2, keepdim=True) / w

        src_centered = src_coords - src_centroid  # B x 3 x N
        tgt_centered = tgt_coords - tgt_centroid

        W = torch.bmm(tgt_centered * weights, src_centered.transpose(2, 1)) / w  # B x 3 x 3

        U, _, V = torch.svd(W)

        det_UV = torch.det(U) * torch.det(V)
        ones = torch.ones(batch_size, 2).type_as(V)
        S = torch.diag_embed(torch.cat((ones, det_UV.unsqueeze(1)), dim=1))  # B x 3 x 3

        # Compute rotation and translation (T_tgt_src)
        R_tgt_src = torch.bmm(U, torch.bmm(S, V.transpose(2, 1)))  # B x 3 x 3
        t_tgt_src_insrc = src_centroid - torch.bmm(R_tgt_src.transpose(2, 1), tgt_centroid)  # B x 3 x 1
        t_src_tgt_intgt = -R_tgt_src.bmm(t_tgt_src_insrc)

        return R_tgt_src.transpose(2, 1), t_src_tgt_intgt

    def convert_to_radar_frame(self, pixel_coords):
        B, N, _ = pixel_coords.size()
        R = torch.tensor([[0, -self.cart_resolution], [self.cart_resolution, 0]]).expand(B, 2, 2).to(self.gpuid)
        t = torch.tensor([[self.cart_min_range],[-self.cart_min_range]]).expand(B, 2, N).to(self.gpuid)
        return (torch.bmm(R, pixel_coords.transpose(2, 1)) + t).transpose(2, 1)
