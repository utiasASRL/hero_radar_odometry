import torch
import torch.nn.functional as F

""" Code from: https://github.com/WangYueFt/dcp/blob/master/model.py """
""" https://igl.ethz.ch/projects/ARAP/svd_rot.pdf """
class SVD(torch.nn.Module):
    def __init__(self, config):
        super(SVD, self).__init__()
        self.window_size = config['window_size']

    def forward(self, keypoint_coords, tgt_coords, weights):
        src_coords = keypoint_coords[::self.window_size]
        batch_size, _, n_points = src_coords.size()  # B x 2 x N
        if src_coords.size(1) < 3:
            pad = 3 - src_coords.size(1)
            F.pad(src_coords, [0, 0, 0, 1])
        if keypoint_coords.size(1) < 3:
            pad = 3 - keypoint_coords.size(1)
            F.pad(keypoint_coords, [0, 0, 0, 1])

        # Compute weighted centroids (elementwise multiplication/division)
        src_centroid = torch.sum(src_coords * weights, dim=2, keepdim=True) / torch.sum(weights, dim=2, keepdim=True)  # B x 3 x 1
        tgt_centroid = torch.sum(tgt_coords * weights, dim=2, keepdim=True) / torch.sum(weights, dim=2, keepdim=True)

        src_centered = src_coords - src_centroid  # B x 3 x N
        tgt_centered = tgt_coords - tgt_centroid

        W = torch.diag_embed(weights.reshape(batch_size, n_points))  # B x N x N
        w = torch.sum(weights, dim=2).unsqueeze(2)                   # B x 1 x 1

        H = (1.0 / w) * torch.bmm(tgt_centered, torch.bmm(W, src_centered.transpose(2, 1).contiguous()))  # B x 3 x 3

        U, S, V = torch.svd(H)

        det_UV = torch.det(U) * torch.det(V)
        ones = torch.ones(batch_size, 2).type_as(V)
        diag = torch.diag_embed(torch.cat((ones, det_UV.unsqueeze(1)), dim=1))  # B x 3 x 3

        # Compute rotation and translation (T_tgt_src)
        R_tgt_src = torch.bmm(U, torch.bmm(diag, V.transpose(2, 1).contiguous()))  # B x 3 x 3
        t_tgt_src_insrc = src_centroid - torch.bmm(R_tgt_src.transpose(2, 1).contiguous(), tgt_centroid)  # B x 3 x 1
        t_src_tgt_intgt = -R_tgt_src.bmm(t_tgt_src_insrc)

        return R_tgt_src, t_src_tgt_intgt
