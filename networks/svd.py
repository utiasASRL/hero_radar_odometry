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
        self.config = config
        self.window_size = config['window_size']
        self.gpuid = config['gpuid']

    def forward(self, src_coords, tgt_coords, weights, convert_from_pixels=True):
        """ This modules used differentiable singular value decomposition to compute the rotations and translations that
            best align matched pointclouds (src and tgt).
        Args:
            src_coords (torch.tensor): (b,N,2) source keypoint locations
            tgt_coords (torch.tensor): (b,N,2) target keypoint locations
            weights (torch.tensor): (b,1,N) weight score associated with each src-tgt match
            convert_from_pixels (bool): if true, input is in pixel coordinates and must be converted to metric
        Returns:
            R_tgt_src (torch.tensor): (b,3,3) rotation from src to tgt
            t_src_tgt_intgt (torch.tensor): (b,3,1) translation from tgt to src as measured in tgt
        """
        if src_coords.size(0) > tgt_coords.size(0):
            BW = src_coords.size(0)
            B = int(BW / self.window_size)
            kp_inds, _ = get_indices(B, self.window_size)
            src_coords = src_coords[kp_inds]
        assert(src_coords.size() == tgt_coords.size())
        B = src_coords.size(0)  # B x N x 2
        if convert_from_pixels:
            src_coords = convert_to_radar_frame(src_coords, self.config)
            tgt_coords = convert_to_radar_frame(tgt_coords, self.config)
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
        except RuntimeError:     # torch.svd sometimes has convergence issues, this has yet to be patched.
            print(W)
            print('Adding turbulence to patch convergence issue')
            U, _, V = torch.svd(W + 1e-4 * W.mean() * torch.rand(1, 3).to(self.gpuid))

        det_UV = torch.det(U) * torch.det(V)
        ones = torch.ones(B, 2).type_as(V)
        S = torch.diag_embed(torch.cat((ones, det_UV.unsqueeze(1)), dim=1))  # B x 3 x 3

        # Compute rotation and translation (T_tgt_src)
        R_tgt_src = torch.bmm(U, torch.bmm(S, V.transpose(2, 1)))  # B x 3 x 3
        t_tgt_src_insrc = src_centroid - torch.bmm(R_tgt_src.transpose(2, 1), tgt_centroid)  # B x 3 x 1
        t_src_tgt_intgt = -R_tgt_src.bmm(t_tgt_src_insrc)

        return R_tgt_src, t_src_tgt_intgt
