import torch
import torch.nn.functional as F
from utils.utils import normalize_coords

class Keypoint(torch.nn.Module):
    """
        Given a dense map of detector scores and weight scores, this modules computes keypoint locations, and their
        associated scores and descriptors. A spatial softmax is used over a regular grid of "patches" to extract a
        single location, score, and descriptor per patch.
    """
    def __init__(self, config):
        super().__init__()
        self.patch_size = config['networks']['keypoint_block']['patch_size']
        self.gpuid = config['gpuid']
        self.width = config['cart_pixel_width']
        v_coords, u_coords = torch.meshgrid([torch.arange(0, self.width), torch.arange(0, self.width)])
        v_coords = v_coords.unsqueeze(0).float()  # 1 x H x W
        u_coords = u_coords.unsqueeze(0).float()
        self.coords = torch.cat((u_coords, v_coords), dim=0).to(self.gpuid)  # 2 x H x W
        BW = config['batch_size'] * config['window_size']
        self.v_patches = F.unfold(v_coords.expand(BW, 1, self.width, self.width), kernel_size=self.patch_size,
                                  stride=self.patch_size).to(self.gpuid)  # BW x patch_elems x num_patches
        self.u_patches = F.unfold(u_coords.expand(BW, 1, self.width, self.width), kernel_size=self.patch_size,
                                  stride=self.patch_size).to(self.gpuid)

    def forward(self, detector_scores, weight_scores, descriptors):
        """
            detector_scores: BWx1xHxW
            weight_scores: BWxSxHxW
            descriptors: BWxCxHxW
        """
        BW, encoder_dim, _, _ = descriptors.size()
        weight_dim = weight_scores.size(1)
        detector_patches = F.unfold(detector_scores, kernel_size=self.patch_size, stride=self.patch_size)
        softmax_attention = F.softmax(detector_patches, dim=1)  # BW x patch_elements x num_patches
        expected_v = torch.sum(self.v_patches * softmax_attention, dim=1)
        expected_u = torch.sum(self.u_patches * softmax_attention, dim=1)
        keypoint_coords = torch.stack([expected_u, expected_v], dim=2)  # BW x num_patches x 2
        num_patches = keypoint_coords.size(1)

        norm_keypoints2D = normalize_coords(keypoint_coords, self.width, self.width).unsqueeze(1)

        keypoint_desc = F.grid_sample(descriptors, norm_keypoints2D, mode='bilinear', align_corners=True)
        keypoint_desc = keypoint_desc.view(BW, encoder_dim, num_patches)  # BW x C x num_patches

        keypoint_scores = F.grid_sample(weight_scores, norm_keypoints2D, mode='bilinear', align_corners=True)
        keypoint_scores = keypoint_scores.view(BW, weight_dim, num_patches)  # BW x 1 x n_patch

        return keypoint_coords, keypoint_scores, keypoint_desc
