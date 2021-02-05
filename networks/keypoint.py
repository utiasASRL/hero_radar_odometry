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
        self.temperature = config['networks']['keypoint_block']['softmax_temp']
        self.gpuid = config['gpuid']
        self.grid_sample = config['networks']['keypoint_block']['grid_sample']
        self.width = config['cart_pixel_width']
        v_coords, u_coords = torch.meshgrid([torch.arange(0, self.width), torch.arange(0, self.width)])
        v_coords = v_coords.unsqueeze(0).float()  # 1 x H x W
        u_coords = u_coords.unsqueeze(0).float()
        self.coords = torch.cat((u_coords, v_coords), dim=0).to(self.gpuid)  # 2 x H x W
        BW = config['batch_size'] * config['window_size']
        self.v_patches = F.unfold(v_coords.expand(BW, 1, self.width, self.width),
                                  kernel_size=self.patch_size,
                                  stride=self.patch_size).to(self.gpuid)  # BW x patch_elems x num_patches
        self.u_patches = F.unfold(u_coords.expand(BW, 1, self.width, self.width),
                                  kernel_size=self.patch_size,
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
        softmax_attention = F.softmax(detector_patches / self.temperature, dim=1)  # BW x patch_elements x num_patches
        expected_v = torch.sum(self.v_patches * softmax_attention, dim=1)
        expected_u = torch.sum(self.u_patches * softmax_attention, dim=1)
        keypoint_coords = torch.stack([expected_u, expected_v], dim=2)  # BW x num_patches x 2
        num_patches = keypoint_coords.size(1)

        if self.grid_sample:
            norm_keypoints2D = normalize_coords(keypoint_coords, self.width, self.width).unsqueeze(1)

            keypoint_desc = F.grid_sample(descriptors, norm_keypoints2D, mode='bilinear')
            keypoint_desc = keypoint_desc.view(BW, encoder_dim, num_patches)  # BW x C x num_patches

            keypoint_scores = F.grid_sample(weight_scores, norm_keypoints2D, mode='bilinear')
            keypoint_scores = keypoint_scores.view(BW, weight_dim, num_patches)  # BW x 1 x n_patch
        else:
            softmax_attention = softmax_attention.unsqueeze(1)
            keypoint_desc = F.unfold(descriptors, kernel_size=self.patch_size, stride=self.patch_size)

            keypoint_desc = keypoint_desc.view(BW, encoder_dim, self.patch_size**2, keypoint_desc.size(2))
            keypoint_desc = torch.sum(keypoint_desc * softmax_attention, dim=2)    # BW x C x num_patches

            keypoint_scores = F.unfold(weight_scores, kernel_size=self.patch_size, stride=self.patch_size)

            keypoint_scores = keypoint_scores.view(BW, weight_dim, self.patch_size**2, keypoint_scores.size(2))
            keypoint_scores = torch.sum(keypoint_scores * softmax_attention, dim=2)    # BW x S x num_patches

        return keypoint_coords, keypoint_scores, keypoint_desc
