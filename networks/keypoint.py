import torch
import torch.nn.functional as F

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
        self.mask_threshold = config['networks']['keypoint_block']['mask_threshold']

    def forward(self, detector_scores, weight_scores, descriptors, mask):

        N, _, height, width = descriptors.size()
        _, _, h, w = mask.size()
        assert(height == h and width == w)

        v_coords, u_coords = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        v_coords = v_coords.unsqueeze(0).float()  # 1 x H x W
        u_coords = u_coords.unsqueeze(0).float()

        v_patches = F.unfold(v_coords.expand(N, 1, height, width),
                             kernel_size=(self.patch_size, self.patch_size),
                             stride=(self.patch_size, self.patch_size)).to(self.gpuid)  # N x patch_elems x num_patches
        u_patches = F.unfold(u_coords.expand(N, 1, height, width),
                             kernel_size=(self.patch_size, self.patch_size),
                             stride=(self.patch_size, self.patch_size)).to(self.gpuid)

        detector_patches = F.unfold(detector_scores, kernel_size=(self.patch_size, self.patch_size),
                                    stride=(self.patch_size, self.patch_size))  # N x patch_elements x num_patches
        mask_patches = F.unfold(mask, kernel_size=(self.patch_size, self.patch_size),
                                    stride=(self.patch_size, self.patch_size))  # N x patch_elements x num_patches

        patch_elems = detector_patches.size(1)
        mask_patches = torch.sum(mask_patches, dim=1) > patch_elems * self.mask_threshold

        softmax_attention = F.softmax(detector_patches / self.temperature, dim=1)  # N x patch_elements x num_patches

        expected_v = torch.sum(v_patches * softmax_attention, dim=1)
        expected_u = torch.sum(u_patches * softmax_attention, dim=1)
        keypoint_coords = torch.stack([expected_u, expected_v], dim=2)

        norm_keypoints2D = normalize_coords(keypoint_coords, width, height).unsqueeze(1)

        keypoint_desc = F.grid_sample(descriptors, norm_keypoints2D, mode='bilinear')
        keypoint_desc = keypoint_desc.reshape(N, descriptors.size(1), keypoint_coords.size(1))  # N x C x num_patches

        keypoint_scores = F.grid_sample(weight_scores, norm_keypoints2D, mode='bilinear')
        keypoint_scores = keypoint_scores.reshape(N, weight_scores.size(1), keypoint_coords.size(1))  # N x 1 x n_patch

        return keypoint_coords, keypoint_scores, keypoint_desc, mask_patches

def normalize_coords(coords_2D, width, height):
    """Normalizes coords_2D (B x N x 2) to be within [-1, 1] """
    batch_size = coords_2D.size(0)
    u_norm = (2 * coords_2D[:, :, 0].reshape(batch_size, -1) / (width - 1)) - 1
    v_norm = (2 * coords_2D[:, :, 1].reshape(batch_size, -1) / (height - 1)) - 1
    return torch.stack([u_norm, v_norm], dim=2)  # B x num_patches x 2
