import torch
import torch.nn.functional as F

class KeypointBlock(torch.nn.Module):

    def __init__(self, config):
        super(KeypointBlock, self).__init__()
        self.config = config
        # KeypointBlock parameters
        self.patch_height = config['networks']['keypoint_block']['patch_height']
        self.patch_width = config['networks']['keypoint_block']['patch_width']
        self.temperature = config['networks']['keypoint_block']['softmax_temp']
        # 2D coordinates
        v_coords, u_coords = torch.meshgrid([torch.arange(0, self.height), torch.arange(0, self.width)])
        v_coords = v_coords.unsqueeze(0).float()  # 1 x H x W
        u_coords = u_coords.unsqueeze(0).float()
        self.register_buffer('v_coords', v_coords)
        self.register_buffer('u_coords', u_coords)

    def forward(self, descriptors, detector_scores, weight_scores):

        N, _, self.height, self.width = descriptors.size()
        # 2D keypoints
        v_patches = F.unfold(self.v_coords.expand(N, 1, self.height, self.width),
                            kernel_size=(self.patch_height, self.patch_width),
                            stride=(self.patch_height, self.patch_width))      # N x num_patch_elements x num_patches
        u_patches = F.unfold(self.u_coords.expand(N, 1, self.height, self.width),
                            kernel_size=(self.patch_height, self.patch_width),
                            stride=(self.patch_height, self.patch_width))

        detector_patches = F.unfold(detector_scores, kernel_size=(self.patch_height, self.patch_width),
                                    stride=(self.patch_height, self.patch_width))  # N x num_patch_elements x num_patches

        softmax_attention = F.softmax(detector_patches/self.temperature, dim=1)  # N x num_patch_elements x num_patches

        expected_v = torch.sum(v_patches*softmax_attention, dim = 1)
        expected_u = torch.sum(u_patches*softmax_attention, dim = 1)
        keypoint_coords = torch.stack([expected_u, expected_v], dim = 2)

        if self.config['networks']['keypoint_block']['grid_sample']:
            # normalize 2d keypoints
            norm_keypoints2D = self.normalize_coords(keypoint_coords, N, self.width, self.height).unsqueeze(1)
            # compute keypoint descriptors
            keypoint_descs = F.grid_sample(descriptors, norm_keypoints2D, mode='bilinear', align_corners=False)
            keypoint_descs = keypoint_descs.reshape(N, descriptors.size(1), keypoint_coords.size(1))  # N x C x num_patches
            # compute keypoint weight scores
            keypoint_weights = F.grid_sample(weight_scores, norm_keypoints2D, mode='bilinear', align_corners=False)
            keypoint_weights = keypoint_weights.reshape(N, weight_scores.size(1), keypoint_coords.size(1)) # N x 1 x num_patches

        else:
            # all at once
            softmax_attention = softmax_attention.unsqueeze(1)

            desc_windows = F.unfold(descriptors,
                                    kernel_size=(self.patch_height, self.patch_width),
                                    stride=(self.patch_height, self.patch_width))
            desc_windows = desc_windows.view(N, descriptors.size(1), self.patch_height*self.patch_width, desc_windows.size(2))
            # N x C x num_patch_elements x num_patches
            keypoint_descs = torch.sum(desc_windows * softmax_attention, dim=2)    # B x C x num_patches

            weight_windows = F.unfold(weight_scores,
                                      kernel_size=(self.patch_height, self.patch_width),
                                      stride=(self.patch_height, self.patch_width))
            weight_windows = weight_windows.view(N, weight_scores.size(1), self.patch_height*self.patch_width, weight_windows.size(2))
            keypoint_weights = torch.sum(weight_windows * softmax_attention, dim=2)    # B x C x num_patches

        return keypoint_coords, keypoint_descs, keypoint_weights

    def normalize_coords(self, coords_2D, batch_size, width, height):
        # B x N x 2
        u_norm = (2 * coords_2D[:, :, 0].reshape(batch_size, -1) / (width - 1)) - 1
        v_norm = (2 * coords_2D[:, :, 1].reshape(batch_size, -1) / (height - 1)) - 1

        # WARNING: grid_sample expects the normalized coordinates as (u, v)
        return torch.stack([u_norm, v_norm], dim=2)  # B x num_patches x 2
