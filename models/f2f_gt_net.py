import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet import UNet

class F2FGTNet(nn.Module):
    def __init__(self, config):
        super(F2FGTNet, self).__init__()
        self.config = config

        self.window_h = 8
        self.window_w = 8
        self.temperature = 1.0
        self.window_size = config.window_size   # should always be 2
        self.batch_size = config.batch_size

        # encoder-decoder network
        self.net = UNet(config.in_feat, 1)

        # image coordinates
        height, width = 64, 822
        v_coords, u_coords = torch.meshgrid([torch.arange(0, height),
                                             torch.arange(0, width)])

        v_coords = v_coords.unsqueeze(0).float() # 1 x H x W
        u_coords = u_coords.unsqueeze(0).float()
        self.register_buffer('v_coords', v_coords)
        self.register_buffer('u_coords', u_coords)

        self.output_loss = 0


    def forward(self, vertex, T_iv):
        logits_pts, scores, features = self.net(vertex)

        batch_size, _, height, width = vertex.size()

        # 2D keypoints
        v_windows = F.unfold(self.v_coords.expand(batch_size, 1, height, width),
                                             kernel_size=(self.window_h, self.window_w),
                                             stride=(self.window_h, self.window_w))      # B x num_wind_elements x num_windows
        u_windows = F.unfold(self.u_coords.expand(batch_size, 1, height, width),
                                             kernel_size=(self.window_h, self.window_w),
                                             stride=(self.window_h, self.window_w))

        logit_pts_windows = F.unfold(logits_pts, kernel_size=(self.window_h, self.window_w),
                                     stride=(self.window_h, self.window_w))  # B x num_wind_elements x num_windows

        softmax_attention = F.softmax(logit_pts_windows / self.temperature, dim=1) # B x num_wind_elements x num_windows

        expected_v = torch.sum(v_windows * softmax_attention, dim=1) # B x num_windows
        expected_u = torch.sum(u_windows * softmax_attention, dim=1)
        keypoints_2D = torch.stack([expected_u, expected_v], dim=2)  # B x num_windows x 2

        # normalize 2d keypoints
        norm_keypoints2D = self.normalize_coords(keypoints_2D, batch_size, width, height).unsqueeze(1)

        # convert to 3D keypoints (should also compute with detector scores)
        # logit_pts_windows_3 = F.unfold(logits_pts.expand(batch_size, 3, height, width),
        #                                kernel_size=(self.window_h, self.window_w),
        #                                stride=(self.window_h, self.window_w))  # B x num_wind_elements x num_windows

        x_windows = F.unfold(vertex[:, 0:1, :, :],
                             kernel_size=(self.window_h, self.window_w),
                             stride=(self.window_h, self.window_w))
        y_windows = F.unfold(vertex[:, 1:2, :, :],
                             kernel_size=(self.window_h, self.window_w),
                             stride=(self.window_h, self.window_w))
        z_windows = F.unfold(vertex[:, 2:3, :, :],
                             kernel_size=(self.window_h, self.window_w),
                             stride=(self.window_h, self.window_w))

        expected_x = torch.sum(x_windows * softmax_attention, dim=1) # B x num_windows
        expected_y = torch.sum(y_windows * softmax_attention, dim=1) # B x num_windows
        expected_z = torch.sum(z_windows * softmax_attention, dim=1) # B x num_windows
        keypoints_3D = torch.stack([expected_x, expected_y, expected_z], dim=2)  # B x num_windows x 3

        # filter no return measurements
        # range = torch.sum(keypoints_3D ** 2, dim=2)

        # sample descriptors, weights of keypoints
        keypoints_scores = F.grid_sample(scores, norm_keypoints2D, mode='bilinear')
        keypoints_scores = keypoints_scores.reshape(batch_size, 1, keypoints_2D.size(1)) # B x C x N
        keypoints_scores = keypoints_scores.transpose(1, 2) # B x N x C

        keypoints_desc = F.grid_sample(features, norm_keypoints2D, mode='bilinear')
        keypoints_desc = keypoints_desc.reshape(batch_size, features.size(1), keypoints_2D.size(1))
        keypoints_desc = F.normalize(keypoints_desc, dim=1) # B x C x N

        # associate between frames (assume frame-to-frame)
        # batch_nr_ids = []
        pseudo_ref = []
        for batch_i in range(self.batch_size):
            # ref is previous frame
            # read is next frame
            # find pseudo ref
            ref_id = 2*batch_i
            read_id = 2*batch_i + 1

            # dot product of descriptors
            wij = keypoints_desc[read_id, :, :].transpose(0, 1)@keypoints_desc[ref_id, :, :]
            wij = torch.softmax(wij/50.0, dim=1)

            # TODO: consistency check

            # pseudos
            pseudo_ref.append( wij@keypoints_3D[ref_id, :, :])
            # keypoints_3D[ref_id, :, :] = wij@keypoints_3D[ref_id, :, :]
            # keypoints_scores[ref_id, :, :] = wij@keypoints_scores[ref_id, :, :]

        return keypoints_3D, keypoints_scores, pseudo_ref

    def normalize_coords(self, coords_2D, batch_size, width, height):
        # B x N x 2
        u_norm = (2 * coords_2D[:, :, 0].reshape(batch_size, -1) / (width - 1)) - 1
        v_norm = (2 * coords_2D[:, :, 1].reshape(batch_size, -1) / (height - 1)) - 1

        # WARNING: grid_sample expects the normalized coordinates as (u, v)
        return torch.stack([u_norm, v_norm], dim=2)  # B x N x 2

    def loss(self, keypoints_3D, keypoints_scores, T_iv, pseudo_ref):
        """
        Runs the loss on outputs of the model
        :param outputs: pointclouds
        :return: loss
        """

        self.output_loss = 0
        total_loss = 0
        for batch_i in range(self.batch_size):
            # ref is previous frame
            # read is next frame
            ref_id = 2*batch_i
            read_id = 2*batch_i + 1

            # get read points (1)
            points2 = keypoints_3D[read_id, :, :]

            # check for no return in read frame
            nr_ids = torch.nonzero(torch.sum(points2, dim=1), as_tuple=False).squeeze()
            points2 = points2[nr_ids, :]

            # get ref points
            # points1 = keypoints_3D[ref_id, nr_ids, :]
            points1 = pseudo_ref[batch_i][nr_ids, :]

            # get read scores
            weights = keypoints_scores[read_id, nr_ids]

            # get gt poses
            # T_i1 = T_iv[ref_id, :, :]
            # T_i2 = T_iv[read_id, :, :]
            T_21 = self.se3_inv(T_iv[read_id, :, :])@T_iv[ref_id, :, :]

            # outlier rejection
            points1_in_2 = points1@T_21[:3, :3].T + T_21[:3, 3].unsqueeze(0)
            error = torch.sum((points1_in_2 - points2) ** 2, dim=1)
            ids = torch.nonzero(error < 4.0 ** 2, as_tuple=False).squeeze()
            # ids = np.arange(points2.size(0))

            total_loss += self.weighted_mse_loss(points1_in_2[ids, :],
                                                 points2[ids, :],
                                                 weights[ids, :])

            # weights loss
            total_loss -= torch.mean(3*weights[ids, :])

        self.output_loss = total_loss
        return total_loss

    def weighted_mse_loss(self, data, target, weight):
        return 3.0*torch.mean(torch.exp(weight) * (data - target) ** 2)
        # return 3.0*torch.mean(weight * (data - target) ** 2)

        # return torch.mean(weight*torch.log(1.0 + torch.sum( (data-target) ** 2, 1).unsqueeze(1)))
        # u = weight * (data - target) ** 2
        # return torch.mean(u / (1 + u))
        # return torch.mean((data - target) ** 2)
        # return ((data-target)**2/data.numel()).sum()

        # e = (data - target).unsqueeze(-1)
        # e2 = e.transpose(1, 2)@W@e
        # return torch.mean(e2/(1.0 + e2))
        # return torch.mean(e.transpose(1, 2)@W@e/12)
        # return torch.mean(e.transpose(1, 2)@e)

    def se3_inv(self, Tf):
        Tinv = torch.zeros_like(Tf)
        Tinv[:3, :3] = Tf[:3, :3].T
        Tinv[:3, 3:] = -Tf[:3, :3].T@Tf[:3, 3:]
        Tinv[3, 3] = 1
        return Tinv