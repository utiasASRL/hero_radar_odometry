from __future__ import division

from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from utils.lie_algebra import se3_inv

class StereoCameraModel(nn.Module):

    def __init__(self):
        super(StereoCameraModel, self).__init__()
        # Initialize focal length and base line to 0
        self.fl = 0.0
        self.b = 0.0

        # Matrices for camera model needed to projecting/reprojecting between
        # the camera and image frames, initialize with zeros.
        M, Q = self.set_camera_model_matrices(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.register_buffer('M', M)
        self.register_buffer('Q', Q)
        self.count = 0

    def set_camera_model_matrices(self, fl, cul, cvl, fr, cur, cvr, b):

        # Matrix needed to project 3D points into stereo camera coordinates
        # [ul, vl, ur, vr]^T = (1/z) * M * [x, y, z, 1]^T (using left camera model)
        #
        # [f, 0, cu,      0]
        # [0, f, cv,      0]
        # [f, 0, cu, -f * b]
        # [0, f, cv,      0]
        #
        M = torch.tensor([[fl, 0.0, cul, 0.0],
                          [0.0, fl, cvl, 0.0],
                          [fr, 0.0, cur, -(fl * b)],
                          [0.0, fr, cvr, 0.0]]).float()

        # Matrix needed to transform image coordinates into 3D points
        # [x, y, z, 1] = (1/d) * Q * [ul, vl, d, 1]^T
        #
        # [b, 0, 0, -b * cu]
        # [0, b, 0, -b * cv]
        # [0, 0, 0,   f * b]
        # [0, 0, 1,       0]
        #
        Q = torch.tensor([[b, 0.0, 0.0, -(b * cul)],
                          [0.0, b, 0.0, -(b * cvl)],
                          [0.0, 0.0, 0.0, fl * b],
                          [0.0, 0.0, 1.0, 0.0]]).float()

        return M, Q

    def normalize_coords(self, coords_2D, batch_size, width, height):
        # B x N x 2
        u_norm = (2 * coords_2D[:, :, 0].reshape(batch_size, -1) / (width - 1)) - 1
        v_norm = (2 * coords_2D[:, :, 1].reshape(batch_size, -1) / (height - 1)) - 1

        # WARNING: grid_sample expects the normalized coordinates as (u, v)
        return torch.stack([u_norm, v_norm], dim=2)  # B x N x 2

    def camera_to_image(self, cam_coords, M):
        """Transform coordinates in the camera frame to the pixel frame.
        Args:
            cam_coords: pixel coordinates defined in the target camera coordinates system -- [B, 3 or 4, N]
            M: Matrix needed to project 3D points into stereo camera coordinates
               [ul, vl, ur, vr]^T = (1/z) * M * [x, y, z, 1]^T -- [B, 4, 4]
        Returns:
            array of [u,v] coordinates for left and right images in the source camera coordinate system -- [B, 4, N]
        """
        batch_size, _, n_points = cam_coords.size()

        # [Ul, Vl, Ur, Vr] = M * [x, y, z, 1]^T
        assert cam_coords.size(1) == 4
        img_coords = M.bmm(cam_coords)

        inv_z = 1.0 / cam_coords[:, 2, :].reshape(batch_size, 1, n_points)  # [B, 1, N], elementwise division
        img_coords = img_coords * inv_z                                     # [B, 4, N], elementwise multiplication

        return img_coords

    def image_to_camera(self, img_coords, disparity, Q):
        """Transform coordinates in the pixel frame to the camera frame.
        Args:
            img_coords: image coordinates to transform -- [B, N, 2]
            disparity: disparity maps -- [B, H, W]
            Q: Matrix needed to transform image coordinates into 3D
               points [x, y, z, 1] = (1/d) * Q * [ul, vl, d, 1]^T  -- [B, 4, 4]
        Returns:
            array of (x,y,z) cam coordinates -- [B, 3, H, W]
        """
        batch_size, height, width = disparity.size()
        disparity = disparity.unsqueeze(1)
        n_points = img_coords.size(1)

        if (torch.sum(disparity == 0.0) > 0):
            print('Warning: 0.0 in point_disparities.')

        # Sample disparity for the image coordiates
        img_coords_norm = self.normalize_coords(img_coords, batch_size, width, height).unsqueeze(1)  # B x 1 x N x 2

        point_disparities = F.grid_sample(disparity, img_coords_norm, mode='nearest',
                                          padding_mode='border')  # B x 1 x 1 x N
        point_disparities = point_disparities.reshape(batch_size, 1, n_points)  # [B, 1, N]

        disp_min = (self.fl * self.b) / 600.0  # Farthest point 600 m
        disp_max = (self.fl * self.b) / 0.1    # Closest point 0.1 m (the max disp is 96, so closes point is actually 4m)
        valid_points = (point_disparities >= disp_min) & (point_disparities <= disp_max)  # [B, 1, N] point in range 0.1 to 333 m

        # Create the [ul, vl, d, 1] vector
        ones = torch.ones(batch_size, n_points).type_as(disparity)
        uvd1_pixel_coords = torch.stack((img_coords[:, :, 0], img_coords[:, :, 1], point_disparities[:, 0, :], ones),
                                        dim=1)  # [B, 4, N]

        # [X, Y, Z, d]^T = Q * [ul, vl, d, 1]^T
        cam_coords = Q.bmm(uvd1_pixel_coords)  # [B, 4, N]

        # [x, y, z, 1]^T = (1/d) * [X, Y, Z, d]^T
        inv_disparity = (1.0 / point_disparities)  # Elementwise division

        cam_coords = cam_coords * inv_disparity  # Elementwise mulitplication
        if (torch.sum(torch.isnan(cam_coords)) > 0):
            print('Warning: Nan cam_coords.')
        if (torch.sum(torch.isinf(cam_coords)) > 0):
            print('Warning: Inf cam_coords.')

        return cam_coords, valid_points

    def camera_model(self, cam_coords, cam_calib):
        """
        Project 3D points into an image.
        Args:
            cam_coords: 3D image coordinates given as [x, y, z] or [x, y, z, 1] -- [B, N, 3 or 4]
            cam_calib: transforms between camera frames, baseline and camera parameter matrices
        Returns:
            img_coords: 2D points in camera frame given as [u_l, v_l, u_r, v_r] -- [B, 4, N]
        """
        batch_size = cam_coords.size(0)

        K2, K3 = cam_calib['K2'].cuda(), cam_calib['K3'].cuda()
        T_c2_c0 = cam_calib['T_c2_c0'][:batch_size, :, :].float().cuda()
        T_c3_c0 = cam_calib['T_c3_c0'][:batch_size, :, :].float().cuda()
        self.b = abs(T_c3_c0.bmm(se3_inv(T_c2_c0))[0, 0, 3])
        self.fl = K2[0, 0, 0]

        if cam_coords.size(1) == 4:
            cam_coords = T_c2_c0.bmm(cam_coords)
        else:
            cam_coords = T_c2_c0[:, :, :3].bmm(cam_coords) + T_c2_c0[:, :, 3].unsqueeze(-1)

        self.M, self.Q =  self.set_camera_model_matrices(K2[0, 0, 0], K2[0, 0, 2], K2[0, 1, 2],
                                                         K3[0, 0, 0], K3[0, 0, 2], K3[0, 1, 2], self.b)

        # Expand fixed matrices to the correct batch size
        M = self.M.expand(batch_size, 4, 4).cuda()

        # Get the camera coordinates in the target frame
        img_coords = self.camera_to_image(cam_coords, M)  # [B,4,N]

        if (torch.sum(torch.isnan(img_coords)) > 0) or (torch.sum(torch.isinf(img_coords)) > 0):
                print('Warning: Nan or Inf values in image coordinate tensor.')

        return img_coords

    def inverse_camera_model(self, img_coords, disparity, cam_calib):
        """
        Inverse warp a source image to the target image plane.
        Args:
            image_coords: 2D image coordinates given as [v, u] -- [B, N, 2]
            disparity: disparity map of the image -- [B, H, W]
            cam_calib: transforms between camera frames, baseline and camera parameter matrices
        Returns:
            cam_coords: 3D points in camera frame given as [x, y, z] -- [B, 3, N]
            valid_points: Boolean array indicating point validity -- [B, 1, N]
        """

        batch_size, height, width = disparity.size()

        K2, K3 = cam_calib['K2'].cuda(), cam_calib['K3'].cuda()
        T_c2_c0 = cam_calib['T_c2_c0'][:batch_size, :, :].float().cuda()
        T_c3_c0 = cam_calib['T_c3_c0'][:batch_size, :, :].float().cuda()
        self.b = abs(T_c3_c0.bmm(se3_inv(T_c2_c0))[0, 0, 3])
        self.fl = K2[0, 0, 0]

        self.M, self.Q = self.set_camera_model_matrices(K2[0, 0, 0], K2[0, 0, 2], K2[0, 1, 2],
                                                        K3[0, 0, 0], K3[0, 0, 2], K3[0, 1, 2], self.b)

        # Expand fixed matrices to the correct batch size
        Q = self.Q.expand(batch_size, 4, 4).cuda()

        # Get the camera coordinates in the target frame
        cam_coords, valid_points = self.image_to_camera(img_coords, disparity, Q)  # [B,4,N]

        if (torch.sum(torch.isnan(cam_coords)) > 0) or (torch.sum(torch.isinf(cam_coords)) > 0):
            print('Warning: Nan or Inf values in camera coordinate tensor.')

        T_c2_c0 = T_c2_c0.expand(batch_size, 4, 4).cuda()
        cam_coords = se3_inv(T_c2_c0).bmm(cam_coords)

        return cam_coords[:, :3, :], valid_points