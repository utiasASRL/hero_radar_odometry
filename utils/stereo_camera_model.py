from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from torchvision import transforms
from PIL import Image, ImageDraw


class StereoCameraModel(nn.Module):

    def __init__(self, cu, cv, f, b):
        super(StereoCameraModel, self).__init__()

        self.cu = cu  # Optical centre u coordinate
        self.cv = cv  # Optical centre v coordinate
        self.f = f  # Camera focal length
        self.b = b  # Stereo camera baseline

        # Matrices for camera model needed to projecting/reprojecting between
        # the camera and image frames
        M, Q = self.set_camera_model_matrices(self.cu, self.cv, self.f, self.b)
        self.register_buffer('M', M)
        self.register_buffer('Q', Q)
        self.count = 0

    def set_camera_model_matrices(self, cu, cv, f, b):

        # Matrix needed to project 3D points into stereo camera coordinates
        # [ul, vl, ur, vr]^T = (1/z) * M * [x, y, z, 1]^T (using left camera model)
        #
        # [f, 0, cu,      0]
        # [0, f, cv,      0]
        # [f, 0, cu, -f * b]
        # [0, f, cv,      0]
        #
        M = torch.tensor([[self.f, 0.0, self.cu, 0.0],
                          [0.0, self.f, self.cv, 0.0],
                          [self.f, 0.0, self.cu, -(self.f * self.b)],
                          [0.0, self.f, self.cv, 0.0]])

        # Matrix needed to transform image coordinates into 3D points
        # [x, y, z, 1] = (1/d) * Q * [ul, vl, d, 1]^T
        #
        # [b, 0, 0, -b * cu]
        # [0, b, 0, -b * cv]
        # [0, 0, 0,   f * b]
        # [0, 0, 1,       0]
        #
        Q = torch.tensor([[self.b, 0.0, 0.0, -(self.b * self.cu)],
                          [0.0, self.b, 0.0, -(self.b * self.cv)],
                          [0.0, 0.0, 0.0, self.f * self.b],
                          [0.0, 0.0, 1.0, 0.0]])

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
            trg_cam_coords: pixel coordinates defined in the target camera coordinates system -- [B, 4, H, W]
            M: Matrix needed to project 3D points into stereo camera coordinates
               [ul, vl, ur, vr]^T = (1/z) * M * [x, y, z, 1]^T -- [B, 4, 4]
            T: 6DoF transformation matrix from target to source
        Returns:
            array of [-1,1] coordinates for left and right images in the source camera coordinate system -- [B, 4, H, W]
        """
        batch_size, _, n_points = cam_coords.size()

        # [Ul, Vl, Ur, Vr] = M * [x, y, z, 1]^T
        img_coords = M.bmm(cam_coords)

        inv_z = 1.0 / cam_coords[:, 2, :].reshape(batch_size, 1, n_points)  # [B, 1, N], elementwise division
        img_coords = img_coords * inv_z  # [B, 4, N], elementwise multiplication

        return img_coords  # torch.stack([img_coords[:, 1, :], img_coords[:, 0, :], img_coords[:, 3, :], img_coords[:, 2, :]], dim=1) # [B, 4, N]

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
        '''if (torch.sum(torch.isnan(img_coords_norm)) > 0):
            print('Warning: Nan img_coords_norm.')
        if (torch.sum(torch.isinf(img_coords_norm)) > 0):
            print('Warning: Inf img_coords_norm.')
        print("Norm img coords out of [-1, 1]: {}".format(torch.sum(torch.abs(img_coords_norm) > 1.0)))'''

        point_disparities = F.grid_sample(disparity, img_coords_norm, mode='nearest',
                                          padding_mode='border')  # B x 1 x 1 x N
        point_disparities = point_disparities.reshape(batch_size, 1, n_points)  # [B, 1, N]
        '''if (torch.sum(torch.isnan(point_disparities)) > 0):
            print('Warning: Nan point_disparities.')
        if (torch.sum(torch.isinf(point_disparities)) > 0):
            print('Warning: Inf point_disparities.')
        if (torch.sum(point_disparities == 0.0) > 0):
            print('Warning: 0.0 in point_disparities.')'''

        disp_min = (self.f * self.b) / 400.0  # Farthers point 400 m
        disp_max = (self.f * self.b) / 0.1  # Closest point 0.1 m
        valid_points = (point_disparities >= disp_min) & (
                    point_disparities <= disp_max)  # [B, 1, N] point in range 0.1 to 333 m
        # print("min disp: {}".format(torch.min(point_disparities[valid_points])), flush=True)
        # print("max disp: {}".format(torch.max(point_disparities[valid_points])), flush=True)

        # print("Disp < thresh: {}".format(torch.sum(point_disparities <= 1e-8, dim=(1,2))))
        # print("Disp neg: {}".format(torch.sum(point_disparities < 0.0, dim=(1,2))))

        '''if self.count < 20:
            disp_img = disparity.detach().cpu()
            disp_img[disp_img <= 1e-8] = 0.0
            disp_img[disp_img > 1e-8] = 1.0
            to_img = transforms.ToPILImage()
            disp_img = to_img(disp_img[0, :, :].reshape(1, height, width))
            draw = ImageDraw.Draw(disp_img)
            for i in range(n_points):
                x = img_coords[0, i, 1]
                y = img_coords[0, i, 0]
                if valid_points[0, 0, i] == 1.0:
                    draw.line([x -5.0, y, x + 5.0, y], fill=128)
                    draw.line([x, y - 5.0, x, y + 5.0], fill=128)
                else:
                    draw.ellipse([x -5.0, y - 5.0, x + 5.0, y + 5.0], outline=128)
            del draw
            disp_img.save("/home/mgr/results/debug/disp_images/img_{}_bool.png".format(self.count), "png")
            disp_img = disparity.detach().cpu()
            disp_img[disp_img <= 1e-8] = 0.0
            disp_img = disp_img[0, :, :].reshape(1, height, width)
            disp_img  = disp_img / torch.max(disp_img)
            disp_img = to_img(disp_img)

            draw = ImageDraw.Draw(disp_img)
            for i in range(n_points):
                x = img_coords[0, i, 1]
                y = img_coords[0, i, 0]
                if (point_disparities[0, 0, i] >= 0.0) and (point_disparities[0, 0, i] <= 1e-8):
                    draw.line([x -5.0, y, x + 5.0, y], fill=128)
                    draw.line([x, y - 5.0, x, y + 5.0], fill=128)
            del draw

            disp_img.save("/home/mgr/results/debug/disp_images/img_{}.png".format(self.count), "png")
            self.count += 1'''

        # Create the [ul, vl, d, 1] vector
        ones = torch.ones(batch_size, n_points).type_as(disparity)
        uvd1_pixel_coords = torch.stack((img_coords[:, :, 0], img_coords[:, :, 1], point_disparities[:, 0, :], ones),
                                        dim=1)  # [B, 4, N]

        # [X, Y, Z, d]^T = Q * [ul, vl, d, 1]^T
        cam_coords = Q.bmm(uvd1_pixel_coords)  # [B, 4, N]

        # [x, y, z, 1]^T = (1/d) * [X, Y, Z, d]^T
        inv_disparity = (1.0 / point_disparities)  # Elementwise division
        '''if (torch.sum(torch.isnan(inv_disparity)) > 0):
            print('Warning: Nan inv_disparity.')
        if (torch.sum(torch.isinf(inv_disparity)) > 0):
            print('Warning: Inf inv_disparity.')'''

        cam_coords = cam_coords * inv_disparity  # Elementwise mulitplication
        if (torch.sum(torch.isnan(cam_coords)) > 0):
            print('Warning: Nan cam_coords.')
        if (torch.sum(torch.isinf(cam_coords)) > 0):
            print('Warning: Inf cam_coords.')

        # print("min z: {}".format(torch.min(cam_coords[:, 2, :].unsqueeze(1)[valid_points])), flush=True)
        # print("max z: {}".format(torch.max(cam_coords[:, 2, :].unsqueeze(1)[valid_points])), flush=True)

        return cam_coords, valid_points

    def camera_model(self, cam_coords):
        """
        Project 3D points into an image.
        Args:
            image_coords: 2D image coordinates given as [v, u] -- [B, N, 2]
            disparity: disparity map of the image -- [B, H, W]
        Returns:
            cam_coords: 3D points in camera frame given as [x, y, z, 1] -- [B, 4, N]
            valid_points: Boolean array indicating point validity -- [B, 1, N]
        """
        batch_size = cam_coords.size(0)

        # Expand fixed matrices to the correct batch size
        M = self.M.expand(batch_size, 4, 4)

        # Get the camera coordinates in the target frame
        img_coords = self.camera_to_image(cam_coords, M)  # [B,4,N]

        if (torch.sum(torch.isnan(img_coords)) > 0) or (torch.sum(torch.isinf(img_coords)) > 0):
            print('Warning: Nan or Inf values in image coordinate tensor.')

        return img_coords

    def inverse_camera_model(self, img_coords, disparity):
        """
        Inverse warp a source image to the target image plane.
        Args:
            image_coords: 2D image coordinates given as [v, u] -- [B, N, 2]
            disparity: disparity map of the image -- [B, H, W]
        Returns:
            cam_coords: 3D points in camera frame given as [x, y, z, 1] -- [B, 4, N]
            valid_points: Boolean array indicating point validity -- [B, 1, N]
        """

        batch_size, height, width = disparity.size()

        # Expand fixed matrices to the correct batch size
        Q = self.Q.expand(batch_size, 4, 4)

        # Get the camera coordinates in the target frame
        cam_coords, valid_points = self.image_to_camera(img_coords, disparity, Q)  # [B,4,N]

        if (torch.sum(torch.isnan(cam_coords)) > 0) or (torch.sum(torch.isinf(cam_coords)) > 0):
            print('Warning: Nan or Inf values in camera coordinate tensor.')

        return cam_coords, valid_points